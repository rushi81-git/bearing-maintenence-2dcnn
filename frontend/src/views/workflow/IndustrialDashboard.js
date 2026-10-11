import React, { useState, useEffect, useRef, useMemo } from 'react';
import {
  Cpu,
  Upload,
  Activity,
  CalendarCheck,
  CheckCircle2,
  AlertTriangle,
  FileSpreadsheet,
  FileText,
  Wrench,
  Lock,
  ArrowRight,
  ShieldCheck,
  Zap,
  Info,
  Layers,
  ChevronRight,
  RefreshCw,
  Plus,
  Volume2,
  VolumeX,
  Gauge,
  Sliders,
  Maximize2,
  Download,
  Flame,
  Check,
  Terminal,
  Crosshair,
  Edit,
  Trash2,
  X,
  Save,
  Image as ImageIcon,
  Crop
} from 'lucide-react';
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
  LineChart,
  Line,
  AreaChart,
  Area
} from 'recharts';
import { parseVibrationCsv, parseVibrationExcel } from '../../utils/csvParser';
import { generateDiagnosticPDF } from '../../utils/pdfReportGenerator';
import { analyzeBearingSignal, createMachine, updateMachine, deleteMachine, createScheduleEntry } from '../../services/api';

// ── Web Audio Acoustic Synthesizer for Bearing Fault Simulation ──
function playBearingAcousticPulse(faultType = 'Inner Race', duration = 3.5) {
  try {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    const ctx = new AudioCtx();
    const now = ctx.currentTime;

    // Repetition frequency: Inner Race ~162 Hz, Outer Race ~107 Hz, Ball ~141 Hz, Normal ~29.5 Hz
    let repFreq = 29.5;
    let impactIntensity = 0.15;
    if (faultType === 'Inner Race') {
      repFreq = 162.2;
      impactIntensity = 0.7;
    } else if (faultType === 'Outer Race') {
      repFreq = 107.5;
      impactIntensity = 0.85;
    } else if (faultType === 'Ball') {
      repFreq = 141.2;
      impactIntensity = 0.6;
    }

    const period = 1 / repFreq;
    const numPulses = Math.min(Math.floor(duration / period), 250);

    for (let i = 0; i < numPulses; i++) {
      const pulseTime = now + (i * period);
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      // High-frequency ringing of bearing outer raceway resonance (~2800 Hz)
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(2800 + (Math.random() * 200 - 100), pulseTime);

      gain.gain.setValueAtTime(impactIntensity, pulseTime);
      gain.gain.exponentialRampToValueAtTime(0.0001, pulseTime + 0.008);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start(pulseTime);
      osc.stop(pulseTime + 0.009);
    }

    setTimeout(() => {
      ctx.close();
    }, (duration + 0.5) * 1000);

    return ctx;
  } catch (_) {
    return null;
  }
}

// ── Simple Fast Power Spectrum for Vibration FFT ──
function computePowerSpectrum(data, fs = 12000, maxFreq = 600) {
  if (!data || data.length < 256) return [];
  const N = Math.min(data.length, 512);
  const spectrum = [];
  const numBins = 60;
  const df = maxFreq / numBins;

  for (let k = 0; k < numBins; k++) {
    const freq = Math.round(k * df);
    let real = 0;
    let imag = 0;
    const omega = 2 * Math.PI * (freq / fs);

    // Discrete Fourier bin projection
    const step = 2;
    for (let n = 0; n < N; n += step) {
      const angle = omega * n;
      real += data[n] * Math.cos(angle);
      imag -= data[n] * Math.sin(angle);
    }
    const mag = Math.sqrt(real * real + imag * imag) / (N / 2);
    spectrum.push({
      freq,
      magnitude: Math.round(mag * 1000) / 1000
    });
  }
  return spectrum;
}

export function IndustrialDashboard({
  machines = [],
  onRefreshMachines,
  onSwitchToSchedule,
  showToast
}) {
  // ── Step 1 State ──
  const [selectedMachineId, setSelectedMachineId] = useState('');
  const [isStep1Confirmed, setIsStep1Confirmed] = useState(false);
  const [machineMode, setMachineMode] = useState('select'); // 'select' | 'add'
  const [newMachineForm, setNewMachineForm] = useState({
    name: '',
    type: 'Induction Motor',
    location: 'Floor Bay 1',
    rpm: '1772',
    bearing_model: 'SKF 6205-2RS (DE)'
  });
  const [isAddingMachine, setIsAddingMachine] = useState(false);

  // ── Edit / Delete State ──
  const [editingMachine, setEditingMachine] = useState(null); // machine object being edited
  const [editForm, setEditForm] = useState({ name: '', type: '', location: '' });
  const [isSavingEdit, setIsSavingEdit] = useState(false);
  const [confirmDeleteId, setConfirmDeleteId] = useState(null); // id awaiting delete confirm
  const [isDeletingMachine, setIsDeletingMachine] = useState(false);

  // ── Step 2 State ──
  const [inputMode, setInputMode] = useState('csv'); // 'csv' | 'image'
  const [bearingLocation, setBearingLocation] = useState('Drive End (DE)');
  const [parsedCsv, setParsedCsv] = useState(null);
  const [selectedColumnIndex, setSelectedColumnIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isStep2Done, setIsStep2Done] = useState(false);
  const [activeChartTab, setActiveChartTab] = useState('waveform'); // 'waveform' | 'spectrum'
  const fileInputRef = useRef(null);

  // 2D CNN FFT Graph Image States
  const [imageBase64, setImageBase64] = useState(null);
  const [imageCropBox, setImageCropBox] = useState(null);
  const [cropInputs, setCropInputs] = useState({ x1: '', y1: '', x2: '', y2: '' });
  const [showCropControls, setShowCropControls] = useState(false);
  const imageInputRef = useRef(null);

  // ── Step 3 & 4 State ──
  const [diagnosisResult, setDiagnosisResult] = useState(null);
  const [sourceFilename, setSourceFilename] = useState('');
  const [dispatchModalOpen, setDispatchModalOpen] = useState(false);
  const [dispatchSuccess, setDispatchSuccess] = useState(false);
  const [audioPlaying, setAudioPlaying] = useState(false);
  const [dispatchForm, setDispatchForm] = useState({
    task_type: 'Bearing Replacement & Overhaul',
    priority: 'High',
    due_date: '',
    technician: 'Bay 1 Lead Mechanical Specialist',
    notes: ''
  });

  // Auto-select first machine if available
  useEffect(() => {
    if (machines.length > 0 && !selectedMachineId) {
      setSelectedMachineId(String(machines[0].id));
    }
  }, [machines, selectedMachineId]);

  const selectedMachine =
    machines.find((m) => String(m.id) === String(selectedMachineId)) ||
    (machines.length > 0 ? machines[0] : null);

  useEffect(() => {
    const d = new Date();
    d.setDate(d.getDate() + 3);
    setDispatchForm((prev) => ({
      ...prev,
      due_date: d.toISOString().split('T')[0]
    }));
  }, []);

  // ── Handlers for Step 1 ──
  const handleConfirmStep1 = () => {
    if (!selectedMachine) {
      showToast('Please select or register a machine first.', 'error');
      return;
    }
    setIsStep1Confirmed(true);
    showToast(`Machine confirmed: ${selectedMachine.name}. Step 2 unlocked.`, 'success');
    setTimeout(() => {
      const step2El = document.getElementById('step-2-section');
      if (step2El) step2El.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }, 100);
  };

  const handleCreateMachine = async (e) => {
    e.preventDefault();
    if (!newMachineForm.name.trim()) {
      showToast('Machine Name is required', 'error');
      return;
    }
    setIsAddingMachine(true);
    try {
      const res = await createMachine({
        name: newMachineForm.name.trim(),
        type: newMachineForm.type,
        location: newMachineForm.location,
        operational_hours: 120,
        temperature: 42.5,
        vibration: 1.2
      });
      if (res.success) {
        showToast(`Machine "${newMachineForm.name}" registered successfully!`, 'success');
        if (onRefreshMachines) await onRefreshMachines();
        if (res.machine_id || res.id) {
          setSelectedMachineId(String(res.machine_id || res.id));
        }
        setMachineMode('select');
        setIsStep1Confirmed(true);
      } else {
        showToast(res.message || 'Failed to create machine', 'error');
      }
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsAddingMachine(false);
    }
  };

  // ── Edit Machine Handler ──
  const handleStartEdit = (m) => {
    setEditingMachine(m);
    setEditForm({
      name: m.name || '',
      type: m.machine_type || m.type || 'Induction Motor',
      location: m.location || ''
    });
    setMachineMode('select');
  };

  const handleUpdateMachine = async (e) => {
    e.preventDefault();
    if (!editForm.name.trim()) {
      showToast('Machine Name is required', 'error');
      return;
    }
    setIsSavingEdit(true);
    try {
      const res = await updateMachine(editingMachine.id, {
        name: editForm.name.trim(),
        type: editForm.type,
        location: editForm.location
      });
      if (res.success !== false) {
        showToast(`Machine "${editForm.name}" updated successfully!`, 'success');
        if (onRefreshMachines) await onRefreshMachines();
        setEditingMachine(null);
      } else {
        showToast(res.message || 'Failed to update machine', 'error');
      }
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsSavingEdit(false);
    }
  };

  // ── Delete Machine Handler ──
  const handleDeleteMachine = async (id) => {
    setIsDeletingMachine(true);
    try {
      const res = await deleteMachine(id);
      if (res.success !== false) {
        showToast('Machine deleted.', 'success');
        if (onRefreshMachines) await onRefreshMachines();
        if (String(selectedMachineId) === String(id)) setSelectedMachineId('');
        setConfirmDeleteId(null);
        setIsStep1Confirmed(false);
      } else {
        showToast(res.message || 'Failed to delete machine', 'error');
      }
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsDeletingMachine(false);
    }
  };

  // ── Handlers for Step 2 ──
  const handleFileProcess = async (file) => {
    if (!file) return;
    const name = file.name.toLowerCase();
    const isExcel = name.endsWith('.xlsx') || name.endsWith('.xls');
    const isCsv   = name.endsWith('.csv') || name.endsWith('.txt');
    if (!isExcel && !isCsv) {
      showToast('Please upload a CSV, TXT, XLSX, or XLS file', 'error');
      return;
    }
    try {
      let parsed;
      if (isExcel) {
        const buffer = await file.arrayBuffer();
        parsed = parseVibrationExcel(buffer, file.name);
      } else {
        const text = await file.text();
        parsed = parseVibrationCsv(text, file.name);
      }
      setParsedCsv(parsed);
      setSelectedColumnIndex(parsed.defaultColumnIndex || 0);
      setSourceFilename(file.name);
      setIsStep2Done(false);
      setDiagnosisResult(null);
      setDispatchSuccess(false);
      showToast(`Loaded ${parsed.columns[parsed.defaultColumnIndex]?.columnName || 'vibration'} (${parsed.totalDataPoints.toLocaleString()} samples)`, 'success');
    } catch (err) {
      showToast(err.message || 'Error parsing CSV file', 'error');
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileProcess(e.dataTransfer.files[0]);
    }
  };

  const handleImageProcess = (file) => {
    if (!file) return;
    const validTypes = ['image/png', 'image/jpeg', 'image/jpg', 'image/bmp', 'image/webp'];
    if (!validTypes.includes(file.type) && !file.name.match(/\.(png|jpe?g|bmp|webp)$/i)) {
      showToast('Please upload an image file (.png, .jpg, .bmp)', 'error');
      return;
    }
    const reader = new FileReader();
    reader.onload = (e) => {
      const b64 = e.target.result;
      setImageBase64(b64);
      setSourceFilename(file.name);
      setIsStep2Done(false);
      setDiagnosisResult(null);
      setDispatchSuccess(false);
      showToast(`Loaded FFT image: ${file.name}`, 'success');
    };
    reader.onerror = () => {
      showToast('Failed to read image file', 'error');
    };
    reader.readAsDataURL(file);
  };

  const handleImageDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleImageProcess(e.dataTransfer.files[0]);
    }
  };

  const handleClearImage = () => {
    setImageBase64(null);
    setImageCropBox(null);
    setCropInputs({ x1: '', y1: '', x2: '', y2: '' });
    if (imageInputRef.current) imageInputRef.current.value = '';
    setIsStep2Done(false);
    setDiagnosisResult(null);
  };

  const handleLoadSampleFftImage = () => {
    try {
      const canvas = document.createElement('canvas');
      canvas.width = 448;
      canvas.height = 300;
      const ctx = canvas.getContext('2d');

      // Clean white background
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      const margin = { top: 20, right: 20, bottom: 40, left: 50 };
      const w = canvas.width - margin.left - margin.right;
      const h = canvas.height - margin.top - margin.bottom;

      // Subtle frequency grid
      ctx.strokeStyle = '#e2e8f0';
      ctx.lineWidth = 1;
      for (let i = 0; i <= 5; i++) {
        const y = margin.top + (h / 5) * i;
        ctx.beginPath();
        ctx.moveTo(margin.left, y);
        ctx.lineTo(margin.left + w, y);
        ctx.stroke();
      }
      for (let i = 0; i <= 6; i++) {
        const x = margin.left + (w / 6) * i;
        ctx.beginPath();
        ctx.moveTo(x, margin.top);
        ctx.lineTo(x, margin.top + h);
        ctx.stroke();
      }

      // Axis labels
      ctx.fillStyle = '#64748b';
      ctx.font = '10px monospace';
      ctx.fillText('0 Hz', margin.left - 10, margin.top + h + 15);
      ctx.fillText('3000 Hz', margin.left + w / 2 - 20, margin.top + h + 15);
      ctx.fillText('6000 Hz', margin.left + w - 30, margin.top + h + 15);
      ctx.fillText('0.0 g', margin.left - 35, margin.top + h);
      ctx.fillText('1.5 g', margin.left - 35, margin.top + 10);

      // Synthesize 256-point FFT line plot with BPFI ~162.2 Hz harmonics
      const numPoints = 256;
      ctx.beginPath();
      ctx.strokeStyle = '#0284c7';
      ctx.lineWidth = 2;

      for (let i = 0; i < numPoints; i++) {
        const freq = (i / numPoints) * 6000;
        let amp = 0.04 + Math.random() * 0.04;

        const bpfi = 162.2;
        for (let harm = 1; harm <= 12; harm++) {
          const target = harm * bpfi;
          const diff = Math.abs(freq - target);
          if (diff < 25) {
            const peak = Math.max(0, 1 - diff / 25) * (1.1 / Math.sqrt(harm));
            amp += peak;
          }
        }
        if (freq > 2600 && freq < 3400) {
          amp += 0.25 * Math.sin((freq - 2600) / 800 * Math.PI) * (0.8 + Math.random() * 0.4);
        }

        amp = Math.min(1.5, amp);
        const x = margin.left + (i / (numPoints - 1)) * w;
        const y = margin.top + h - (amp / 1.5) * h;

        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();

      const b64 = canvas.toDataURL('image/png');
      setImageBase64(b64);
      setSourceFilename('CWRU_IR_007_FFT_Spectrum.png');
      setIsStep2Done(false);
      setDiagnosisResult(null);
      setDispatchSuccess(false);
      showToast('Loaded benchmark CWRU FFT Graph Image (Inner Race defect)', 'info');
    } catch (e) {
      showToast('Error generating benchmark FFT image', 'error');
    }
  };

  const handleApplyCrop = () => {
    const x1 = parseInt(cropInputs.x1, 10);
    const y1 = parseInt(cropInputs.y1, 10);
    const x2 = parseInt(cropInputs.x2, 10);
    const y2 = parseInt(cropInputs.y2, 10);
    if (!isNaN(x1) && !isNaN(y1) && !isNaN(x2) && !isNaN(y2) && x2 > x1 && y2 > y1) {
      setImageCropBox([x1, y1, x2, y2]);
      showToast(`Crop box applied: [${x1}, ${y1}, ${x2}, ${y2}]`, 'success');
    } else {
      setImageCropBox(null);
      showToast('Invalid crop coordinates — must satisfy x2 > x1 and y2 > y1', 'error');
    }
  };

  const handleRunDiagnosis = async () => {
    if (inputMode === 'image') {
      if (!imageBase64) {
        showToast('Please attach or drop an FFT vibration graph image first', 'error');
        return;
      }
      setIsAnalyzing(true);
      try {
        const cropBox = (imageCropBox && imageCropBox.length === 4) ? imageCropBox : null;
        const res = await analyzeBearingSignal({
          image_base64: imageBase64,
          crop_box: cropBox,
          input_type: 'fft_image',
          source_type: 'fft_graph_image',
          source_filename: sourceFilename || 'fft_graph_plot.png',
          machine_id: selectedMachine ? selectedMachine.id : 1,
          bearing_location: bearingLocation
        });

        if (res.success && res.data) {
          setDiagnosisResult(res.data);
          setIsStep2Done(true);
          showToast('2D CNN Graph AI Diagnosis complete!', 'success');
          setTimeout(() => {
            const step3El = document.getElementById('step-3-section');
            if (step3El) step3El.scrollIntoView({ behavior: 'smooth', block: 'start' });
          }, 120);
        } else {
          showToast(res.message || 'Diagnostic inference failed', 'error');
        }
      } catch (err) {
        showToast(err.message || 'Diagnosis execution error', 'error');
      } finally {
        setIsAnalyzing(false);
      }
      return;
    }

    if (!parsedCsv) {
      showToast('Please attach or drop a vibration CSV file first', 'error');
      return;
    }
    const col = parsedCsv.columns[selectedColumnIndex];
    if (!col || !col.data || col.data.length < 1024) {
      showToast('Need at least 1,024 vibration data points for valid diagnosis', 'error');
      return;
    }

    setIsAnalyzing(true);
    try {
      const fs = parsedCsv.detectedSamplingRate || 12000;
      const res = await analyzeBearingSignal({
        signal: col.data.slice(0, 4096),
        machine_id: selectedMachine ? selectedMachine.id : 1,
        sampling_rate_hz: fs,
        signal_unit: 'g',
        source_type: 'csv',
        source_filename: sourceFilename || 'vibration_telemetry.csv',
        model_mode: 'auto',
        use_classical_ml: false,
        bearing_location: bearingLocation
      });

      if (res.success && res.data) {
        setDiagnosisResult(res.data);
        setIsStep2Done(true);
        showToast('Industrial AI Diagnostic run complete!', 'success');
        setTimeout(() => {
          const step3El = document.getElementById('step-3-section');
          if (step3El) step3El.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }, 120);
      } else {
        showToast(res.message || 'Diagnostic inference failed', 'error');
      }
    } catch (err) {
      showToast(err.message || 'Diagnosis execution error', 'error');
    } finally {
      setIsAnalyzing(false);
    }
  };

  // Quick-load benchmark dataset (12kHz Inner Race defect with clean physics envelope)
  const handleLoadSampleDataset = () => {
    const n = 2048;
    const fs = 12000;
    const bpfi = 162.2;
    const rows = ['Sample_No,Time_s,Vibration_Acceleration_g,Fault_Type,Defect_Size_in,Sampling_Rate_Hz'];
    for (let i = 0; i < n; i++) {
      const t = i / fs;
      const carrier = Math.sin(2 * Math.PI * 3200 * t);
      const envelope = Math.pow(Math.max(0, Math.sin(2 * Math.PI * bpfi * t)), 8);
      const impact = 4.2 * envelope * carrier;
      const noise = (Math.random() - 0.5) * 0.35;
      const val = (impact + noise).toFixed(6);
      rows.push(`${i + 1},${t.toFixed(6)},${val},IR,0.007,12000.0`);
    }
    const csvContent = rows.join('\n');
    const parsed = parseVibrationCsv(csvContent, 'IR_007_12k_DriveEnd.csv');
    setParsedCsv(parsed);
    setSelectedColumnIndex(parsed.defaultColumnIndex || 0);
    setSourceFilename('IR_007_12k_DriveEnd.csv');
    setIsStep2Done(false);
    setDiagnosisResult(null);
    setDispatchSuccess(false);
    showToast('Loaded benchmark Inner Race (0.007") 12 kHz dataset', 'info');
  };

  // ── Handlers for Step 4 ──
  const handleDownloadPdf = () => {
    if (!diagnosisResult) return;
    generateDiagnosticPDF({
      diagnosis: diagnosisResult,
      machine: selectedMachine,
      filename: sourceFilename
    });
    showToast('Executive Diagnostic PDF Report downloaded', 'success');
  };

  const handleDispatchSubmit = async (e) => {
    e.preventDefault();
    if (!selectedMachine) return;
    try {
      const res = await createScheduleEntry({
        machine_id: selectedMachine.id,
        task_type: dispatchForm.task_type,
        task_description: `${dispatchForm.task_type} - ${diagnosisResult?.fault_type || 'Fault'} (${diagnosisResult?.fault_size_inches || '0.007'}") on ${bearingLocation}. Notes: ${dispatchForm.notes}`,
        priority: dispatchForm.priority,
        due_date: dispatchForm.due_date,
        assigned_technician: dispatchForm.technician
      });

      if (res.success) {
        setDispatchSuccess(true);
        setDispatchModalOpen(false);
        showToast('Work order successfully dispatched to Maintenance Schedule!', 'success');
      } else {
        showToast(res.message || 'Failed to dispatch work order', 'error');
      }
    } catch (err) {
      showToast(err.message || 'Error dispatching task', 'error');
    }
  };

  const handlePlayAudio = () => {
    if (audioPlaying) return;
    setAudioPlaying(true);
    playBearingAcousticPulse(diagnosisResult?.fault_type || 'Inner Race', 3.0);
    setTimeout(() => setAudioPlaying(false), 3200);
  };

  // Waveform preview points (150 samples)
  const waveformPreviewData = useMemo(() => {
    if (!parsedCsv || !parsedCsv.columns[selectedColumnIndex]) return [];
    const col = parsedCsv.columns[selectedColumnIndex];
    const data = col.data;
    const step = Math.max(1, Math.floor(data.length / 150));
    const points = [];
    for (let i = 0; i < data.length && points.length < 150; i += step) {
      points.push({ index: i, val: Math.round(data[i] * 1000) / 1000 });
    }
    return points;
  }, [parsedCsv, selectedColumnIndex]);

  // FFT Power Spectrum preview points
  const spectrumPreviewData = useMemo(() => {
    if (!parsedCsv || !parsedCsv.columns[selectedColumnIndex]) return [];
    const col = parsedCsv.columns[selectedColumnIndex];
    const fs = parsedCsv.detectedSamplingRate || 12000;
    return computePowerSpectrum(col.data, fs, 500);
  }, [parsedCsv, selectedColumnIndex]);

  // Overall condition calculations
  const isHealthy = diagnosisResult?.fault_type === 'Normal';
  const faultColor = isHealthy ? '#10d9a0' : (diagnosisResult?.severity === 'Severe' ? '#ef4444' : '#f59e0b');
  const healthScore = isHealthy ? 96 : Math.max(12, Math.round(100 - (diagnosisResult?.prediction_probability || 0.85) * 85));

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24, maxWidth: 1260, margin: '0 auto', width: '100%', paddingBottom: 80 }}>
      
      {/* ═════════════════════════════════════════════════════════════════════════ */}
      {/* MAXIMALIST TOP SCADA CONTROL ROOM STATUS RIBBON                           */}
      {/* ═════════════════════════════════════════════════════════════════════════ */}
      <div
        style={{
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-lg)',
          border: '1px solid var(--border-subtle)',
          padding: '18px 24px',
          boxShadow: 'var(--shadow-md)',
          position: 'relative',
          overflow: 'hidden'
        }}
      >
        {/* Top Ticker Row */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, marginBottom: 14 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <span
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 6,
                padding: '4px 10px',
                borderRadius: 'var(--radius-full)',
                background: 'rgba(16, 217, 160, 0.15)',
                border: '1px solid rgba(16, 217, 160, 0.4)',
                color: '#10d9a0',
                fontSize: 11,
                fontWeight: 800,
                fontFamily: 'var(--font-mono)',
                letterSpacing: '0.05em'
              }}
            >
              <span style={{ width: 7, height: 7, borderRadius: '50%', background: '#10d9a0', boxShadow: '0 0 8px #10d9a0' }} />
              SCADA ONLINE
            </span>
            <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
              Workshop Floor Bay Continuous Condition Monitoring
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
            <div>GRID SYNC: <strong style={{ color: '#38bdf8' }}>50.0 Hz</strong></div>
            <div>MOTOR POLES: <strong style={{ color: '#38bdf8' }}>4-Pole AC</strong></div>
            <div>STANDARDS: <strong style={{ color: '#818cf8' }}>ISO 10816-3 Class II</strong></div>
          </div>
        </div>

        {/* Stepper Pipeline Controls */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, paddingTop: 12, borderTop: '1px solid rgba(255,255,255,0.06)' }}>
          <div>
            <h1 style={{ fontSize: 22, fontWeight: 900, margin: 0, letterSpacing: '0.02em', textTransform: 'uppercase', color: 'var(--text-primary)' }}>
              Dual-Bearing Vibration Diagnostics Console
            </h1>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
              High-frequency impact envelope demodulation & neural fault classification
            </div>
          </div>

          {/* Stepper Pills */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 14px',
                borderRadius: 0,
                fontSize: 12,
                fontWeight: 800,
                background: isStep1Confirmed ? 'var(--status-healthy-bg)' : 'var(--bg-surface-raised)',
                color: isStep1Confirmed ? 'var(--status-healthy)' : 'var(--text-primary)',
                border: `2px solid ${isStep1Confirmed ? 'var(--status-healthy)' : 'var(--card-border-color)'}`,
                boxShadow: 'var(--shadow-xs)'
              }}
            >
              {isStep1Confirmed ? <CheckCircle2 size={14} /> : <span>01</span>}
              <span>Machine Asset</span>
            </div>

            <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 14px',
                borderRadius: 0,
                fontSize: 12,
                fontWeight: 800,
                background: isStep2Done ? 'var(--status-healthy-bg)' : isStep1Confirmed ? 'var(--bg-surface-raised)' : 'var(--bg-surface-sunken)',
                color: isStep2Done ? 'var(--status-healthy)' : isStep1Confirmed ? 'var(--text-primary)' : 'var(--text-muted)',
                border: `2px solid ${isStep2Done ? 'var(--status-healthy)' : isStep1Confirmed ? 'var(--card-border-color)' : 'var(--border-subtle)'}`,
                boxShadow: isStep1Confirmed ? 'var(--shadow-xs)' : 'none'
              }}
            >
              {isStep2Done ? <CheckCircle2 size={14} /> : isStep1Confirmed ? <span>02</span> : <Lock size={12} />}
              <span>CSV Ingestion</span>
            </div>

            <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 14px',
                borderRadius: 0,
                fontSize: 12,
                fontWeight: 800,
                background: isStep2Done ? 'var(--status-healthy-bg)' : 'var(--bg-surface-sunken)',
                color: isStep2Done ? 'var(--status-healthy)' : 'var(--text-muted)',
                border: `2px solid ${isStep2Done ? 'var(--status-healthy)' : 'var(--border-subtle)'}`,
                boxShadow: isStep2Done ? 'var(--shadow-xs)' : 'none'
              }}
            >
              {isStep2Done ? <CheckCircle2 size={14} /> : <Lock size={12} />}
              <span>Diagnostic Bar Graphs</span>
            </div>

            <ChevronRight size={14} style={{ color: 'var(--text-muted)' }} />

            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 6,
                padding: '6px 14px',
                borderRadius: 0,
                fontSize: 12,
                fontWeight: 800,
                background: dispatchSuccess ? 'var(--status-healthy-bg)' : isStep2Done ? 'var(--bg-surface-raised)' : 'var(--bg-surface-sunken)',
                color: dispatchSuccess ? 'var(--status-healthy)' : isStep2Done ? 'var(--text-primary)' : 'var(--text-muted)',
                border: `2px solid ${dispatchSuccess ? 'var(--status-healthy)' : isStep2Done ? 'var(--card-border-color)' : 'var(--border-subtle)'}`,
                boxShadow: dispatchSuccess || isStep2Done ? 'var(--shadow-xs)' : 'none'
              }}
            >
              {dispatchSuccess ? <CheckCircle2 size={14} /> : <span>04</span>}
              <span>Work Order Dispatch</span>
            </div>
          </div>
        </div>
      </div>

      {/* ═════════════════════════════════════════════════════════════════════════ */}
      {/* STEP 1: SELECT OR ADD MACHINE ASSET                                       */}
      {/* ═════════════════════════════════════════════════════════════════════════ */}
      <section
        id="step-1-section"
        style={{
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-lg)',
          border: isStep1Confirmed ? '3px solid var(--status-healthy-border)' : '3px solid var(--card-border-color)',
          overflow: 'hidden',
          boxShadow: 'var(--shadow-card)',
          position: 'relative'
        }}
      >
        {/* Step 1 Header */}
        <div
          style={{
            padding: '18px 24px',
            background: isStep1Confirmed ? 'var(--status-healthy-bg)' : 'var(--bg-surface-raised)',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 12
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 0,
                background: isStep1Confirmed ? '#10d9a0' : '#38bdf8',
                color: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 900,
                fontSize: 16,
                boxShadow: 'var(--shadow-xs)'
              }}
            >
              {isStep1Confirmed ? <CheckCircle2 size={20} /> : '01'}
            </div>
            <div>
              <h2 style={{ fontSize: 17, fontWeight: 800, margin: 0 }}>
                Step 1: Machinery Asset Selection & Bearing Specifications
              </h2>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Target motor mechanical parameters, load zone, and bearing housing model numbers
              </span>
            </div>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            {isStep1Confirmed && (
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 800,
                  color: '#10d9a0',
                  background: 'rgba(16, 217, 160, 0.12)',
                  padding: '5px 14px',
                  borderRadius: 'var(--radius-full)',
                  border: '1px solid rgba(16, 217, 160, 0.4)',
                  fontFamily: 'var(--font-mono)'
                }}
              >
                CONFIGURED: {selectedMachine?.name}
              </span>
            )}
            <button
              type="button"
              onClick={() => setIsStep1Confirmed(false)}
              style={{
                fontSize: 12,
                color: '#38bdf8',
                background: 'transparent',
                border: 'none',
                cursor: 'pointer',
                fontWeight: 700,
                display: isStep1Confirmed ? 'inline-block' : 'none'
              }}
            >
              Change Machinery
            </button>
          </div>
        </div>

        {/* Step 1 Body */}
        <div style={{ padding: '24px' }}>
          
          {/* Sub-mode switcher */}
          <div style={{ display: 'flex', gap: 14, marginBottom: 22 }}>
            <button
              type="button"
              id="tab-choose-machine"
              onClick={() => setMachineMode('select')}
              style={{
                flex: 1,
                padding: '14px 18px',
                borderRadius: 'var(--radius-md)',
                border: machineMode === 'select' ? '3px solid var(--text-primary)' : '2px solid var(--card-border-color)',
                background: machineMode === 'select' ? 'var(--text-primary)' : 'var(--bg-surface)',
                color: machineMode === 'select' ? 'var(--bg-surface)' : 'var(--text-secondary)',
                fontWeight: 800,
                fontSize: 14,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 10,
                boxShadow: machineMode === 'select' ? 'var(--shadow-md)' : 'var(--shadow-xs)'
              }}
            >
              <Cpu size={20} />
              <span>Choose Enrolled Machine ({machines.length})</span>
            </button>

            <button
              type="button"
              id="tab-add-machine"
              onClick={() => setMachineMode('add')}
              style={{
                flex: 1,
                padding: '14px 18px',
                borderRadius: 'var(--radius-md)',
                border: machineMode === 'add' ? '3px solid var(--text-primary)' : '2px solid var(--card-border-color)',
                background: machineMode === 'add' ? 'var(--text-primary)' : 'var(--bg-surface)',
                color: machineMode === 'add' ? 'var(--bg-surface)' : 'var(--text-secondary)',
                fontWeight: 800,
                fontSize: 14,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: 10,
                boxShadow: machineMode === 'add' ? 'var(--shadow-md)' : 'var(--shadow-xs)'
              }}
            >
              <Plus size={20} />
              <span>+ Register New Workshop Machine</span>
            </button>
          </div>

          {/* Mode A: Enrolled Machine Grid with Detailed Mechanical Badges */}
          {machineMode === 'select' && (
            <div>
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fill, minmax(290px, 1fr))',
                  gap: 18,
                  marginBottom: 20
                }}
              >
                {machines.map((m) => {
                  const isSelected = String(m.id) === String(selectedMachineId);
                  const isConfirmingDelete = String(confirmDeleteId) === String(m.id);
                  return (
                    <div
                      key={m.id}
                      id={`machine-card-${m.id}`}
                      onClick={() => { if (!isConfirmingDelete) setSelectedMachineId(String(m.id)); }}
                      style={{
                        padding: '18px 20px',
                        borderRadius: 'var(--radius-md)',
                        border: isSelected ? '3px solid var(--text-primary)' : isConfirmingDelete ? '3px solid #ef4444' : '2px solid var(--card-border-color)',
                        background: isConfirmingDelete ? 'rgba(239,68,68,0.06)' : isSelected ? 'var(--bg-surface-raised)' : 'var(--bg-surface)',
                        cursor: isConfirmingDelete ? 'default' : 'pointer',
                        transition: 'all 0.18s ease',
                        position: 'relative',
                        boxShadow: isSelected ? 'var(--shadow-md)' : 'var(--shadow-xs)'
                      }}
                    >
                      {/* Row 1: type badge + active badge + action buttons */}
                      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
                        <span
                          style={{
                            fontSize: 11,
                            fontWeight: 800,
                            textTransform: 'uppercase',
                            padding: '3px 9px',
                            borderRadius: 4,
                            background: isSelected ? 'var(--text-primary)' : 'var(--bg-surface-sunken)',
                            color: isSelected ? 'var(--bg-surface)' : 'var(--text-secondary)',
                            fontFamily: 'var(--font-mono)'
                          }}
                        >
                          {m.machine_type || m.type || 'Induction Motor'}
                        </span>
                        {/* Edit / Delete action buttons */}
                        <div
                          style={{ display: 'flex', gap: 6 }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <button
                            type="button"
                            title="Edit machine"
                            onClick={() => handleStartEdit(m)}
                            style={{
                              padding: '4px 8px',
                              borderRadius: 0,
                              border: '1.5px solid var(--card-border-color)',
                              background: 'var(--bg-surface)',
                              color: 'var(--text-secondary)',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 4,
                              fontSize: 11,
                              fontWeight: 700
                            }}
                          >
                            <Edit size={12} /> Edit
                          </button>
                          <button
                            type="button"
                            title="Delete machine"
                            onClick={() => setConfirmDeleteId(isConfirmingDelete ? null : m.id)}
                            style={{
                              padding: '4px 8px',
                              borderRadius: 0,
                              border: '1.5px solid #ef4444',
                              background: isConfirmingDelete ? '#ef4444' : 'transparent',
                              color: isConfirmingDelete ? '#fff' : '#ef4444',
                              cursor: 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 4,
                              fontSize: 11,
                              fontWeight: 700
                            }}
                          >
                            <Trash2 size={12} /> {isConfirmingDelete ? 'Cancel' : 'Delete'}
                          </button>
                        </div>
                      </div>

                      {/* Delete confirmation prompt */}
                      {isConfirmingDelete && (
                        <div
                          style={{
                            marginBottom: 12,
                            padding: '10px 12px',
                            background: 'rgba(239,68,68,0.1)',
                            border: '1.5px solid #ef4444',
                            borderRadius: 'var(--radius-sm)',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            gap: 10
                          }}
                          onClick={(e) => e.stopPropagation()}
                        >
                          <span style={{ fontSize: 12, fontWeight: 700, color: '#ef4444' }}>
                            Permanently delete this machine?
                          </span>
                          <button
                            type="button"
                            disabled={isDeletingMachine}
                            onClick={() => handleDeleteMachine(m.id)}
                            style={{
                              padding: '5px 14px',
                              borderRadius: 0,
                              border: '2px solid #ef4444',
                              background: '#ef4444',
                              color: '#fff',
                              fontWeight: 800,
                              fontSize: 12,
                              cursor: isDeletingMachine ? 'wait' : 'pointer',
                              display: 'flex',
                              alignItems: 'center',
                              gap: 6
                            }}
                          >
                            {isDeletingMachine ? <RefreshCw size={12} className="spin" /> : <Trash2 size={12} />}
                            Confirm Delete
                          </button>
                        </div>
                      )}

                      <div style={{ fontSize: 17, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 4 }}>
                        {m.name}
                      </div>

                      <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginBottom: 14 }}>
                        📍 {m.location || 'Bay 1 - Induction Motor Stand'}
                      </div>

                      <div
                        style={{
                          padding: '10px 12px',
                          background: 'rgba(0,0,0,0.3)',
                          borderRadius: 'var(--radius-sm)',
                          fontSize: 11,
                          fontFamily: 'var(--font-mono)',
                          color: 'var(--text-secondary)',
                          lineHeight: 1.6,
                          border: '1px solid rgba(255,255,255,0.04)'
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>Rated Speed:</span>
                          <strong style={{ color: 'var(--text-primary)' }}>1772 RPM (2 HP)</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>Drive End (DE):</span>
                          <strong style={{ color: '#38bdf8' }}>SKF 6205-2RS JEM</strong>
                        </div>
                        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                          <span>Fan End (FE):</span>
                          <strong style={{ color: '#818cf8' }}>SKF 6203 Deep Groove</strong>
                        </div>
                      </div>

                      {isSelected && (
                        <span
                          style={{
                            position: 'absolute',
                            bottom: 14,
                            right: 14,
                            color: '#38bdf8',
                            display: 'flex',
                            alignItems: 'center',
                            gap: 4,
                            fontSize: 11,
                            fontWeight: 800
                          }}
                        >
                          <CheckCircle2 size={14} /> ACTIVE
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>

              {/* ── Inline Edit Panel ── */}
              {editingMachine && (
                <form
                  onSubmit={handleUpdateMachine}
                  style={{
                    marginTop: 20,
                    background: 'var(--bg-surface-raised)',
                    padding: '22px 24px',
                    borderRadius: 'var(--radius-md)',
                    border: '2px solid var(--text-primary)',
                    boxShadow: 'var(--shadow-md)'
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16 }}>
                    <h3 style={{ fontSize: 15, fontWeight: 800, margin: 0, color: '#38bdf8', textTransform: 'uppercase', letterSpacing: 1 }}>
                      Edit: {editingMachine.name}
                    </h3>
                    <button
                      type="button"
                      onClick={() => setEditingMachine(null)}
                      style={{
                        background: 'transparent',
                        border: 'none',
                        cursor: 'pointer',
                        color: 'var(--text-secondary)',
                        display: 'flex',
                        alignItems: 'center'
                      }}
                    >
                      <X size={18} />
                    </button>
                  </div>

                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 16, marginBottom: 16 }}>
                    <div>
                      <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Machine Name *</label>
                      <input
                        type="text"
                        required
                        value={editForm.name}
                        onChange={(e) => setEditForm({ ...editForm, name: e.target.value })}
                        style={{
                          width: '100%',
                          padding: '10px 12px',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border-subtle)',
                          background: 'var(--bg-base)',
                          color: 'var(--text-primary)',
                          fontSize: 14
                        }}
                      />
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Equipment Category</label>
                      <select
                        value={editForm.type}
                        onChange={(e) => setEditForm({ ...editForm, type: e.target.value })}
                        style={{
                          width: '100%',
                          padding: '10px 12px',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border-subtle)',
                          background: 'var(--bg-base)',
                          color: 'var(--text-primary)',
                          fontSize: 14
                        }}
                      >
                        <option value="Induction Motor">Induction Motor (Dual Bearing)</option>
                        <option value="Centrifugal Pump">Centrifugal Pump</option>
                        <option value="CNC Lathe Spindle">CNC Lathe Spindle</option>
                        <option value="Air Compressor">Air Compressor</option>
                        <option value="Gearbox Shaft">Gearbox Input Shaft</option>
                      </select>
                    </div>

                    <div>
                      <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>Workshop Bay / Location</label>
                      <input
                        type="text"
                        value={editForm.location}
                        onChange={(e) => setEditForm({ ...editForm, location: e.target.value })}
                        style={{
                          width: '100%',
                          padding: '10px 12px',
                          borderRadius: 'var(--radius-sm)',
                          border: '1px solid var(--border-subtle)',
                          background: 'var(--bg-base)',
                          color: 'var(--text-primary)',
                          fontSize: 14
                        }}
                      />
                    </div>
                  </div>

                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
                    <button
                      type="button"
                      onClick={() => setEditingMachine(null)}
                      style={{
                        padding: '10px 18px',
                        borderRadius: 'var(--radius-sm)',
                        border: '1px solid var(--border-subtle)',
                        background: 'transparent',
                        color: 'var(--text-secondary)',
                        cursor: 'pointer',
                        fontWeight: 700
                      }}
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      disabled={isSavingEdit}
                      style={{
                        padding: '10px 22px',
                        borderRadius: 0,
                        border: '2px solid var(--card-border-color)',
                        background: 'var(--text-primary)',
                        color: 'var(--bg-surface)',
                        fontWeight: 800,
                        cursor: isSavingEdit ? 'wait' : 'pointer',
                        display: 'flex',
                        alignItems: 'center',
                        gap: 8,
                        boxShadow: 'var(--shadow-btn)'
                      }}
                    >
                      {isSavingEdit ? <RefreshCw size={15} className="spin" /> : <Save size={15} />}
                      Save Changes
                    </button>
                  </div>
                </form>
              )}
            </div>
          )}

          {/* Mode B: Add New Machine Form */}
          {machineMode === 'add' && (
            <form
              onSubmit={handleCreateMachine}
              style={{
                background: 'var(--bg-surface-raised)',
                padding: '24px',
                borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-subtle)',
                marginBottom: 20
              }}
            >
              <h3 style={{ fontSize: 16, fontWeight: 800, marginTop: 0, marginBottom: 16, color: '#38bdf8' }}>
                Register Machinery Baseline Specs
              </h3>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 16, marginBottom: 16 }}>
                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>
                    Machine Name / Identifier *
                  </label>
                  <input
                    type="text"
                    required
                    placeholder="e.g. Induction Motor - Test Stand 3"
                    value={newMachineForm.name}
                    onChange={(e) => setNewMachineForm({ ...newMachineForm, name: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '11px 13px',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)',
                      background: 'var(--bg-base)',
                      color: 'var(--text-primary)',
                      fontSize: 14
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>
                    Equipment Category
                  </label>
                  <select
                    value={newMachineForm.type}
                    onChange={(e) => setNewMachineForm({ ...newMachineForm, type: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '11px 13px',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)',
                      background: 'var(--bg-base)',
                      color: 'var(--text-primary)',
                      fontSize: 14
                    }}
                  >
                    <option value="Induction Motor">Induction Motor (Dual Bearing)</option>
                    <option value="Centrifugal Pump">Centrifugal Pump</option>
                    <option value="CNC Lathe Spindle">CNC Lathe Spindle</option>
                    <option value="Air Compressor">Air Compressor</option>
                    <option value="Gearbox Shaft">Gearbox Input Shaft</option>
                  </select>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 6 }}>
                    Workshop Bay / Location
                  </label>
                  <input
                    type="text"
                    placeholder="e.g. Bay 2 - Machining Line"
                    value={newMachineForm.location}
                    onChange={(e) => setNewMachineForm({ ...newMachineForm, location: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '11px 13px',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)',
                      background: 'var(--bg-base)',
                      color: 'var(--text-primary)',
                      fontSize: 14
                    }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
                <button
                  type="button"
                  onClick={() => setMachineMode('select')}
                  style={{
                    padding: '11px 18px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                    background: 'transparent',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontWeight: 700
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isAddingMachine}
                  style={{
                    padding: '11px 22px',
                    borderRadius: 0,
                    border: '2px solid var(--card-border-color)',
                    background: 'var(--text-primary)',
                    color: 'var(--bg-surface)',
                    fontWeight: 800,
                    cursor: isAddingMachine ? 'wait' : 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    boxShadow: 'var(--shadow-btn)'
                  }}
                >
                  {isAddingMachine ? <RefreshCw size={16} className="spin" /> : <Plus size={16} />}
                  <span>Save Machine & Continue</span>
                </button>
              </div>
            </form>
          )}

          {/* Confirm Machine Button */}
          <div
            style={{
              paddingTop: 18,
              borderTop: '1px solid var(--border-subtle)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              flexWrap: 'wrap',
              gap: 14
            }}
          >
            <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
              Selected Target: <strong style={{ color: 'var(--text-primary)' }}>{selectedMachine?.name || 'None'}</strong> ({selectedMachine?.location || 'Floor Bay'})
            </div>
            <button
              type="button"
              id="btn-confirm-step1"
              onClick={handleConfirmStep1}
              style={{
                padding: '14px 28px',
                borderRadius: 0,
                background: 'var(--text-primary)',
                color: 'var(--bg-surface)',
                border: '2px solid var(--card-border-color)',
                fontWeight: 800,
                fontSize: 14,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                boxShadow: 'var(--shadow-btn)'
              }}
            >
              <span>Confirm Machine & Unlock CSV Ingestion (Step 2)</span>
              <ArrowRight size={18} />
            </button>
          </div>
        </div>
      </section>

      {/* ═════════════════════════════════════════════════════════════════════════ */}
      {/* STEP 2: CSV TELEMETRY & INTERACTIVE MOTOR SCHEMATIC                      */}
      {/* ═════════════════════════════════════════════════════════════════════════ */}
      <section
        id="step-2-section"
        style={{
          background: 'var(--bg-surface)',
          borderRadius: 'var(--radius-lg)',
          border: isStep2Done
            ? '2px solid rgba(16, 217, 160, 0.6)'
            : isStep1Confirmed
            ? '2px solid #38bdf8'
            : '1px solid var(--border-subtle)',
          overflow: 'hidden',
          opacity: isStep1Confirmed ? 1 : 0.65,
          pointerEvents: isStep1Confirmed ? 'auto' : 'none',
          transition: 'all 0.25s ease',
          boxShadow: isStep1Confirmed ? 'var(--shadow-card)' : 'none'
        }}
      >
        {/* Step 2 Header */}
        <div
          style={{
            padding: '18px 24px',
            background: isStep2Done ? 'var(--status-healthy-bg)' : 'var(--bg-surface-raised)',
            borderBottom: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 12
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 0,
                background: isStep2Done ? 'var(--status-healthy)' : isStep1Confirmed ? 'var(--text-primary)' : 'var(--bg-surface-sunken)',
                color: '#ffffff',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 900,
                fontSize: 16,
                boxShadow: 'var(--shadow-xs)'
              }}
            >
              {isStep2Done ? <CheckCircle2 size={20} /> : isStep1Confirmed ? '02' : <Lock size={16} />}
            </div>
            <div>
              <h2 style={{ fontSize: 18, fontWeight: 800, margin: 0, letterSpacing: '-0.01em' }}>
                Step 2: Vibration Telemetry & FFT Graph Ingestion
              </h2>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                Target bearing accelerometer acquisition point, ingest raw vibration telemetry or 2D FFT graph images
              </span>
            </div>
          </div>

          {!isStep1Confirmed && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: '#fbbf24', fontWeight: 700 }}>
              <Lock size={14} /> Complete Step 1 First
            </div>
          )}
        </div>

        {/* Step 2 Body */}
        <div style={{ padding: '24px' }}>
          {/* ── Mode Switcher: CSV Time-Series vs 2D FFT Graph Image ── */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 10,
              marginBottom: 20,
              background: 'var(--bg-surface-raised)',
              padding: '6px',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-subtle)',
              width: 'fit-content',
              flexWrap: 'wrap'
            }}
          >
            <button
              type="button"
              id="tab-csv-mode"
              onClick={() => setInputMode('csv')}
              style={{
                padding: '8px 18px',
                borderRadius: 'var(--radius-sm)',
                border: 'none',
                background: inputMode === 'csv' ? 'var(--text-primary)' : 'transparent',
                color: inputMode === 'csv' ? 'var(--bg-surface)' : 'var(--text-secondary)',
                fontSize: 13,
                fontWeight: 800,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                transition: 'all 0.15s ease'
              }}
            >
              <FileSpreadsheet size={16} />
              <span>CSV Vibration Telemetry (1D CNN)</span>
            </button>

            <button
              type="button"
              id="tab-image-mode"
              onClick={() => setInputMode('image')}
              style={{
                padding: '8px 18px',
                borderRadius: 'var(--radius-sm)',
                border: inputMode === 'image' ? '1px solid #6366f1' : 'none',
                background: inputMode === 'image' ? '#6366f1' : 'transparent',
                color: inputMode === 'image' ? '#ffffff' : 'var(--text-secondary)',
                fontSize: 13,
                fontWeight: 800,
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: 8,
                boxShadow: inputMode === 'image' ? '0 0 14px rgba(99, 102, 241, 0.4)' : 'none',
                transition: 'all 0.15s ease'
              }}
            >
              <ImageIcon size={16} />
              <span>FFT Graph Image (2D CNN Vision)</span>
              <span
                style={{
                  fontSize: 10,
                  fontWeight: 900,
                  background: inputMode === 'image' ? '#ffffff' : '#6366f1',
                  color: inputMode === 'image' ? '#6366f1' : '#ffffff',
                  padding: '2px 7px',
                  borderRadius: 'var(--radius-full)'
                }}
              >
                4-CLASS
              </span>
            </button>
          </div>

          {inputMode === 'csv' && (
            <>
              {/* ── CSV Upload Dropzone & Benchmark Loader ── */}
              <div style={{ marginBottom: 22 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10 }}>
              <label style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-primary)' }}>
                Ingest Vibration Telemetry Data (CSV / Excel):
              </label>
              <button
                type="button"
                id="btn-load-sample"
                onClick={handleLoadSampleDataset}
                style={{
                  fontSize: 12,
                  fontWeight: 800,
                  color: '#38bdf8',
                  background: 'rgba(56, 189, 248, 0.15)',
                  border: '1px solid #38bdf8',
                  borderRadius: 'var(--radius-full)',
                  padding: '6px 14px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  boxShadow: '0 0 10px rgba(56, 189, 248, 0.2)'
                }}
              >
                <Zap size={14} />
                <span>Load Benchmark CWRU 12kHz Inner Race CSV</span>
              </button>
            </div>

            <input
              type="file"
              ref={fileInputRef}
              accept=".csv,.txt,.xlsx,.xls"
              style={{ display: 'none' }}
              onChange={(e) => {
                if (e.target.files && e.target.files[0]) {
                  handleFileProcess(e.target.files[0]);
                }
              }}
            />

            <div
              id="csv-dropzone"
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: isDragging ? '3px dashed var(--text-primary)' : '3px dashed var(--card-border-color)',
                background: isDragging ? 'var(--accent-subtle)' : 'var(--bg-surface-sunken)',
                borderRadius: 'var(--radius-lg)',
                padding: '38px 24px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: 'none'
              }}
            >
              <div
                style={{
                  width: 58,
                  height: 58,
                  borderRadius: '50%',
                  background: parsedCsv ? 'rgba(16, 217, 160, 0.2)' : 'rgba(56, 189, 248, 0.15)',
                  color: parsedCsv ? '#10d9a0' : '#38bdf8',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 12px',
                  boxShadow: parsedCsv ? '0 0 20px rgba(16, 217, 160, 0.4)' : '0 0 20px rgba(56, 189, 248, 0.3)'
                }}
              >
                {parsedCsv ? <CheckCircle2 size={32} /> : <Upload size={32} />}
              </div>

              <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 4 }}>
                {parsedCsv ? `Loaded File: ${sourceFilename}` : 'Click to Browse or Drag & Drop Vibration CSV / Excel File'}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', maxWidth: 540, margin: '0 auto' }}>
                Supports <code>.csv</code>, <code>.txt</code>, and <code>.xlsx / .xls</code> files. Auto-parser filters out row index counters (<code>Sample_No</code>, <code>Time_s</code>) and extracts vibration acceleration telemetry.
              </div>
            </div>
          </div>

          {/* ── Telemetry Matrix & Dual Charts (Waveform + Frequency Spectrum) ── */}
          {parsedCsv && (
            <div
              style={{
                background: 'var(--bg-surface-raised)',
                borderRadius: 'var(--radius-md)',
                padding: '20px',
                border: '1px solid var(--border-subtle)',
                marginBottom: 22
              }}
            >
              {/* Density Metrics Matrix */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 12, marginBottom: 18 }}>
                <div style={{ background: 'var(--bg-base)', padding: '12px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Sampling Rate</span>
                  <span style={{ fontSize: 16, fontWeight: 900, color: '#38bdf8', fontFamily: 'var(--font-mono)' }}>
                    {parsedCsv.detectedSamplingRate ? `${parsedCsv.detectedSamplingRate.toLocaleString()} Hz` : '12,000 Hz'}
                  </span>
                </div>

                <div style={{ background: 'var(--bg-base)', padding: '12px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Target Channel</span>
                  <span style={{ fontSize: 15, fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                    {parsedCsv.columns[selectedColumnIndex]?.columnName || 'Vibration_g'}
                  </span>
                </div>

                <div style={{ background: 'var(--bg-base)', padding: '12px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Buffer Points</span>
                  <span style={{ fontSize: 16, fontWeight: 900, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>
                    {parsedCsv.totalDataPoints.toLocaleString()} pts
                  </span>
                </div>

                <div style={{ background: 'var(--bg-base)', padding: '12px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>DC Offset Drift</span>
                  <span style={{ fontSize: 15, fontWeight: 800, color: '#10d9a0', fontFamily: 'var(--font-mono)' }}>
                    Zeroed (0.000 g)
                  </span>
                </div>

                {parsedCsv.detectedDefectSize && (
                  <div style={{ background: 'var(--bg-base)', padding: '12px 14px', borderRadius: 8, border: '1px solid rgba(255,255,255,0.05)' }}>
                    <span style={{ fontSize: 11, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Continuous Defect</span>
                    <span style={{ fontSize: 15, fontWeight: 900, color: '#fb923c', fontFamily: 'var(--font-mono)' }}>
                      {parsedCsv.detectedDefectSize}" ({Math.round(parsedCsv.detectedDefectSize * 25.4 * 1000) / 1000} mm)
                    </span>
                  </div>
                )}
              </div>

              {/* Chart Mode Tabs: Waveform vs Frequency Spectrum */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 12 }}>
                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    type="button"
                    onClick={() => setActiveChartTab('waveform')}
                    style={{
                      padding: '6px 14px',
                      borderRadius: 'var(--radius-sm)',
                      border: 'none',
                      background: activeChartTab === 'waveform' ? '#38bdf8' : 'rgba(255,255,255,0.06)',
                      color: activeChartTab === 'waveform' ? '#060b14' : 'var(--text-secondary)',
                      fontSize: 12,
                      fontWeight: 800,
                      cursor: 'pointer'
                    }}
                  >
                    Time-Domain Waveform (g)
                  </button>

                  <button
                    type="button"
                    onClick={() => setActiveChartTab('spectrum')}
                    style={{
                      padding: '6px 14px',
                      borderRadius: 'var(--radius-sm)',
                      border: 'none',
                      background: activeChartTab === 'spectrum' ? '#38bdf8' : 'rgba(255,255,255,0.06)',
                      color: activeChartTab === 'spectrum' ? '#060b14' : 'var(--text-secondary)',
                      fontSize: 12,
                      fontWeight: 800,
                      cursor: 'pointer'
                    }}
                  >
                    Frequency Spectrum FFT (Hz)
                  </button>
                </div>

                <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                  {activeChartTab === 'waveform' ? '150-sample telemetry envelope' : 'Spectral peaks: BPFI ~162.2 Hz | 2X ~324 Hz'}
                </span>
              </div>

              {/* Interactive Telemetry Chart */}
              <div style={{ width: '100%', height: 180, background: 'var(--bg-base)', borderRadius: 8, padding: '10px 0', border: '1px solid rgba(255,255,255,0.04)' }}>
                <ResponsiveContainer width="100%" height="100%">
                  {activeChartTab === 'waveform' ? (
                    <LineChart data={waveformPreviewData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="index" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} unit="g" />
                      <Tooltip
                        contentStyle={{ background: '#0d1525', border: '1px solid rgba(56, 189, 248, 0.4)', borderRadius: 8, fontSize: 12 }}
                        formatter={(val) => [`${val} g`, 'Acceleration']}
                      />
                      <Line type="monotone" dataKey="val" stroke="#38bdf8" dot={false} strokeWidth={2} isAnimationActive={false} />
                    </LineChart>
                  ) : (
                    <AreaChart data={spectrumPreviewData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                      <XAxis dataKey="freq" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} unit="Hz" />
                      <YAxis tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
                      <Tooltip
                        contentStyle={{ background: '#0d1525', border: '1px solid rgba(56, 189, 248, 0.4)', borderRadius: 8, fontSize: 12 }}
                        formatter={(val) => [`${val}`, 'Amplitude']}
                      />
                      <Area type="monotone" dataKey="magnitude" stroke="#818cf8" fill="rgba(129, 140, 248, 0.25)" isAnimationActive={false} />
                    </AreaChart>
                  )}
                </ResponsiveContainer>
              </div>
            </div>
          )}
        </>
      )}

      {/* ── Mode 2: FFT Graph Image Dropzone & Ingestion ── */}
      {inputMode === 'image' && (
        <div style={{ marginBottom: 22 }} className="animate-fade-in">
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10, flexWrap: 'wrap', gap: 10 }}>
            <div>
              <label style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-primary)' }}>
                Ingest FFT Vibration Graph Image (.png / .jpg / .bmp):
              </label>
              <span style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block', marginTop: 2 }}>
                Diagnostic Bandwidth: 0 to 6,000 Hz · Linear Amplitude in g · Line Graph
              </span>
            </div>

            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              <button
                type="button"
                id="btn-load-sample-image"
                onClick={handleLoadSampleFftImage}
                style={{
                  fontSize: 12,
                  fontWeight: 800,
                  color: '#a855f7',
                  background: 'rgba(168, 85, 247, 0.15)',
                  border: '1px solid #a855f7',
                  borderRadius: 'var(--radius-full)',
                  padding: '6px 14px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  boxShadow: '0 0 10px rgba(168, 85, 247, 0.2)'
                }}
              >
                <Zap size={14} />
                <span>Load Benchmark CWRU FFT Graph Sample</span>
              </button>

              {imageBase64 && (
                <button
                  type="button"
                  onClick={handleClearImage}
                  style={{
                    fontSize: 12,
                    fontWeight: 800,
                    color: '#ef4444',
                    background: 'rgba(239, 68, 68, 0.12)',
                    border: '1px solid #ef4444',
                    borderRadius: 'var(--radius-full)',
                    padding: '6px 14px',
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6
                  }}
                >
                  <Trash2 size={14} />
                  <span>Clear Image</span>
                </button>
              )}
            </div>
          </div>

          <input
            type="file"
            ref={imageInputRef}
            accept=".png,.jpg,.jpeg,.bmp,.webp,image/*"
            style={{ display: 'none' }}
            onChange={(e) => {
              if (e.target.files && e.target.files[0]) {
                handleImageProcess(e.target.files[0]);
              }
            }}
          />

          {!imageBase64 ? (
            <div
              id="image-dropzone"
              onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={handleImageDrop}
              onClick={() => imageInputRef.current?.click()}
              style={{
                border: isDragging ? '3px dashed #6366f1' : '3px dashed var(--card-border-color)',
                background: isDragging ? 'rgba(99, 102, 241, 0.1)' : 'var(--bg-surface-sunken)',
                borderRadius: 'var(--radius-lg)',
                padding: '38px 24px',
                textAlign: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease',
                boxShadow: 'none'
              }}
            >
              <div
                style={{
                  width: 58,
                  height: 58,
                  borderRadius: '50%',
                  background: 'rgba(99, 102, 241, 0.15)',
                  color: '#6366f1',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  margin: '0 auto 12px',
                  boxShadow: '0 0 20px rgba(99, 102, 241, 0.3)'
                }}
              >
                <ImageIcon size={32} />
              </div>

              <div style={{ fontSize: 16, fontWeight: 800, color: 'var(--text-primary)', marginBottom: 4 }}>
                Click to Browse or Drag & Drop FFT Vibration Graph Image
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', maxWidth: 540, margin: '0 auto' }}>
                Supports <code>.png</code>, <code>.jpg</code>, <code>.bmp</code> image exports from vibration analyzers or oscilloscope FFT screens. The 2D CNN preprocessor auto-crops and letterboxes to 224×224 pixels.
              </div>
              <div style={{ marginTop: 12, display: 'inline-block', fontSize: 11, fontWeight: 800, background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', padding: '4px 12px', borderRadius: 4, fontFamily: 'var(--font-mono)' }}>
                4-Class 2D CNN · Healthy, Inner Race Fault, Outer Race Fault, Ball Fault
              </div>
            </div>
          ) : (
            /* Image Preview & Crop Controls Card */
            <div
              style={{
                background: 'var(--bg-surface-raised)',
                borderRadius: 'var(--radius-md)',
                padding: '20px',
                border: '1px solid var(--border-subtle)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14, flexWrap: 'wrap', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                  <div style={{ width: 10, height: 10, borderRadius: '50%', background: '#10d9a0', boxShadow: '0 0 8px #10d9a0' }} />
                  <div>
                    <span style={{ fontSize: 14, fontWeight: 800, color: 'var(--text-primary)' }}>
                      Loaded Image: {sourceFilename}
                    </span>
                    <span style={{ fontSize: 12, color: 'var(--text-muted)', display: 'block' }}>
                      Ready for 2D CNN inference · Preprocessing pipeline will normalize to 224×224
                    </span>
                  </div>
                </div>

                <div style={{ display: 'flex', gap: 8 }}>
                  <button
                    type="button"
                    onClick={() => setShowCropControls(!showCropControls)}
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: 'var(--text-secondary)',
                      background: showCropControls ? 'var(--bg-surface-sunken)' : 'transparent',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '5px 12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6
                    }}
                  >
                    <Crop size={14} />
                    <span>{showCropControls ? 'Hide Crop Box' : 'Set Crop Area (Optional)'}</span>
                  </button>

                  <button
                    type="button"
                    onClick={() => imageInputRef.current?.click()}
                    style={{
                      fontSize: 12,
                      fontWeight: 700,
                      color: 'var(--text-primary)',
                      background: 'var(--bg-surface)',
                      border: '1px solid var(--border-subtle)',
                      borderRadius: 'var(--radius-sm)',
                      padding: '5px 12px',
                      cursor: 'pointer',
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6
                    }}
                  >
                    <RefreshCw size={12} />
                    <span>Change Image</span>
                  </button>
                </div>
              </div>

              {/* Optional Manual Crop Controls */}
              {showCropControls && (
                <div
                  style={{
                    background: 'var(--bg-base)',
                    padding: '12px 16px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-strong)',
                    marginBottom: 16
                  }}
                >
                  <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
                    Specify Plot Bounding Box (pixels) to exclude analyzer chrome:
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
                    <input
                      type="number"
                      placeholder="X1"
                      value={cropInputs.x1}
                      onChange={(e) => setCropInputs({ ...cropInputs, x1: e.target.value })}
                      style={{ width: 70, padding: '5px 8px', fontSize: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', borderRadius: 4 }}
                    />
                    <input
                      type="number"
                      placeholder="Y1"
                      value={cropInputs.y1}
                      onChange={(e) => setCropInputs({ ...cropInputs, y1: e.target.value })}
                      style={{ width: 70, padding: '5px 8px', fontSize: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', borderRadius: 4 }}
                    />
                    <input
                      type="number"
                      placeholder="X2"
                      value={cropInputs.x2}
                      onChange={(e) => setCropInputs({ ...cropInputs, x2: e.target.value })}
                      style={{ width: 70, padding: '5px 8px', fontSize: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', borderRadius: 4 }}
                    />
                    <input
                      type="number"
                      placeholder="Y2"
                      value={cropInputs.y2}
                      onChange={(e) => setCropInputs({ ...cropInputs, y2: e.target.value })}
                      style={{ width: 70, padding: '5px 8px', fontSize: 12, background: 'var(--bg-surface)', border: '1px solid var(--border-subtle)', color: 'var(--text-primary)', borderRadius: 4 }}
                    />
                    <button
                      type="button"
                      onClick={handleApplyCrop}
                      style={{ padding: '5px 12px', fontSize: 12, fontWeight: 700, background: '#6366f1', color: '#fff', border: 'none', borderRadius: 4, cursor: 'pointer' }}
                    >
                      Apply Crop
                    </button>
                    {imageCropBox && (
                      <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: '#10d9a0' }}>
                        Active: [{imageCropBox.join(', ')}]
                      </span>
                    )}
                  </div>
                </div>
              )}

              {/* Display Image Preview */}
              <div
                style={{
                  background: '#ffffff',
                  borderRadius: 'var(--radius-sm)',
                  border: '2px solid var(--border-strong)',
                  padding: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  maxHeight: 340,
                  overflow: 'hidden'
                }}
              >
                <img
                  src={imageBase64}
                  alt="Loaded FFT Graph Preview"
                  style={{
                    maxWidth: '100%',
                    maxHeight: 300,
                    objectFit: 'contain'
                  }}
                />
              </div>

              {/* Analyzer Specifications Badge */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(170px, 1fr))', gap: 10, marginTop: 14 }}>
                <div style={{ background: 'var(--bg-base)', padding: '10px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Frequency Span</span>
                  <span style={{ fontSize: 14, fontWeight: 800, color: '#6366f1', fontFamily: 'var(--font-mono)' }}>0 – 6,000 Hz</span>
                </div>
                <div style={{ background: 'var(--bg-base)', padding: '10px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Amplitude Axis</span>
                  <span style={{ fontSize: 14, fontWeight: 800, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)' }}>Linear (g peak)</span>
                </div>
                <div style={{ background: 'var(--bg-base)', padding: '10px 12px', borderRadius: 6, border: '1px solid rgba(255,255,255,0.05)' }}>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)', display: 'block', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>Neural Target</span>
                  <span style={{ fontSize: 14, fontWeight: 800, color: '#10d9a0', fontFamily: 'var(--font-mono)' }}>224×224 Conv2D</span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Action Button: Execute AI Diagnosis */}
      <div
        style={{
          paddingTop: 18,
          borderTop: '1px solid var(--border-subtle)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 14
        }}
      >
        <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
          Target: <strong style={{ color: 'var(--text-primary)' }}>{bearingLocation}</strong> on <strong style={{ color: 'var(--text-primary)' }}>{selectedMachine?.name}</strong>
        </div>

        <button
          type="button"
          id="btn-execute-diagnosis"
          disabled={(inputMode === 'csv' && !parsedCsv) || (inputMode === 'image' && !imageBase64) || isAnalyzing}
          onClick={handleRunDiagnosis}
          style={{
            padding: '14px 32px',
            borderRadius: 0,
            background: ((inputMode === 'csv' && parsedCsv) || (inputMode === 'image' && imageBase64))
              ? (inputMode === 'image' ? '#6366f1' : 'var(--text-primary)')
              : 'var(--bg-surface-raised)',
            color: ((inputMode === 'csv' && parsedCsv) || (inputMode === 'image' && imageBase64))
              ? '#ffffff'
              : 'var(--text-muted)',
            border: '2px solid var(--card-border-color)',
            fontWeight: 900,
            fontSize: 15,
            cursor: ((inputMode === 'csv' && parsedCsv) || (inputMode === 'image' && imageBase64)) && !isAnalyzing ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            gap: 10,
            boxShadow: ((inputMode === 'csv' && parsedCsv) || (inputMode === 'image' && imageBase64)) ? 'var(--shadow-btn)' : 'none'
          }}
        >
          {isAnalyzing ? (
            <>
              <RefreshCw size={20} className="spin" />
              <span>{inputMode === 'image' ? 'Running 2D CNN Inference...' : 'Computing Harmonic Envelopes...'}</span>
            </>
          ) : (
            <>
              {inputMode === 'image' ? <ImageIcon size={20} /> : <Activity size={20} />}
              <span>{inputMode === 'image' ? 'Execute 2D CNN Graph AI Diagnosis (Step 3)' : 'Execute Industrial AI Diagnosis (Step 3)'}</span>
              <ArrowRight size={18} />
            </>
          )}
        </button>
      </div>
        </div>
      </section>

      {/* ═════════════════════════════════════════════════════════════════════════ */}
      {/* STEP 3: DIAGNOSTIC AUDIT & BAR GRAPHS (ON SAME PAGE)                     */}
      {/* ═════════════════════════════════════════════════════════════════════════ */}
      <section
        id="step-3-section"
        style={{
          background: 'var(--bg-surface)',
          borderRadius: 0,
          border: isStep2Done ? `3px solid ${faultColor}` : '3px solid var(--card-border-color)',
          overflow: 'hidden',
          opacity: isStep2Done ? 1 : 0.65,
          pointerEvents: isStep2Done ? 'auto' : 'none',
          transition: 'all 0.25s ease',
          boxShadow: isStep2Done ? 'var(--shadow-card)' : 'none'
        }}
      >
        {/* Step 3 Header */}
        <div
          style={{
            padding: '18px 24px',
            background: isStep2Done ? (isHealthy ? 'var(--status-healthy-bg)' : 'var(--status-critical-bg)') : 'var(--bg-surface-raised)',
            borderBottom: '2px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 12
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
            <div
              style={{
                width: 36,
                height: 36,
                borderRadius: 0,
                background: isStep2Done ? faultColor : 'var(--text-muted)',
                color: '#ffffff',
                border: '2px solid #000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontWeight: 900,
                fontSize: 16,
                boxShadow: isStep2Done ? 'var(--shadow-xs)' : 'none'
              }}
            >
              {isStep2Done ? (isHealthy ? <CheckCircle2 size={20} /> : <AlertTriangle size={20} />) : <Lock size={16} />}
            </div>
            <div>
              <h2 style={{ fontSize: 17, fontWeight: 900, textTransform: 'uppercase', margin: 0, color: 'var(--text-primary)' }}>
                Step 3: Machine Health Assessment & Vibration Severity (Bar Graphs)
              </h2>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Simple, readable fault probability distribution & ISO 10816-3 mechanical severity zones
              </span>
            </div>
          </div>

          {!isStep2Done && (
            <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, color: 'var(--status-warning)', fontWeight: 800 }}>
              <Lock size={14} /> COMPLETE STEP 2 FIRST
            </div>
          )}
        </div>

        {/* Step 3 Body */}
        {diagnosisResult && (
          <div style={{ padding: '24px' }}>
            
            {/* ── Status Hero Banner with Acoustic Sound Player ── */}
            <div
              style={{
                padding: '24px',
                borderRadius: 0,
                background: isHealthy ? 'var(--status-healthy-bg)' : 'var(--status-critical-bg)',
                border: `3px solid ${faultColor}`,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 20,
                marginBottom: 24,
                boxShadow: 'var(--shadow-card)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 18 }}>
                <div
                  style={{
                    width: 58,
                    height: 58,
                    borderRadius: 0,
                    border: '2px solid #000',
                    background: faultColor,
                    color: '#ffffff',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    flexShrink: 0,
                    boxShadow: 'var(--shadow-xs)'
                  }}
                >
                  {isHealthy ? <CheckCircle2 size={36} /> : <AlertTriangle size={36} />}
                </div>
                <div>
                  <div style={{ fontSize: 12, fontWeight: 900, textTransform: 'uppercase', letterSpacing: '0.08em', color: faultColor }}>
                    {isHealthy ? 'System Health Normal' : `${diagnosisResult.severity.toUpperCase()} FAULT DETECTED`}
                  </div>
                  <h3 style={{ fontSize: 24, fontWeight: 900, margin: '2px 0 4px', color: 'var(--text-primary)' }}>
                    {diagnosisResult.fault_type === 'Inner Race'
                      ? 'Inner Race Fault (BPFI Harmonic Impact Pattern)'
                      : diagnosisResult.fault_type === 'Ball'
                      ? 'Ball Bearing Element Fault (BSF Impact)'
                      : diagnosisResult.fault_type === 'Outer Race'
                      ? 'Outer Race Fault (BPFO Impact)'
                      : 'Healthy Operation (Baseline Dynamic Balance)'}
                  </h3>
                  <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                    Asset: <strong>{selectedMachine?.name}</strong> • Bearing Housing: <strong>{bearingLocation}</strong>
                  </div>
                </div>
              </div>

              {/* Confidence Score & Audio Signature Button */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 10 }}>
                <div
                  style={{
                    fontSize: 16,
                    fontWeight: 900,
                    padding: '8px 18px',
                    borderRadius: 0,
                    border: '2px solid #000',
                    background: faultColor,
                    color: '#060b14',
                    fontFamily: 'var(--font-mono)',
                    boxShadow: 'var(--shadow-xs)'
                  }}
                >
                  {(diagnosisResult.prediction_probability * 100).toFixed(1)}% AI Confidence
                </div>

                {/* Acoustic Sound Player Button */}
                <button
                  type="button"
                  id="btn-play-sound"
                  onClick={handlePlayAudio}
                  style={{
                    padding: '8px 14px',
                    borderRadius: 0,
                    background: audioPlaying ? 'var(--text-primary)' : 'var(--bg-surface-raised)',
                    color: audioPlaying ? 'var(--bg-surface)' : 'var(--text-primary)',
                    border: '2px solid var(--card-border-color)',
                    fontSize: 12,
                    fontWeight: 800,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6,
                    transition: 'all 0.15s ease',
                    boxShadow: 'var(--shadow-xs)'
                  }}
                >
                  {audioPlaying ? <VolumeX size={15} /> : <Volume2 size={15} />}
                  <span>{audioPlaying ? 'Synthesizing Knocking...' : '▶ Listen to Bearing Acoustic Sound'}</span>
                </button>
              </div>
            </div>

            {/* ── Verified 2D CNN Preprocessed Graph Banner (If Image Inference) ── */}
            {diagnosisResult.processed_image_base64 && (
              <div
                style={{
                  background: 'var(--bg-surface)',
                  borderRadius: 0,
                  padding: '18px 22px',
                  border: '2px solid #6366f1',
                  marginBottom: 24,
                  boxShadow: '0 0 16px rgba(99, 102, 241, 0.2)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: 16
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
                  <img
                    src={diagnosisResult.processed_image_base64}
                    alt="Verified 2D CNN Preprocessed Graph"
                    style={{
                      width: 84,
                      height: 84,
                      objectFit: 'contain',
                      background: '#000000',
                      border: '2px solid #6366f1',
                      borderRadius: 4
                    }}
                  />
                  <div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                      <span style={{ fontSize: 11, fontWeight: 900, textTransform: 'uppercase', color: '#6366f1', letterSpacing: '0.05em' }}>
                        VERIFIED 2D CNN MODEL INPUT (224×224)
                      </span>
                      <span style={{ fontSize: 10, background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', padding: '1px 6px', borderRadius: 4, fontWeight: 800 }}>
                        Plot Area Aligned (224×224)
                      </span>
                    </div>
                    <h4 style={{ fontSize: 16, fontWeight: 900, margin: '4px 0 2px', color: 'var(--text-primary)' }}>
                      Input: {diagnosisResult.source_filename || 'FFT Spectrum Line Plot'}
                    </h4>
                    <div style={{ fontSize: 12, color: 'var(--text-secondary)', maxWidth: 560 }}>
                      The uploaded FFT graph plot area was auto-detected, aligned to 224×224 pixels, and evaluated across 4 classes (`Healthy`, `Inner Race`, `Outer Race`, `Ball`) by the dedicated 2D CNN.
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: 6 }}>
                  <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', background: 'rgba(99, 102, 241, 0.15)', color: '#818cf8', padding: '4px 10px', borderRadius: 4, fontWeight: 800, border: '1px solid rgba(99, 102, 241, 0.3)' }}>
                    Model: 2D CNN Vision Classifier
                  </span>
                  <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                    Inference: {diagnosisResult.inference_time_ms || '<60'} ms
                  </span>
                </div>
              </div>
            )}

            {/* ── TWO PROMINENT BAR CHARTS (RECHARTS) ── */}
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))', gap: 20, marginBottom: 24 }}>
              
              {/* Bar Chart 1: Fault Probability Distribution */}
              <div
                style={{
                  background: 'var(--bg-surface)',
                  borderRadius: 0,
                  padding: '22px',
                  border: '2px solid var(--card-border-color)',
                  boxShadow: 'var(--shadow-card)'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                  <div>
                    <h4 style={{ margin: 0, fontSize: 16, fontWeight: 900, textTransform: 'uppercase', color: 'var(--text-primary)' }}>
                      Fault Probability Distribution (Bar Chart)
                    </h4>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                      Neural classification across mechanical defect modes
                    </span>
                  </div>
                  <span style={{ fontSize: 12, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)', fontWeight: 800 }}>
                    100% Total
                  </span>
                </div>

                {(() => {
                  const probData = diagnosisResult.probabilities || [
                    { name: 'Normal',     value: isHealthy ? 95 : 4,                                   fill: '#10d9a0' },
                    { name: 'Inner Race', value: diagnosisResult.fault_type === 'Inner Race' ? 84 : 4, fill: '#38bdf8' },
                    { name: 'Ball Fault', value: diagnosisResult.fault_type === 'Ball' ? 85 : 3,       fill: '#fbbf24' },
                    { name: 'Outer Race', value: diagnosisResult.fault_type === 'Outer Race' ? 85 : 3, fill: '#f43f5e' }
                  ];
                  return (
                    <div style={{ width: '100%', height: 240 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={probData} margin={{ top: 10, right: 15, left: -20, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                          <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} interval={0} />
                          <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} domain={[0, 100]} unit="%" />
                          <Tooltip
                            contentStyle={{ background: 'var(--bg-surface)', border: '2px solid var(--card-border-color)', borderRadius: 0, boxShadow: 'var(--shadow-sm)', color: 'var(--text-primary)', fontSize: 12 }}
                            formatter={(val) => [`${val}%`, 'Likelihood']}
                          />
                          <Bar dataKey="value" radius={[0, 0, 0, 0]}>
                            {probData.map((entry, idx) => (
                              <Cell key={`cell-${idx}`} fill={entry.fill || 'var(--text-primary)'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  );
                })()}
              </div>

              {/* Bar Chart 2: ISO 10816-3 Vibration Severity Scale */}
              <div
                style={{
                  background: 'var(--bg-surface)',
                  borderRadius: 0,
                  padding: '22px',
                  border: '2px solid var(--card-border-color)',
                  boxShadow: 'var(--shadow-card)'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 14 }}>
                  <div>
                    <h4 style={{ margin: 0, fontSize: 16, fontWeight: 900, textTransform: 'uppercase', color: 'var(--text-primary)' }}>
                      ISO 10816-3 Vibration Severity (Bar Chart)
                    </h4>
                    <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                      Standardized mechanical velocity / acceleration limits
                    </span>
                  </div>
                  <span
                    style={{
                      fontSize: 12,
                      fontWeight: 800,
                      padding: '3px 10px',
                      borderRadius: 0,
                      border: '1px solid #000',
                      boxShadow: 'var(--shadow-xs)',
                      background: faultColor,
                      color: '#060b14',
                      fontFamily: 'var(--font-mono)'
                    }}
                  >
                    {diagnosisResult.features?.rms ? `RMS: ${diagnosisResult.features.rms} g` : 'FFT Graph Spectrum'}
                  </span>
                </div>

                {(() => {
                  const isoData = diagnosisResult.iso_levels || [
                    { name: 'Zone A', limit: 0.12, fill: '#10d9a0' },
                    { name: 'Zone B', limit: 0.22, fill: '#38bdf8' },
                    { name: 'Zone C', limit: 0.45, fill: '#fbbf24' },
                    { name: 'Zone D', limit: 0.70, fill: '#ef4444' }
                  ];
                  return (
                    <div style={{ width: '100%', height: 240 }}>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={isoData} margin={{ top: 10, right: 15, left: -20, bottom: 20 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                          <XAxis dataKey="name" tick={{ fontSize: 12, fill: 'var(--text-secondary)' }} interval={0} />
                          <YAxis tick={{ fontSize: 11, fill: 'var(--text-muted)' }} domain={[0, 0.8]} unit="g" />
                          <Tooltip
                            contentStyle={{ background: 'var(--bg-surface)', border: '2px solid var(--card-border-color)', borderRadius: 0, boxShadow: 'var(--shadow-sm)', color: 'var(--text-primary)', fontSize: 12 }}
                            formatter={(val) => [`${val} g`, 'Severity Threshold']}
                          />
                          <Bar dataKey="limit" radius={[0, 0, 0, 0]}>
                            {isoData.map((entry, idx) => (
                              <Cell key={`cell-iso-${idx}`} fill={entry.fill || 'var(--text-primary)'} />
                            ))}
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  );
                })()}
              </div>
            </div>

            {/* ── Mechanical Workshop Recommendations ── */}
            <div
              style={{
                background: 'var(--bg-surface)',
                borderRadius: 0,
                padding: '22px',
                border: '2px solid var(--card-border-color)',
                boxShadow: 'var(--shadow-card)',
                marginBottom: 24
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 14 }}>
                <Wrench size={20} style={{ color: 'var(--text-primary)' }} />
                <h4 style={{ margin: 0, fontSize: 16, fontWeight: 900, textTransform: 'uppercase', color: 'var(--text-primary)' }}>
                  Practical Maintenance Directives for Workshop Mechanics:
                </h4>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))', gap: 16 }}>
                <div style={{ background: 'var(--bg-surface-raised)', padding: '16px', borderRadius: 0, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 11, fontWeight: 800, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                    DEFECT CHARACTERISTICS
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6 }}>
                    {diagnosisResult.fault_type === 'Inner Race'
                      ? 'Ball Pass Frequency Inner Race (BPFI ~162.2 Hz) harmonics identified. Sharp acceleration spikes indicate localized micro-spalling on the rotating inner ring track.'
                      : diagnosisResult.fault_type === 'Outer Race'
                      ? 'Periodic impacts matching stationary outer ring defect (BPFO ~107.5 Hz). High radial vibration transferred to machine housing.'
                      : diagnosisResult.fault_type === 'Ball'
                      ? 'Ball spin frequency (BSF ~141.2 Hz) detected. Rolling balls display surface pitting or severe grease contamination.'
                      : 'Baseline smooth operation. Hydrodynamic lubrication barrier intact with minimal friction losses.'}
                  </div>
                </div>

                <div style={{ background: 'var(--bg-surface-raised)', padding: '16px', borderRadius: 0, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 11, fontWeight: 800, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                    WORKSHOP ACTION REQUIRED
                  </div>
                  <div style={{ fontSize: 13, color: 'var(--text-primary)', lineHeight: 1.6 }}>
                    {diagnosisResult.recommendation ||
                      'Schedule bearing overhaul and grease relubrication within recommended maintenance window.'}
                  </div>
                </div>

                <div style={{ background: 'var(--bg-surface-raised)', padding: '16px', borderRadius: 0, border: '1px solid var(--border-subtle)' }}>
                  <div style={{ fontSize: 11, fontWeight: 800, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                    TARGET INTERVENTION TIMELINE
                  </div>
                  <div style={{ fontSize: 15, fontWeight: 900, color: faultColor, lineHeight: 1.5 }}>
                    {diagnosisResult.urgency || (isHealthy ? 'Routine 90-Day Inspection' : 'Intervene within 7 Days')}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 4 }}>
                    Prevent catastrophic rotor imbalance and thermal shaft seizure.
                  </div>
                </div>
              </div>
            </div>

            {/* ═════════════════════════════════════════════════════════════════════════ */}
            {/* STEP 4: AUDIT REPORTING & MAINTENANCE DISPATCH (BOTTOM OF SAME PAGE)      */}
            {/* ═════════════════════════════════════════════════════════════════════════ */}
            <div
              style={{
                paddingTop: 24,
                borderTop: '2px solid rgba(255,255,255,0.08)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                flexWrap: 'wrap',
                gap: 16
              }}
            >
              <div>
                <h4 style={{ margin: '0 0 4px', fontSize: 17, fontWeight: 900 }}>
                  Step 4: Executive Audit Reporting & Corrective Dispatch
                </h4>
                <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                  Download official stamped engineering audit PDF or dispatch task directly to workshop schedule
                </span>
              </div>

              <div style={{ display: 'flex', gap: 14, flexWrap: 'wrap' }}>
                <button
                  type="button"
                  id="btn-download-pdf"
                  onClick={handleDownloadPdf}
                  style={{
                    padding: '13px 22px',
                    borderRadius: 0,
                    border: '2px solid var(--card-border-color)',
                    background: 'var(--bg-surface)',
                    color: 'var(--text-primary)',
                    fontWeight: 800,
                    fontSize: 14,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    transition: 'all 0.15s ease',
                    boxShadow: 'var(--shadow-btn)'
                  }}
                >
                  <FileText size={18} />
                  <span>Download Executive Audit PDF</span>
                </button>

                <button
                  type="button"
                  id="btn-open-dispatch-modal"
                  onClick={() => setDispatchModalOpen(true)}
                  style={{
                    padding: '13px 26px',
                    borderRadius: 0,
                    border: '2px solid var(--card-border-color)',
                    background: 'var(--text-primary)',
                    color: 'var(--bg-surface)',
                    fontWeight: 900,
                    fontSize: 14,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 10,
                    boxShadow: 'var(--shadow-btn)'
                  }}
                >
                  <CalendarCheck size={18} />
                  <span>Dispatch Machine to Maintenance</span>
                </button>
              </div>
            </div>

            {/* Dispatch Success Confirmation Banner */}
            {dispatchSuccess && (
              <div
                style={{
                  marginTop: 22,
                  padding: '18px 24px',
                  borderRadius: 0,
                  background: 'var(--status-healthy-bg)',
                  border: '2px solid var(--status-healthy)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  flexWrap: 'wrap',
                  gap: 14,
                  boxShadow: 'var(--shadow-card)'
                }}
              >
                <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
                  <CheckCircle2 size={26} style={{ color: 'var(--status-healthy)' }} />
                  <div>
                    <div style={{ fontSize: 15, fontWeight: 900, color: 'var(--status-healthy)' }}>
                      WORK ORDER SUCCESSFULLY DISPATCHED TO WORKSHOP SCHEDULE!
                    </div>
                    <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                      Maintenance task registered for {selectedMachine?.name} with assigned priority.
                    </div>
                  </div>
                </div>

                <button
                  type="button"
                  id="btn-view-in-schedule"
                  onClick={onSwitchToSchedule}
                  style={{
                    padding: '10px 18px',
                    borderRadius: 0,
                    background: 'var(--status-healthy)',
                    color: '#000000',
                    border: '2px solid #000000',
                    fontWeight: 900,
                    fontSize: 13,
                    cursor: 'pointer',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    boxShadow: 'var(--shadow-btn)'
                  }}
                >
                  <span>View in Maintenance Schedule</span>
                  <ArrowRight size={16} />
                </button>
              </div>
            )}
          </div>
        )}
      </section>

      {/* ═════════════════════════════════════════════════════════════════════════ */}
      {/* DISPATCH WORK ORDER MODAL                                                */}
      {/* ═════════════════════════════════════════════════════════════════════════ */}
      {dispatchModalOpen && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0,0,0,0.75)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 1000,
            padding: 20
          }}
        >
          <div
            style={{
              background: 'var(--bg-surface)',
              borderRadius: 0,
              border: '3px solid var(--card-border-color)',
              width: '100%',
              maxWidth: 560,
              padding: '26px',
              boxShadow: 'var(--shadow-xl)'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 18 }}>
              <div>
                <h3 style={{ margin: 0, fontSize: 19, fontWeight: 900, textTransform: 'uppercase', color: 'var(--text-primary)' }}>
                  Dispatch Corrective Maintenance Order
                </h3>
                <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  Directly queue task into workshop maintenance log
                </span>
              </div>
              <button
                type="button"
                onClick={() => setDispatchModalOpen(false)}
                style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', cursor: 'pointer', fontSize: 22 }}
              >
                ✕
              </button>
            </div>

            <form onSubmit={handleDispatchSubmit}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 14, marginBottom: 22 }}>
                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                    Machine Target
                  </label>
                  <input
                    type="text"
                    disabled
                    value={`${selectedMachine?.name} (${selectedMachine?.location})`}
                    style={{
                      width: '100%',
                      padding: '11px 13px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-base)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-muted)',
                      fontSize: 13
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                    Work Order Task Description *
                  </label>
                  <input
                    type="text"
                    required
                    value={dispatchForm.task_type}
                    onChange={(e) => setDispatchForm({ ...dispatchForm, task_type: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '11px 13px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-base)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-primary)',
                      fontSize: 13
                    }}
                  />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                  <div>
                    <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                      Priority Level
                    </label>
                    <select
                      value={dispatchForm.priority}
                      onChange={(e) => setDispatchForm({ ...dispatchForm, priority: e.target.value })}
                      style={{
                        width: '100%',
                        padding: '11px 13px',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-base)',
                        border: '1px solid var(--border-subtle)',
                        color: 'var(--text-primary)',
                        fontSize: 13
                      }}
                    >
                      <option value="Critical">Critical (Immediate Stop)</option>
                      <option value="High">High (Within 48 Hours)</option>
                      <option value="Medium">Medium (Scheduled Next Overhaul)</option>
                      <option value="Low">Low (Routine Checkup)</option>
                    </select>
                  </div>

                  <div>
                    <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                      Scheduled Due Date *
                    </label>
                    <input
                      type="date"
                      required
                      value={dispatchForm.due_date}
                      onChange={(e) => setDispatchForm({ ...dispatchForm, due_date: e.target.value })}
                      style={{
                        width: '100%',
                        padding: '11px 13px',
                        borderRadius: 'var(--radius-sm)',
                        background: 'var(--bg-base)',
                        border: '1px solid var(--border-subtle)',
                        color: 'var(--text-primary)',
                        fontSize: 13
                      }}
                    />
                  </div>
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                    Assigned Lead Technician
                  </label>
                  <input
                    type="text"
                    value={dispatchForm.technician}
                    onChange={(e) => setDispatchForm({ ...dispatchForm, technician: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '11px 13px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-base)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-primary)',
                      fontSize: 13
                    }}
                  />
                </div>

                <div>
                  <label style={{ display: 'block', fontSize: 12, fontWeight: 700, marginBottom: 4 }}>
                    Technical Notes
                  </label>
                  <textarea
                    rows={3}
                    placeholder="e.g. Inspect SKF 6205 inner race for micro-spalling. Replace grease pack with Mobil Polyrex EM."
                    value={dispatchForm.notes}
                    onChange={(e) => setDispatchForm({ ...dispatchForm, notes: e.target.value })}
                    style={{
                      width: '100%',
                      padding: '11px 13px',
                      borderRadius: 'var(--radius-sm)',
                      background: 'var(--bg-base)',
                      border: '1px solid var(--border-subtle)',
                      color: 'var(--text-primary)',
                      fontSize: 13,
                      resize: 'vertical'
                    }}
                  />
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 12 }}>
                <button
                  type="button"
                  onClick={() => setDispatchModalOpen(false)}
                  style={{
                    padding: '11px 18px',
                    borderRadius: 0,
                    border: '2px solid var(--card-border-color)',
                    background: 'var(--bg-surface-raised)',
                    color: 'var(--text-secondary)',
                    fontWeight: 700,
                    cursor: 'pointer',
                    boxShadow: 'var(--shadow-xs)'
                  }}
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  id="btn-confirm-dispatch-order"
                  style={{
                    padding: '11px 24px',
                    borderRadius: 0,
                    border: '2px solid var(--card-border-color)',
                    background: 'var(--text-primary)',
                    color: 'var(--bg-surface)',
                    fontWeight: 900,
                    cursor: 'pointer',
                    boxShadow: 'var(--shadow-btn)'
                  }}
                >
                  Confirm & Dispatch Work Order
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}







