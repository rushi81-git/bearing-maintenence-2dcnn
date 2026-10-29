import React, { useState, useMemo, useCallback } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  Activity,
  ArrowLeft,
  ArrowRight,
  Sparkles,
  Download,
  Sliders,
  CheckCircle2,
  AlertTriangle,
  RotateCcw,
  Zap,
  Gauge
} from 'lucide-react';
import { WaveformChart } from '../../components/WaveformChart';
import { SpectrumChart } from '../../components/SpectrumChart';
import { parseVibrationCSV } from '../../utils/csvParser';
import { computeFFT } from '../../utils/fft';
import { analyzeBearingSignal, fetchDemoSamples } from '../../services/api';

export function CsvUploadStep({
  selectedMachine,
  onBack,
  onDiagnosisComplete,
  showToast
}) {
  const [sourceFilename, setSourceFilename] = useState('');
  const [activeSignal, setActiveSignal] = useState(null);
  const [samplingRateHz, setSamplingRateHz] = useState(48000);
  const [signalUnit, setSignalUnit] = useState('g');
  const [parsedCsv, setParsedCsv] = useState(null);
  const [selectedColumnIndex, setSelectedColumnIndex] = useState(0);
  const [isDragging, setIsDragging] = useState(false);
  const [analyzing, setAnalyzing] = useState(false);
  const [activeTab, setActiveTab] = useState('waveform'); // 'waveform' | 'spectrum'

  // Oscilloscope controls
  const [zoomSize, setZoomSize] = useState(1024);
  const [windowOffset, setWindowOffset] = useState(0);

  // Compute FFT spectrum for active signal
  const spectrumData = useMemo(() => {
    if (!activeSignal || activeSignal.length === 0) return [];
    return computeFFT(activeSignal.slice(0, 1024), samplingRateHz);
  }, [activeSignal, samplingRateHz]);

  // Compute immediate signal statistical features for the summary cards
  const signalStats = useMemo(() => {
    if (!activeSignal || activeSignal.length === 0) return null;
    const n = activeSignal.length;
    let sum = 0;
    let sumSq = 0;
    let max = -Infinity;
    let min = Infinity;

    for (let i = 0; i < n; i++) {
      const v = activeSignal[i];
      sum += v;
      sumSq += v * v;
      if (v > max) max = v;
      if (v < min) min = v;
    }

    const mean = sum / n;
    const rms = Math.sqrt(sumSq / n);
    const peakToPeak = max - min;
    const peak = Math.max(Math.abs(max), Math.abs(min));
    const crestFactor = rms > 0 ? peak / rms : 0;

    let sumDiff4 = 0;
    let sumDiff2 = 0;
    for (let i = 0; i < n; i++) {
      const diff = activeSignal[i] - mean;
      sumDiff2 += diff * diff;
      sumDiff4 += Math.pow(diff, 4);
    }
    const variance = sumDiff2 / n;
    const std = Math.sqrt(variance);
    const kurtosis = std > 0 ? (sumDiff4 / n) / Math.pow(variance, 2) : 3.0;

    return {
      sampleCount: n,
      durationSec: (n / samplingRateHz).toFixed(3),
      rms: rms.toFixed(4),
      peakToPeak: peakToPeak.toFixed(4),
      crestFactor: crestFactor.toFixed(2),
      kurtosis: kurtosis.toFixed(2)
    };
  }, [activeSignal, samplingRateHz]);

  // Handle CSV file upload
  const handleFileUpload = useCallback(
    (e) => {
      const file = e.target.files?.[0];
      if (!file) return;

      const filename = file.name;
      const reader = new FileReader();

      reader.onload = (evt) => {
        try {
          const text = evt.target?.result;
          const parsed = parseVibrationCSV(text, filename);

          setParsedCsv(parsed);
          setSourceFilename(filename);
          setSelectedColumnIndex(parsed.defaultColumn.columnIndex);
          setActiveSignal(parsed.defaultColumn.data);
          if (parsed.detectedSamplingRate) {
            setSamplingRateHz(parsed.detectedSamplingRate);
          }
          setWindowOffset(0);

          showToast(
            `Loaded ${parsed.defaultColumn.sampleCount.toLocaleString()} samples from "${parsed.defaultColumn.columnName}" (${parsed.detectedSamplingRate || samplingRateHz} Hz)`,
            'success'
          );
        } catch (err) {
          showToast(err.message || 'Failed to parse CSV file', 'error');
        }
      };

      reader.readAsText(file);
    },
    [showToast, samplingRateHz]
  );

  // Handle Channel / Column Switch
  const handleColumnChange = useCallback(
    (colIdx) => {
      setSelectedColumnIndex(colIdx);
      if (parsedCsv && parsedCsv.columns) {
        const col = parsedCsv.columns.find((c) => c.columnIndex === colIdx);
        if (col) {
          setActiveSignal(col.data);
          showToast(`Switched active signal to column: "${col.columnName}"`, 'info');
        }
      }
    },
    [parsedCsv, showToast]
  );

  // Handle Drag & Drop
  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) {
      handleFileUpload({ target: { files: [file] } });
    }
  };

  // Quick Preset Sample Loader (1-Click test)
  const handleLoadPreset = async (presetType) => {
    try {
      showToast(`Loading ${presetType} vibration signal...`, 'info');
      const res = await fetchDemoSamples();
      if (res.success && res.samples) {
        let sampleKey = 'Normal';
        if (presetType === 'Inner Race Fault') sampleKey = 'IR_014';
        if (presetType === 'Ball Fault') sampleKey = 'Ball_014';
        if (presetType === 'Outer Race Fault') sampleKey = 'OR_014';

        const s = res.samples[sampleKey];
        if (s && s.signal && s.signal.length > 0) {
          setActiveSignal(s.signal);
          setSourceFilename(`sample_${sampleKey.toLowerCase()}.csv`);
          setSamplingRateHz(s.sampling_rate_hz || 48000);
          setParsedCsv(null);
          showToast(`Loaded ${presetType} telemetry sample (${s.signal.length} points)`, 'success');
          return;
        }
      }

      // Fallback synthesizer if sample not available
      const generated = generateDemoWaveform(presetType);
      setActiveSignal(generated);
      setSourceFilename(`sample_${presetType.toLowerCase().replace(/\s+/g, '_')}.csv`);
      setParsedCsv(null);
      showToast(`Generated synthetic ${presetType} sample telemetry`, 'success');
    } catch (err) {
      showToast('Error loading sample preset', 'error');
    }
  };

  // Helper synthetic generator
  const generateDemoWaveform = (type) => {
    const N = 2048;
    const sig = [];
    const fs = 48000;
    const fr = 29.95; // 1797 RPM
    for (let i = 0; i < N; i++) {
      const t = i / fs;
      let val = (Math.random() - 0.5) * 0.08;
      val += 0.03 * Math.sin(2 * Math.PI * fr * t);
      if (type.includes('Inner Race')) {
        const bpfi = 5.415 * fr;
        const mod = Math.pow(Math.max(0, Math.sin(2 * Math.PI * bpfi * t)), 12);
        val += 0.32 * mod * Math.sin(2 * Math.PI * 3200 * t);
      } else if (type.includes('Ball')) {
        const bsf = 2.357 * fr;
        const mod = Math.pow(Math.max(0, Math.sin(2 * Math.PI * bsf * t)), 8);
        val += 0.24 * mod * Math.sin(2 * Math.PI * 2600 * t);
      } else if (type.includes('Outer Race')) {
        const bpfo = 3.585 * fr;
        const mod = Math.pow(Math.max(0, Math.sin(2 * Math.PI * bpfo * t)), 10);
        val += 0.28 * mod * Math.sin(2 * Math.PI * 2900 * t);
      }
      sig.push(Math.round(val * 10000) / 10000);
    }
    return sig;
  };

  // Download Sample CSV Helper
  const handleDownloadSampleCsv = () => {
    const signalToExport = activeSignal || generateDemoWaveform('Inner Race Fault');
    let csvContent = 'Time_Sec,Acceleration_g\n';
    const dt = 1 / samplingRateHz;
    signalToExport.forEach((val, idx) => {
      csvContent += `${(idx * dt).toFixed(6)},${val}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.setAttribute('href', url);
    link.setAttribute('download', 'sample_bearing_vibration.csv');
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    showToast('Downloaded sample vibration CSV', 'success');
  };

  // Execute Diagnosis CTA
  const handleExecuteDiagnosis = async () => {
    if (!activeSignal || activeSignal.length < 1024) {
      showToast('At least 1,024 vibration samples are required for neural diagnosis', 'error');
      return;
    }

    setAnalyzing(true);
    try {
      const res = await analyzeBearingSignal({
        signal: activeSignal,
        machine_id: selectedMachine?.id || null,
        sampling_rate_hz: samplingRateHz,
        signal_unit: signalUnit,
        source_type: 'csv',
        source_filename: sourceFilename || 'vibration_telemetry.csv',
        model_mode: 'auto'
      });

      if (res.success && res.data) {
        showToast('Diagnosis completed successfully!', 'success');
        onDiagnosisComplete({
          diagnosis: res.data,
          signal: activeSignal,
          filename: sourceFilename || 'vibration_telemetry.csv'
        });
      } else {
        showToast(res.message || 'Diagnosis failed', 'error');
      }
    } catch (err) {
      showToast(err.message || 'Error during inference diagnosis', 'error');
    } finally {
      setAnalyzing(false);
    }
  };

  const sampleCount = activeSignal ? activeSignal.length : 0;
  const isSignalReady = sampleCount >= 1024;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }} className="animate-fade-in">
      {/* ── Active Machine Banner ── */}
      <div
        className="card"
        style={{
          background: 'var(--bg-surface)',
          padding: '16px 24px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 12,
          borderLeft: '4px solid var(--accent-primary)'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: 14 }}>
          <button
            type="button"
            onClick={onBack}
            className="btn btn-outline"
            style={{ padding: '6px 12px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
            title="Return to Machine Selection"
          >
            <ArrowLeft size={14} />
            <span>Change Machine</span>
          </button>

          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: 700, textTransform: 'uppercase' }}>
                Active Machine
              </span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>·</span>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{selectedMachine?.location || 'Floor Bay A'}</span>
            </div>
            <h3 style={{ fontSize: 17, fontWeight: 800, fontFamily: 'var(--font-heading)', color: 'var(--text-primary)', marginTop: 2 }}>
              {selectedMachine?.name || 'Spindle Machine'} ({selectedMachine?.machine_type || selectedMachine?.type || 'Equipment'})
            </h3>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <button
            type="button"
            onClick={handleDownloadSampleCsv}
            className="btn btn-outline"
            style={{ padding: '6px 12px', fontSize: 12, display: 'flex', alignItems: 'center', gap: 6 }}
          >
            <Download size={13} />
            <span>Download Sample CSV</span>
          </button>
        </div>
      </div>

      {/* ── CSV Upload Card ── */}
      <div
        className="card"
        style={{
          background: 'var(--bg-surface)',
          padding: '24px 28px',
          display: 'flex',
          flexDirection: 'column',
          gap: 16
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
          <div>
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: 700, textTransform: 'uppercase' }}>
              // Step 2: Upload CSV Telemetry
            </span>
            <h3 style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-heading)', marginTop: 2 }}>
              Upload Accelerometer CSV Reading
            </h3>
            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 2 }}>
              Drop a vibration time-series CSV file or select a pre-calibrated test benchmark.
            </p>
          </div>

          {/* Quick Preset Buttons */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
            <span style={{ fontSize: 11, color: 'var(--text-muted)', fontWeight: 600 }}>Quick Presets:</span>
            <button
              type="button"
              onClick={() => handleLoadPreset('Normal Baseline')}
              className="btn btn-secondary"
              style={{ padding: '4px 10px', fontSize: 11 }}
            >
              Normal
            </button>
            <button
              type="button"
              onClick={() => handleLoadPreset('Inner Race Fault')}
              className="btn btn-secondary"
              style={{ padding: '4px 10px', fontSize: 11 }}
            >
              Inner Race
            </button>
            <button
              type="button"
              onClick={() => handleLoadPreset('Ball Fault')}
              className="btn btn-secondary"
              style={{ padding: '4px 10px', fontSize: 11 }}
            >
              Ball Fault
            </button>
            <button
              type="button"
              onClick={() => handleLoadPreset('Outer Race Fault')}
              className="btn btn-secondary"
              style={{ padding: '4px 10px', fontSize: 11 }}
            >
              Outer Race
            </button>
          </div>
        </div>

        {/* Drag & Drop Zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setIsDragging(true); }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          style={{
            border: isDragging ? '2px dashed var(--accent-primary)' : '2px dashed var(--border-strong)',
            background: isDragging ? 'var(--accent-subtle)' : 'var(--bg-surface-raised)',
            borderRadius: 'var(--radius-lg)',
            padding: '28px 20px',
            textAlign: 'center',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            position: 'relative'
          }}
        >
          <input
            type="file"
            accept=".csv,.txt"
            onChange={handleFileUpload}
            style={{
              position: 'absolute',
              inset: 0,
              opacity: 0,
              cursor: 'pointer',
              width: '100%',
              height: '100%'
            }}
          />

          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8 }}>
            <div
              style={{
                width: 48,
                height: 48,
                borderRadius: 'var(--radius-full)',
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: 'var(--accent-primary)'
              }}
            >
              <UploadCloud size={24} />
            </div>

            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)' }}>
                {sourceFilename ? `Active: ${sourceFilename}` : 'Click to browse or drop CSV vibration file here'}
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-secondary)', marginTop: 2 }}>
                Supports standard comma/tab/space-delimited single or multi-channel accelerometer signals (≥ 1,024 samples)
              </div>
            </div>

            {sourceFilename && (
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginTop: 4 }}>
                <span
                  style={{
                    display: 'inline-flex',
                    alignItems: 'center',
                    gap: 6,
                    padding: '3px 10px',
                    borderRadius: 'var(--radius-full)',
                    background: 'var(--status-healthy-bg)',
                    color: 'var(--status-healthy)',
                    fontSize: 11,
                    fontWeight: 700,
                    border: '1px solid var(--status-healthy-border)'
                  }}
                >
                  <CheckCircle2 size={13} />
                  <span>{sampleCount.toLocaleString()} samples loaded</span>
                </span>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Signal Telemetry Graphs & Charts (Shown on the same page!) ── */}
      {isSignalReady && (
        <div
          className="card animate-fade-in"
          style={{
            background: 'var(--bg-surface)',
            padding: '24px 28px',
            display: 'flex',
            flexDirection: 'column',
            gap: 20
          }}
        >
          {/* Signal Header with Tabs */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12 }}>
            <div>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: 700, textTransform: 'uppercase' }}>
                // Real-time Signal Telemetry Visualizer
              </span>
              <h3 style={{ fontSize: 18, fontWeight: 800, fontFamily: 'var(--font-heading)', marginTop: 2 }}>
                Waveform &amp; Spectral Harmonics
              </h3>
            </div>

            {/* Waveform / Spectrum Toggle */}
            <div
              style={{
                display: 'flex',
                background: 'var(--bg-surface-raised)',
                padding: 4,
                borderRadius: 'var(--radius-lg)',
                border: '1px solid var(--border-subtle)',
                gap: 4
              }}
            >
              <button
                type="button"
                onClick={() => setActiveTab('waveform')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-md)',
                  fontSize: 12,
                  fontWeight: 700,
                  border: 'none',
                  cursor: 'pointer',
                  background: activeTab === 'waveform' ? 'var(--accent-primary)' : 'transparent',
                  color: activeTab === 'waveform' ? '#ffffff' : 'var(--text-secondary)'
                }}
              >
                <Activity size={14} />
                <span>Waveform (Oscilloscope)</span>
              </button>

              <button
                type="button"
                onClick={() => setActiveTab('spectrum')}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 6,
                  padding: '6px 14px',
                  borderRadius: 'var(--radius-md)',
                  fontSize: 12,
                  fontWeight: 700,
                  border: 'none',
                  cursor: 'pointer',
                  background: activeTab === 'spectrum' ? 'var(--accent-primary)' : 'transparent',
                  color: activeTab === 'spectrum' ? '#ffffff' : 'var(--text-secondary)'
                }}
              >
                <Zap size={14} />
                <span>FFT Frequency Spectrum</span>
              </button>
            </div>
          </div>

          {/* Statistical Metric Pills */}
          {signalStats && (
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                gap: 10
              }}
            >
              <div style={{ padding: '10px 14px', background: 'var(--bg-surface-raised)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  Total Points
                </div>
                <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text-primary)', marginTop: 2 }}>
                  {signalStats.sampleCount.toLocaleString()}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>{signalStats.durationSec} sec @ 48kHz</div>
              </div>

              <div style={{ padding: '10px 14px', background: 'var(--bg-surface-raised)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  RMS Vibration
                </div>
                <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--accent-primary)', marginTop: 2 }}>
                  {signalStats.rms} g
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Continuous energy</div>
              </div>

              <div style={{ padding: '10px 14px', background: 'var(--bg-surface-raised)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  Kurtosis
                </div>
                <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text-primary)', marginTop: 2 }}>
                  {signalStats.kurtosis}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Gaussian baseline ~3.0</div>
              </div>

              <div style={{ padding: '10px 14px', background: 'var(--bg-surface-raised)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  Crest Factor
                </div>
                <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text-primary)', marginTop: 2 }}>
                  {signalStats.crestFactor}
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Peak / RMS ratio</div>
              </div>

              <div style={{ padding: '10px 14px', background: 'var(--bg-surface-raised)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', textTransform: 'uppercase', fontWeight: 700 }}>
                  Peak-to-Peak
                </div>
                <div style={{ fontSize: 15, fontWeight: 800, color: 'var(--text-primary)', marginTop: 2 }}>
                  {signalStats.peakToPeak} g
                </div>
                <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Full excursion span</div>
              </div>
            </div>
          )}

          {/* Interactive Chart Container */}
          <div
            style={{
              background: 'var(--bg-base)',
              borderRadius: 'var(--radius-lg)',
              padding: '16px',
              border: '1px solid var(--border-subtle)'
            }}
          >
            {activeTab === 'waveform' ? (
              <WaveformChart
                signal={activeSignal.slice(windowOffset, windowOffset + zoomSize)}
                samplingRateHz={samplingRateHz}
                signalUnit={signalUnit}
                height={260}
                showRMSLine={true}
              />
            ) : (
              <SpectrumChart
                spectrumData={spectrumData}
                samplingRateHz={samplingRateHz}
                height={260}
                activeOverlay="all"
              />
            )}
          </div>

          {/* ── EXECUTE DIAGNOSIS BUTTON (Hero CTA) ── */}
          <div
            style={{
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              paddingTop: 16,
              borderTop: '1px solid var(--border-subtle)',
              flexWrap: 'wrap',
              gap: 14
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
              <div
                style={{
                  width: 8,
                  height: 8,
                  borderRadius: '50%',
                  background: 'var(--status-healthy)',
                  boxShadow: '0 0 8px var(--status-healthy)'
                }}
              />
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                Signal verified &amp; ready for neural feature extraction
              </span>
            </div>

            <button
              type="button"
              disabled={analyzing}
              onClick={handleExecuteDiagnosis}
              className="btn btn-primary"
              style={{
                padding: '12px 32px',
                fontSize: 15,
                fontWeight: 800,
                display: 'flex',
                alignItems: 'center',
                gap: 10,
                boxShadow: '0 0 20px rgba(99, 102, 241, 0.45)',
                letterSpacing: '0.02em'
              }}
            >
              <Activity size={18} className={analyzing ? 'spin' : ''} />
              <span>{analyzing ? 'Analyzing Telemetry Dynamics...' : 'Execute Diagnosis'}</span>
              {!analyzing && <ArrowRight size={17} />}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
