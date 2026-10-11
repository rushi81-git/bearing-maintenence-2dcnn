import React, { useState, useEffect, useMemo, useCallback } from 'react';
import { BearingStepper } from '../components/bearing/BearingStepper';
import { MachineSelector } from '../components/bearing/MachineSelector';
import { SignalUploadCard } from '../components/bearing/SignalUploadCard';
import { SignalPreviewCard } from '../components/bearing/SignalPreviewCard';
import { DemoSamplesDrawer } from '../components/bearing/DemoSamplesDrawer';
import { ProcessingPipeline, PIPELINE_STEPS } from '../components/bearing/ProcessingPipeline';
import { DiagnosisPanel } from '../components/bearing/DiagnosisPanel';
import { MultiWindowDistribution } from '../components/bearing/MultiWindowDistribution';
import { TechnicalAuditSection } from '../components/bearing/TechnicalAuditSection';
import { parseVibrationCSV } from '../utils/csvParser';
import { computeFFT } from '../utils/fft';
import { analyzeBearingSignal, fetchDemoSamples } from '../services/api';
import { RotateCcw, Activity, FileSpreadsheet, ArrowRight, Image as ImageIcon } from 'lucide-react';

export function BearingAnalysis({
  machines = [],
  showToast,
  preSelectedMachineId = null,
  onScheduleMaintenance = null
}) {
  // Machine Selection
  const [selectedMachineId, setSelectedMachineId] = useState(preSelectedMachineId || (machines[0]?.id || ''));

  // Signal State
  const [sourceFilename, setSourceFilename] = useState('');
  const [sourceType, setSourceType] = useState('csv'); // 'csv' | 'demo' | 'simulated' | 'fft_graph_image'
  const [parsedCsv, setParsedCsv] = useState(null);
  const [selectedColumnIndex, setSelectedColumnIndex] = useState(0);
  const [activeSignal, setActiveSignal] = useState(null);
  const [activeImageBase64, setActiveImageBase64] = useState(null);
  const [activeCropBox, setActiveCropBox] = useState(null);
  const [samplingRateHz, setSamplingRateHz] = useState(48000);
  const [signalUnit, setSignalUnit] = useState('g');

  // Demo Samples Modal
  const [isDemoDrawerOpen, setIsDemoDrawerOpen] = useState(false);
  const [demoSamplesCache, setDemoSamplesCache] = useState(null);

  // Workflow State & Stepper
  const [currentStep, setCurrentStep] = useState(1);
  const [analyzing, setAnalyzing] = useState(false);
  const [pipelineStepIndex, setPipelineStepIndex] = useState(0);
  const [diagnosisResult, setDiagnosisResult] = useState(null);

  // Update selected machine if prop changes or machine list loads
  useEffect(() => {
    if (preSelectedMachineId) {
      setSelectedMachineId(preSelectedMachineId);
    } else if (!selectedMachineId && machines.length > 0) {
      setSelectedMachineId(machines[0].id);
    }
  }, [preSelectedMachineId, machines, selectedMachineId]);

  // Pre-load demo samples catalog
  useEffect(() => {
    fetchDemoSamples()
      .then(res => {
        if (res.success) setDemoSamplesCache(res.samples);
      })
      .catch(() => {});
  }, []);

  // Compute FFT spectrum for active signal (first 1024 points)
  const spectrumData = useMemo(() => {
    if (!activeSignal || activeSignal.length === 0) return [];
    return computeFFT(activeSignal.slice(0, 1024), samplingRateHz);
  }, [activeSignal, samplingRateHz]);

  // Handle CSV file upload
  const handleFileUpload = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const filename = file.name;
    const reader = new FileReader();

    reader.onload = (evt) => {
      try {
        const text = evt.target?.result;
        const parsed = parseVibrationCSV(text);

        setParsedCsv(parsed);
        setSourceFilename(filename);
        setSourceType('csv');
        setSelectedColumnIndex(parsed.defaultColumn.columnIndex);
        setActiveSignal(parsed.defaultColumn.data);
        setDiagnosisResult(null);
        setCurrentStep(2); // Progress to preview step

        if (parsed.hasMultiColumnWarning) {
          showToast(
            `Multi-column report detected — flattened ${parsed.defaultColumn.sampleCount.toLocaleString()} data points from ${filename}. Select the AUTO-CONCAT stream or individual columns below.`,
            'info'
          );
        } else {
          showToast(
            `Successfully loaded ${parsed.defaultColumn.sampleCount.toLocaleString()} samples from ${filename}`,
            'success'
          );
        }
      } catch (err) {
        showToast(err.message || 'Failed to parse CSV file', 'error');
      }
    };

    reader.readAsText(file);
  }, [showToast]);

  // Handle Column selection change if CSV contains multiple channels
  // colIdx === -1 means the auto-concatenated multi-column stream
  const handleColumnChange = useCallback((colIdx) => {
    setSelectedColumnIndex(colIdx);
    if (parsedCsv) {
      // Find by exact columnIndex (supports -1 sentinel for concat column)
      const col = parsedCsv.columns.find(c => c.columnIndex === colIdx);
      if (col) {
        setActiveSignal(col.data);
        setDiagnosisResult(null);
        setCurrentStep(col.sampleCount >= 1024 ? 2 : 1);
        if (col.isConcatenated) {
          showToast(
            `Using flattened multi-column stream (${col.sampleCount.toLocaleString()} points). Signal represents concatenated numeric measurements, not raw time-domain data.`,
            'info'
          );
        }
      }
    }
  }, [parsedCsv, showToast]);

  // Load a curated CWRU Demo Sample
  const handleLoadDemo = useCallback((faultKey) => {
    if (!demoSamplesCache || !demoSamplesCache[faultKey]) {
      showToast('Demo sample not found in catalog.', 'error');
      return;
    }

    const sample = demoSamplesCache[faultKey];
    if (!sample.signal || sample.signal.length === 0) {
      showToast('Backend ML service is required for demo telemetry.', 'error');
      return;
    }

    setParsedCsv(null);
    setActiveSignal(sample.signal);
    setSamplingRateHz(sample.sampling_rate_hz || 48000);
    setSignalUnit(sample.signal_unit || 'g');
    setSourceFilename(`cwru_benchmark_${faultKey.toLowerCase()}.csv`);
    setSourceType('demo');
    setDiagnosisResult(null);
    setCurrentStep(2); // Move to Preview

    showToast(`Loaded CWRU Demonstration Sample (${faultKey})`, 'info');
  }, [demoSamplesCache, showToast]);

  // Handle Kinematic Signal Synthesis
  const handleSynthesizeSignal = useCallback((simData) => {
    setParsedCsv(null);
    setActiveSignal(simData.signal);
    setSamplingRateHz(simData.samplingRateHz || 48000);
    setSignalUnit(simData.signalUnit || 'g');
    setSourceFilename(simData.filename);
    setSourceType('simulated');
    setDiagnosisResult(null);
    setCurrentStep(2); // Progress to preview

    showToast(`Kinematic synthesis generated: ${simData.faultKey} (${simData.meta.rpm} RPM)`, 'success');
  }, [showToast]);

  // Handle FFT Graph Image upload
  const handleImageUpload = useCallback(({ imageBase64, filename }) => {
    setActiveImageBase64(imageBase64);
    setSourceFilename(filename || 'fft_graph_image.png');
    setSourceType('fft_graph_image');
    setActiveSignal(null);
    setParsedCsv(null);
    setDiagnosisResult(null);
    setCurrentStep(2); // Move to preview
    showToast(`Loaded FFT Graph Image: ${filename}`, 'success');
  }, [showToast]);

  // Clear current image
  const handleClearImage = useCallback(() => {
    setActiveImageBase64(null);
    setActiveCropBox(null);
    setSourceFilename('');
    setSourceType('csv');
    setDiagnosisResult(null);
    setCurrentStep(1);
  }, []);

  // Clear current signal
  const handleClearSignal = useCallback(() => {
    setParsedCsv(null);
    setActiveSignal(null);
    setActiveImageBase64(null);
    setActiveCropBox(null);
    setSourceFilename('');
    setSourceType('csv');
    setDiagnosisResult(null);
    setCurrentStep(1);
  }, []);

  // Execute Bearing Analysis Pipeline
  const handleAnalyze = async () => {
    const isImage = sourceType === 'fft_graph_image' || !!activeImageBase64;

    if (!isImage && (!activeSignal || activeSignal.length < 1024)) {
      showToast('At least 1024 vibration samples are required for neural inference.', 'error');
      return;
    }
    if (isImage && !activeImageBase64) {
      showToast('Please upload an FFT graph image before analyzing.', 'error');
      return;
    }

    setAnalyzing(true);
    setPipelineStepIndex(0);
    setDiagnosisResult(null);

    // Progressive pipeline timer
    const stepInterval = setInterval(() => {
      setPipelineStepIndex(prev => {
        if (prev < PIPELINE_STEPS.length - 1) return prev + 1;
        return prev;
      });
    }, 220);

    try {
      const payload = isImage ? {
        input_type: 'fft_graph_image',
        image_base64: activeImageBase64,
        crop_box: activeCropBox,
        machine_id: selectedMachineId || null,
        source_type: 'fft_graph_image',
        source_filename: sourceFilename
      } : {
        signal: activeSignal,
        machine_id: selectedMachineId || null,
        sampling_rate_hz: samplingRateHz,
        signal_unit: signalUnit,
        source_type: sourceType,
        source_filename: sourceFilename,
        model_mode: 'auto'
      };

      const res = await analyzeBearingSignal(payload);

      clearInterval(stepInterval);

      if (res.success) {
        setDiagnosisResult(res.data);
        setCurrentStep(3); // Move to Diagnosis step
        showToast('Bearing diagnostic analysis completed successfully!', 'success');
      } else {
        showToast(res.message || 'Diagnosis failed', 'error');
      }
    } catch (err) {
      clearInterval(stepInterval);
      showToast(`Inference error: ${err.message}`, 'error');
    } finally {
      setAnalyzing(false);
    }
  };

  const selectedMachine = machines.find(m => String(m.id) === String(selectedMachineId));
  const sampleCount = activeSignal ? activeSignal.length : 0;
  const canGoToStep2 = (activeSignal && activeSignal.length >= 1024) || !!activeImageBase64;
  const canGoToStep3 = !!diagnosisResult;

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 20 }} className="animate-fade-in">
      {/* ── Top Stepper ── */}
      <BearingStepper
        currentStep={currentStep}
        setStep={setCurrentStep}
        canGoToStep2={canGoToStep2}
        canGoToStep3={canGoToStep3}
      />

      {/* ── 2-Column Bearing Analysis Workspace ── */}
      <div
        className="bearing-workspace-grid"
        style={{
          display: 'grid',
          gridTemplateColumns: 'minmax(320px, 380px) 1fr',
          gap: 20,
          alignItems: 'start'
        }}
      >
        {/* ── LEFT COLUMN: Equipment Config + Signal Ingestion ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          {/* Machine Selection Card */}
          <div className="card" style={{ background: 'var(--bg-surface)' }}>
            <MachineSelector
              machines={machines}
              selectedMachineId={selectedMachineId}
              onSelectMachine={setSelectedMachineId}
            />
          </div>

          {/* Vibration Ingestion Card with Mode Switcher & Synthesizer */}
          <SignalUploadCard
            sourceFilename={sourceFilename}
            sourceType={sourceType}
            sampleCount={sampleCount}
            parsedCsv={parsedCsv}
            selectedColumnIndex={selectedColumnIndex}
            onColumnChange={handleColumnChange}
            samplingRateHz={samplingRateHz}
            onSamplingRateChange={setSamplingRateHz}
            signalUnit={signalUnit}
            onSignalUnitChange={setSignalUnit}
            onFileUpload={handleFileUpload}
            onClearSignal={handleClearSignal}
            onOpenDemoDrawer={() => setIsDemoDrawerOpen(true)}
            onSynthesizeSignal={handleSynthesizeSignal}
            activeImageBase64={activeImageBase64}
            onImageUpload={handleImageUpload}
            onClearImage={handleClearImage}
            cropBox={activeCropBox}
            onCropBoxChange={setActiveCropBox}
          />
        </div>

        {/* ── RIGHT COLUMN: Telemetry Preview, Analysis & Diagnosis Results ── */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>
          
          {/* Active Processing Pipeline Animation (shown while analyzing) */}
          {analyzing && (
            <ProcessingPipeline currentStepIndex={pipelineStepIndex} />
          )}

          {/* FFT Graph Image Preview Card (shown when an FFT image is loaded) */}
          {!analyzing && activeImageBase64 && !diagnosisResult && (
            <div
              className="card"
              style={{
                padding: '24px',
                display: 'flex',
                flexDirection: 'column',
                gap: 16,
                background: 'var(--bg-surface)'
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 10 }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', fontWeight: 700, color: 'var(--accent-primary)', textTransform: 'uppercase' }}>
                      2D CNN Inference Pipeline · Experimental
                    </span>
                    <span style={{ fontSize: 10, padding: '1px 6px', borderRadius: 4, background: 'rgba(99,102,241,0.15)', color: 'var(--accent-primary)', fontWeight: 700 }}>
                      4-Class Softmax
                    </span>
                  </div>
                  <h3 style={{ fontSize: 18, fontWeight: 800, marginTop: 4 }}>
                    FFT Spectrum Line Graph Preview
                  </h3>
                </div>

                <button
                  type="button"
                  onClick={handleAnalyze}
                  disabled={analyzing}
                  className="btn btn-primary"
                  style={{ padding: '8px 16px', display: 'flex', alignItems: 'center', gap: 8 }}
                >
                  <span>Run 2D CNN Diagnosis</span>
                  <ArrowRight size={14} />
                </button>
              </div>

              {/* Image Display */}
              <div
                style={{
                  background: '#ffffff',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-strong)',
                  padding: '12px',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  minHeight: 280,
                  boxShadow: 'inset 0 2px 4px rgba(0,0,0,0.05)'
                }}
              >
                <img
                  src={activeImageBase64}
                  alt="FFT Graph Image"
                  style={{
                    maxWidth: '100%',
                    maxHeight: 340,
                    objectFit: 'contain',
                    borderRadius: 'var(--radius-sm)'
                  }}
                />
              </div>

              {/* Specifications & Parameter Checklist */}
              <div
                style={{
                  display: 'grid',
                  gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))',
                  gap: 10,
                  background: 'var(--bg-surface-raised)',
                  padding: '12px 14px',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: 11
                }}
              >
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', fontWeight: 700 }}>Target Classes</div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>Healthy, Inner Race, Outer Race, Ball</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', fontWeight: 700 }}>Supported Bandwidth</div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>0 to 6,000 Hz (linear amplitude in g)</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', fontWeight: 700 }}>Preprocessing</div>
                  <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginTop: 2 }}>Aspect-ratio preserved + 224×224 padding</div>
                </div>
                <div>
                  <div style={{ color: 'var(--text-muted)', fontSize: 10, textTransform: 'uppercase', fontWeight: 700 }}>Model Scope</div>
                  <div style={{ fontWeight: 600, color: 'var(--status-mild)', marginTop: 2 }}>Current defect state (not RUL)</div>
                </div>
              </div>
            </div>
          )}

          {/* Signal Telemetry Preview Card (Oscilloscope + FFT + Sonification + CTA) */}
          {!analyzing && !activeImageBase64 && (
            <SignalPreviewCard
              signal={activeSignal}
              samplingRateHz={samplingRateHz}
              signalUnit={signalUnit}
              sourceFilename={sourceFilename}
              sourceType={sourceType}
              spectrumData={spectrumData}
              onAnalyze={handleAnalyze}
              analyzing={analyzing}
              onOpenDemoDrawer={() => setIsDemoDrawerOpen(true)}
            />
          )}

          {/* Post-Inference Diagnosis & Recommendations */}
          {!analyzing && diagnosisResult && (
            <>
              {/* Strong Hero Diagnosis Panel with ISO Gauge */}
              <DiagnosisPanel
                diagnosisResult={diagnosisResult}
                selectedMachine={selectedMachine}
                onReset={handleClearSignal}
                onScheduleMaintenance={onScheduleMaintenance}
              />

              {/* Multi-Window Continuity Aggregation & Timeline Scrubber */}
              <MultiWindowDistribution diagnosisResult={diagnosisResult} />

              {/* Expandable Technical Audit Section */}
              <TechnicalAuditSection diagnosisResult={diagnosisResult} />
            </>
          )}
        </div>
      </div>

      {/* ── Secondary Demo Samples Modal / Drawer ── */}
      <DemoSamplesDrawer
        isOpen={isDemoDrawerOpen}
        onClose={() => setIsDemoDrawerOpen(false)}
        onLoadDemo={handleLoadDemo}
        activeFilename={sourceFilename}
      />
    </div>
  );
}
