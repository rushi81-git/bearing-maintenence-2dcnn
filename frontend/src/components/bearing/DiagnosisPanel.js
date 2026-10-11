import React, { useState } from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  Clock,
  ShieldCheck,
  Zap,
  Layers,
  Wrench,
  Sparkles,
  ArrowRight,
  TrendingUp,
  Cpu,
  Gauge,
  CalendarPlus,
  Info,
  Image as ImageIcon
} from 'lucide-react';
import { StatusBadge } from '../StatusBadge';
import { ISOSeverityGauge } from './ISOSeverityGauge';

export function DiagnosisPanel({
  diagnosisResult,
  selectedMachine,
  onReset,
  onScheduleMaintenance
}) {
  const [selectedFeature, setSelectedFeature] = useState(null);

  if (!diagnosisResult) return null;

  const isHealthy = diagnosisResult.bearing_status === 'Healthy' || diagnosisResult.fault_type === 'Normal';
  const severity = diagnosisResult.severity || (isHealthy ? 'Healthy' : 'Moderate');
  const confidencePercent = (diagnosisResult.prediction_probability * 100).toFixed(1);

  // Status-dependent theme colors
  let statusColor = 'var(--status-healthy)';
  let statusBg = 'var(--status-healthy-bg)';
  let statusBorder = 'var(--status-healthy-border)';

  if (severity === 'Mild') {
    statusColor = 'var(--status-mild)';
    statusBg = 'var(--status-mild-bg)';
    statusBorder = 'var(--status-mild-border)';
  } else if (severity === 'Moderate') {
    statusColor = 'var(--status-moderate)';
    statusBg = 'var(--status-moderate-bg)';
    statusBorder = 'var(--status-moderate-border)';
  } else if (severity === 'Severe') {
    statusColor = 'var(--status-severe)';
    statusBg = 'var(--status-severe-bg)';
    statusBorder = 'var(--status-severe-border)';
  }

  // Format fault diameter display
  const faultSizeInches = diagnosisResult.fault_size_inches;
  const faultSizeDisplay = faultSizeInches && faultSizeInches > 0
    ? `${faultSizeInches}" (${diagnosisResult.fault_size_mm} mm)`
    : 'None (Baseline)';

  const measuredRms = diagnosisResult.features?.rms ?? diagnosisResult.rms ?? 0.15;

  const FEATURE_INFO = {
    rms: {
      name: 'Root Mean Square (RMS)',
      desc: 'Reflects total continuous vibration power. Standard indicator for ISO 10816 machinery health.',
      healthyRange: '< 0.15 g',
      current: `${measuredRms} g`
    },
    kurtosis: {
      name: 'Kurtosis (Impulsiveness)',
      desc: 'Measures peakedness/tailedness of signal distribution. Gaussian baseline is ~3.0; values > 4.0 indicate sharp bearing impact shocks.',
      healthyRange: '2.8 - 3.2',
      current: `${diagnosisResult.features?.kurtosis ?? diagnosisResult.kurtosis ?? 'N/A'}`
    },
    crest_factor: {
      name: 'Crest Factor (Peak / RMS)',
      desc: 'Detects isolated transient spalling impacts before overall RMS vibration increases significantly.',
      healthyRange: '3.0 - 4.5',
      current: `${diagnosisResult.features?.crest_factor ?? diagnosisResult.crest_factor ?? 'N/A'}`
    },
    peak_to_peak: {
      name: 'Peak-to-Peak Amplitude',
      desc: 'Full excursion range between maximum positive and negative acceleration excursions.',
      healthyRange: '< 0.80 g',
      current: `${diagnosisResult.features?.peak_to_peak ?? diagnosisResult.peak_to_peak ?? 'N/A'} g`
    }
  };

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16
      }}
      className="animate-fade-in"
    >
      {/* ── Primary Hero Diagnosis Card ── */}
      <div
        className="card"
        style={{
          borderLeft: `5px solid ${statusColor}`,
          padding: '24px 28px',
          background: 'var(--bg-surface)',
          boxShadow: 'var(--shadow-card)'
        }}
      >
        <div
          style={{
            display: 'flex',
            alignItems: 'flex-start',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 20
          }}
        >
          {/* Main Status & Classification Details */}
          <div style={{ flex: '1 1 340px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8, flexWrap: 'wrap' }}>
              <StatusBadge status={diagnosisResult.bearing_status} size="md" />
              <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                {diagnosisResult.windows_analyzed} {diagnosisResult.windows_analyzed === 1 ? 'window analyzed' : 'windows evaluated'}
              </span>
              {selectedMachine && (
                <span style={{ fontSize: 11, color: 'var(--accent-primary)', fontWeight: 600, background: 'var(--accent-subtle)', padding: '2px 8px', borderRadius: 'var(--radius-full)' }}>
                  Asset: {selectedMachine.name}
                </span>
              )}
            </div>

            <h2
              style={{
                fontFamily: 'var(--font-heading)',
                fontSize: 24,
                fontWeight: 800,
                letterSpacing: '-0.02em',
                color: 'var(--text-primary)',
                lineHeight: 1.2
              }}
            >
              {isHealthy ? 'Normal Bearing Condition' : `${diagnosisResult.fault_type} Fault Detected`}
            </h2>

            {/* Diagnostic Core Key-Value Attributes */}
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
                gap: 10,
                marginTop: 16
              }}
            >
              <div
                style={{
                  padding: '10px 12px',
                  background: 'var(--bg-surface-raised)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Fault Category
                </div>
                <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                  {diagnosisResult.fault_type}
                </div>
              </div>

              <div
                style={{
                  padding: '10px 12px',
                  background: 'var(--bg-surface-raised)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Defect Diameter
                </div>
                <div style={{ fontSize: 14, fontWeight: 700, color: 'var(--text-primary)', fontFamily: 'var(--font-mono)', marginTop: 2 }}>
                  {faultSizeDisplay}
                </div>
              </div>

              <div
                style={{
                  padding: '10px 12px',
                  background: 'var(--bg-surface-raised)',
                  borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  Severity Level
                </div>
                <div style={{ fontSize: 14, fontWeight: 700, color: statusColor, textTransform: 'capitalize', marginTop: 2 }}>
                  {severity}
                </div>
              </div>
            </div>
          </div>

          {/* Model Probability Pill Card */}
          <div
            style={{
              background: 'var(--bg-surface-raised)',
              border: '1px solid var(--border-subtle)',
              borderRadius: 'var(--radius-lg)',
              padding: '16px 20px',
              textAlign: 'right',
              minWidth: 190,
              boxShadow: 'var(--shadow-xs)'
            }}
          >
            <div style={{ fontSize: 10, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
              Confidence Rating
            </div>
            <div
              style={{
                fontSize: 30,
                fontWeight: 800,
                color: 'var(--accent-primary)',
                fontFamily: 'var(--font-mono)',
                marginTop: 2,
                letterSpacing: '-0.03em'
              }}
            >
              {confidencePercent}%
            </div>
            <div style={{ fontSize: 11, color: 'var(--text-secondary)', marginTop: 2 }}>
              Softmax Probability
            </div>
          </div>
        </div>

        {/* ── ISO 10816-3 Gauge Sub-Component ── */}
        <div style={{ marginTop: 18 }}>
          <ISOSeverityGauge rmsValue={measuredRms} unit="g" />
        </div>

        {/* ── Autonomous Model Fit Engine Insight Strip ── */}
        <div
          style={{
            marginTop: 14,
            padding: '12px 16px',
            background: 'var(--bg-surface-raised)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            flexWrap: 'wrap',
            gap: 12
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div
              style={{
                width: 28,
                height: 28,
                borderRadius: 'var(--radius-sm)',
                background: 'var(--accent-subtle)',
                color: 'var(--accent-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                flexShrink: 0
              }}
            >
              <Cpu size={16} />
            </div>
            <div>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--accent-primary)', fontWeight: 700, textTransform: 'uppercase' }}>
                  Autonomous Engine Fit
                </span>
                <span
                  style={{
                    fontSize: 9,
                    fontFamily: 'var(--font-mono)',
                    padding: '1px 6px',
                    borderRadius: 'var(--radius-sm)',
                    background: 'var(--status-healthy-bg)',
                    color: 'var(--status-healthy)',
                    fontWeight: 700,
                    border: '1px solid var(--status-healthy-border)'
                  }}
                >
                  BEST FIT ACTIVE
                </span>
              </div>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)', marginTop: 2 }}>
                {diagnosisResult.auto_fit_details?.selected_model_name || diagnosisResult.model_version || '1D Deep CNN'}
              </div>
            </div>
          </div>

          <div style={{ fontSize: 11, color: 'var(--text-secondary)', maxWidth: 480, lineHeight: 1.4 }}>
            {diagnosisResult.auto_fit_details?.selection_rationale || 'The system autonomously evaluated raw telemetry dynamics and routed inference to the optimal best-fit architecture.'}
          </div>
        </div>

        {/* ── Verified Model Input Image (FFT 2D CNN Verification) ── */}
        {diagnosisResult.processed_image_base64 && (
          <div
            style={{
              marginTop: 16,
              padding: '16px 20px',
              background: 'var(--bg-surface-raised)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: 12
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <ImageIcon size={16} style={{ color: 'var(--accent-primary)' }} />
                <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                  Verified 2D CNN Model Input Image (224×224 RGB)
                </span>
              </div>
              <span
                style={{
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--accent-primary)',
                  background: 'var(--accent-subtle)',
                  padding: '2px 8px',
                  borderRadius: 'var(--radius-full)'
                }}
              >
                Exact Input Received by Neural Network
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: 16, flexWrap: 'wrap' }}>
              <img
                src={diagnosisResult.processed_image_base64}
                alt="224x224 Processed Input Received by 2D CNN"
                style={{
                  width: 120,
                  height: 120,
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-strong)',
                  background: '#ffffff',
                  boxShadow: 'var(--shadow-xs)'
                }}
              />
              <div style={{ flex: '1 1 260px', display: 'flex', flexDirection: 'column', gap: 6, fontSize: 11, color: 'var(--text-secondary)' }}>
                <div><strong>Resolution:</strong> 224 × 224 pixels (3 color channels RGB)</div>
                <div><strong>Diagnostic Span:</strong> 0 to 6,000 Hz (CWRU bearing defect harmonic range)</div>
                <div><strong>Scaling & Window:</strong> Linear acceleration (g), Hann periodic window</div>
                <div style={{ fontSize: 10, color: 'var(--status-mild)', marginTop: 4 }}>
                  ⚠️ <em>Notice: Classification assesses current fault condition only, not remaining useful life (RUL). Softmax activations are closed-set scores.</em>
                </div>
              </div>
            </div>

            {/* 4-Class Softmax Score Grid */}
            {diagnosisResult.class_scores && (
              <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 10, marginTop: 4 }}>
                <div style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 8, textTransform: 'uppercase', fontFamily: 'var(--font-mono)' }}>
                  4-Class Neural Activation Breakdown:
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: 8 }}>
                  {Object.entries(diagnosisResult.class_scores).map(([cName, cScore]) => {
                    const isWinner = cName === diagnosisResult.predicted_class;
                    return (
                      <div
                        key={cName}
                        style={{
                          padding: '8px 10px',
                          borderRadius: 'var(--radius-sm)',
                          background: isWinner ? 'var(--accent-subtle)' : 'var(--bg-surface)',
                          border: `1px solid ${isWinner ? 'var(--accent-primary)' : 'var(--border-subtle)'}`
                        }}
                      >
                        <div style={{ fontSize: 10, color: 'var(--text-muted)', fontWeight: 600 }}>{cName}</div>
                        <div style={{ fontSize: 14, fontWeight: 700, fontFamily: 'var(--font-mono)', color: isWinner ? 'var(--accent-primary)' : 'var(--text-primary)', marginTop: 2 }}>
                          {(cScore * 100).toFixed(1)}%
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* ── Actionable Maintenance Recommendation Box & Work Order CTA ── */}
        <div
          style={{
            marginTop: 16,
            padding: '16px 20px',
            background: 'var(--bg-surface-raised)',
            borderRadius: 'var(--radius-md)',
            border: `1px solid ${statusBorder}`,
            display: 'flex',
            alignItems: 'flex-start',
            gap: 14
          }}
        >
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: 'var(--radius-sm)',
              background: statusBg,
              color: statusColor,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
              marginTop: 2
            }}
          >
            {isHealthy ? <ShieldCheck size={18} /> : <AlertTriangle size={18} />}
          </div>

          <div style={{ flex: 1 }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
              <span
                style={{
                  fontSize: 12,
                  fontWeight: 700,
                  color: 'var(--text-primary)',
                  textTransform: 'uppercase',
                  letterSpacing: '0.04em'
                }}
              >
                Maintenance Directive · {diagnosisResult.urgency || 'Routine'} Urgency
              </span>
              <span
                style={{
                  fontSize: 11,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--text-secondary)',
                  background: 'var(--bg-surface)',
                  padding: '2px 8px',
                  borderRadius: 'var(--radius-sm)',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                Interval: {diagnosisResult.check_interval || 'Standard'}
              </span>
            </div>

            <p style={{ fontSize: 13, color: 'var(--text-secondary)', marginTop: 6, lineHeight: 1.6 }}>
              {diagnosisResult.recommendation}
            </p>

            {/* Quick Work Order Creation Button (if fault detected) */}
            {!isHealthy && onScheduleMaintenance && (
              <div style={{ marginTop: 12 }}>
                <button
                  type="button"
                  onClick={() => onScheduleMaintenance({
                    machineId: selectedMachine?.id,
                    machineName: selectedMachine?.name || 'Monitored Spindle',
                    task: `Bearing Overhaul: Inspect for ${diagnosisResult.fault_type} spalling (${faultSizeDisplay}). ${diagnosisResult.recommendation}`,
                    priority: diagnosisResult.urgency === 'Immediate' ? 'High' : 'Medium'
                  })}
                  className="btn btn-outline"
                  style={{
                    padding: '6px 14px',
                    fontSize: 12,
                    fontWeight: 600,
                    display: 'flex',
                    alignItems: 'center',
                    gap: 6
                  }}
                >
                  <CalendarPlus size={14} />
                  <span>Dispatch Work Order to Scheduler</span>
                </button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
