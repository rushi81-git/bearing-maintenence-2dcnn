import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileSpreadsheet,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Trash2,
  Sparkles,
  Sliders,
  FileText,
  Image as ImageIcon,
  Crop
} from 'lucide-react';
import { SignalSynthesizerCard } from './SignalSynthesizerCard';

export function SignalUploadCard({
  sourceFilename,
  sourceType,
  sampleCount,
  parsedCsv,
  selectedColumnIndex,
  onColumnChange,
  samplingRateHz,
  onSamplingRateChange,
  signalUnit,
  onSignalUnitChange,
  onFileUpload,
  onClearSignal,
  onOpenDemoDrawer,
  onSynthesizeSignal,
  // FFT Graph Image props
  activeImageBase64 = null,
  onImageUpload = null,
  onClearImage = null,
  cropBox = null,
  onCropBoxChange = null
}) {
  const [activeMode, setActiveMode] = useState(sourceType === 'fft_graph_image' ? 'fft_image' : 'upload'); // 'upload' | 'synthesizer' | 'fft_image'
  const [isDragging, setIsDragging] = useState(false);
  const [showCropControls, setShowCropControls] = useState(false);
  const fileInputRef = useRef(null);
  const imageInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = e.dataTransfer?.files;
    if (files && files.length > 0) {
      const file = files[0];
      if (activeMode === 'fft_image') {
        handleImageFile(file);
      } else if (fileInputRef.current) {
        onFileUpload({ target: { files: [file] } });
      }
    }
  };

  const handleImageFile = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      alert('Please upload a valid PNG or JPG image file.');
      return;
    }
    const reader = new FileReader();
    reader.onload = (evt) => {
      const b64 = evt.target?.result;
      if (onImageUpload) {
        onImageUpload({
          imageBase64: b64,
          filename: file.name
        });
      }
    };
    reader.readAsDataURL(file);
  };

  const isSignalValid = sampleCount >= 1024;
  const isDemo = sourceType === 'demo';
  const isSimulated = sourceType === 'simulated';
  const isImageMode = activeMode === 'fft_image';

  return (
    <div
      className="card"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 16
      }}
    >
      {/* Card Header & Mode Switcher */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
        <label
          style={{
            fontSize: 11,
            fontWeight: 700,
            color: 'var(--text-secondary)',
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            fontFamily: 'var(--font-mono)'
          }}
        >
          2. Signal / FFT Ingestion
        </label>

        {/* Try Demo Drawer Button */}
        <button
          type="button"
          onClick={onOpenDemoDrawer}
          className="btn btn-ghost"
          style={{
            padding: '3px 8px',
            fontSize: 11,
            color: 'var(--status-mild)',
            background: 'var(--status-mild-bg)',
            borderRadius: 'var(--radius-sm)'
          }}
        >
          <Sparkles size={12} />
          Demo Catalog
        </button>
      </div>

      {/* Segmented Mode Picker: File Upload vs Kinematic Synthesizer vs FFT Graph Image */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr 1.2fr',
          background: 'var(--bg-surface-sunken)',
          padding: 3,
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
          gap: 2
        }}
      >
        <button
          type="button"
          onClick={() => setActiveMode('upload')}
          style={{
            padding: '6px 6px',
            fontSize: 10,
            fontWeight: 600,
            fontFamily: 'var(--font-sans)',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: activeMode === 'upload' ? 'var(--bg-surface-raised)' : 'transparent',
            color: activeMode === 'upload' ? 'var(--accent-primary)' : 'var(--text-muted)',
            boxShadow: activeMode === 'upload' ? 'var(--shadow-xs)' : 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4
          }}
        >
          <UploadCloud size={12} />
          <span>Upload CSV</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveMode('synthesizer')}
          style={{
            padding: '6px 6px',
            fontSize: 10,
            fontWeight: 600,
            fontFamily: 'var(--font-sans)',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: activeMode === 'synthesizer' ? 'var(--bg-surface-raised)' : 'transparent',
            color: activeMode === 'synthesizer' ? 'var(--accent-primary)' : 'var(--text-muted)',
            boxShadow: activeMode === 'synthesizer' ? 'var(--shadow-xs)' : 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4
          }}
        >
          <Sliders size={12} />
          <span>Synthesizer</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveMode('fft_image')}
          style={{
            padding: '6px 6px',
            fontSize: 10,
            fontWeight: 700,
            fontFamily: 'var(--font-sans)',
            borderRadius: 'var(--radius-sm)',
            border: 'none',
            background: activeMode === 'fft_image' ? 'var(--bg-surface-raised)' : 'transparent',
            color: activeMode === 'fft_image' ? 'var(--accent-primary)' : 'var(--text-muted)',
            boxShadow: activeMode === 'fft_image' ? 'var(--shadow-xs)' : 'none',
            cursor: 'pointer',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 4
          }}
          title="FFT Graph Image (2D CNN) — Experimental"
        >
          <ImageIcon size={12} />
          <span>FFT Graph Image</span>
          <span style={{ fontSize: 8, padding: '1px 3px', borderRadius: 4, background: 'rgba(99,102,241,0.15)', color: 'var(--accent-primary)' }}>EXP</span>
        </button>
      </div>

      {/* Mode 1: File Upload */}
      {activeMode === 'upload' && (
        <>
          {!sourceFilename || (sourceFilename === 'machine_vibration.csv' && sampleCount === 0) ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
              style={{
                border: `2px dashed ${isDragging ? 'var(--accent-primary)' : 'var(--border-strong)'}`,
                borderRadius: 'var(--radius-lg)',
                padding: '28px 18px',
                textAlign: 'center',
                background: isDragging ? 'var(--accent-subtle)' : 'var(--bg-surface-sunken)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                position: 'relative',
                transition: 'all 0.2s ease'
              }}
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".csv,.txt"
                onChange={onFileUpload}
                style={{ display: 'none' }}
              />

              <div
                style={{
                  width: 44,
                  height: 44,
                  borderRadius: 'var(--radius-full)',
                  background: 'var(--accent-subtle)',
                  color: 'var(--accent-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: 10,
                  transform: isDragging ? 'scale(1.1)' : 'scale(1)',
                  transition: 'transform 0.2s ease'
                }}
              >
                <UploadCloud size={22} />
              </div>

              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                {isDragging ? 'Drop CSV File Here' : 'Click or Drag Vibration CSV'}
              </div>

              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4 }}>
                Supports single or multi-channel time-series data (.csv, .txt)
              </div>

              <div
                style={{
                  marginTop: 12,
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--accent-primary)',
                  background: 'var(--bg-surface)',
                  padding: '3px 8px',
                  borderRadius: 'var(--radius-full)',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                Minimum 1024 numeric samples required
              </div>
            </div>
          ) : (
            /* Uploaded File Details Card */
            <div
              style={{
                background: 'var(--bg-surface-raised)',
                border: `1px solid ${isSignalValid ? 'var(--status-healthy-border)' : 'var(--status-mild-border)'}`,
                borderRadius: 'var(--radius-md)',
                padding: '14px',
                display: 'flex',
                flexDirection: 'column',
                gap: 12
              }}
            >
              {/* File Name & Status Row */}
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
                  <div
                    style={{
                      width: 34,
                      height: 34,
                      borderRadius: 'var(--radius-md)',
                      background: isDemo ? 'rgba(251, 191, 36, 0.12)' : isSimulated ? 'var(--accent-subtle)' : 'var(--bg-surface-sunken)',
                      color: isDemo ? 'var(--status-mild)' : 'var(--accent-primary)',
                      display: 'flex',
                      alignItems: 'center',
                      justifyContent: 'center',
                      flexShrink: 0
                    }}
                  >
                    {isDemo ? <Sparkles size={16} /> : isSimulated ? <Sliders size={16} /> : <FileSpreadsheet size={16} />}
                  </div>

                  <div style={{ minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: 'var(--text-primary)',
                        fontFamily: 'var(--font-mono)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}
                      title={sourceFilename}
                    >
                      {sourceFilename}
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--text-muted)' }}>
                      <span>{isDemo ? 'CWRU Benchmark' : isSimulated ? 'Synthesized' : 'Custom CSV'}</span>
                      <span>·</span>
                      <span style={{ fontFamily: 'var(--font-mono)', color: isSignalValid ? 'var(--status-healthy)' : 'var(--status-severe)', fontWeight: 600 }}>
                        {sampleCount.toLocaleString()} pts
                      </span>
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexShrink: 0 }}>
                  <button
                    type="button"
                    onClick={() => fileInputRef.current?.click()}
                    className="btn btn-outline"
                    style={{ padding: '5px 8px', fontSize: 11 }}
                    title="Replace CSV File"
                  >
                    <RefreshCw size={11} />
                    Change
                  </button>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".csv,.txt"
                    onChange={onFileUpload}
                    style={{ display: 'none' }}
                  />
                  <button
                    type="button"
                    onClick={onClearSignal}
                    className="btn btn-ghost"
                    style={{ padding: '5px', color: 'var(--status-severe)' }}
                    title="Clear Signal"
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>

              {/* Multi-Column Condition Report Advisory Banner */}
              {parsedCsv && parsedCsv.hasMultiColumnWarning && (
                <div
                  style={{
                    display: 'flex',
                    alignItems: 'flex-start',
                    gap: 10,
                    padding: '10px 12px',
                    background: 'rgba(251, 191, 36, 0.08)',
                    border: '1px solid var(--status-mild-border)',
                    borderRadius: 'var(--radius-sm)'
                  }}
                >
                  <AlertTriangle size={14} style={{ color: 'var(--status-mild)', flexShrink: 0, marginTop: 1 }} />
                  <div style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                    <strong style={{ color: 'var(--status-mild)' }}>Multi-column report detected</strong> — no single
                    column has ≥ 1024 samples. A flattened signal stream is offered as the first option below.
                    For best accuracy, provide a raw time-domain vibration CSV (one amplitude value per row).
                  </div>
                </div>
              )}

              {/* Multi-Channel Selection (if CSV has multiple columns) */}
              {parsedCsv && parsedCsv.columnCount > 1 && (
                <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: 10 }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 6 }}>
                    Select Signal Stream / Channel:
                  </label>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                    {parsedCsv.columns.map((col) => {
                      const isActive = selectedColumnIndex === col.columnIndex;
                      const isReady = col.sampleCount >= 1024;
                      return (
                        <div
                          key={col.columnIndex}
                          onClick={() => onColumnChange(col.columnIndex)}
                          style={{
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'space-between',
                            padding: '7px 10px',
                            borderRadius: 'var(--radius-sm)',
                            border: `1px solid ${isActive ? 'var(--accent-primary)' : 'var(--border-subtle)'}`,
                            background: isActive ? 'var(--accent-subtle)' : 'var(--bg-surface-sunken)',
                            cursor: 'pointer',
                            fontSize: 11,
                            gap: 8
                          }}
                        >
                          <div style={{ display: 'flex', alignItems: 'center', gap: 6, minWidth: 0, flex: 1 }}>
                            <FileText size={13} style={{ color: isActive ? 'var(--accent-primary)' : 'var(--text-muted)', flexShrink: 0 }} />
                            <span
                              style={{
                                fontWeight: 600,
                                fontFamily: 'var(--font-mono)',
                                overflow: 'hidden',
                                textOverflow: 'ellipsis',
                                whiteSpace: 'nowrap'
                              }}
                              title={col.columnName}
                            >
                              {col.columnName}
                            </span>
                            {col.isConcatenated && (
                              <span
                                style={{
                                  fontSize: 9,
                                  padding: '1px 5px',
                                  borderRadius: 'var(--radius-full)',
                                  background: 'var(--status-mild-bg)',
                                  color: 'var(--status-mild)',
                                  border: '1px solid var(--status-mild-border)',
                                  fontWeight: 700,
                                  flexShrink: 0
                                }}
                              >
                                AUTO-CONCAT
                              </span>
                            )}
                          </div>
                          <span
                            style={{
                              color: isReady ? 'var(--status-healthy)' : 'var(--status-severe)',
                              fontFamily: 'var(--font-mono)',
                              fontSize: 10,
                              fontWeight: 600,
                              flexShrink: 0
                            }}
                          >
                            {col.sampleCount.toLocaleString()} pts{!isReady ? ' ⚠' : ''}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}
        </>
      )}

      {/* Mode 2: Kinematic Synthesizer */}
      {activeMode === 'synthesizer' && (
        <SignalSynthesizerCard onSynthesizeSignal={onSynthesizeSignal} />
      )}

      {/* Mode 3: FFT Graph Image (2D CNN) — Experimental */}
      {activeMode === 'fft_image' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          {!activeImageBase64 ? (
            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              onClick={() => imageInputRef.current?.click()}
              style={{
                border: `2px dashed ${isDragging ? 'var(--accent-primary)' : 'var(--border-strong)'}`,
                borderRadius: 'var(--radius-lg)',
                padding: '24px 16px',
                textAlign: 'center',
                background: isDragging ? 'var(--accent-subtle)' : 'var(--bg-surface-sunken)',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
            >
              <input
                ref={imageInputRef}
                type="file"
                accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                onChange={(e) => handleImageFile(e.target.files?.[0])}
                style={{ display: 'none' }}
              />

              <div
                style={{
                  width: 44,
                  height: 44,
                  borderRadius: 'var(--radius-full)',
                  background: 'rgba(99, 102, 241, 0.12)',
                  color: 'var(--accent-primary)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  marginBottom: 10
                }}
              >
                <ImageIcon size={22} />
              </div>

              <div style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)' }}>
                {isDragging ? 'Drop FFT Graph Image Here' : 'Click or Drag FFT Graph Image (.png, .jpg)'}
              </div>

              <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 4, maxWidth: 280 }}>
                Upload an FFT vibration spectrum line graph image (0–6,000 Hz, linear acceleration in g).
              </div>

              <div
                style={{
                  marginTop: 10,
                  fontSize: 10,
                  fontFamily: 'var(--font-mono)',
                  color: 'var(--accent-primary)',
                  background: 'var(--bg-surface)',
                  padding: '3px 8px',
                  borderRadius: 'var(--radius-full)',
                  border: '1px solid var(--border-subtle)'
                }}
              >
                2D CNN · 4-Class Classification (Healthy, IR, OR, Ball)
              </div>
            </div>
          ) : (
            /* Uploaded Image Preview & Details Card */
            <div
              style={{
                background: 'var(--bg-surface-raised)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-md)',
                padding: '12px',
                display: 'flex',
                flexDirection: 'column',
                gap: 12
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 10 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, minWidth: 0 }}>
                  <img
                    src={activeImageBase64}
                    alt="FFT Graph Preview"
                    style={{
                      width: 50,
                      height: 50,
                      objectFit: 'cover',
                      borderRadius: 'var(--radius-sm)',
                      border: '1px solid var(--border-subtle)',
                      background: '#fff'
                    }}
                  />
                  <div style={{ minWidth: 0 }}>
                    <div
                      style={{
                        fontSize: 12,
                        fontWeight: 700,
                        color: 'var(--text-primary)',
                        fontFamily: 'var(--font-mono)',
                        whiteSpace: 'nowrap',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis'
                      }}
                    >
                      {sourceFilename || 'fft_graph_image.png'}
                    </div>
                    <div style={{ fontSize: 11, color: 'var(--status-healthy)', fontWeight: 600 }}>
                      FFT Graph Image Ready for 2D CNN
                    </div>
                  </div>
                </div>

                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <button
                    type="button"
                    onClick={() => setShowCropControls(prev => !prev)}
                    className="btn btn-outline"
                    style={{ padding: '4px 8px', fontSize: 11 }}
                    title="Manual Graph Area Crop"
                  >
                    <Crop size={12} />
                    Crop
                  </button>
                  <button
                    type="button"
                    onClick={() => imageInputRef.current?.click()}
                    className="btn btn-outline"
                    style={{ padding: '4px 8px', fontSize: 11 }}
                  >
                    <RefreshCw size={11} />
                    Change
                  </button>
                  <input
                    ref={imageInputRef}
                    type="file"
                    accept=".png,.jpg,.jpeg,image/png,image/jpeg"
                    onChange={(e) => handleImageFile(e.target.files?.[0])}
                    style={{ display: 'none' }}
                  />
                  <button
                    type="button"
                    onClick={onClearImage}
                    className="btn btn-ghost"
                    style={{ padding: '4px 6px', color: 'var(--status-severe)' }}
                  >
                    <Trash2 size={13} />
                  </button>
                </div>
              </div>

              {/* Manual Graph Area Crop Controls */}
              {showCropControls && (
                <div
                  style={{
                    background: 'var(--bg-surface-sunken)',
                    padding: '10px',
                    borderRadius: 'var(--radius-sm)',
                    border: '1px solid var(--border-subtle)',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: 8
                  }}
                >
                  <div style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-primary)', display: 'flex', alignItems: 'center', gap: 6 }}>
                    <Crop size={12} />
                    <span>Manual Graph Area Bounding Box (px):</span>
                  </div>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 6 }}>
                    {['X1', 'Y1', 'X2', 'Y2'].map((coord, idx) => (
                      <div key={coord}>
                        <label style={{ fontSize: 9, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>{coord}</label>
                        <input
                          type="number"
                          placeholder="0"
                          value={cropBox ? cropBox[idx] : ''}
                          onChange={(e) => {
                            const val = parseInt(e.target.value, 10);
                            const newBox = cropBox ? [...cropBox] : [0, 0, 500, 500];
                            newBox[idx] = isNaN(val) ? 0 : val;
                            if (onCropBoxChange) onCropBoxChange(newBox);
                          }}
                          style={{
                            width: '100%',
                            padding: '4px 6px',
                            fontSize: 11,
                            fontFamily: 'var(--font-mono)',
                            background: 'var(--bg-surface)',
                            border: '1px solid var(--border-subtle)',
                            borderRadius: 4,
                            color: 'var(--text-primary)'
                          }}
                        />
                      </div>
                    ))}
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6 }}>
                    <button
                      type="button"
                      onClick={() => onCropBoxChange && onCropBoxChange(null)}
                      className="btn btn-ghost"
                      style={{ fontSize: 10, padding: '2px 6px' }}
                    >
                      Reset Full Image
                    </button>
                  </div>
                </div>
              )}

              {/* Specification Notice */}
              <div
                style={{
                  background: 'rgba(99, 102, 241, 0.06)',
                  border: '1px solid rgba(99, 102, 241, 0.2)',
                  borderRadius: 'var(--radius-sm)',
                  padding: '8px 10px',
                  fontSize: 10,
                  color: 'var(--text-secondary)',
                  lineHeight: 1.4
                }}
              >
                <strong style={{ color: 'var(--accent-primary)' }}>Required Analyzer Settings:</strong> Span 0–6,000 Hz, linear acceleration amplitude in g, Hann window.
                <br />
                <em>Note: Model predicts current fault state, not Remaining Useful Life (RUL).</em>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Sampling Rate & Unit Tuning Controls */}
      <div
        style={{
          display: 'grid',
          gridTemplateColumns: '1fr 1fr',
          gap: 10,
          borderTop: '1px solid var(--border-subtle)',
          paddingTop: 12
        }}
      >
        <div>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
            Sampling Rate:
          </label>
          <select
            className="select-field"
            style={{ width: '100%', padding: '6px 8px', fontSize: 11 }}
            value={samplingRateHz}
            onChange={(e) => onSamplingRateChange(parseInt(e.target.value, 10))}
          >
            <option value="48000">48,000 Hz (CWRU High-Res)</option>
            <option value="12000">12,000 Hz (Standard CWRU)</option>
            <option value="25600">25,600 Hz (Industrial ISO)</option>
            <option value="10000">10,000 Hz (Telemetry)</option>
          </select>
        </div>

        <div>
          <label style={{ fontSize: 11, fontWeight: 600, color: 'var(--text-secondary)', display: 'block', marginBottom: 4 }}>
            Vibration Unit:
          </label>
          <select
            className="select-field"
            style={{ width: '100%', padding: '6px 8px', fontSize: 11 }}
            value={signalUnit}
            onChange={(e) => onSignalUnitChange(e.target.value)}
          >
            <option value="g">g (Gravitational Accel)</option>
            <option value="m/s²">m/s² (SI Acceleration)</option>
            <option value="mm/s">mm/s (Vibration Velocity)</option>
            <option value="mV">mV (Piezoelectric Sensor)</option>
          </select>
        </div>
      </div>
    </div>
  );
}
