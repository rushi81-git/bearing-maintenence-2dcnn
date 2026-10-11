"""
=============================================================
ml/preprocessing/preprocess.py
CWRU Bearing Fault Diagnosis — Unified Preprocessing Module

Standardizes 1024-sample vibration window preprocessing for both:
  - Branch A: Classical ML (12 statistical time-domain features)
  - Branch B: 1D Deep CNN (Normalized raw waveform [-1, 1])

Guarantees:
  1. Universal 1024-sample window size across training and inference.
  2. Exact mathematical formulas for all features (including shape factor = rms / mean(|x|)).
  3. Epsilon protection against zero / constant / near-zero signals.
  4. Leakage-safe block-based partitioning (70% train, 10% val, 20% test).
  5. Scaler fitted ONLY on training split (no scaler leakage).
  6. Fixed canonical feature ordering.
=============================================================
"""

import os
import sys
import json
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis as scipy_kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder

warnings.filterwarnings('ignore')

PREPROCESS_DIR   = os.path.dirname(os.path.abspath(__file__))
BASE_DIR         = os.path.dirname(PREPROCESS_DIR)
ROOT_DIR         = os.path.dirname(BASE_DIR)
DATA_DIR         = os.path.join(BASE_DIR, 'data')
MODELS_DIR       = os.path.join(BASE_DIR, 'saved_models')
CWRU_DATASET_DIR = os.path.join(ROOT_DIR, 'CWRU_dataset')
os.makedirs(MODELS_DIR, exist_ok=True)

# Default to CWRU_dataset directory if present, otherwise ml/data
PRIMARY_NPZ = os.path.join(CWRU_DATASET_DIR, 'CWRU_48k_load_1_CNN_data.npz')
FALLBACK_NPZ = os.path.join(DATA_DIR, 'CWRU_48k_load_1_CNN_data.npz')
NPZ_PATH = PRIMARY_NPZ if os.path.exists(PRIMARY_NPZ) else FALLBACK_NPZ
CSV_PATH = os.path.join(DATA_DIR, 'feature_time_48k_2048_load_1.csv')

# ─── Constants ────────────────────────────────────────────────────────────────
RANDOM_SEED       = 42
WINDOW_LENGTH     = 1024
TEST_RATIO        = 0.20   # 20% held-out test
VAL_RATIO         = 0.10   # 10% validation
TRAIN_RATIO       = 0.70   # 70% training
EPSILON           = 1e-10  # Numerical stability guard

# Canonical Feature Order (Fixed and immutable for inference consistency)
CANONICAL_FEATURES = [
    "max",
    "min",
    "mean",
    "std",
    "rms",
    "skewness",
    "kurtosis",
    "crest_factor",
    "shape_factor",
    "peak_to_peak",
    "mean_absolute_value",
    "variance"
]

# 10 Standard CWRU Classes
CWRU_CLASSES = [
    'Normal',
    'Ball_007',
    'Ball_014',
    'Ball_021',
    'IR_007',
    'IR_014',
    'IR_021',
    'OR_007',
    'OR_014',
    'OR_021'
]

# Fault Metadata Mappings
FAULT_METADATA = {
    'Normal':   {'fault_type': 'Normal',     'fault_size_inches': 0.0,   'fault_size_mm': 0.0,   'severity': 'Healthy',  'bearing_status': 'Healthy'},
    'Ball_007': {'fault_type': 'Ball',       'fault_size_inches': 0.007, 'fault_size_mm': 0.178, 'severity': 'Mild',     'bearing_status': 'Fault Detected'},
    'Ball_014': {'fault_type': 'Ball',       'fault_size_inches': 0.014, 'fault_size_mm': 0.356, 'severity': 'Moderate', 'bearing_status': 'Fault Detected'},
    'Ball_021': {'fault_type': 'Ball',       'fault_size_inches': 0.021, 'fault_size_mm': 0.533, 'severity': 'Severe',   'bearing_status': 'Fault Detected'},
    'IR_007':   {'fault_type': 'Inner Race', 'fault_size_inches': 0.007, 'fault_size_mm': 0.178, 'severity': 'Mild',     'bearing_status': 'Fault Detected'},
    'IR_014':   {'fault_type': 'Inner Race', 'fault_size_inches': 0.014, 'fault_size_mm': 0.356, 'severity': 'Moderate', 'bearing_status': 'Fault Detected'},
    'IR_021':   {'fault_type': 'Inner Race', 'fault_size_inches': 0.021, 'fault_size_mm': 0.533, 'severity': 'Severe',   'bearing_status': 'Fault Detected'},
    'OR_007':   {'fault_type': 'Outer Race', 'fault_size_inches': 0.007, 'fault_size_mm': 0.178, 'severity': 'Mild',     'bearing_status': 'Fault Detected'},
    'OR_014':   {'fault_type': 'Outer Race', 'fault_size_inches': 0.014, 'fault_size_mm': 0.356, 'severity': 'Moderate', 'bearing_status': 'Fault Detected'},
    'OR_021':   {'fault_type': 'Outer Race', 'fault_size_inches': 0.021, 'fault_size_mm': 0.533, 'severity': 'Severe',   'bearing_status': 'Fault Detected'},
}


def normalize_class_name(name: str) -> str:
    """Normalize legacy class naming variations (e.g. 'Normal_1' -> 'Normal', 'OR_007_6_1' -> 'OR_007')."""
    s = str(name).strip()
    if s.startswith('Normal'):
        return 'Normal'
    if s.startswith('Ball_007'):
        return 'Ball_007'
    if s.startswith('Ball_014'):
        return 'Ball_014'
    if s.startswith('Ball_021'):
        return 'Ball_021'
    if s.startswith('IR_007'):
        return 'IR_007'
    if s.startswith('IR_014'):
        return 'IR_014'
    if s.startswith('IR_021'):
        return 'IR_021'
    if s.startswith('OR_007'):
        return 'OR_007'
    if s.startswith('OR_014'):
        return 'OR_014'
    if s.startswith('OR_021'):
        return 'OR_021'
    return s


def compute_time_domain_features(signal: np.ndarray, remove_dc: bool = False) -> dict:
    """
    Compute 12 canonical statistical time-domain features from a 1D vibration signal.

    Formulas:
      1. max: max(x)
      2. min: min(x)
      3. mean: mean(x)
      4. std: sample standard deviation (ddof=1)
      5. rms: sqrt(mean(x^2))
      6. skewness: Fisher-Pearson 3rd moment
      7. kurtosis: Fisher excess kurtosis
      8. crest_factor: max(|x|) / (rms + eps)
      9. shape_factor: rms / (mean(|x|) + eps)  <- CORRECT FORMULA
      10. peak_to_peak: max(x) - min(x)
      11. mean_absolute_value: mean(|x|)
      12. variance: var(x, ddof=1)
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    if len(x) == 0:
        raise ValueError("Cannot extract features from empty signal array.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Signal contains non-finite values (NaN or Inf).")

    if remove_dc:
        x = x - np.mean(x)

    n = len(x)
    sig_max = float(np.max(x))
    sig_min = float(np.min(x))
    sig_mean = float(np.mean(x))
    sig_std = float(np.std(x, ddof=1)) if n > 1 else 0.0
    rms = float(np.sqrt(np.mean(x ** 2)))
    mav = float(np.mean(np.abs(x)))
    peak_abs = float(np.max(np.abs(x)))

    # Moments with guards
    if sig_std > EPSILON and n > 2:
        sig_skew = float(skew(x, bias=False))
        sig_kurt = float(scipy_kurtosis(x, fisher=True, bias=False))
    else:
        sig_skew = 0.0
        sig_kurt = 0.0

    crest_factor = float(peak_abs / (rms + EPSILON))
    shape_factor = float(rms / (mav + EPSILON))
    peak_to_peak = float(sig_max - sig_min)
    variance = float(sig_std ** 2)

    return {
        "max": sig_max,
        "min": sig_min,
        "mean": sig_mean,
        "std": sig_std,
        "rms": rms,
        "skewness": sig_skew,
        "kurtosis": sig_kurt,
        "crest_factor": crest_factor,
        "shape_factor": shape_factor,
        "peak_to_peak": peak_to_peak,
        "mean_absolute_value": mav,
        "variance": variance
    }


def extract_features_matrix(signals: np.ndarray, remove_dc: bool = False, verbose: bool = False) -> np.ndarray:
    """
    Extract 12 features for an array of 1D signals of shape (N, 1024).
    Returns ordered numpy array of shape (N, 12) adhering to CANONICAL_FEATURES.
    """
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 1:
        signals = signals.reshape(1, -1)
    elif signals.ndim == 3 and signals.shape[1] == 32 and signals.shape[2] == 32:
        signals = signals.reshape(signals.shape[0], -1)

    n_samples = len(signals)
    feature_matrix = np.zeros((n_samples, len(CANONICAL_FEATURES)), dtype=np.float64)

    for i, sig in enumerate(signals):
        if verbose and (i + 1) % 1000 == 0:
            print(f"  Extracted {i + 1}/{n_samples} feature rows...")
        f_dict = compute_time_domain_features(sig, remove_dc=remove_dc)
        feature_matrix[i] = [f_dict[col] for col in CANONICAL_FEATURES]

    return feature_matrix


def normalize_signals_for_cnn(signals: np.ndarray) -> np.ndarray:
    """
    Per-sample normalization for 1D CNN: scales each window to [-1, 1].
    signal / max(|signal| + eps)
    """
    signals = np.asarray(signals, dtype=np.float32)
    if signals.ndim == 3 and signals.shape[1] == 32 and signals.shape[2] == 32:
        signals = signals.reshape(signals.shape[0], -1)

    max_abs = np.max(np.abs(signals), axis=1, keepdims=True)
    max_abs = np.where(max_abs < EPSILON, 1.0, max_abs)
    return signals / max_abs


def augment_vibration_signals(signals: np.ndarray, labels: np.ndarray, seed: int = RANDOM_SEED) -> tuple:
    """
    Physically grounded data augmentation for 1D raw vibration signals:
      1. Random circular phase shift (+/- 96 samples) to simulate arbitrary sampling start times.
      2. Additive Gaussian noise (SNR 25-35 dB) to simulate real workshop sensor/electrical noise.
      3. Subtle amplitude perturbations (0.94x - 1.06x) for load/gain variations.
    Doubles training set diversity while preserving ground-truth fault harmonic physics.
    """
    rng = np.random.default_rng(seed)
    n_samples, length = signals.shape
    aug_signals = np.zeros_like(signals)

    for i in range(n_samples):
        sig = signals[i].copy()
        # 1. Circular phase shift
        shift = int(rng.integers(-96, 96))
        sig = np.roll(sig, shift)
        # 2. Amplitude scaling
        scale = float(rng.uniform(0.94, 1.06))
        sig = sig * scale
        # 3. Additive Gaussian noise
        std = float(np.std(sig))
        noise_level = std * float(rng.uniform(0.015, 0.035))
        noise = rng.normal(0, noise_level, size=length)
        sig = sig + noise

        aug_signals[i] = sig

    combined_signals = np.vstack([signals, aug_signals])
    combined_labels = np.concatenate([labels, labels])

    shuffle_idx = rng.permutation(len(combined_signals))
    return combined_signals[shuffle_idx], combined_labels[shuffle_idx]


def load_and_preprocess_cwru_1024(npz_path: str = None, augment_train: bool = True, verbose: bool = True):
    """
    Loads CWRU dataset (4600 samples, 1024 points each), applies block splitting,
    optionally generates augmented training signals for maximum generalization,
    extracts 1024-point classical features, fits scaler ONLY on train set, and prepares
    all artifacts.
    """
    if npz_path is None:
        npz_path = NPZ_PATH

    if not os.path.exists(npz_path):
        if os.path.exists(FALLBACK_NPZ):
            npz_path = FALLBACK_NPZ
        else:
            raise FileNotFoundError(f"CWRU dataset not found at {npz_path}")

    d = np.load(npz_path)
    data = d['data']    # (4600, 32, 32)
    raw_labels = d['labels']  # (4600,)

    # Flatten to (4600, 1024)
    signals = data.reshape(data.shape[0], -1).astype(np.float64)
    labels = np.array([normalize_class_name(lbl) for lbl in raw_labels])

    if verbose:
        print("=" * 60)
        print("LOADED CWRU 1024-POINT DATASET")
        print(f"Total samples: {len(signals)}, Length per sample: {signals.shape[1]}")
        print(f"Signal range: {signals.min():.4f}g to {signals.max():.4f}g")
        print(f"Class distribution: {pd.Series(labels).value_counts().to_dict()}")
        print("=" * 60)

    # ── Leakage-Safe Block Split ──────────────────────────────────────────────
    # For each class (460 sequential samples from continuous recording):
    # First 70% -> Train (322)
    # Next  10% -> Val   (46)
    # Last  20% -> Test  (92)
    train_indices, val_indices, test_indices = [], [], []

    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n = len(cls_idx)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)

        train_indices.extend(cls_idx[:n_train])
        val_indices.extend(cls_idx[n_train:n_train + n_val])
        test_indices.extend(cls_idx[n_train + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    # Raw signals split
    sig_train = signals[train_indices]
    sig_val   = signals[val_indices]
    sig_test  = signals[test_indices]

    y_train = labels[train_indices]
    y_val   = labels[val_indices]
    y_test  = labels[test_indices]

    # ── Branch A: Feature Extraction on 1024 Windows ─────────────────────────
    if verbose:
        print("\nExtracting 12 time-domain features on 1024 windows...")
    X_train_raw = extract_features_matrix(sig_train, verbose=False)
    X_val_raw   = extract_features_matrix(sig_val, verbose=False)
    X_test_raw  = extract_features_matrix(sig_test, verbose=False)

    # Scaler fit strictly on training set (No Scaler Leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_val_scaled   = scaler.transform(X_val_raw)
    X_test_scaled  = scaler.transform(X_test_raw)

    # Label Encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(CWRU_CLASSES)
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc   = label_encoder.transform(y_val)
    y_test_enc  = label_encoder.transform(y_test)

    # ── Branch B: 1D CNN Normalized Signals ──────────────────────────────────
    sig_train_norm = normalize_signals_for_cnn(sig_train)
    sig_val_norm   = normalize_signals_for_cnn(sig_val)
    sig_test_norm  = normalize_signals_for_cnn(sig_test)

    # ── Training Data Augmentation (Generalization Enhancement) ──────────────
    if augment_train:
        if verbose:
            print("Applying phase-shift, noise injection & scale augmentation to train split...")
        sig_train_aug_raw, y_train_aug = augment_vibration_signals(sig_train, y_train, seed=RANDOM_SEED)
        sig_train_aug_norm = normalize_signals_for_cnn(sig_train_aug_raw)
        y_train_aug_enc = label_encoder.transform(y_train_aug)
        if verbose:
            print(f"  Augmented training set: {len(sig_train_aug_norm)} samples (2x expansion)")
    else:
        sig_train_aug_raw = sig_train
        sig_train_aug_norm = sig_train_norm
        y_train_aug = y_train
        y_train_aug_enc = y_train_enc

    # ── Save Universal Artifacts ─────────────────────────────────────────────
    scaler_path = os.path.join(MODELS_DIR, 'scaler_cwru_1024_v1.joblib')
    scaler_pkl_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(scaler_pkl_path, 'wb') as f:
        pickle.dump(scaler, f)

    le_path = os.path.join(MODELS_DIR, 'label_encoder_v1.joblib')
    le_pkl_path = os.path.join(MODELS_DIR, 'label_encoder.pkl')
    with open(le_path, 'wb') as f:
        pickle.dump(label_encoder, f)
    with open(le_pkl_path, 'wb') as f:
        pickle.dump(label_encoder, f)

    class_mapping = {
        str(i): {
            'class_name': cls,
            'fault_type': FAULT_METADATA[cls]['fault_type'],
            'fault_size_inches': FAULT_METADATA[cls]['fault_size_inches'],
            'fault_size_mm': FAULT_METADATA[cls]['fault_size_mm'],
            'severity': FAULT_METADATA[cls]['severity'],
            'bearing_status': FAULT_METADATA[cls]['bearing_status'],
        }
        for i, cls in enumerate(label_encoder.classes_)
    }
    with open(os.path.join(MODELS_DIR, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)

    config = {
        'window_length': WINDOW_LENGTH,
        'feature_columns': CANONICAL_FEATURES,
        'classes': list(label_encoder.classes_),
        'n_classes': len(label_encoder.classes_),
        'split_method': 'block_split (70% train, 10% val, 20% test - leakage safe)',
        'sampling_rate_hz': 48000,
        'signal_unit': 'g',
        'train_samples': len(train_indices),
        'val_samples': len(val_indices),
        'test_samples': len(test_indices),
    }
    with open(os.path.join(MODELS_DIR, 'preprocessing_config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    if verbose:
        print("\nPreprocessing and artifacts generated successfully:")
        print(f"  Train samples: {len(X_train_scaled)}, Val: {len(X_val_scaled)}, Test: {len(X_test_scaled)}")
        print(f"  Feature shape: {X_train_scaled.shape}")
        print(f"  Signals shape for CNN: {sig_train_norm.shape}")

    return {
        'X_train': X_train_scaled,
        'X_val': X_val_scaled,
        'X_test': X_test_scaled,
        'y_train': y_train_enc,
        'y_val': y_val_enc,
        'y_test': y_test_enc,
        'y_train_labels': y_train,
        'y_val_labels': y_val,
        'y_test_labels': y_test,
        'sig_train': sig_train_norm,
        'sig_train_aug': sig_train_aug_norm,
        'y_train_aug': y_train_aug_enc,
        'sig_val': sig_val_norm,
        'sig_test': sig_test_norm,
        'raw_sig_train': sig_train,
        'raw_sig_val': sig_val,
        'raw_sig_test': sig_test,
        'scaler': scaler,
        'label_encoder': label_encoder,
        'class_mapping': class_mapping,
        'config': config
    }


# ─── SPECTRUM PREPROCESSING & CANONICAL PIPELINE ──────────────────────────────
CANONICAL_SPECTRUM_BINS = 512
CANONICAL_SPECTRUM_MAX_FREQ = 6000.0
CANONICAL_SPECTRUM_FREQS = np.linspace(0.0, CANONICAL_SPECTRUM_MAX_FREQ, CANONICAL_SPECTRUM_BINS)

CANONICAL_SPECTRAL_FEATURES = [
    "spectral_centroid",
    "spectral_spread",
    "spectral_crest_factor",
    "spectral_rolloff_85",
    "spectral_rolloff_95",
    "peak_frequency",
    "peak_amplitude",
    "band_energy_0_500",
    "band_energy_500_1500",
    "band_energy_1500_3000",
    "band_energy_3000_6000",
    "total_spectral_energy",
    "mean_spectral_amplitude",
    "std_spectral_amplitude"
]


def compute_fft_spectrum(signal: np.ndarray, sampling_rate_hz: int = 48000) -> tuple:
    """
    Documented, consistent FFT pipeline for raw vibration windows:
      1. Detrend: x - mean(x) (DC bias removal)
      2. Window: Periodic Hanning window w[n] = 0.5 * (1 - cos(2*pi*n/N))
      3. Real FFT: rfft for positive frequencies
      4. Single-sided peak amplitude scaling: A[k] = 2.0 / sum(w) * |X[k]|, DC scaled by 1.0 / sum(w)
      5. Returns (freqs_hz, amplitudes_g_peak)
    """
    x = np.asarray(signal, dtype=np.float64).ravel()
    n = len(x)
    if n < 16:
        raise ValueError(f"Signal too short for FFT ({n} samples; minimum 16 required).")

    x_detrend = x - np.mean(x)
    win = np.hanning(n)
    win_sum = float(np.sum(win))
    if win_sum < EPSILON:
        win_sum = 1.0
    win_scale = 2.0 / win_sum

    fft_vals = np.fft.rfft(x_detrend * win)
    amplitudes = np.abs(fft_vals) * win_scale
    amplitudes[0] /= 2.0  # DC bin is not doubled

    freqs = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate_hz))
    return freqs, amplitudes


def extract_spectral_features(freqs: np.ndarray, amplitudes: np.ndarray) -> dict:
    """
    Extract 14 canonical spectral features from frequency-amplitude spectrum.
    Adheres strictly to vibration condition monitoring standards.
    """
    f = np.asarray(freqs, dtype=np.float64)
    a = np.asarray(amplitudes, dtype=np.float64)

    total_amp = float(np.sum(a)) + EPSILON
    power = a ** 2
    total_energy = float(np.sum(power)) + EPSILON

    # Spectral centroid (spectral center of gravity in Hz)
    centroid = float(np.sum(f * a) / total_amp)

    # Spectral spread (spectral bandwidth / standard deviation around centroid in Hz)
    spread = float(np.sqrt(np.sum(((f - centroid) ** 2) * a) / total_amp))

    # Spectral crest factor (peak amplitude relative to spectral RMS)
    spec_rms = float(np.sqrt(np.mean(power))) + EPSILON
    peak_amp = float(np.max(a))
    crest_factor = float(peak_amp / spec_rms)

    # Dominant peak frequency
    peak_idx = int(np.argmax(a))
    peak_freq = float(f[peak_idx])

    # Spectral roll-off (frequencies below which 85% and 95% of energy is concentrated)
    cum_power = np.cumsum(power)
    idx_85 = int(np.searchsorted(cum_power, 0.85 * total_energy))
    idx_95 = int(np.searchsorted(cum_power, 0.95 * total_energy))
    rolloff_85 = float(f[min(idx_85, len(f) - 1)])
    rolloff_95 = float(f[min(idx_95, len(f) - 1)])

    # Energy in key diagnostic sub-bands
    e_0_500 = float(np.sum(power[f <= 500.0]) / total_energy)
    e_500_1500 = float(np.sum(power[(f > 500.0) & (f <= 1500.0)]) / total_energy)
    e_1500_3000 = float(np.sum(power[(f > 1500.0) & (f <= 3000.0)]) / total_energy)
    e_3000_6000 = float(np.sum(power[(f > 3000.0) & (f <= 6000.0)]) / total_energy)

    mean_amp = float(np.mean(a))
    std_amp = float(np.std(a))

    return {
        "spectral_centroid": centroid,
        "spectral_spread": spread,
        "spectral_crest_factor": crest_factor,
        "spectral_rolloff_85": rolloff_85,
        "spectral_rolloff_95": rolloff_95,
        "peak_frequency": peak_freq,
        "peak_amplitude": peak_amp,
        "band_energy_0_500": e_0_500,
        "band_energy_500_1500": e_500_1500,
        "band_energy_1500_3000": e_1500_3000,
        "band_energy_3000_6000": e_3000_6000,
        "total_spectral_energy": total_energy,
        "mean_spectral_amplitude": mean_amp,
        "std_spectral_amplitude": std_amp
    }


def validate_spectrum_input(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    unit: str = "g, peak",
    spectrum_type: str = "amplitude"
) -> dict:
    """
    Validate numeric data, frequency order and spacing, units, spectrum type,
    and frequency range. Gives clear errors for unsupported or ambiguous exports.
    """
    f = np.asarray(frequencies, dtype=np.float64).ravel()
    a = np.asarray(amplitudes, dtype=np.float64).ravel()

    if len(f) == 0 or len(a) == 0:
        raise ValueError("Spectrum data is empty. Both frequency and amplitude columns are required.")

    if len(f) != len(a):
        raise ValueError(
            f"Frequency and amplitude lengths mismatch: {len(f)} frequencies vs {len(a)} amplitudes."
        )

    if len(f) < 16:
        raise ValueError(f"Spectrum has too few frequency bins ({len(f)}; minimum 16 required).")

    if not np.all(np.isfinite(f)):
        raise ValueError("Frequency array contains NaN or Infinite values.")

    if not np.all(np.isfinite(a)):
        raise ValueError("Amplitude array contains NaN or Infinite values.")

    # Frequency order check: strictly monotonically increasing
    diffs = np.diff(f)
    if np.any(diffs <= 0):
        invalid_idx = np.where(diffs <= 0)[0][0]
        raise ValueError(
            f"Frequencies must be strictly monotonically increasing. "
            f"Violation at index {invalid_idx}: {f[invalid_idx]} >= {f[invalid_idx + 1]}."
        )

    if f[0] < 0:
        raise ValueError(f"Frequencies must be non-negative. Minimum frequency is {f[0]} Hz.")

    # Amplitude check: must be non-negative
    if np.any(a < -1e-6):
        neg_count = np.sum(a < -1e-6)
        raise ValueError(
            f"Amplitudes must be non-negative (received {neg_count} negative values, min: {np.min(a):.4f}). "
            "A negative signal indicates a raw time-domain waveform or corrupt data, not a magnitude spectrum."
        )

    # Frequency spacing check: uniform within tolerance
    median_step = float(np.median(diffs))
    if median_step <= 0:
        raise ValueError("Invalid frequency step: step size must be positive.")
    step_variation = np.max(np.abs(diffs - median_step))
    if step_variation > 0.10 * median_step and len(diffs) > 10:
        raise ValueError(
            f"Non-uniform frequency spacing detected. Frequency step varies by up to "
            f"{step_variation:.4f} Hz from median step {median_step:.4f} Hz (>10% variation)."
        )

    # Unit and spectrum type validation
    u_clean = str(unit).lower().strip()
    s_type = str(spectrum_type).lower().strip()

    if "psd" in u_clean or "g^2" in u_clean or "g2" in u_clean or "psd" in s_type or "power" in s_type:
        raise ValueError(
            "Power Spectral Density (PSD / g²/Hz) is not currently supported. "
            "Please export linear acceleration amplitude (g, peak or g, RMS)."
        )

    if "db" in u_clean or "log" in u_clean or "db" in s_type:
        raise ValueError(
            "Logarithmic (dB) scale spectrum is not supported. Please export linear acceleration amplitude."
        )

    # Frequency range validation
    max_f = float(f[-1])
    min_f = float(f[0])

    if min_f > 100.0:
        raise ValueError(
            f"Spectrum starting frequency is {min_f:.1f} Hz. "
            "Diagnosis requires low-frequency coverage starting near 0 Hz (≤100 Hz) to inspect 1X/2X shaft harmonics."
        )

    if max_f < 6000.0:
        raise ValueError(
            f"Spectrum maximum frequency is {max_f:.1f} Hz, which is below the required 6,000 Hz diagnostic bandwidth. "
            "High-frequency bearing defect impact and resonance bands cannot be evaluated. "
            "Please provide a spectrum export with at least 0–6,000 Hz coverage (minimum 12 kHz sampling rate)."
        )

    return {
        "valid": True,
        "bin_count": len(f),
        "min_freq_hz": min_f,
        "max_freq_hz": max_f,
        "frequency_step_hz": median_step,
        "unit": unit,
        "spectrum_type": spectrum_type,
        "peak_amplitude": float(np.max(a)),
        "mean_amplitude": float(np.mean(a))
    }


def transform_spectrum_to_canonical(freqs: np.ndarray, amplitudes: np.ndarray) -> np.ndarray:
    """
    Interpolates linear amplitude spectrum onto the canonical frequency grid
    CANONICAL_SPECTRUM_FREQS (512 bins, 0 to 6000 Hz) and appends 14 canonical
    spectral features -> returns (526,) feature vector.
    """
    f = np.asarray(freqs, dtype=np.float64).ravel()
    a = np.asarray(amplitudes, dtype=np.float64).ravel()

    # Interpolate to canonical 512-bin grid
    canonical_bins = np.interp(CANONICAL_SPECTRUM_FREQS, f, a)

    # Extract canonical spectral features
    features_dict = extract_spectral_features(CANONICAL_SPECTRUM_FREQS, canonical_bins)
    feat_values = np.array([features_dict[col] for col in CANONICAL_SPECTRAL_FEATURES], dtype=np.float64)

    return np.concatenate([canonical_bins, feat_values])


def load_and_preprocess_cwru_spectrum(npz_path: str = None, verbose: bool = True):
    """
    Generates training spectra from CWRU 1024-point raw recordings using the documented
    FFT pipeline, applies identical leakage-safe 70/10/20 block splitting, and prepares
    canonical spectrum datasets for spectrum classifier training.
    """
    if npz_path is None:
        npz_path = NPZ_PATH
    if not os.path.exists(npz_path):
        if os.path.exists(FALLBACK_NPZ):
            npz_path = FALLBACK_NPZ
        else:
            raise FileNotFoundError(f"CWRU dataset not found at {npz_path}")

    d = np.load(npz_path)
    signals = d['data'].reshape(d['data'].shape[0], -1).astype(np.float64)
    labels = np.array([normalize_class_name(lbl) for lbl in d['labels']])

    # Leakage-safe block split: 70% Train, 10% Val, 20% Test
    train_indices, val_indices, test_indices = [], [], []
    for cls in np.unique(labels):
        cls_idx = np.where(labels == cls)[0]
        n = len(cls_idx)
        n_train = int(n * TRAIN_RATIO)
        n_val = int(n * VAL_RATIO)
        train_indices.extend(cls_idx[:n_train])
        val_indices.extend(cls_idx[n_train:n_train + n_val])
        test_indices.extend(cls_idx[n_train + n_val:])

    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    test_indices = np.array(test_indices)

    sig_train = signals[train_indices]
    sig_val = signals[val_indices]
    sig_test = signals[test_indices]

    y_train = labels[train_indices]
    y_val = labels[val_indices]
    y_test = labels[test_indices]

    label_encoder = LabelEncoder()
    label_encoder.fit(CWRU_CLASSES)
    y_train_enc = label_encoder.transform(y_train)
    y_val_enc = label_encoder.transform(y_val)
    y_test_enc = label_encoder.transform(y_test)

    if verbose:
        print("=" * 60)
        print("EXTRACTING SPECTRUM DATASET (CANONICAL 512 BINS + 14 SPECTRAL FEATURES)")
        print(f"Train samples: {len(sig_train)}, Val: {len(sig_val)}, Test: {len(sig_test)}")

    def batch_transform(sig_matrix):
        out = np.zeros((len(sig_matrix), CANONICAL_SPECTRUM_BINS + len(CANONICAL_SPECTRAL_FEATURES)), dtype=np.float64)
        for i, sig in enumerate(sig_matrix):
            freqs, amps = compute_fft_spectrum(sig, sampling_rate_hz=48000)
            out[i] = transform_spectrum_to_canonical(freqs, amps)
        return out

    X_train_spec = batch_transform(sig_train)
    X_val_spec = batch_transform(sig_val)
    X_test_spec = batch_transform(sig_test)

    if verbose:
        print(f"Feature vector shape: {X_train_spec.shape}")
        print("=" * 60)

    return {
        'X_train': X_train_spec,
        'X_val': X_val_spec,
        'X_test': X_test_spec,
        'y_train': y_train_enc,
        'y_val': y_val_enc,
        'y_test': y_test_enc,
        'y_train_labels': y_train,
        'y_val_labels': y_val,
        'y_test_labels': y_test,
        'label_encoder': label_encoder
    }


if __name__ == '__main__':
    load_and_preprocess_cwru_1024(verbose=True)

