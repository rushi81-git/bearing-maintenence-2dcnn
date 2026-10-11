"""
=============================================================
ml/preprocessing/preprocess_fft_images.py
CWRU Bearing Fault Diagnosis — 2D FFT Graph Image Preprocessing

Features:
  1. 4-Class Mapping:
     - Normal -> 'Healthy'
     - IR_007, IR_014, IR_021 -> 'Inner Race Fault'
     - OR_007, OR_014, OR_021 -> 'Outer Race Fault'
     - Ball_007, Ball_014, Ball_021 -> 'Ball Fault'
  2. Leakage-safe temporal block partitioning per source recording
     (70% train, 10% val, 20% test) with reproducible seed and manifest saving.
  3. Window-to-FFT spectrum pipeline:
     - Detrend (mean subtraction)
     - Periodic Hann window
     - rfft single-sided amplitude scaling (g peak)
     - Diagnostic frequency range (0 to 6,000 Hz)
  4. Render to 224x224 RGB line graph images (consistent canvas, no text/labels).
  5. Realistic rendering variations for train set (line thickness, dark line colors, subtle background).
=============================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PREPROCESS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR       = os.path.dirname(PREPROCESS_DIR)
DATA_DIR       = os.path.join(BASE_DIR, 'data')
MODELS_DIR     = os.path.join(BASE_DIR, 'saved_models')
os.makedirs(MODELS_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from preprocessing.preprocess import compute_fft_spectrum, normalize_class_name

RANDOM_SEED          = 42
SAMPLING_RATE_HZ     = 48000
WINDOW_LENGTH        = 1024
CANONICAL_MAX_FREQ   = 6000.0   # Diagnostic bandwidth covering up to ~35x shaft harmonics
CANONICAL_MAX_AMP    = 1.50     # Fitted on CWRU training set (max observed is ~1.32g)
IMAGE_SIZE           = (224, 224)

# 4-Class Diagnostic Taxonomy
FOUR_CLASSES = ['Healthy', 'Inner Race Fault', 'Outer Race Fault', 'Ball Fault']
CLASS_TO_IDX = {cls_name: i for i, cls_name in enumerate(FOUR_CLASSES)}
IDX_TO_CLASS = {i: cls_name for i, cls_name in enumerate(FOUR_CLASSES)}

# Detailed CWRU Recording Metadata
CWRU_RECORDING_METADATA = {
    'Normal':   {'rec_id': '098', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.000"'},
    'Ball_007': {'rec_id': '122', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.007"'},
    'Ball_014': {'rec_id': '123', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.014"'},
    'Ball_021': {'rec_id': '124', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.021"'},
    'IR_007':   {'rec_id': '109', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.007"'},
    'IR_014':   {'rec_id': '110', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.014"'},
    'IR_021':   {'rec_id': '111', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.021"'},
    'OR_007':   {'rec_id': '135', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.007"'},
    'OR_014':   {'rec_id': '136', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.014"'},
    'OR_021':   {'rec_id': '137', 'channel': 'DE', 'sampling_rate': 48000, 'hp': 1, 'rpm': 1772, 'fault_size': '0.021"'}
}


def map_to_four_classes(raw_label: str) -> str:
    """Maps 10 CWRU condition labels to the 4 canonical diagnostic classes."""
    norm = normalize_class_name(raw_label)
    if norm.startswith('Normal'):
        return 'Healthy'
    if norm.startswith('IR'):
        return 'Inner Race Fault'
    if norm.startswith('OR'):
        return 'Outer Race Fault'
    if norm.startswith('Ball'):
        return 'Ball Fault'
    raise ValueError(f"Unrecognized CWRU label: '{raw_label}'")


def build_split_indices(labels: np.ndarray, train_ratio=0.70, val_ratio=0.10, test_ratio=0.20):
    """
    Leakage-safe temporal block partitioning per source recording:
      - 70% Train (first 322 sequential windows)
      - 10% Val   (next 46 sequential windows)
      - 20% Test  (final 92 sequential windows)
    Guarantees no window overlap between splits.
    """
    train_idx, val_idx, test_idx = [], [], []
    split_details = {}

    for cls in np.unique(labels):
        idxs = np.where(labels == cls)[0]
        n = len(idxs)
        n_tr = int(n * train_ratio)
        n_va = int(n * val_ratio)

        tr = idxs[:n_tr]
        va = idxs[n_tr:n_tr + n_va]
        te = idxs[n_tr + n_va:]

        train_idx.extend(tr)
        val_idx.extend(va)
        test_idx.extend(te)

        mapped_cls = map_to_four_classes(cls)
        split_details[cls] = {
            'mapped_4class': mapped_cls,
            'total_samples': n,
            'train_indices': [int(tr[0]), int(tr[-1])],
            'val_indices': [int(va[0]), int(va[-1])],
            'test_indices': [int(te[0]), int(te[-1])],
            'metadata': CWRU_RECORDING_METADATA.get(cls, {})
        }

    manifest = {
        'dataset_source': 'CWRU 48k Drive End Accelerometer, 1 HP Load (1772 RPM)',
        'window_length': WINDOW_LENGTH,
        'sampling_rate_hz': SAMPLING_RATE_HZ,
        'train_ratio': train_ratio,
        'val_ratio': val_ratio,
        'test_ratio': test_ratio,
        'random_seed': RANDOM_SEED,
        'total_samples': len(labels),
        'split_counts': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx)
        },
        'class_breakdown': {
            split_name: {
                c: int(np.sum(np.array([map_to_four_classes(labels[i]) for i in idx_list]) == c))
                for c in FOUR_CLASSES
            }
            for split_name, idx_list in [('train', train_idx), ('val', val_idx), ('test', test_idx)]
        },
        'recordings': split_details,
        'split_limitation_note': (
            "Each continuous CWRU recording was partitioned into strictly isolated temporal blocks "
            "(70% train, 10% val, 20% test). Because the CWRU 48k 1 HP benchmark contains a single continuous "
            "baseline recording for Healthy and single recordings per defect severity, a pure across-recording "
            "split would leave Healthy completely absent from either training or testing. Temporal block "
            "partitioning prevents window overlap and simulates evaluating a continuous machine monitoring run."
        )
    }

    manifest_path = os.path.join(MODELS_DIR, 'fft_image_split_manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    return np.array(train_idx), np.array(val_idx), np.array(test_idx), manifest


def render_fft_line_graph(
    freqs: np.ndarray,
    amplitudes: np.ndarray,
    image_size=(224, 224),
    max_freq_hz=CANONICAL_MAX_FREQ,
    max_amplitude=CANONICAL_MAX_AMP,
    line_color='black',
    bg_color='white',
    linewidth=1.2
) -> np.ndarray:
    """
    Renders an FFT frequency-versus-amplitude line graph as a 224x224 RGB image.
    Strictly standard across classes: consistent graph area, no titles, labels, or class identifiers.
    """
    fig = plt.figure(figsize=(image_size[0] / 100.0, image_size[1] / 100.0), dpi=100)
    fig.patch.set_facecolor(bg_color)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(bg_color)

    # Filter to canonical frequency range
    mask = freqs <= max_freq_hz
    f_sub = freqs[mask]
    a_sub = amplitudes[mask]

    ax.plot(f_sub, a_sub, color=line_color, linewidth=linewidth)
    ax.set_xlim(0, max_freq_hz)
    ax.set_ylim(0, max_amplitude)
    ax.axis('off')

    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    rgb = rgba[:image_size[1], :image_size[0], :3].copy()
    plt.close(fig)

    return rgb


def batch_render_fft_images(
    signals: np.ndarray,
    sampling_rate_hz: int = SAMPLING_RATE_HZ,
    is_training: bool = False,
    rng_seed: int = RANDOM_SEED,
    verbose: bool = True
) -> np.ndarray:
    """
    Converts 1D vibration signals into a batch of 224x224x3 uint8 FFT graph images.
    If is_training=True, subtle realistic visual variations are applied uniformly.
    """
    n = len(signals)
    images = np.zeros((n, IMAGE_SIZE[0], IMAGE_SIZE[1], 3), dtype=np.uint8)
    rng = np.random.default_rng(rng_seed)

    # Realistic visual variation options (applied uniformly across all classes)
    line_colors = ['#000000', '#1e293b', '#0f172a', '#111827', '#0a0a0a']
    bg_colors   = ['#ffffff', '#fafafa', '#f8fafc', '#f9fafb']

    # Pre-setup persistent canvas for high rendering throughput
    fig = plt.figure(figsize=(IMAGE_SIZE[0] / 100.0, IMAGE_SIZE[1] / 100.0), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    line, = ax.plot([], [], color='black', linewidth=1.2)
    ax.set_xlim(0, CANONICAL_MAX_FREQ)
    ax.set_ylim(0, CANONICAL_MAX_AMP)
    ax.axis('off')

    for i in range(n):
        if verbose and (i + 1) % 500 == 0:
            print(f"  Rendered {i + 1}/{n} FFT graph images...")

        sig = signals[i]
        freqs, amps = compute_fft_spectrum(sig, sampling_rate_hz=sampling_rate_hz)

        mask = freqs <= CANONICAL_MAX_FREQ
        f_sub = freqs[mask]
        a_sub = amps[mask]

        if is_training:
            # Realistic physical perturbations (load/sensor calibration +/- 5%)
            scale = float(rng.uniform(0.95, 1.05))
            a_sub = a_sub * scale
            lc = line_colors[int(rng.integers(0, len(line_colors)))]
            bg = bg_colors[int(rng.integers(0, len(bg_colors)))]
            lw = float(rng.uniform(1.0, 1.6))
            ax.set_facecolor(bg)
            fig.patch.set_facecolor(bg)
            line.set_color(lc)
            line.set_linewidth(lw)
        else:
            ax.set_facecolor('#ffffff')
            fig.patch.set_facecolor('#ffffff')
            line.set_color('#000000')
            line.set_linewidth(1.2)

        line.set_data(f_sub, a_sub)
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        images[i] = rgba[:IMAGE_SIZE[1], :IMAGE_SIZE[0], :3]

    plt.close(fig)
    return images


def load_cwru_fft_image_dataset(cache_file: str = None, verbose: bool = True):
    """
    Loads CWRU data, applies 4-class mapping, isolates temporal blocks,
    renders 224x224 FFT line graph images, and returns train/val/test splits.
    Caches pre-rendered images in ml/data/cwru_fft_224_images.npz for fast repeat loading.
    """
    if cache_file is None:
        cache_file = os.path.join(DATA_DIR, 'cwru_fft_224_images.npz')

    npz_path = os.path.join(DATA_DIR, 'CWRU_48k_load_1_CNN_data.npz')
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"CWRU dataset not found at {npz_path}")

    d = np.load(npz_path)
    signals = d['data'].reshape(d['data'].shape[0], -1).astype(np.float64)
    raw_labels = np.array([normalize_class_name(lbl) for lbl in d['labels']])
    mapped_labels = np.array([map_to_four_classes(l) for l in raw_labels])
    tr_idx, va_idx, te_idx, manifest = build_split_indices(raw_labels)

    y_train = np.array([CLASS_TO_IDX[mapped_labels[i]] for i in tr_idx])
    y_val   = np.array([CLASS_TO_IDX[mapped_labels[i]] for i in va_idx])
    y_test  = np.array([CLASS_TO_IDX[mapped_labels[i]] for i in te_idx])

    if os.path.exists(cache_file):
        if verbose:
            print(f"Loading cached FFT graph images from {cache_file}...")
        c = np.load(cache_file)
        X_train = c['X_train']
        X_val   = c['X_val']
        X_test  = c['X_test']
        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'manifest': manifest,
            'classes': FOUR_CLASSES,
            'class_to_idx': CLASS_TO_IDX,
            'idx_to_class': IDX_TO_CLASS
        }

    if verbose:
        print("=" * 65)
        print("GENERATING FFT GRAPH TRAINING IMAGES (224x224 RGB)")
        print(f"Total: {len(signals)} | Train: {len(tr_idx)} | Val: {len(va_idx)} | Test: {len(te_idx)}")
        print(f"4-Class breakdown: {manifest['class_breakdown']}")
        print("=" * 65)

    X_train = batch_render_fft_images(signals[tr_idx], is_training=True, verbose=verbose)
    X_val   = batch_render_fft_images(signals[va_idx], is_training=False, verbose=verbose)
    X_test  = batch_render_fft_images(signals[te_idx], is_training=False, verbose=verbose)

    # Save cache
    try:
        np.savez_compressed(cache_file, X_train=X_train, X_val=X_val, X_test=X_test)
        if verbose:
            print(f"Cached pre-rendered images to {cache_file}")
    except Exception as e:
        if verbose:
            print(f"Warning: Failed to cache images: {e}")

    return {
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        'manifest': manifest,
        'classes': FOUR_CLASSES,
        'class_to_idx': CLASS_TO_IDX,
        'idx_to_class': IDX_TO_CLASS
    }


if __name__ == '__main__':
    # Self-test check
    test_sig = np.random.randn(1024) * 0.2
    freqs, amps = compute_fft_spectrum(test_sig, 48000)
    img = render_fft_line_graph(freqs, amps)
    assert img.shape == (224, 224, 3), f"Bad image shape: {img.shape}"
    assert img.dtype == np.uint8, f"Bad dtype: {img.dtype}"
    print(f"Self-check passed: FFT graph rendered successfully {img.shape}, dtype={img.dtype}")
