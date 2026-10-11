"""
=============================================================
ml/training/train_spectrum.py
CWRU Bearing Fault Diagnosis — Frequency Spectrum Classifier

Trains a dedicated spectrum classifier on frequency–amplitude spectra
generated from labeled CWRU vibration recordings:
  - Pipeline: detrend, periodic Hanning window, rfft, single-sided peak amplitude (g)
  - Transformation: Canonical 512-bin grid (0–6000 Hz) + 14 canonical spectral features (526-dim)
  - Partitioning: Leakage-Safe Block Split (70% Train, 10% Val, 20% Test)
  - Model: Random Forest Ensemble (200 trees)
=============================================================
"""

import os
import sys
import json
import pickle
import datetime
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(TRAINING_DIR)
MODELS_DIR   = os.path.join(BASE_DIR, 'saved_models')
EVAL_DIR     = os.path.join(BASE_DIR, 'evaluation')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from preprocessing.preprocess import (
    load_and_preprocess_cwru_spectrum,
    CANONICAL_SPECTRUM_BINS,
    CANONICAL_SPECTRUM_MAX_FREQ,
    CANONICAL_SPECTRAL_FEATURES,
    CWRU_CLASSES,
    FAULT_METADATA
)

RANDOM_SEED = 42


def train_and_evaluate_spectrum_model(verbose=True):
    print("=" * 70)
    print("STARTING CWRU SPECTRUM CLASSIFIER TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load dataset with strict leakage-safe block partitioning
    data = load_and_preprocess_cwru_spectrum(verbose=verbose)
    X_train = data['X_train']
    y_train = data['y_train']
    X_val   = data['X_val']
    y_val   = data['y_val']
    X_test  = data['X_test']
    y_test  = data['y_test']
    label_encoder = data['label_encoder']
    classes = list(label_encoder.classes_)

    if verbose:
        print(f"\nTraining on {len(X_train)} samples across {len(classes)} classes...")
        print(f"Feature dimension: {X_train.shape[1]} (512 frequency bins + 14 spectral features)")

    # 2. Train Random Forest Model
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    # 3. Evaluate on Validation Split (separate from Test)
    val_preds = clf.predict(X_val)
    val_probs = clf.predict_proba(X_val)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_f1_macro = float(f1_score(y_val, val_preds, average='macro'))
    val_f1_weighted = float(f1_score(y_val, val_preds, average='weighted'))

    print("\n" + "-" * 50)
    print("VALIDATION SPLIT RESULTS (460 samples, 10% held-out block)")
    print("-" * 50)
    print(f"  Validation Accuracy:    {val_acc * 100:.2f}%")
    print(f"  Validation Macro F1:    {val_f1_macro:.4f}")
    print(f"  Validation Weighted F1: {val_f1_weighted:.4f}")

    # 4. Evaluate on Held-Out Test Split (strictly unseen 20% block)
    test_preds = clf.predict(X_test)
    test_probs = clf.predict_proba(X_test)
    test_acc = float(accuracy_score(y_test, test_preds))
    test_prec_macro = float(precision_score(y_test, test_preds, average='macro'))
    test_rec_macro = float(recall_score(y_test, test_preds, average='macro'))
    test_f1_macro = float(f1_score(y_test, test_preds, average='macro'))
    test_f1_weighted = float(f1_score(y_test, test_preds, average='weighted'))

    print("\n" + "=" * 50)
    print("HELD-OUT TEST SPLIT RESULTS (920 samples, 20% continuous block)")
    print("=" * 50)
    print(f"  Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"  Test Precision:   {test_prec_macro:.4f}")
    print(f"  Test Recall:      {test_rec_macro:.4f}")
    print(f"  Test Macro F1:    {test_f1_macro:.4f}")
    print(f"  Test Weighted F1: {test_f1_weighted:.4f}")

    test_report = classification_report(
        y_test, test_preds,
        target_names=classes,
        output_dict=True
    )
    print("\nDetailed Test Classification Report:")
    print(classification_report(y_test, test_preds, target_names=classes))

    # 5. Save model artifacts
    model_save_path = os.path.join(MODELS_DIR, 'spectrum_classifier_v1.joblib')
    with open(model_save_path, 'wb') as f:
        pickle.dump(clf, f)

    config_path = os.path.join(MODELS_DIR, 'spectrum_config.json')
    spectrum_config = {
        'model_name': 'Random Forest Spectrum Classifier',
        'model_version': 'spectrum_classifier_v1',
        'pipeline': 'Detrend + Periodic Hanning Window + Single-Sided Peak FFT Amplitude (g)',
        'canonical_bins': CANONICAL_SPECTRUM_BINS,
        'canonical_max_freq_hz': CANONICAL_SPECTRUM_MAX_FREQ,
        'spectral_feature_names': CANONICAL_SPECTRAL_FEATURES,
        'total_input_features': CANONICAL_SPECTRUM_BINS + len(CANONICAL_SPECTRAL_FEATURES),
        'classes': classes,
        'validation_metrics': {
            'samples': len(y_val),
            'accuracy': val_acc,
            'macro_f1': val_f1_macro,
            'weighted_f1': val_f1_weighted
        },
        'test_metrics': {
            'samples': len(y_test),
            'accuracy': test_acc,
            'macro_precision': test_prec_macro,
            'macro_recall': test_rec_macro,
            'macro_f1': test_f1_macro,
            'weighted_f1': test_f1_weighted
        },
        'trained_at': datetime.datetime.now().isoformat()
    }
    with open(config_path, 'w') as f:
        json.dump(spectrum_config, f, indent=2)

    # 6. Update master metadata file
    meta_path = os.path.join(MODELS_DIR, 'model_metadata_v1.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            master_meta = json.load(f)
        if 'models' not in master_meta:
            master_meta['models'] = {}
        master_meta['models']['spectrum_classifier'] = {
            'model_name': 'Spectrum Random Forest',
            'model_version': 'spectrum_classifier_v1',
            'test_accuracy': test_acc,
            'test_macro_f1': test_f1_macro,
            'val_accuracy': val_acc,
            'val_macro_f1': val_f1_macro,
            'feature_dim': CANONICAL_SPECTRUM_BINS + len(CANONICAL_SPECTRAL_FEATURES),
            'classes': classes
        }
        with open(meta_path, 'w') as f:
            json.dump(master_meta, f, indent=2)

    # 7. Save evaluation results
    eval_results_path = os.path.join(EVAL_DIR, 'spectrum_model_results.json')
    with open(eval_results_path, 'w') as f:
        json.dump({
            'validation': spectrum_config['validation_metrics'],
            'test': spectrum_config['test_metrics'],
            'classification_report': test_report
        }, f, indent=2)

    print(f"\nArtifacts successfully saved:")
    print(f"  Model:   {os.path.basename(model_save_path)}")
    print(f"  Config:  {os.path.basename(config_path)}")
    print(f"  Results: {os.path.basename(eval_results_path)}")
    return spectrum_config


if __name__ == '__main__':
    train_and_evaluate_spectrum_model(verbose=True)
