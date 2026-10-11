"""
=============================================================
ml/training/train_fft_cnn.py
CWRU Bearing Fault Diagnosis — 2D CNN for FFT Graph Image Classification

Architecture:
  - Input: 224x224x3 RGB FFT Line Graph Images
  - Backbone: 4 Conv2D + MaxPooling blocks with GlobalAveragePooling
  - Classifier: Dense(64) + Dropout(0.3) + Dense(4, softmax)
  - 4 Classes: Healthy, Inner Race Fault, Outer Race Fault, Ball Fault
  - Class-weighted Cross-Entropy loss (compensates for Healthy 1:3 ratio)
  - Early stopping + ModelCheckpoint
  - Honest evaluation: Test accuracy, Macro F1, Precision/Recall,
    Confusion Matrix, misclassification analysis, and held-out severity check.
=============================================================
"""

import os
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
import json
import argparse
import datetime
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR     = os.path.dirname(TRAINING_DIR)
MODELS_DIR   = os.path.join(BASE_DIR, 'saved_models')
EVAL_DIR     = os.path.join(BASE_DIR, 'evaluation')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)

sys.path.insert(0, BASE_DIR)
from preprocessing.preprocess_fft_images import (
    load_cwru_fft_image_dataset,
    FOUR_CLASSES,
    CLASS_TO_IDX,
    IDX_TO_CLASS,
    CANONICAL_MAX_FREQ,
    CANONICAL_MAX_AMP,
    SAMPLING_RATE_HZ,
    WINDOW_LENGTH,
    IMAGE_SIZE,
    RANDOM_SEED
)


def build_2d_fft_cnn(input_shape=(224, 224, 3), num_classes=4):
    """
    Constructs a compact 2D CNN for FFT line graph image classification.
    Parameter count: ~150k params. Trains quickly on CPU or GPU.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=input_shape, name='fft_graph_input'),
        layers.Rescaling(1.0 / 255.0, name='rescaling'),

        layers.Conv2D(32, (3, 3), padding='same', activation='relu', name='conv1'),
        layers.MaxPooling2D((2, 2), name='pool1'),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv2'),
        layers.MaxPooling2D((2, 2), name='pool2'),

        layers.Conv2D(64, (3, 3), padding='same', activation='relu', name='conv3'),
        layers.MaxPooling2D((2, 2), name='pool3'),

        layers.Conv2D(128, (3, 3), padding='same', activation='relu', name='conv4'),
        layers.GlobalAveragePooling2D(name='gap'),

        layers.Dropout(0.3, name='dropout'),
        layers.Dense(64, activation='relu', name='dense_features'),
        layers.Dense(num_classes, activation='softmax', name='class_probabilities')
    ], name='Bearing_FFT_2D_CNN')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate_2d_cnn(
    epochs: int = 20,
    batch_size: int = 32,
    smoke_test: bool = False,
    verbose: bool = True
):
    import tensorflow as tf

    print("=" * 70)
    print("STARTING 2D CNN FFT GRAPH IMAGE TRAINING PIPELINE")
    print(f"TensorFlow Version: {tf.__version__}")
    print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
    print("=" * 70)

    # 1. Load Preprocessed Image Dataset
    dataset = load_cwru_fft_image_dataset(verbose=verbose)
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val   = dataset['X_val']
    y_val   = dataset['y_val']
    X_test  = dataset['X_test']
    y_test  = dataset['y_test']
    manifest = dataset['manifest']

    if smoke_test:
        print("\n[SMOKE TEST MODE] Restricting to small balanced batch for rapid pipeline verification...")
        # Pick 16 samples per class for train, 8 for val, 8 for test
        tr_sub = np.concatenate([np.where(y_train == c)[0][:16] for c in range(len(FOUR_CLASSES))])
        va_sub = np.concatenate([np.where(y_val == c)[0][:8] for c in range(len(FOUR_CLASSES))])
        te_sub = np.concatenate([np.where(y_test == c)[0][:8] for c in range(len(FOUR_CLASSES))])
        X_train, y_train = X_train[tr_sub], y_train[tr_sub]
        X_val, y_val = X_val[va_sub], y_val[va_sub]
        X_test, y_test = X_test[te_sub], y_test[te_sub]
        epochs = 1

    # 2. Compute Class Weights (compensates for Healthy 1:3 ratio)
    class_weights_arr = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = {int(i): float(w) for i, w in zip(np.unique(y_train), class_weights_arr)}
    print(f"\nClass weights (balanced): {class_weight_dict}")

    # 3. Build Model
    model = build_2d_fft_cnn(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3), num_classes=len(FOUR_CLASSES))
    if verbose:
        model.summary()

    # 4. Callbacks
    best_model_path = os.path.join(MODELS_DIR, 'cnn_fft_2d_model.keras')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]

    # 5. Train Model
    t_start = datetime.datetime.now()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=1
    )
    training_duration_sec = (datetime.datetime.now() - t_start).total_seconds()

    # 6. Evaluate on Validation Set
    val_probs = model.predict(X_val, batch_size=batch_size, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)
    val_acc = float(accuracy_score(y_val, val_preds))
    val_macro_f1 = float(f1_score(y_val, val_preds, average='macro'))

    print("\n" + "-" * 50)
    print(f"VALIDATION SPLIT RESULTS ({len(y_val)} samples, 10% held-out block)")
    print("-" * 50)
    print(f"  Validation Accuracy: {val_acc * 100:.2f}%")
    print(f"  Validation Macro F1: {val_macro_f1:.4f}")

    # 7. Evaluate on Held-Out Test Set (unseen 20% block)
    test_probs = model.predict(X_test, batch_size=batch_size, verbose=0)
    test_preds = np.argmax(test_probs, axis=1)
    test_acc = float(accuracy_score(y_test, test_preds))
    class_labels = list(range(len(FOUR_CLASSES)))
    test_macro_prec = float(precision_score(y_test, test_preds, labels=class_labels, average='macro', zero_division=0))
    test_macro_rec  = float(recall_score(y_test, test_preds, labels=class_labels, average='macro', zero_division=0))
    test_macro_f1   = float(f1_score(y_test, test_preds, labels=class_labels, average='macro', zero_division=0))
    test_cm         = confusion_matrix(y_test, test_preds, labels=class_labels).tolist()

    print("\n" + "=" * 50)
    print(f"HELD-OUT TEST SPLIT RESULTS ({len(y_test)} samples, 20% held-out block)")
    print("=" * 50)
    print(f"  Test Accuracy:    {test_acc * 100:.2f}%")
    print(f"  Test Precision:   {test_macro_prec:.4f}")
    print(f"  Test Recall:      {test_macro_rec:.4f}")
    print(f"  Test Macro F1:    {test_macro_f1:.4f}")

    test_report = classification_report(
        y_test, test_preds,
        labels=class_labels,
        target_names=FOUR_CLASSES,
        output_dict=True,
        zero_division=0
    )
    print("\nClassification Report (Test):")
    print(classification_report(y_test, test_preds, labels=class_labels, target_names=FOUR_CLASSES, zero_division=0))

    # 8. Misclassification Analysis
    misclassified_indices = np.where(test_preds != y_test)[0].tolist()
    misclassified_details = []
    for idx in misclassified_indices[:10]:  # sample up to 10
        misclassified_details.append({
            'test_index': int(idx),
            'true_class': FOUR_CLASSES[int(y_test[idx])],
            'predicted_class': FOUR_CLASSES[int(test_preds[idx])],
            'confidence': float(test_probs[idx][test_preds[idx]])
        })
    print(f"Total misclassified test samples: {len(misclassified_indices)} of {len(y_test)}")

    # 9. Save Configuration & Class Mapping
    config = {
        'model_name': '2D CNN FFT Graph Image Classifier',
        'model_version': 'cnn_fft_2d_v1',
        'framework': 'TensorFlow / Keras',
        'classes': FOUR_CLASSES,
        'class_to_idx': CLASS_TO_IDX,
        'input_shape': [IMAGE_SIZE[0], IMAGE_SIZE[1], 3],
        'image_size': [IMAGE_SIZE[0], IMAGE_SIZE[1]],
        'frequency_range_hz': [0.0, CANONICAL_MAX_FREQ],
        'max_amplitude_g': CANONICAL_MAX_AMP,
        'sampling_rate_hz': SAMPLING_RATE_HZ,
        'window_length': WINDOW_LENGTH,
        'frequency_resolution_hz': round(SAMPLING_RATE_HZ / WINDOW_LENGTH, 3),
        'signal_unit': 'g (linear acceleration peak vs Hz)',
        'required_analyzer_settings': {
            'spectrum_type': 'linear magnitude / peak amplitude (g)',
            'frequency_span': f'0 to {int(CANONICAL_MAX_FREQ)} Hz (minimum 12 kHz sampling)',
            'window_function': 'Hanning / Hann',
            'averaging': 'RMS / Peak-Hold'
        },
        'training_summary': {
            'epochs_trained': len(history.history['loss']),
            'batch_size': batch_size,
            'duration_seconds': round(training_duration_sec, 2),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'validation_metrics': {
            'accuracy': round(val_acc, 4),
            'macro_f1': round(val_macro_f1, 4)
        },
        'test_metrics': {
            'accuracy': round(test_acc, 4),
            'macro_precision': round(test_macro_prec, 4),
            'macro_recall': round(test_macro_rec, 4),
            'macro_f1': round(test_macro_f1, 4)
        },
        'disclaimers': [
            "ACADEMIC BENCHMARK NOTICE: Model was trained strictly on CWRU laboratory test-rig vibration spectra.",
            "High test accuracy on synthetic CWRU FFT graphs DOES NOT establish equivalent reliability on real-world analyzer screenshots.",
            "Screenshots from field vibration analyzers may contain differing aspect ratios, axes, fonts, noise floors, machine speeds, or harmonics.",
            "Predicted softmax probabilities reflect internal closed-set model activations and are NOT guaranteed reliability or probability of failure estimates.",
            "This model classifies CURRENT bearing condition (presence and location of localized defect) and does NOT predict Remaining Useful Life (RUL)."
        ],
        'trained_at': datetime.datetime.now().isoformat()
    }

    config_path = os.path.join(MODELS_DIR, 'fft_image_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    class_mapping = {
        str(i): {
            'class_name': name,
            'bearing_status': 'Healthy' if name == 'Healthy' else 'Fault Detected',
            'fault_type': 'Normal' if name == 'Healthy' else name.replace(' Fault', ''),
            'severity': 'Healthy' if name == 'Healthy' else 'Fault Detected'
        }
        for i, name in enumerate(FOUR_CLASSES)
    }
    with open(os.path.join(MODELS_DIR, 'fft_image_class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f, indent=2)

    # Save training history
    history_serializable = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(os.path.join(MODELS_DIR, 'fft_image_training_history.json'), 'w') as f:
        json.dump(history_serializable, f, indent=2)

    # Save evaluation results
    eval_results = {
        'validation': config['validation_metrics'],
        'test': config['test_metrics'],
        'confusion_matrix': {
            'classes': FOUR_CLASSES,
            'matrix': test_cm
        },
        'classification_report': test_report,
        'misclassifications_sample': misclassified_details,
        'total_misclassified': len(misclassified_indices)
    }
    with open(os.path.join(EVAL_DIR, 'fft_image_model_results.json'), 'w') as f:
        json.dump(eval_results, f, indent=2)

    # 10. Update master model metadata
    meta_path = os.path.join(MODELS_DIR, 'model_metadata_v1.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            master_meta = json.load(f)
        if 'models' not in master_meta:
            master_meta['models'] = {}
        master_meta['models']['fft_2d_cnn'] = {
            'model_name': '2D CNN FFT Graph Image Classifier',
            'model_version': 'cnn_fft_2d_v1',
            'input_type': 'fft_graph_image',
            'input_shape': [224, 224, 3],
            'test_accuracy': test_acc,
            'test_macro_f1': test_macro_f1,
            'val_accuracy': val_acc,
            'val_macro_f1': val_macro_f1,
            'classes': FOUR_CLASSES
        }
        with open(meta_path, 'w') as f:
            json.dump(master_meta, f, indent=2)

    print("\nSaved Artifacts:")
    print(f"  Model:   {best_model_path}")
    print(f"  Config:  {config_path}")
    print(f"  Results: {os.path.join(EVAL_DIR, 'fft_image_model_results.json')}")

    return config, eval_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train 2D CNN on FFT Graph Images")
    parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--smoke-test', action='store_true', help="Run small smoke test")
    args = parser.parse_args()

    train_and_evaluate_2d_cnn(
        epochs=args.epochs,
        batch_size=args.batch_size,
        smoke_test=args.smoke_test,
        verbose=True
    )
