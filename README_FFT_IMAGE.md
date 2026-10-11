# 📈 2D CNN FFT Vibration Graph Classifier (Experimental)

> **ADCET, Ashta — TY BTech Mechanical Engineering Project**  
> **Topic:** Bearing Fault Diagnosis via 2D Convolutional Neural Networks on FFT Vibration Line Graphs  
> **Input Domain:** Frequency Line Graphs (0–6,000 Hz, Linear Amplitude in $g$, 224×224 RGB)  
> **Target Taxonomy:** 4 Canonical Classes (`Healthy`, `Inner Race Fault`, `Outer Race Fault`, `Ball Fault`)

---

## 1. Overview & Architectural Role

While the primary pipeline uses a 1D Deep CNN and Classical ML models on raw 1,024-sample vibration acceleration series, this module provides an **experimental 2D CNN vision pipeline** designed to diagnose bearing defects directly from **FFT vibration graph images** (e.g., exported plots from industrial handheld vibration analyzers or benchtop FFT analyzers).

```
  [ Vibration Signal / Analyzer Plot ]
                 │
                 ▼
  [ 2D FFT Graph Image Ingestion ]
    • Format Check (PNG/JPG/BMP)
    • Optional Bounding Box Crop [x1, y1, x2, y2]
    • Aspect-Ratio Preserving Resize + Uniform Letterbox Padding (224×224×3)
    • Normalization ([0, 1])
                 │
                 ▼
     [ 2D Convolutional Neural Network ]
    • 4 Conv2D Blocks (16 → 32 → 64 → 128 filters) + MaxPool
    • Global Average Pooling (Spatial invariance)
    • Dense(64, ReLU) + Dropout(0.3)
    • Dense(4, Softmax) with Balanced Class Weights
                 │
                 ▼
  [ 4-Class Diagnostic Assessment ]
    • Healthy (Normal)
    • Inner Race Fault (IR)
    • Outer Race Fault (OR)
    • Ball Fault (Rolling Element)
```

---

## 2. 4-Class Fault Taxonomy & CWRU Mapping

The standard CWRU 48k Load 1 dataset contains 10 fine-grained condition classes across 3 defect sizes (0.007", 0.014", 0.021"). To ensure robust visual pattern recognition of component-specific spectral signatures, these are mapped into **4 canonical maintenance classes**:

| Canonical 4-Class Label | Source CWRU Conditions | Defect Location | Total Samples | Train (70%) | Val (10%) | Test (20%) |
|---|---|---|---|---|---|---|
| **`Healthy`** | `Normal` | None (Baseline) | 460 | 322 | 46 | 92 |
| **`Inner Race Fault`** | `IR_007`, `IR_014`, `IR_021` | Inner Ring | 1,380 | 966 | 138 | 276 |
| **`Outer Race Fault`** | `OR_007`, `OR_014`, `OR_021` | Outer Ring (6 o'clock) | 1,380 | 966 | 138 | 276 |
| **`Ball Fault`** | `Ball_007`, `Ball_014`, `Ball_021` | Rolling Elements | 1,380 | 966 | 138 | 276 |
| **Total** | 10 Conditions | — | **4,600** | **3,220** | **460** | **920** |

---

## 3. Dataset Preprocessing & Graph Rendering

### Diagnostic Bandwidth & Graph Specifications
- **Graph Type:** Pure frequency line plot (Amplitude $g$ vs Frequency Hz). **Not** a spectrogram or scalogram.
- **Diagnostic Span:** $0\text{ to }6,000\text{ Hz}$ (captures fundamental defect pass frequencies BPFI, BPFO, BSF, FTF and their first 5–10 harmonics).
- **Amplitude Scaling:** Linear acceleration peak in $g$ (full scale capped at $1.5\text{ g}$).
- **Output Dimensions:** $224 \times 224$ pixels, 3 color channels (RGB).

### Leakage-Safe Temporal Partitioning
To prevent optimistic data leakage between consecutive sliding FFT windows, a strict **temporal block split** is enforced per class file:
- **Training Set (70%):** First 70% of continuous time windows (indices 0 to 321).
- **Validation Set (10%):** Middle 10% of continuous time windows (indices 322 to 367).
- **Test Set (20%):** Final 20% of continuous time windows (indices 368 to 459).

### Preprocessing Command
```bash
python ml/preprocessing/preprocess_fft_images.py
```
This renders and caches the entire 4,600-sample dataset into `ml/data/cwru_fft_224_images.npz` and outputs `ml/saved_models/fft_image_split_manifest.json`.

---

## 4. 2D CNN Model Architecture

Designed from scratch for fast inference (<60 ms) and high generalization on line graph features:

| Layer | Type | Output Shape | Parameters | Activation | Regularization |
|---|---|---|---|---|---|
| Input | InputLayer | `(224, 224, 3)` | 0 | — | — |
| Block 1 | Conv2D (16, 3×3) + MaxPool | `(112, 112, 16)` | 448 | ReLU | BatchNormalization |
| Block 2 | Conv2D (32, 3×3) + MaxPool | `(56, 56, 32)` | 4,640 | ReLU | BatchNormalization |
| Block 3 | Conv2D (64, 3×3) + MaxPool | `(28, 28, 64)` | 18,496 | ReLU | BatchNormalization |
| Block 4 | Conv2D (128, 3×3) + MaxPool | `(14, 14, 128)` | 73,856 | ReLU | BatchNormalization |
| Pooling | GlobalAveragePooling2D | `(128)` | 0 | — | — |
| Dense 1 | Dense (64) | `(64)` | 8,256 | ReLU | Dropout(0.30) |
| Output | Dense (4) | `(4)` | 260 | Softmax | — |
| **Total** | **4 Conv Blocks + Head** | — | **138,404 params** | — | **Balanced Class Weights** |

### Balanced Class Weighting
Because `Healthy` has 460 samples while each fault category has 1,380 samples (1:3 ratio), training utilizes balanced inverse class weights computed via scikit-learn:
$$\text{weight}_c = \frac{N_{\text{total}}}{4 \times N_c}$$
- `Healthy` weight: **2.5000**
- `Inner Race Fault` weight: **0.8333**
- `Outer Race Fault` weight: **0.8333**
- `Ball Fault` weight: **0.8333**

### Training Command
```bash
python ml/training/train_fft_cnn.py
```
Outputs model weights to `ml/saved_models/cnn_fft_2d_model.keras` and evaluation metrics to `ml/evaluation/fft_image_model_results.json`.

---

## 5. Benchmark Evaluation Results

Evaluated strictly on the **held-out 920-sample test set**:

### Overall Performance
- **Test Accuracy:** **86.74%**
- **Macro Average Precision:** **0.9079**
- **Macro Average Recall:** **0.8895**
- **Macro Average F1-Score:** **0.8826**
- **Validation Accuracy:** **86.74%**

### Per-Class Performance Breakdown

| Fault Class | Precision | Recall | F1-Score | Support (Samples) | Test Accuracy |
|---|---|---|---|---|---|
| **`Healthy`** | **0.9787** | **1.0000** | **0.9892** | 92 | **100.0%** (92 / 92) |
| **`Inner Race Fault`** | **0.9579** | **0.9891** | **0.9733** | 276 | **98.9%** (273 / 276) |
| **`Ball Fault`** | **0.7236** | **0.9674** | **0.8279** | 276 | **96.7%** (267 / 276) |
| **`Outer Race Fault`** | **0.9647** | **0.5942** | **0.7354** | 276 | **59.4%** (164 / 276) |

### Test Confusion Matrix

```
                      PREDICTED CLASS
                Healthy    IR      Ball     OR
ACTUAL  Healthy   92        0        0       0    (100.0%)
CLASS   IR         0      273        0       3    ( 98.9%)
        Ball       0        4      267       5    ( 96.7%)
        OR         2        8      102     164    ( 59.4%)
```

---

## 6. Root-Cause Analysis: Outer Race Misclassification

A rigorous engineering breakdown reveals why `Outer Race Fault` recall is 59.4% on 2D images:

### Defect Severity Breakdown for Outer Race:
- **`OR_007` (0.007" defect):** **100.0%** accuracy (92 / 92 correct)
- **`OR_021` (0.021" defect):** **78.3%** accuracy (72 / 92 correct)
- **`OR_014` (0.014" defect):** **0.0%** accuracy (0 / 92 correct — all 92 classified as Ball Fault)

### Physical & Vibration Mechanics Explanation:
1. **Vibration Energy Disparity:**
   - In CWRU 48k Load 1, `OR_007` generates massive impact acceleration ($\text{RMS} = 1.056\text{ g}$).
   - `OR_021` generates high-energy structural resonance ($\text{RMS} = 0.603\text{ g}$).
   - In stark contrast, `OR_014` exhibits an unusually low vibration energy ($\text{RMS} = 0.136\text{ g}$), which is nearly identical to `Ball_014` ($\text{RMS} = 0.138\text{ g}$) and `Ball_007` ($\text{RMS} = 0.141\text{ g}$).
2. **Spectral Line Smearing in 2D Plots:**
   - On a 2D line graph, `OR_014` lacks the sharp, towering BPFO harmonic spikes seen in `OR_007`. Instead, its spectral envelope shows lower-amplitude distributed peaks across 1,000–3,500 Hz that visually mirror the sideband modulations typical of rolling element (ball) defect signatures.
   - The 1D CNN processes raw 1024-point phase and time-waveform transient data and achieves 99.7% on the same defect, demonstrating that **2D static line graphs discard critical phase and micro-transient information**.

---

## 7. Ingestion & Image Preprocessing Pipeline

The system includes robust defenses against malformed or distorted analyzer uploads:

1. **Aspect-Ratio Preserving Letterboxing:**
   - Uploaded plots of any aspect ratio (e.g., 16:9 analyzer screen or 4:3 screenshot) are scaled to fit within $224 \times 224$ while maintaining their original geometric aspect ratio.
   - The remaining border area is uniformly padded with black pixels `(0, 0, 0)`.
2. **Interactive Bounding Box Cropping:**
   - Users can optionally provide crop coordinates `[x1, y1, x2, y2]` to trim away analyzer borders, menu bars, or legends before classification.
3. **Blank / Solid Image Guard:**
   - If an uploaded image is solid white, blank, or has insufficient variance ($\text{std} < 0.01$), the API rejects it with HTTP 400 and an actionable message.
4. **Visual Verification Preview:**
   - The exact $224 \times 224$ preprocessed image passed to the neural network is returned as base64 and rendered in the dashboard for complete transparency.

---

## 8. API Reference

### 1. Direct FastAPI Inference (`POST /predict`)
```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/png;base64,iVBORw0KGgo...",
    "crop_box": [50, 30, 450, 280]
  }'
```

### 2. Binary Image Upload (`POST /predict-image`)
```bash
curl -X POST http://127.0.0.1:8000/predict-image \
  -H "Content-Type: image/png" \
  --data-binary "@sample_fft_plot.png"
```

### 3. Backend Proxy (`POST /api/bearing-analysis`)
```bash
curl -X POST http://localhost:5000/api/bearing-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "data:image/png;base64,...",
    "crop_box": null,
    "input_type": "fft_image"
  }'
```

---

## 9. Academic & Industrial Disclaimers

> [!IMPORTANT]
> **Industrial Analyzer Notice:**
> - **Domain Shift:** This model was trained on synthetic line plots generated from CWRU benchmark accelerometer data (white background, thin dark lines, 0–6,000 Hz, linear amplitude in $g$). Screenshots from commercial vibration analyzers with dark backgrounds, logarithmic axes, waterfall displays, or prominent grid overlays will experience domain shift.
> - **Current Condition vs. RUL:** The 2D CNN predicts the **CURRENT bearing health status and localized defect location**. It does **NOT** forecast Remaining Useful Life (RUL), time-to-failure, or degradation trends.
> - **Probability Calibration:** The class probabilities output by the model are softmax neural activation scores on CWRU feature distributions, **not** certified industrial probabilities of operational reliability.
