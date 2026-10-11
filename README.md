# ⚙️ AI-Driven Smart Workshop Maintenance & Bearing Fault Diagnosis System (v3.0)

> **ADCET, Ashta — TY BTech Mechanical Engineering Project**  
> **Guide:** Ms. R.P. Mali  
> **Topic:** Predictive Maintenance, Industrial Vibration Analysis, Deep Learning (1D CNN) & Machine Learning

---

## 📋 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [The CWRU Bearing Vibration Dataset](#3-the-cwru-bearing-vibration-dataset)
4. [Fault Classes & Taxonomy](#4-fault-classes--taxonomy)
5. [System Architecture](#5-system-architecture)
6. [Data Preprocessing & Leakage Prevention](#6-data-preprocessing--leakage-prevention)
7. [Feature Engineering & Shape Factor Fix](#7-feature-engineering--shape-factor-fix)
8. [Models Tested & Benchmark Results](#8-models-tested--benchmark-results)
9. [1D Deep Convolutional Neural Network](#9-1d-deep-convolutional-neural-network)
10. [Confusion Matrix Analysis](#10-confusion-matrix-analysis)
10b. [2D CNN FFT Graph Classifier (Experimental)](#10b-2d-cnn-fft-graph-classifier-experimental) — [Detailed Guide](file:///c:/Users/Rushikesh/OneDrive/문서/Desktop/minor/workshop-maintenance-system/README_FFT_IMAGE.md)
11. [How to Train and Run](#11-how-to-train-and-run)
12. [Inference API & Real-Time Sensor Integration](#12-inference-api--real-time-sensor-integration)
13. [Old Project vs. Upgraded Project Comparison](#13-old-project-vs-upgraded-project-comparison)
14. [Academic & Industrial Limitations (Domain Shift)](#14-academic--industrial-limitations-domain-shift)
15. [Future Roadmap for Real Vibration Sensors](#15-future-roadmap-for-real-vibration-sensors)

---

## 1. Project Overview

This project implements an end-to-end **Industrial Bearing Fault Detection, Classification, and Severity Analysis System** using real vibration accelerometer data from the Case Western Reserve University (CWRU) Bearing Data Center.

The application integrates:
- **Python ML/DL Core**: Classical ML (SVM, Random Forest, Gradient Boosting) + 1D Deep Convolutional Neural Network (CNN).
- **FastAPI Inference Microservice**: Real-time signal preprocessing and diagnosis serving on port 8000.
- **Node.js & Express REST API**: Workshop machine fleet management, parameter logging, and maintenance scheduling.
- **React 18 Dashboard**: Live vibration waveform rendering, FFT spectrum visualizer (0–24 kHz), 10-class probability distributions, and fault localization badges.

---

## 2. Problem Statement

Rotating machinery in industrial workshops (lathes, milling machines, presses, compressors) relies critically on rolling element bearings. Bearing failure accounts for over **45% of all motor breakdowns**.

### The Core Challenge:
- **Traditional Time-Based Maintenance**: Replaces bearings too early (wasting money) or too late (causing catastrophic downtime).
- **Previous System Weakness**: Used manual heuristic weighted scoring without real accelerometer signal processing.
- **This Upgrade**: Processes raw vibration signals ($F_s = 48\text{ kHz}$) to scientifically identify:
  1. Is there a fault? (**Fault Detection**)
  2. Which component is failing? (**Fault Diagnosis**: Ball, Inner Race, Outer Race)
  3. How severe is the defect? (**Severity Classification**: 0.007", 0.014", 0.021" diameter)

---

## 3. The CWRU Bearing Vibration Dataset

The dataset originates from the **Case Western Reserve University (CWRU) Bearing Data Center**, a world-standard benchmark in predictive maintenance research.

### Test Rig Specifications:
- **Motor**: 2 HP Reliance Electric induction motor.
- **Speed & Load**: 1,772 RPM at 1 HP load (Load 1).
- **Bearing Type**: SKF 6205-2RS JEM deep-groove ball bearing (Drive End).
- **Transducer**: Accelerometer placed at 12 o'clock on the drive end housing.
- **Sampling Frequency**: $48,000\text{ samples/second}$ (48 kHz).
- **Defects**: Electro-Discharge Machined (EDM) single-point faults seeded into the bearing elements.

---

## 4. Fault Classes & Taxonomy

The system classifies **10 balanced conditions** (460 raw signal windows / 230 feature windows per class):

| Class Label | Defect Location | Fault Diameter (in) | Severity Category | Vibration RMS (Mean) |
|---|---|---|---|---|
| `Normal` | None (Healthy Baseline) | N/A | Healthy | 0.066 g |
| `Ball_007` | Rolling Element (Ball) | 0.007" (0.178 mm) | Mild | 0.141 g |
| `Ball_014` | Rolling Element (Ball) | 0.014" (0.356 mm) | Moderate | 0.138 g |
| `Ball_021` | Rolling Element (Ball) | 0.021" (0.533 mm) | Severe | 0.200 g |
| `IR_007` | Inner Race | 0.007" (0.178 mm) | Mild | 0.279 g |
| `IR_014` | Inner Race | 0.014" (0.356 mm) | Moderate | 0.197 g |
| `IR_021` | Inner Race | 0.021" (0.533 mm) | Severe | 0.606 g |
| `OR_007` | Outer Race (6 o'clock) | 0.007" (0.178 mm) | Mild | 1.056 g |
| `OR_014` | Outer Race (6 o'clock) | 0.014" (0.356 mm) | Moderate | 0.136 g |
| `OR_021` | Outer Race (6 o'clock) | 0.021" (0.533 mm) | Severe | 0.603 g |

---

## 5. System Architecture

```
                       [ Workshop Bearing Vibration Sensor ]
                                      │ (1024 samples)
                                      ▼
                        [ Signal Preprocessing ]
                    (DC Offset Removal, Normalization)
                                      │
                   ┌──────────────────┴──────────────────┐
                   ▼                                     ▼
        [ Branch A: Feature ML ]              [ Branch B: 1D Deep CNN ]
      - Time-Domain Features                - 1,024-Point Raw Signal Vector
      - StandardScaler                      - Conv1D(64) + Conv1D(32) + Conv1D(16)
      - SVM (RBF) Classifier                - GlobalAveragePooling + Dense
                   │                                     │
                   └──────────────────┬──────────────────┘
                                      ▼
                         [ FastAPI ML Inference ] (Port 8000)
                                      │
                                      ▼
                       [ Node.js Backend API ] (Port 5000)
                                      │
                                      ▼
                      [ React 18 Dynamic Dashboard ] (Port 3000)
                  - Time Waveform & FFT Spectrum (0-24 kHz)
                  - Fault Localization & Severity Gauge
                  - 10-Class Probability Distribution
                  - Actionable Maintenance Prescriptions
```

---

## 6. Data Preprocessing & Leakage Prevention

### ⚠️ The Data Leakage Vulnerability in CWRU
When continuous vibration signals are windowed into small slices (1024 or 2048 points), sequential windows from the same recording are highly correlated.
- **Flawed Approach**: Random `train_test_split()` shuffles rows, putting neighboring slices of the *same* continuous recording into both training and testing sets. This produces artificially high accuracy (leakage).
- **Our Solution**: **Block-Based Split**. We partition the continuous signal chronologically:
  - First **70%** of each recording $\rightarrow$ Training
  - Next **10%** of each recording $\rightarrow$ Validation
  - Final **20%** of each recording $\rightarrow$ Held-Out Test Set
- **Result**: Zero neighboring samples cross between train and test.

---

## 7. Feature Engineering & Shape Factor Fix

### 🛠️ The Shape Factor (`form`) Bug Fix
In the original CSV dataset, the `form` feature was mistakenly computed as:
$$\text{form}_{\text{flawed}} = \frac{\text{RMS}}{\text{mean}(x)}$$
Because vibration signals oscillate symmetrically around 0, $\text{mean}(x) \approx 0$. Dividing by near-zero caused extreme artificial outliers ($\text{form} > 300$).

We corrected this to the standard **Vibration Shape Factor**:
$$\text{Shape Factor} = \frac{\text{RMS}}{\text{mean}(|x|)}$$

### Extracted Feature Vector (12 Features):
1. **Maximum Value ($\max$)**: Peak positive shock amplitude.
2. **Minimum Value ($\min$)**: Peak negative shock amplitude.
3. **Mean ($\mu$)**: DC bias / offset.
4. **Standard Deviation ($\sigma$)**: Dynamic oscillation energy.
5. **Root Mean Square ($\text{RMS}$)**: Total signal power.
6. **Skewness ($S$)**: 3rd statistical moment (waveform asymmetry).
7. **Kurtosis ($K$)**: 4th statistical moment (impulsive peakiness).
8. **Crest Factor**: $\frac{\text{Peak}}{\text{RMS}}$ (impact detection).
9. **Shape Factor**: $\frac{\text{RMS}}{\text{MAV}}$ (waveform profile).
10. **Peak-to-Peak**: $\max(x) - \min(x)$.
11. **Mean Absolute Value ($\text{MAV}$)**: Average rectified signal.
12. **Variance ($\sigma^2$)**: Second central moment.

---

## 8. Models Tested & Benchmark Results

All models evaluated strictly on the **held-out 20% block test set** (zero leakage):

| Model Architecture | Input Data Type | 5-Fold CV Accuracy | Test Accuracy | Macro F1 | Weighted F1 |
|---|---|---|---|---|---|
| **1D Deep CNN (Ours)** | **1,024 Raw Signal** | **99.2% ± 0.4%** | **99.67%** | **99.67%** | **99.67%** |
| **SVM (RBF Kernel)** | **12 Features** | **95.2% ± 1.5%** | **96.96%** | **96.95%** | **96.95%** |
| **Gradient Boosting** | **12 Features** | **95.1% ± 0.9%** | **96.96%** | **96.93%** | **96.93%** |
| **Random Forest (300 Trees)** | **12 Features** | **95.3% ± 1.0%** | **96.09%** | **96.05%** | **96.05%** |
| **Logistic Regression (Baseline)** | **12 Features** | **91.7% ± 1.0%** | **95.22%** | **95.18%** | **95.18%** |

---

## 9. 1D Deep Convolutional Neural Network

The 1D CNN processes raw 1,024-point vibration arrays directly, learning multi-scale feature hierarchies without manual feature engineering.

### Architecture Summary:
```
Input: (1024, 1) Normalized Vibration Signal
  ├── Conv1D (32 filters, kernel=64, L2=1e-4) + BatchNorm + ReLU + MaxPool(4)
  ├── Conv1D (64 filters, kernel=32, L2=1e-4) + BatchNorm + ReLU + MaxPool(4)
  ├── Conv1D (128 filters, kernel=16, L2=1e-4) + BatchNorm + ReLU + MaxPool(4)
  ├── Conv1D (128 filters, kernel=8,  L2=1e-4) + BatchNorm + ReLU
  ├── GlobalAveragePooling1D
  ├── Dense (128) + ReLU + Dropout(0.3)
  ├── Dense (64)  + ReLU + Dropout(0.2)
  └── Dense (10, Softmax)  → 10-Class Probabilities
```
- **Total Parameters**: 356,554 (1.36 MB)
- **Early Stopping**: Epoch 36 (restored best weights at epoch 21)
- **Loss**: Categorical Crossentropy with Adam optimizer ($\text{lr} = 10^{-3}$, reduced on plateau).

---

## 10. Confusion Matrix Analysis

### 1D CNN Class-Wise Performance (920 Test Samples):
- `Normal`: **100% Precision, 100% Recall** (Zero false alarms).
- `Inner Race (IR_007, IR_014, IR_021)`: **100% F1 Score**.
- `Outer Race (OR_007, OR_014, OR_021)`: **99.5% F1 Score**.
- `Ball Faults (Ball_007, Ball_014, Ball_021)`: **99.3% F1 Score**.

Confusion matrices and training curves are saved in:
- [`ml/evaluation/confusion_matrix_CNN.png`](file:///c:/workshop-maintenance-system/ml/evaluation/confusion_matrix_CNN.png)
- [`ml/evaluation/confusion_matrix_SVM_RBF.png`](file:///c:/workshop-maintenance-system/ml/evaluation/confusion_matrix_SVM_RBF.png)
- [`ml/evaluation/cnn_training_curves.png`](file:///c:/workshop-maintenance-system/ml/evaluation/cnn_training_curves.png)

---

## 10b. 2D CNN FFT Graph Classifier (Experimental)

> Full documentation, architectural details, and root-cause analysis are provided in:  
> 📖 [README_FFT_IMAGE.md](file:///c:/Users/Rushikesh/OneDrive/문서/Desktop/minor/workshop-maintenance-system/README_FFT_IMAGE.md)

In addition to the primary 1D time-domain CNN and classical ML pipeline, the system includes a dedicated **2D CNN vision classifier** that accepts **FFT frequency line graph images** (e.g. from handheld vibration analyzers).

### Summary Specifications:
- **Input Format**: 224×224 RGB line graph (0–6,000 Hz, linear amplitude in $g$).
- **Taxonomy**: 4 Canonical classes (`Healthy`, `Inner Race Fault`, `Outer Race Fault`, `Ball Fault`).
- **Architecture**: 4 Conv2D + MaxPool blocks (16 → 32 → 64 → 128 filters) + GlobalAveragePooling2D + Dense(64) + Dense(4, Softmax) (~138k parameters).
- **Class Balancing**: Balanced inverse class weights (1:3 Healthy-to-fault ratio).
- **Held-Out Test Accuracy**: **86.74%** (Macro F1: **0.8826**).
  - `Healthy`: **100.0%** (92/92, F1: 0.99)
  - `Inner Race Fault`: **98.9%** (273/276, F1: 0.97)
  - `Ball Fault`: **96.7%** (267/276, F1: 0.83)
  - `Outer Race Fault`: **59.4%** (164/276, F1: 0.74 — `OR_014` low-energy defect envelope overlaps ball modulation).

---

## 11. How to Train and Run

### Step 1 — Python ML Environment
```bash
cd c:/workshop-maintenance-system
pip install -r requirements.txt
```

### Step 2 — Train Models (Optional - Models Already Pretrained)
```bash
# Preprocess and train classical ML models (SVM, RF, Gradient Boosting)
python ml/training/train_ml.py

# Train 1D Deep CNN (Raw vibration waveform)
python ml/training/train_cnn.py

# Preprocess FFT graph images (Generates 224x224 cached dataset)
python ml/preprocessing/preprocess_fft_images.py

# Train 2D CNN (FFT graph line plots)
python ml/training/train_fft_cnn.py
```

### Step 3 — Start the Python FastAPI ML Server (Port 8000)
```bash
python -m uvicorn ml.api.ml_server:app --port 8000 --host 127.0.0.1
```

### Step 4 — Start the Node.js Backend (Port 5000)
```bash
cd backend
npm start
```

### Step 5 — Start the React Frontend (Port 3000)
```bash
cd frontend
npm start
```
Open **`http://localhost:3000`** in your browser.

---

## 12. Inference API & Real-Time Sensor Integration

### Prediction Function API
```python
from ml.inference.predict import predict_vibration

# Send any 1,024 vibration readings (float list or numpy array)
result = predict_vibration(signal_1024, use_cnn=True)
```

### Example Structured Output:
```json
{
  "bearing_status": "Fault Detected",
  "fault_type": "Inner Race",
  "fault_size": "0.014",
  "severity": "Moderate",
  "confidence": 0.9950,
  "vibration_rms": 0.1970,
  "crest_factor": 4.3300,
  "kurtosis": 1.2000,
  "recommendation": "Schedule bearing inspection and lubrication check within 2 weeks. Focus inspection on the inner race (shaft-side ring).",
  "urgency": "Medium",
  "check_interval": "Inspect within 14 days",
  "model_used": "1D CNN (356,554 params)",
  "top_predictions": [
    { "class": "IR_014", "probability": 0.9950 },
    { "class": "IR_007", "probability": 0.0035 },
    { "class": "Ball_014", "probability": 0.0012 }
  ]
}
```

---

## 13. Old Project vs. Upgraded Project Comparison

| Dimension | Old Project (v2.0) | Upgraded Project (v3.0 CWRU AI) |
|---|---|---|
| **Data Source** | None (Manually typed values) | Genuine CWRU Bearing Accelerometer Dataset ($F_s = 48\text{ kHz}$) |
| **Input Signal** | 4 Scalar numbers (temperature, vibration mm/s, power, hours) | High-Resolution 1,024-Point Raw Vibration Signal Arrays |
| **Classification Output** | Generic `Healthy / Moderate / Critical` (Hand-tuned rules) | Scientific: **Fault Detection + Defect Type + Exact Severity Size** |
| **Model Type** | Rule-Based Hardcoded Weighted Sum | **1D Deep CNN (99.7%) + SVM RBF (97.0%) + Random Forest** |
| **Confidence Metric** | Synthetically calculated from distance to threshold | **True Softmax Model Probability (0.0% – 100.0%)** |
| **Data Splitting** | None | **Leak-Proof Block Partitioning (No adjacent slice leakage)** |
| **Defect Localization** | ❌ Cannot identify component | ✅ Identifies **Ball vs. Inner Race vs. Outer Race** |
| **Severity Sizing** | ❌ Guessed from general score | ✅ Exact Defect Diameter (**0.007" / 0.014" / 0.021"**) |
| **Signal Visualization** | Simple scalar metrics | **Interactive Waveform Chart + FFT Frequency Spectrum (0-24 kHz)** |
| **Real Sensor Readiness** | ❌ No signal ingestion API | ✅ Structured `predict_vibration(signal)` REST Microservice |

---

## 14. Academic & Industrial Limitations (Domain Shift)

> [!WARNING]
> **Academic Viva Defense Point:**
> CWRU test accuracy ($\approx 99.7\%$) does **not** equal guaranteed $99.7\%$ accuracy on a random factory machine.

### Key Factors for Domain Shift:
1. **Steady-State EDM Seeded Faults**: CWRU faults were created using EDM drilling. Real industrial bearings develop fatigue spalling and micro-cracking gradually over millions of stress cycles.
2. **Controlled Test Rig Dynamics**: CWRU data was recorded on a rigid laboratory bedplate with clean electric motor drive. Real workshop machines experience variable cutting loads, gearbox harmonics, and structural resonances.
3. **Constant Operating Load (1 HP)**: Models trained at 1 HP require fine-tuning or domain adaptation when applied to variable speed/variable load machinery.

---

## 15. Future Roadmap for Real Vibration Sensors

To transition from CWRU to real workshop machinery:
1. **Sensor Installation**: Mount a piezoelectric or MEMS accelerometer (e.g., ADXL345 or IMI 603C01) on the motor bearing housing with magnetic/stud mount.
2. **Edge Acquisition**: Use an ESP32 or Raspberry Pi with ADC to sample at $\ge 20\text{ kHz}$.
3. **Data Streaming**: Stream 1,024-point signal packets via MQTT or HTTP POST to `/api/bearing-analysis`.
4. **Transfer Learning / Fine-Tuning**: Collect baseline normal and known fault data from the target machine, freeze lower CNN convolutional layers, and retrain the classification head.

---

## 🎓 Academic Credentials & Credits
- **Institution**: Annasaheb Dange College of Engineering and Technology (ADCET), Ashta
- **Department**: Department of Mechanical Engineering
- **Project Type**: TY BTech Mechanical Engineering Project
- **Project Guide**: Ms. R.P. Mali
- **Dataset Reference**: Case Western Reserve University Bearing Data Center (SKF 6205-2RS JEM)
