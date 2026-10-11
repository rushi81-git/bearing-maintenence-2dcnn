"""
=============================================================
ml/api/ml_server.py
CWRU Bearing Fault Diagnosis — FastAPI Online Inference Service

Features:
  - POST /predict: Analyzes 1024-sample windows (single or multi-window)
  - GET /demo-samples: Curated CWRU demo waveforms
  - GET /model-metadata: System metadata & benchmark metrics
  - GET /health: Health check and status
=============================================================
"""

import os
import sys
import time
import numpy as np
from typing import List, Optional, Dict, Any

API_DIR  = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(API_DIR)
sys.path.insert(0, BASE_DIR)

from fastapi import FastAPI, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator

from inference.predict import (
    analyze_vibration_signal,
    predict_single_window,
    get_curated_demo_samples,
    load_inference_artifacts,
    detect_and_route_input,
    diagnose_fft_image,
    WINDOW_LENGTH
)

# ─── FastAPI App ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="CWRU Bearing Fault Detection and Diagnosis API",
    description="Online inference service for real-time vibration signal analysis and bearing fault classification.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Pydantic Models ─────────────────────────────────────────────────────────
class PredictRequest(BaseModel):
    signal: Optional[List[float]] = Field(None, description="Raw vibration signal (minimum 1024 numeric samples)")
    frequencies: Optional[List[float]] = Field(None, description="Frequency bins in Hz for spectrum analysis")
    amplitudes: Optional[List[float]] = Field(None, description="Amplitude values for spectrum analysis")
    image_base64: Optional[str] = Field(None, description="Base64-encoded FFT graph image (PNG/JPG)")
    crop_box: Optional[List[int]] = Field(None, description="Optional crop box coordinates [x1, y1, x2, y2]")
    input_type: Optional[str] = Field("auto", description="Input format: 'auto', 'raw_vibration', 'frequency_spectrum', or 'fft_graph_image'")
    sampling_rate_hz: Optional[int] = Field(48000, description="Accelerometer sampling rate in Hz")
    signal_unit: Optional[str] = Field("g", description="Vibration unit (default: g acceleration)")
    source_type: Optional[str] = Field("csv", description="Input source ('csv', 'spectrum_analyzer', 'fft_graph_image', 'sensor_stream', 'demo')")
    source_filename: Optional[str] = Field(None, description="Original filename of input file")
    model_mode: Optional[str] = Field("auto", description="Model selection mode: 'auto' (autonomous best-fit), 'cnn', or 'classical'")
    use_classical_ml: Optional[bool] = Field(False, description="Legacy fallback flag")

    @validator('signal')
    def validate_signal_array(cls, v):
        if v is not None:
            if len(v) < WINDOW_LENGTH:
                raise ValueError(f"At least {WINDOW_LENGTH} valid vibration samples are required. Received {len(v)} samples.")
            arr = np.array(v)
            if not np.all(np.isfinite(arr)):
                raise ValueError("Vibration signal contains NaN or Infinite values.")
        return v


class PredictResponse(BaseModel):
    success: bool
    input_type: str = "raw_vibration"
    model_used: Optional[str] = "1D Deep CNN"
    bearing_status: str
    predicted_class: str
    fault_type: str
    fault_size_inches: float
    fault_size_mm: float
    severity: str
    prediction_probability: float
    windows_analyzed: int
    total_samples: int
    discarded_trailing_samples: int
    sampling_rate_hz: int
    signal_unit: str
    source_type: str
    features: Optional[Dict[str, float]] = {}
    class_scores: Optional[Dict[str, float]] = None
    probabilities: Optional[List[Dict[str, Any]]] = None
    processed_image_base64: Optional[str] = None
    analyzer_requirements: Optional[Dict[str, Any]] = None
    dominant_window_count: int
    prediction_distribution: List[Dict[str, Any]]
    window_predictions: List[Dict[str, Any]]
    model_version: str
    auto_fit_details: Optional[Dict[str, Any]] = None
    recommendation: str
    urgency: str
    check_interval: str
    disclaimer: str
    inference_time_ms: float


# ─── Endpoints ───────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    """Health check endpoint confirming API status and model availability."""
    try:
        artifacts = load_inference_artifacts()
        cnn_loaded = artifacts.get('cnn_model') is not None
        classical_loaded = artifacts.get('classical_model') is not None
        spectrum_loaded = artifacts.get('spectrum_model') is not None
        fft_2d_loaded = artifacts.get('fft_2d_cnn_model') is not None
        return {
            "status": "healthy",
            "timestamp": time.time(),
            "window_size": WINDOW_LENGTH,
            "models_available": {
                "1d_cnn": cnn_loaded,
                "classical_ml": classical_loaded,
                "spectrum_classifier": spectrum_loaded,
                "fft_2d_cnn": fft_2d_loaded
            },
            "classes": list(artifacts['label_encoder'].classes_)
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e),
            "timestamp": time.time()
        }


@app.post("/predict", response_model=PredictResponse)
def predict_bearing_condition(req: PredictRequest):
    """
    Main prediction endpoint: routes raw vibration signals to 1D CNN / classical ML,
    frequency–amplitude spectrum files to dedicated spectrum classifier,
    and FFT graph images to 2D CNN.
    """
    t0 = time.perf_counter()
    try:
        result = detect_and_route_input(
            signal=req.signal,
            frequencies=req.frequencies,
            amplitudes=req.amplitudes,
            image_base64=req.image_base64,
            crop_box=req.crop_box,
            input_type=req.input_type or "auto",
            sampling_rate_hz=req.sampling_rate_hz or 48000,
            signal_unit=req.signal_unit or "g",
            source_type=req.source_type or "csv",
            source_filename=req.source_filename,
            model_mode=req.model_mode or "auto",
            use_classical_ml=bool(req.use_classical_ml)
        )
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)

        return PredictResponse(
            success=True,
            inference_time_ms=elapsed_ms,
            **result
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference error: {str(err)}")


@app.post("/predict-image")
async def predict_fft_image_endpoint(request: Request):
    """
    Accepts an FFT graph image either as binary bytes (Content-Type: image/* or octet-stream)
    or JSON payload with image_base64 and optional crop_box.
    Runs graph preprocessing (crop, aspect-ratio preserved resize to 224x224, padding),
    and classifies condition across 4 classes using the dedicated 2D CNN.
    """
    t0 = time.perf_counter()
    try:
        content_type = request.headers.get("content-type", "")
        crop_box = None
        source_filename = None
        if "application/json" in content_type:
            body = await request.json()
            image_b64 = body.get("image_base64")
            crop_box = body.get("crop_box")
            source_filename = body.get("filename")
            if not image_b64:
                raise ValueError("JSON request must contain 'image_base64'")
            result = diagnose_fft_image(
                image_base64=image_b64,
                crop_box=crop_box,
                source_filename=source_filename
            )
        else:
            raw_bytes = await request.body()
            if not raw_bytes:
                raise ValueError("No image data provided in request body")
            result = diagnose_fft_image(
                image_bytes=raw_bytes,
                crop_box=None,
                source_filename="uploaded_image.png"
            )

        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "success": True,
            "inference_time_ms": elapsed_ms,
            **result
        }
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(val_err))
    except Exception as err:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Image inference error: {str(err)}")



@app.get("/demo-samples")
def get_demo_samples():
    """Returns curated CWRU demonstration samples for UI testing and demos."""
    try:
        samples = get_curated_demo_samples()
        return {
            "success": True,
            "count": len(samples),
            "samples": samples
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


@app.get("/model-metadata")
def get_metadata():
    """Returns master training metadata, class mappings, and benchmark metrics."""
    try:
        artifacts = load_inference_artifacts()
        return {
            "success": True,
            "metadata": artifacts.get('metadata', {}),
            "config": artifacts.get('config', {})
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))


if __name__ == '__main__':
    import uvicorn
    uvicorn.run("ml_server:app", host="127.0.0.1", port=8000, reload=False)
