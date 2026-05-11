"""API FastAPI — khung phục vụ dự đoán phân khúc giá điện thoại."""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel


class PredictRequest(BaseModel):
    battery_power: float = 0.0
    blue: float = 0.0
    clock_speed: float = 0.0
    dual_sim: float = 0.0
    fc: float = 0.0
    four_g: float = 0.0
    int_memory: float = 0.0
    m_dep: float = 0.0
    mobile_wt: float = 0.0
    n_cores: float = 0.0
    pc: float = 0.0
    px_height: float = 0.0
    px_width: float = 0.0
    ram: float = 0.0
    sc_h: float = 0.0
    sc_w: float = 0.0
    talk_time: float = 0.0
    three_g: float = 0.0
    touch_screen: float = 0.0
    wifi: float = 0.0


class PredictResponse(BaseModel):
    segment: str | int
    detail: dict[str, Any] | None = None


# Load model thật tại đây (joblib/pickle) khi có artifact
_model: Any | None = None
_scaler: Any | None = None

SEGMENT_LABELS: dict[int, str] = {
    0: "Low",
    1: "Mid",
    2: "High",
    3: "Very High",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _scaler
    root = Path(__file__).resolve().parent.parent
    model_path = root / "models" / "best_svm_model.pkl"
    scaler_path = root / "models" / "scaler.pkl"

    _model = joblib.load(model_path)
    _scaler = joblib.load(scaler_path)
    yield
    _model = None
    _scaler = None


app = FastAPI(
    title="Phone Price Segment API",
    description="Dự đoán phân khúc giá điện thoại và tư vấn người dùng.",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    """Endpoint dự đoán — nối với mô hình đã train."""
    if _model is None or _scaler is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Huấn luyện và mount artifact trước.",
        )

    features = np.array(
        [
            [
                req.battery_power,
                req.blue,
                req.clock_speed,
                req.dual_sim,
                req.fc,
                req.four_g,
                req.int_memory,
                req.m_dep,
                req.mobile_wt,
                req.n_cores,
                req.pc,
                req.px_height,
                req.px_width,
                req.ram,
                req.sc_h,
                req.sc_w,
                req.talk_time,
                req.three_g,
                req.touch_screen,
                req.wifi,
            ]
        ],
        dtype=float,
    )

    features_scaled = _scaler.transform(features)
    pred_idx = int(_model.predict(features_scaled)[0])
    pred_proba = _model.predict_proba(features_scaled)
    confidence_pct = float(np.max(pred_proba) * 100.0)

    segment = SEGMENT_LABELS.get(pred_idx, str(pred_idx))
    return PredictResponse(
        segment=segment,
        detail={
            "class_id": pred_idx,
            "confidence": round(confidence_pct, 2),
        },
    )
