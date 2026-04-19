"""API FastAPI — khung phục vụ dự đoán phân khúc giá điện thoại."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Ví dụ payload; mở rộng theo đặc trưng thực tế sau khi huấn luyện."""

    features: list[float] = Field(
        ...,
        description="Vector đặc trưng đã tiền xử lý (cùng thứ tự với lúc train).",
    )


class PredictResponse(BaseModel):
    segment: str | int
    detail: dict[str, Any] | None = None


# Load model thật tại đây (joblib/pickle) khi có artifact
_model: Any | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # TODO: load model và preprocessor
    global _model
    _model = None
    yield
    _model = None


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
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model chưa được load. Huấn luyện và mount artifact trước.",
        )
    # TODO: preprocessor.transform(req.features) rồi _model.predict
    raise HTTPException(status_code=501, detail="Chưa triển khai predict.")
