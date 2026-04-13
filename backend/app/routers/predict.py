"""
Prediction API Routes
"""
import logging
from fastapi import APIRouter, HTTPException
from ..schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    AspectSpan,
    HealthResponse,
    ModelInfoResponse,
)
from ..services.predictor import predictor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["prediction"])


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and loaded models."""
    return HealthResponse(
        status="ok",
        models_loaded=predictor.available_models,
        device=str(predictor.device),
    )


@router.get("/models", response_model=list[ModelInfoResponse])
async def list_models():
    """List available models with their info."""
    models_info = []

    models_info.append(ModelInfoResponse(
        name="bigru_crf",
        description="BiGRU-CRF — Lightweight joint model (~12MB). Tốc độ nhanh, phù hợp CPU.",
        size_mb=12.0,
        loaded="bigru_crf" in predictor.models,
    ))
    models_info.append(ModelInfoResponse(
        name="phobert_crf",
        description="PhoBERT-CRF — High-accuracy model (~540MB). Cần GPU cho tốc độ tốt.",
        size_mb=540.0,
        loaded="phobert_crf" in predictor.models,
    ))

    return models_info


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Analyze Vietnamese product review for aspect-based sentiment.
    
    Returns detected aspect-sentiment spans with text highlights.
    """
    model_name = request.model or "bigru_crf"

    if model_name not in predictor.models:
        available = predictor.available_models
        raise HTTPException(
            status_code=400,
            detail=f"Model '{model_name}' not available. Loaded: {available}",
        )

    try:
        spans = predictor.predict(request.text, model_name)
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Build summary
    sentiment_counts = {"POSITIVE": 0, "NEUTRAL": 0, "NEGATIVE": 0}
    aspect_set = set()
    for span in spans:
        sentiment_counts[span["sentiment"]] = sentiment_counts.get(span["sentiment"], 0) + 1
        aspect_set.add(span["aspect"])

    summary = {
        "total_spans": len(spans),
        "unique_aspects": len(aspect_set),
        "sentiment_counts": sentiment_counts,
    }

    return PredictionResponse(
        text=request.text,
        model_used=model_name,
        spans=[AspectSpan(**s) for s in spans],
        summary=summary,
    )
