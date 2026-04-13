"""
Pydantic schemas for API request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional


class PredictionRequest(BaseModel):
    """Request body for prediction endpoint."""
    text: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Vietnamese product review text to analyze",
        json_schema_extra={"examples": ["Máy đẹp, pin trâu, camera chụp rõ nét"]},
    )
    model: Optional[str] = Field(
        default="bigru_crf",
        description="Model to use: 'bigru_crf' or 'phobert_crf'",
    )


class AspectSpan(BaseModel):
    """A single detected aspect-sentiment span."""
    aspect: str = Field(..., description="Aspect category (e.g., CAMERA, BATTERY)")
    sentiment: str = Field(..., description="Sentiment: POSITIVE, NEUTRAL, or NEGATIVE")
    text: str = Field(..., description="Text span from the review")
    start: int = Field(..., description="Start character position")
    end: int = Field(..., description="End character position")


class PredictionResponse(BaseModel):
    """Response body for prediction endpoint."""
    text: str = Field(..., description="Original input text")
    model_used: str = Field(..., description="Name of the model used")
    spans: List[AspectSpan] = Field(default_factory=list, description="Detected aspect-sentiment spans")
    summary: dict = Field(default_factory=dict, description="Summary statistics")


class HealthResponse(BaseModel):
    """Response for health check endpoint."""
    status: str = "ok"
    models_loaded: List[str] = Field(default_factory=list)
    device: str = "cpu"


class ModelInfoResponse(BaseModel):
    """Information about available models."""
    name: str
    description: str
    size_mb: float
    loaded: bool
