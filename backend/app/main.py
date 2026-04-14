"""
🇻🇳 Vietnamese ABSA — FastAPI Backend
Aspect-Based Sentiment Analysis for Vietnamese Product Reviews

Run: uvicorn backend.app.main:app --reload
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from .config import CORS_ORIGINS, HOST, PORT
from .routers.predict import router as predict_router
from .services.predictor import predictor

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, cleanup on shutdown."""
    logger.info("🚀 Starting Vietnamese ABSA Backend...")
    logger.info(f"📱 Device: {predictor.device}")

    # Load BiGRU-CRF (default — lightweight)
    predictor.load_bigru()
    
    # Load PhoBERT-CRF (heavyweight)
    predictor.load_phobert()

    logger.info(f"✅ Models loaded: {predictor.available_models}")
    yield
    logger.info("👋 Shutting down...")


app = FastAPI(
    title="Vietnamese ABSA API",
    description=(
        "Aspect-Based Sentiment Analysis for Vietnamese product reviews. "
        "Phân tích cảm xúc theo khía cạnh cho đánh giá sản phẩm tiếng Việt."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow frontend to call API
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(predict_router)

# Serve frontend static files
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
    logger.info(f"📂 Serving frontend from {frontend_dir}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("backend.app.main:app", host=HOST, port=PORT, reload=True)
