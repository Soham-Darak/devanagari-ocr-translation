
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.health import router as health_router
from app.api.routes.ocr import router as ocr_router
from app.api.routes.translation import router as translation_router
from app.core.config import get_settings
from app.services.local_translation_service import LocalTranslationService
from app.services.ocr_service import OCRService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

logger = logging.getLogger("lekha")


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings

    # Initialize the existing OCR model.
    # Initialize OCR models.
    app.state.ocr_service = OCRService(
        checkpoint_path=settings.checkpoint_path,
        device=settings.model_device,
    )

    try:
        app.state.ocr_service.load_model()
        logger.info("OCR service initialization completed.")
    except Exception:
        logger.exception("OCR service initialization failed.")
        

    # Initialize the local translation model.
    app.state.translation_service = LocalTranslationService(
        model_name=settings.translation_model_name,
        device=settings.translation_device,
        max_characters=settings.max_translation_characters,
    )

    try:
        app.state.translation_service.load_model()
    except Exception:
        logger.exception(
            "Local translation model unavailable at startup."
        )

    yield


settings = get_settings()

app = FastAPI(
    title=settings.app_name,
    description=(
        "API for Devanagari character classification, PDF text extraction, "
        "and local English translation."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

app.include_router(health_router, prefix=settings.api_prefix)
app.include_router(ocr_router, prefix=settings.api_prefix)
app.include_router(translation_router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": settings.app_name,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
