
from fastapi import APIRouter, Request


router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check(request: Request) -> dict[str, object]:
    settings = request.app.state.settings
    ocr_service = request.app.state.ocr_service

    return {
        "status": "ok",
        "service": settings.app_name,
        "ocr_model": {
            "status": "loaded" if ocr_service.is_loaded else "unavailable",
        },
    }

