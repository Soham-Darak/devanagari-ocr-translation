
from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.translation import (
    TranslationRequest,
    TranslationResponse,
)
from app.services.translation_service import (
    TranslationError,
    TranslationService,
)


router = APIRouter(tags=["Translation"])


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    payload: TranslationRequest,
    request: Request,
) -> TranslationResponse:
    settings = request.app.state.settings

    service = TranslationService(settings)

    try:
        result = await service.translate(
            text=payload.text,
            source_language=payload.source_language,
            target_language=payload.target_language,
        )
    except TranslationError as exc:
        message = str(exc)

        if "not configured" in message:
            code = status.HTTP_503_SERVICE_UNAVAILABLE
        elif "limit" in message.lower() or "exceeds" in message.lower():
            code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        elif "timed out" in message.lower() or "connect" in message.lower():
            code = status.HTTP_502_BAD_GATEWAY
        elif "authentication failed" in message.lower():
            code = status.HTTP_502_BAD_GATEWAY
        else:
            code = status.HTTP_502_BAD_GATEWAY

        raise HTTPException(status_code=code, detail=message) from exc

    return TranslationResponse(
        success=True,
        original_text=payload.text,
        translated_text=result["translated_text"],
        source_language=result["source_language"],
        target_language=result["target_language"],
    )

