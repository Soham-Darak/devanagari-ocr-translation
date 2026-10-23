
from fastapi import APIRouter, HTTPException, Request, status
from starlette.concurrency import run_in_threadpool

from app.schemas.translation import (
    TranslationRequest,
    TranslationResponse,
)
from app.services.translation_service import TranslationError


router = APIRouter(tags=["Translation"])


@router.post("/translate", response_model=TranslationResponse)
async def translate_text(
    payload: TranslationRequest,
    request: Request,
) -> TranslationResponse:
    service = request.app.state.translation_service

    try:
        result = await run_in_threadpool(
            service.translate,
            text=payload.text,
            source_language=payload.source_language,
            target_language=payload.target_language,
        )
    except TranslationError as exc:
        message = str(exc)

        if "limit" in message.lower() or "exceeds" in message.lower():
            code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
        elif (
            "unavailable" in message.lower()
            or "not enabled" in message.lower()
        ):
            code = status.HTTP_503_SERVICE_UNAVAILABLE
        else:
            code = status.HTTP_422_UNPROCESSABLE_ENTITY

        raise HTTPException(
            status_code=code,
            detail=message,
        ) from exc

    return TranslationResponse(
        success=True,
        original_text=payload.text,
        translated_text=result["translated_text"],
        source_language=result["source_language"],
        target_language=result["target_language"],
    )
