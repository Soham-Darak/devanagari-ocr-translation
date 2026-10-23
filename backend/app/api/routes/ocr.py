
from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status

from app.schemas.ocr import OCRResponse, PageOCRResult
from app.services.ocr_service import (
    OCRCapabilityUnavailable,
    OCRModelUnavailable,
    OCRService,
)
from app.services.pdf_service import PDFProcessingError, extract_pdf_text


router = APIRouter(tags=["OCR"])

ALLOWED_IMAGE_TYPES = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/bmp",
    "image/tiff",
}


@router.post("/ocr", response_model=OCRResponse)
async def perform_ocr(
    request: Request,
    file: UploadFile = File(...),
) -> OCRResponse:
    settings = request.app.state.settings
    ocr_service: OCRService = request.app.state.ocr_service

    content = await file.read(settings.max_upload_size_mb * 1024 * 1024 + 1)
    await file.close()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty.",
        )

    if len(content) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Maximum upload size is {settings.max_upload_size_mb} MB.",
        )

    filename = (file.filename or "upload").replace("\\", "/").split("/")[-1]
    is_pdf = content.startswith(b"%PDF-")

    if is_pdf:
        try:
            extraction = extract_pdf_text(
                content,
                max_pages=settings.max_pdf_pages,
            )
        except PDFProcessingError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

        pages = [
            PageOCRResult(
                page_number=page.page_number,
                text=page.text,
                extraction_method=page.extraction_method,
                warning=page.warning,
            )
            for page in extraction.pages
        ]

        return OCRResponse(
            success=True,
            filename=filename,
            input_type="pdf",
            page_count=len(pages),
            extracted_text=extraction.combined_text,
            pages=pages,
            warnings=extraction.warnings,
        )

    # Check actual image decoding rather than trusting the filename or MIME.
    try:
        image = OCRService.open_image(content)
    except OCRCapabilityUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=str(exc),
        ) from exc

    if file.content_type and file.content_type not in ALLOWED_IMAGE_TYPES:
        # The bytes are a readable image, but unsupported formats should
        # not be silently accepted under an unrelated MIME type.
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image content type.",
        )

    try:
        extracted_text, warnings = ocr_service.extract_from_image(image)
    except OCRModelUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except OCRCapabilityUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return OCRResponse(
        success=True,
        filename=filename,
        input_type="image",
        extracted_text=extracted_text,
        pages=[
            PageOCRResult(
                page_number=1,
                text=extracted_text,
                extraction_method="character_classifier",
            )
        ],
        warnings=warnings,
    )
