
from dataclasses import dataclass

import fitz


@dataclass
class PDFPage:
    page_number: int
    text: str
    extraction_method: str
    warning: str | None = None


@dataclass
class PDFExtraction:
    pages: list[PDFPage]
    warnings: list[str]

    @property
    def combined_text(self) -> str:
        return "\n\n".join(
            page.text for page in self.pages if page.text.strip()
        )


class PDFProcessingError(ValueError):
    pass


def extract_pdf_text(
    content: bytes,
    max_pages: int = 20,
) -> PDFExtraction:
    if not content.startswith(b"%PDF-"):
        raise PDFProcessingError("The uploaded file is not a valid PDF.")

    try:
        document = fitz.open(stream=content, filetype="pdf")
    except Exception as exc:
        raise PDFProcessingError(
            "The PDF is corrupted or cannot be opened."
        ) from exc

    try:
        if document.needs_pass:
            raise PDFProcessingError(
                "This PDF is password-protected. Upload an unlocked copy."
            )

        if document.page_count == 0:
            raise PDFProcessingError("The PDF contains no pages.")

        if document.page_count > max_pages:
            raise PDFProcessingError(
                f"The PDF has {document.page_count} pages. "
                f"The current limit is {max_pages}."
            )

        pages: list[PDFPage] = []
        warnings: list[str] = []

        for index, page in enumerate(document):
            text = page.get_text("text").strip()

            if text:
                pages.append(
                    PDFPage(
                        page_number=index + 1,
                        text=text,
                        extraction_method="embedded_text",
                    )
                )
            else:
                message = (
                    "This page has no selectable text. Full-page OCR "
                    "is not yet available for scanned documents."
                )
                pages.append(
                    PDFPage(
                        page_number=index + 1,
                        text="",
                        extraction_method="unavailable",
                        warning=message,
                    )
                )
                warnings.append(f"Page {index + 1}: {message}")

        return PDFExtraction(pages=pages, warnings=warnings)

    finally:
        document.close()
