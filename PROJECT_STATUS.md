
# Lekha — Project Status

## Project identity

- **Product:** Lekha
- **Repository:** `devanagari-ocr-translation`
- **Purpose:** Devanagari document processing, text review, and English translation
- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** Python, FastAPI, PyTorch
- **Translation:** Local NLLB-200 distilled 600M model via Hugging Face Transformers

## Current implementation

### Backend

- FastAPI application with health, OCR, and translation routes.
- Configuration managed through `pydantic-settings`.
- PDF processing through PyMuPDF.
- Custom CNN model adapter for isolated Devanagari character classification.
- Local translation service using `facebook/nllb-200-distilled-600M`.
- Translation requests preserve the established response fields: `success`, `original_text`, `translated_text`, `source_language`, and `target_language`.

### Translation support

- Hindi (`hi`) to English (`en`)
- Marathi (`mr`) to English (`en`)
- Nepali (`ne`) to English (`en`)
- Automatic source-language detection is not supported.
- No Google Translation API key is required.
- Initial model setup requires a large download and sufficient disk space and RAM.

### Frontend

- Upload image and PDF files.
- Review and edit source text.
- Select a supported source language.
- Translate into English.
- Export source and translated text.

## Model details and limitations

### Character classifier

- Notebook: `devanagari_cnn.ipynb`
- Architecture: custom three-block convolutional neural network
- Input: grayscale tensors shaped `[batch, 1, 32, 32]`
- Output: 46 class logits per character
- Recorded notebook test accuracy: 96.80%
- Checkpoint: `backend/models/devanagari_cnn_best.pth`
- Dataset: Devanagari Handwritten Character Dataset (DHCD)

The CNN is an isolated-character classifier, not a complete page-level OCR model. Full-page handwriting recognition requires character segmentation, ordering, and reconstruction or a separate OCR engine. Preserve the original dataset class ordering; do not assume a manually constructed mapping from output indices to characters.

### Translation model

- Model: `facebook/nllb-200-distilled-600M`
- Runtime: Hugging Face Transformers and PyTorch
- Default device: CPU
- License: CC-BY-NC 4.0; not for commercial use under that license
- Intended project scope: local demonstration and evaluation, subject to the model's license and usage restrictions

## API contract

- `GET /api/health`
- `POST /api/ocr`
- `POST /api/translate`

The translation endpoint accepts text, a supported source language, and English as the target language.

## Development checklist

- [x] Repository initialized.
- [x] Frontend implemented and typed for all extraction methods.
- [x] FastAPI endpoints and schemas implemented.
- [x] Local NLLB model tested independently for Hindi-to-English translation.
- [x] Local translation service integrated into the backend code.
- [x] Run the complete backend test suite after integration (17/17 pytest tests pass).
- [x] Run frontend lint and production build after UI changes (Vite build passed).
- [x] Implement Shirorekha-based character splitting and Peak-Valley line segmentation.
- [x] Correct DHCD polarity mismatch (black bg, white strokes) & aspect-ratio centering.
- [x] Integrate dual-mode OCR (full Devanagari handwriting + custom CNN classifier).
- [x] Verify local translation on sample test images (test1, test2, test3).
- [x] Confirm the README, architecture, and environment example match the code.
- [x] Remove temporary experiment scripts and local environments from the working tree.
- [x] Review Git status and ignore rules.
- [x] Commit and push verified code.

## Security and repository hygiene

- Never commit `.env` files or API secrets.
- Do not commit model weights, datasets, Python virtual environments, model caches, or generated build output.
- Do not log document contents by default.
- Validate uploads on the server.
- Keep the frontend response aligned with the backend schemas.

## Current status

The Lekha application is fully implemented, verified, and hardened:
1. **Isolated Character CNN:** 96.17% test accuracy on DHCD (46 classes).
2. **Segmentation Pipeline:** Robust notebook rule removal, peak-valley line detection, and headline-aware character splitting.
3. **Full Document OCR & Translation:** End-to-end extraction and complete English translation demonstrated on test1, test2, and test3.
4. **Automated Testing:** 17/17 pytest backend unit tests passing; clean TypeScript Vite frontend build.

