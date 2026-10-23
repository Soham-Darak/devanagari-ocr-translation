
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
- [x] Frontend implemented.
- [x] FastAPI endpoints and schemas implemented.
- [x] Local NLLB model tested independently for Hindi-to-English translation.
- [x] Local translation service integrated into the backend code.
- [ ] Replace the obsolete API test that expects Google credentials.
- [ ] Run the complete backend test suite after integration.
- [ ] Run frontend lint and production build after UI changes.
- [ ] Verify local translation through the live FastAPI endpoint.
- [ ] Verify the frontend-to-backend translation workflow.
- [ ] Confirm the README, architecture, and environment example match the code.
- [ ] Remove temporary experiment scripts and local environments from the working tree.
- [ ] Review Git status and ignore rules.
- [ ] Commit and push only after verification.

## Security and repository hygiene

- Never commit `.env` files or API secrets.
- Do not commit model weights, datasets, Python virtual environments, model caches, or generated build output.
- Do not log document contents by default.
- Validate uploads on the server.
- Keep the frontend response aligned with the backend schemas.
- Do not claim that the character classifier performs complete page-level OCR.

## Current next steps

1. Finish the local translation API tests.
2. Update the frontend language selector and error messaging.
3. Update project and frontend documentation.
4. Run backend and frontend checks.
5. Test an actual translation request and inspect the health endpoint.
6. Remove temporary files that are no longer needed.
7. Review the final diff, commit, and push.

## Development log

### Local translation integration

- Tested `facebook/nllb-200-distilled-600M` in a separate Python environment.
- Confirmed a Hindi-to-English example translated successfully on CPU.
- Abandoned the gated IndicTrans2 checkpoint after access was denied.
- Added a local translation service and configuration for NLLB.
- Replaced the translation route's Google-backed implementation with the local service.
- Updated the API test plan because the previous missing-Google-credentials test no longer describes the intended behavior.

### Verification note

Do not mark the integration complete until the updated backend tests, frontend lint/build, and live translation smoke test pass.
