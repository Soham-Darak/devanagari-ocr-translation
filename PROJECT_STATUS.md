
# Lekha — Project Status

## Project identity

- Product: Lekha
- Repository: devanagari-ocr-translation
- Purpose: Devanagari character recognition, document text extraction,
  and English translation.
- Frontend: React, TypeScript, Vite, Tailwind CSS.
- Backend: Python, FastAPI, PyTorch.
- Translation: Official Google Cloud Translation API v2.

## Source model

- Notebook: devanagari_cnn.ipynb
- Architecture: Custom three-block convolutional neural network.
- Input: Grayscale tensor of shape [batch, 1, 32, 32].
- Output: 46 classification logits per character.
- Recorded notebook test accuracy: 96.80%.
- Checkpoint filename: devanagari_cnn_best.pth.
- Checkpoint availability: Must be supplied separately.
- Training dataset: Devanagari Handwritten Character Dataset (DHCD).

## Important model constraints

The CNN classifies individual character images. It is not a complete
document OCR model.

Full-page OCR requires segmentation and character reconstruction or a
separate document OCR adapter.

The notebook's ImageFolder class ordering must be preserved. Never map
output indices to Devanagari characters using a manually assumed order.

Use the saved checkpoint's class_names and validate the mapping.

## Security requirements

- Never commit API keys or .env files.
- Keep Google Translation credentials on the backend.
- Never fabricate OCR or translation results.
- Do not log document contents by default.
- Validate uploads on the server.
- Do not commit model weights or datasets.

## API contract

- GET /api/health
- POST /api/ocr
- POST /api/translate

## Development milestones

- [ ] Repository initialized.
- [ ] Project documentation committed.
- [ ] Backend configuration and dependencies installed.
- [ ] Actual CNN adapter implemented and tested.
- [ ] Image OCR endpoint implemented.
- [ ] PDF extraction and rendering implemented.
- [ ] Google Translation service implemented.
- [ ] Backend automated tests pass.
- [ ] Frontend implemented.
- [ ] Frontend type check and production build pass.
- [ ] Real checkpoint smoke test passes.
- [ ] End-to-end workflow verified.

## Current blockers

1. The trained .pth checkpoint was not included with the uploaded notebook.
2. Full-document recognition is not provided by the CNN alone.
3. Google Cloud Translation credentials must be configured locally.

## Current development stage

Stage 1: Repository setup and model integration foundation.

## Development log

Update this section after each meaningful change.

### Initial audit
- Inspected the uploaded Jupyter notebook.
- Identified the CNN architecture and input/output contract.
- Identified the need to preserve ImageFolder class ordering.
- Identified missing checkpoint and full-document OCR limitations.
