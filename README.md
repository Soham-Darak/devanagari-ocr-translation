
# Lekha — Devanagari OCR & English Translation

Lekha is a web application for recognizing Devanagari characters, extracting
text from documents, and translating reviewed text into English.

## Technology stack

- Frontend: React, TypeScript, Vite, Tailwind CSS
- Backend: Python, FastAPI, PyTorch
- PDF processing: PyMuPDF
- Translation: Official Google Cloud Translation API v2
- Testing: Pytest and frontend build/type checks

## Important model limitation

The supplied deep-learning notebook implements a 46-class character
classifier. It does not, by itself, recognize complete words or documents.

The application must not represent individual-character predictions as
verified full-document OCR. Full-document recognition requires an
additional segmentation/reconstruction pipeline or a separate OCR adapter.

## Prerequisites

- Python 3.11
- Node.js and npm
- Git
- A compatible PyTorch installation
- A Google Cloud project with Cloud Translation enabled for translation

## Repository layout

- `backend/`: API, model adapter, PDF processing, and translation
- `frontend/`: React web application
- `notebooks/`: notebook integration notes
- `PROJECT_STATUS.md`: current progress and handover context
- `PROJECT_ARCHITECTURE.md`: system architecture

## Model checkpoint

Obtain `devanagari_cnn_best.pth` from the original notebook environment
and place it at:

`backend/models/devanagari_cnn_best.pth`

Model weights are intentionally excluded from Git.

## Environment configuration

Copy `backend/.env.example` to `backend/.env` and configure the values
for your machine. Never commit real API keys.

## Development status

See `PROJECT_STATUS.md` for completed milestones, blockers, and next steps.

Do not consider OCR or translation operational until their real dependencies
are configured and their tests pass.
