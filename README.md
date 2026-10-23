
# Lekha — Devanagari OCR & English Translation

Lekha is a web application for processing Devanagari documents, reviewing extracted text, and translating supported languages into English.

## Features

- Upload PNG, JPG/JPEG, WEBP, and PDF files.
- Extract embedded text from supported PDFs.
- Process document pages through the backend OCR pipeline.
- Review and edit extracted text before translation.
- Translate Hindi, Marathi, and Nepali text into English using a locally run NLLB-200 model.
- Export source text and translated text as `.txt` files.
- Use the FastAPI backend without a Google Translation API key.

## Technology stack

- **Frontend:** React, TypeScript, Vite, Tailwind CSS
- **Backend:** Python, FastAPI, PyTorch
- **Translation:** Hugging Face Transformers and `facebook/nllb-200-distilled-600M`
- **PDF processing:** PyMuPDF
- **Testing:** Pytest, ESLint, TypeScript checks, and Vite production build

## Important limitations

### OCR

The custom CNN checkpoint is a 46-class isolated-character classifier. It is not, by itself, a full-page handwriting OCR model. Complete document recognition requires reliable character segmentation and reconstruction or a separate document OCR implementation.

PDFs containing embedded text may be extractable without handwriting recognition. Do not treat a character-classification result as verified full-document OCR.

### Translation

- Supported source languages: Hindi (`hi`), Marathi (`mr`), and Nepali (`ne`).
- Target language: English (`en`).
- Automatic source-language detection is not currently enabled.
- The first model load requires downloading several gigabytes of model files unless they are already cached.
- CPU inference may be slow, and the model requires substantial RAM.
- The NLLB-200 distilled model is licensed **CC-BY-NC 4.0 (non-commercial)**. Review the model's license and intended-use restrictions before commercial use.

## Prerequisites

- Python 3.11 recommended
- Node.js and npm
- Git
- Sufficient disk space and RAM for the translation model
- A compatible PyTorch installation

## Repository layout

- `backend/`: FastAPI routes, OCR and translation services, configuration, and tests
- `frontend/`: React application
- `notebooks/`: notebook and model integration material
- `PROJECT_STATUS.md`: development status and next steps
- `PROJECT_ARCHITECTURE.md`: architecture and API contracts

## Setup

### 1. Backend

Open PowerShell at the repository root:

```powershell
cd backend
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

Place the CNN checkpoint at:

`backend/models/devanagari_cnn_best.pth`

Model weights and local environment files must not be committed.

The first translation-model load downloads `facebook/nllb-200-distilled-600M` from Hugging Face. Ensure the machine has internet access during the initial download and enough disk space for the model cache.

Start the API from the `backend` directory:

```powershell
uvicorn app.main:app --reload
```

API documentation: `http://127.0.0.1:8000/docs`

Health check: `http://127.0.0.1:8000/api/health`

### 2. Frontend

Open a second PowerShell terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the URL printed by Vite, usually `http://localhost:5173`.

### 3. Configuration

Backend settings are read from `backend/.env`. The frontend must not contain model credentials or private backend configuration.

The default translation model is:

`facebook/nllb-200-distilled-600M`

The default translation device is CPU. Change it only after confirming the installed PyTorch build and hardware support the chosen device.

## Testing

From the `backend` directory:

```powershell
python -m pytest app/tests -v
```

From the `frontend` directory:

```powershell
npm run lint
npm run build
```

Before committing, verify the translation endpoint with a short Hindi, Marathi, or Nepali sentence and confirm the result is returned in English.

## License and model attribution

The project code and the downloaded translation model may have different license terms. Check the license of each dependency and model before redistribution or commercial deployment. NLLB-200 distilled 600M is distributed under CC-BY-NC 4.0.
