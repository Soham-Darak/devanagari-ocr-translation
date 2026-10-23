
# Lekha Frontend

React and TypeScript frontend for the Lekha Devanagari OCR and English translation application.

## Requirements

- Node.js and npm
- The Lekha FastAPI backend running locally

## Setup

From the `frontend` directory:

```powershell
npm install
npm run dev
```

Open the local URL printed by Vite, usually `http://localhost:5173`.

## Commands

```powershell
npm run dev
npm run lint
npm run build
npm run preview
```

## Application workflow

1. Upload a supported image or PDF.
2. Run document processing.
3. Review and edit the source text.
4. Choose Hindi, Marathi, or Nepali.
5. Translate the text into English.
6. Export the source or translated text.

## Backend integration

The frontend calls the backend's `/api/ocr` and `/api/translate` endpoints. The development server is expected to proxy `/api` requests to FastAPI.

The translation request uses:
- `text`
- `source_language`: `hi`, `mr`, or `ne`
- `target_language`: `en`

The translation model runs on the backend. Do not add Google Translation credentials or model weights to the frontend.

## Limitations

Automatic source-language detection is not currently supported. The OCR CNN classifies isolated characters and does not, by itself, guarantee full-page handwriting recognition.
