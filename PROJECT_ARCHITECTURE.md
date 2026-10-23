
# Lekha Architecture

## Overview

Lekha is a web application with a React frontend and a FastAPI backend. OCR and translation are separate backend capabilities.

```text
React + TypeScript frontend
          |
          | HTTP / JSON and multipart requests
          v
      FastAPI API
          |
          +-- Input validation
          |
          +-- OCR service
          |     +-- Image preprocessing
          |     +-- PDF page processing
          |     +-- Embedded PDF text extraction
          |     +-- CNN character classification adapter
          |     +-- Optional full-document OCR adapter
          |
          +-- Local translation service
                +-- Hugging Face tokenizer
                +-- NLLB-200 distilled 600M model
                +-- Hindi / Marathi / Nepali to English
```

## Frontend responsibilities

- Upload supported images and PDFs.
- Preview selected images and display document information.
- Display editable source text.
- Allow the user to choose Hindi, Marathi, or Nepali as the source language.
- Request English translation.
- Display loading, validation, and error states.
- Copy or download source and translated text.

The frontend must not require a Google Translation API key. The translation model runs on the backend, not in the browser.

## Backend responsibilities

- Validate uploaded file type, size, and PDF page count.
- Extract embedded text from PDFs where available.
- Process supported image/PDF input through the OCR service.
- Load the CNN checkpoint when available.
- Load the local translation model and tokenizer.
- Enforce the configured translation character limit.
- Translate supported source languages into English.
- Return explicit errors when a model is unavailable or a request is invalid.
- Avoid logging document contents by default and clean up temporary resources.

## OCR limitation

The custom CNN is a 46-class isolated-character classifier. It does not locate arbitrary characters in a full page, infer reading order, or reconstruct arbitrary words by itself.

Full-document handwriting OCR requires a segmentation and reconstruction pipeline or another document OCR adapter. The UI and API must communicate OCR limitations rather than present unverified predictions as accurate document transcription.

## Translation behavior

The translation service uses `facebook/nllb-200-distilled-600M` through Hugging Face Transformers.

Supported source-language codes:

| Language | Code |
|---|---|
| Hindi | `hi` |
| Marathi | `mr` |
| Nepali | `ne` |

Target language: English (`en`).

Automatic language detection is not enabled. The client must send a supported source language. The model is loaded by the backend and may consume several gigabytes of disk space and substantial RAM. CPU inference can be slow.

The model's CC-BY-NC 4.0 license restricts commercial use; review the license before commercial deployment.

## API contracts

### `GET /api/health`

Returns API and model availability without exposing secrets.

### `POST /api/ocr`

Content type: `multipart/form-data`

Field:
- `file`: supported image or PDF.

Returns the OCR result, including extracted text, page-level results, and warnings where applicable. The exact fields are defined by the backend response schema.

### `POST /api/translate`

Content type: `application/json`

Example request:

```json
{
  "text": "मेरा नाम सोहम है।",
  "source_language": "hi",
  "target_language": "en"
}
```

Returns a response containing:
- `success`
- `original_text`
- `translated_text`
- `source_language`
- `target_language`

The response contract is shared with the frontend. Only English is currently supported as a target language.

## Configuration

Backend settings are read from environment variables and `backend/.env`.

Important translation settings:
- `TRANSLATION_MODEL_NAME`
- `TRANSLATION_DEVICE`
- `MAX_TRANSLATION_CHARACTERS`

The default model is `facebook/nllb-200-distilled-600M`, and the default device is `cpu`.

Keep `.env`, downloaded model weights, datasets, and local Python environments out of Git.

## Data handling

Document contents should be processed only as required for OCR and translation. Do not log source document contents by default. Temporary files must be cleaned up after use. Translation runs locally in the backend after the model has been downloaded; translation text is not sent to Google Cloud Translation.
