
# Lekha Architecture

## Overview

Lekha is a local-first web application with separate OCR and translation
services.

React frontend
    |
    | HTTP / JSON and multipart requests
    v
FastAPI backend
    |
    +-- Input validation
    |
    +-- OCR service
    |     |
    |     +-- Image preprocessing
    |     +-- PDF page processing
    |     +-- CNN character classification adapter
    |     +-- Optional full-document OCR adapter
    |
    +-- Translation service
          |
          +-- Google Cloud Translation API v2

## Frontend responsibilities

- Upload images and PDFs.
- Preview selected files.
- Accept manually entered Devanagari text.
- Display editable OCR output.
- Preserve original OCR results.
- Request English translation.
- Copy and download results.
- Display loading, validation, and error states.

## Backend responsibilities

- Validate file contents, formats, sizes, and PDF page counts.
- Process PDF pages in order.
- Load the CNN once when its checkpoint is available.
- Return explicit errors when required model assets are missing.
- Translate text only when the user requests translation.
- Keep credentials and model files off the frontend.
- Clean up temporary resources.

## OCR limitation

The supplied CNN is a 46-class single-character classifier. It does
not locate and reconstruct arbitrary words or document layouts.

The character classifier and any full-document OCR adapter must remain
separate components with clearly identified capabilities.

## API contracts

### GET /api/health

Returns API status and model availability without exposing secrets.

### POST /api/ocr

Multipart field:
- file: an image or PDF.

Returns:
- success
- filename
- input type
- page count, where applicable
- extracted text
- page-level results
- warnings, where appropriate

A model-dependent operation must return a clear error if the necessary
model asset or OCR capability is unavailable.

### POST /api/translate

JSON fields:
- text
- source_language: language code or auto
- target_language: en

Returns:
- original text
- translated text
- source language information, where available
- target language
- success status

## Configuration

Backend configuration comes from environment variables.

The frontend only receives the public backend base URL. It must never
receive the Google Translation API key.

## Data handling

Uploaded documents are processed in memory or in temporary files.
Temporary files must be removed after processing.

Document text is sent to Google Cloud Translation only when translation
is requested.
