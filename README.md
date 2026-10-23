# Lekha — Devanagari OCR & English Translation

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.119-green.svg)](https://fastapi.tiangolo.com/)
[![React 18](https://img.shields.io/badge/React-18-cyan.svg)](https://react.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)](https://pytorch.org/)
[![NLLB-200](https://img.shields.io/badge/Translation-NLLB--200--600M-purple.svg)](https://huggingface.co/facebook/nllb-200-distilled-600M)

**Lekha** is an end-to-end Machine Learning web application designed to process Devanagari documents (handwritten and printed), perform Optical Character Recognition (OCR), and translate extracted Hindi, Marathi, and Nepali text into English using an entirely local, offline neural translation pipeline.

---

## 1. Project Overview & Architecture

The application consists of a decoupled architecture with a modern React frontend and a FastAPI backend serving deep learning inference models locally without external cloud API dependencies.

```
┌─────────────────────────────────────────────────────────────┐
│                    React 18 + Vite Frontend                 │
│        (Drag & Drop, Preview, Edit Text, Dual-pane UI)      │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP /api/ocr & /api/translate
┌──────────────────────────────▼──────────────────────────────┐
│                    FastAPI Backend Service                  │
│                                                             │
│  ┌───────────────────────┐      ┌────────────────────────┐  │
│  │   OCR Engine Routing  │      │  NLLB-200 Translation  │  │
│  │                       │      │ (facebook/nllb-200-    │  │
│  │ • Embedded PDF text   │      │  distilled-600M)       │  │
│  │ • Full Devanagari OCR │      │                        │  │
│  │   (vowels + matras)   │      │ Hindi (hi)    ──┐      │  │
│  │ • Custom Devanagari   │      │ Marathi (mr)  ──┼─► En │  │
│  │   46-class CNN with   │      │ Nepali (ne)   ──┘      │  │
│  │   Shirorekha splitting│      └────────────────────────┘  │
│  └───────────────────────┘                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Features

- **Document Ingestion:** Supports PNG, JPG/JPEG, WEBP, and PDF files up to 10 MB.
- **Dual-Engine OCR:**
  - **Full Document Handwriting OCR (`engine="auto"`):** Transcribes continuous handwritten or printed text including all independent vowels, vowel diacritics (matras), conjuncts, anusvara, and punctuation.
  - **Custom Isolated-Character CNN (`engine="cnn"`):** 46-class deep convolutional neural network trained on the Devanagari Handwritten Character Dataset (DHCD) with Shirorekha-based character splitting and polarity correction.
  - **Embedded PDF Extraction:** Instant text extraction for vector and digital PDFs via PyMuPDF.
- **Offline Neural Translation:** Local inference using Meta's `facebook/nllb-200-distilled-600M` model via Hugging Face Transformers. No Google Cloud credentials or external API keys needed.
- **Interactive Review:** Source text is fully editable before translation, allowing users to correct noisy handwriting or format text.
- **Exporting:** Download recognized Devanagari text and translated English text as `.txt` files with a single click.

---

## 3. Deep Learning & Computer Vision Pipeline

### 3.1 Custom 46-Class Devanagari CNN
- **Architecture:** 3 convolutional blocks ($32 \to 64 \to 128$ filters) with Batch Normalization, ReLU activation, $2\times2$ Max Pooling, Spatial Dropout, and Fully Connected dense layers ($128\times4\times4 \to 256 \to 46$).
- **Parameters:** 824,206 trainable weights.
- **Dataset:** Devanagari Handwritten Character Dataset (DHCD), covering 36 base consonants (क–ज्ञ) and 10 numerals (०–९).
- **Evaluation Accuracy:** **96.17%** on 4,600 held-out test images.

### 3.2 Full-Page Classical CV Segmentation
When running the custom CNN pipeline on full notebook pages:
1. **Rule Line Removal:** Horizontal morphological filtering (`cv2.MORPH_OPEN` with wide horizontal kernels) detects and subtracts ruled notebook lines without destroying intersecting character descenders.
2. **Peak-Valley Line Band Detection:** Uses smoothed horizontal projection to identify Shirorekha headline density peaks and cuts at inter-line valleys, reliably segmenting all text lines.
3. **Shirorekha-Aware Word Splitting:** Within each connected component (word), vertical projection below the headline detects character boundaries and slices joined Devanagari script into individual glyphs.
4. **Aspect-Ratio Centering & Polarity Inversion:** Auto-detects paper scans (dark ink on bright background), inverts to match DHCD training polarity (bright stroke on black background), scales glyphs to fit a $26\times26$ bounding box, and centers them on a $32\times32$ canvas.

---

## 4. Local Installation & Running Guide

### Prerequisites
- **Python 3.10 or 3.11** installed
- **Node.js (v18+)** and **npm** installed
- **Git** installed

---

### Step 1: Clone Repository
```powershell
git clone https://github.com/Soham-Darak/devanagari-ocr-translation.git
cd devanagari-ocr-translation
```

---

### Step 2: Backend Setup & Execution

1. Navigate to the `backend/` directory:
   ```powershell
   cd backend
   ```

2. Create and activate a Python virtual environment:
   - **Windows (PowerShell):**
     ```powershell
     py -3.11 -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **Linux / macOS:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. Upgrade pip and install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Configure environment:
   ```powershell
   Copy-Item .env.example .env
   ```

5. Ensure the CNN checkpoint is present:
   ```text
   backend/models/devanagari_cnn_best.pth
   ```

6. Start the FastAPI server:
   ```powershell
   uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
   ```
   - API Docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Health Check: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)

---

### Step 3: Frontend Setup & Execution

1. Open a new terminal and navigate to the `frontend/` directory:
   ```powershell
   cd frontend
   ```

2. Install dependencies:
   ```powershell
   npm install
   ```

3. Start the Vite development server:
   ```powershell
   npm run dev
   ```

4. Open your browser and navigate to the printed local URL:
   ```text
   http://localhost:5173
   ```

---

### Step 4: Running Tests & Validation

- **Run Backend Unit Tests (17 tests):**
  ```powershell
  cd backend
  .\.venv\Scripts\python.exe -m pytest app/tests -v
  ```

- **Evaluate CNN Isolated Character Accuracy (96.17% on DHCD):**
  ```powershell
  cd backend
  .\.venv\Scripts\python.exe evaluate_cnn.py
  ```

- **Run Frontend Production Build Check:**
  ```powershell
  cd frontend
  npm run build
  ```

---

## 5. Demonstration & Benchmark on Test Images

The repository includes 3 representative test images in the root directory to validate end-to-end OCR and English translation.

### Test 1: `test1.png` (School Announcement Notice)
- **Document Type:** Handwritten institutional notice on unlined paper.
- **Extracted Devanagari Text:**
  ```text
  विद्यालय सूचना दिनांकः २४ अक्टूबर २०२६
  सभी विद्यार्थियों को सूचित किया जाता है कि सोमवार को सुबह १० बजे पुस्तकालय में एक विशेष कार्यक्रम आयोजित किया जाएगा। सभी विद्यार्थियों से समय पर उपस्थित होने का अनुरोध है।
  प्रधानाचार्य
  ```
- **Translated English Text:**
  ```text
  School notification date: 24 October 2026. All students are informed that a special event will be held at the library on Monday at 10:00 am. All students are requested to attend on time. Principal.
  ```

---

### Test 2: `test2.png` (Handwritten Essay on Lined Paper)
- **Document Type:** Multi-line cursive student essay (*मेरा भारत*) on ruled notebook paper with horizontal blue lines.
- **Extracted Devanagari Text:**
  ```text
  मेरा भारत
  भारत एक महान और विविधताओं से भरा देश है। यहाँ कई भाषाएँ, संस्कृतियाँ, त्योहार और परम्पराएँ हैं। भारत की पहचान उसकी एकता में है, जो उसे पूरे विश्व में विशेष बनाती है।
  ```
- **Translated English Text:**
  ```text
  My India.
  India is a great and diverse country with many languages, cultures, festivals and traditions. The identity of India is in its unity, which makes it special throughout the world.
  ```

---

### Test 3: `test3.jpeg` (Marathi Literature / Examination Excerpt)
- **Document Type:** Marathi cursive literary interpretation question paper.
- **Extracted Devanagari Text (Marathi):**
  ```text
  काव्यपंक्तीतील विचारसौंदर्य स्पष्ट करा... जेव्हा त्यावेळेस पाऊस तीव्र असतो, तितके मनात प्रसन्न वातावरण तयार होते.
  ```
- **Translated English Text:**
  ```text
  Explain the aesthetic and emotional thought in the poetic line... When the rain is intense, a pleasant and joyful atmosphere is created in the heart.
  ```

---

## 6. Project Structure

```text
devanagari-ocr-translation/
├── .gitignore                     # Git exclusion rules (models, caches, venvs)
├── README.md                      # Comprehensive project documentation
├── PROJECT_ARCHITECTURE.md        # Technical architecture and API contracts
├── PROJECT_STATUS.md              # Milestones, verification, and roadmap
├── test1.png                      # Test sample 1 (Hindi notice)
├── test2.png                      # Test sample 2 (Handwritten essay on ruled paper)
├── test3.jpeg                     # Test sample 3 (Marathi literary handwriting)
├── backend/
│   ├── .env.example               # Template environment configuration
│   ├── requirements.txt           # Python backend dependencies
│   ├── evaluate_cnn.py            # CNN benchmark evaluation script
│   ├── app/
│   │   ├── main.py                # FastAPI application entrypoint & lifespan
│   │   ├── core/config.py         # Settings & environment validation
│   │   ├── api/routes/
│   │   │   ├── health.py          # GET /api/health
│   │   │   ├── ocr.py             # POST /api/ocr (engine="auto" | "cnn")
│   │   │   └── translation.py     # POST /api/translate
│   │   ├── schemas/               # Pydantic request/response schemas
│   │   ├── services/
│   │   │   ├── model_adapter.py   # DevanagariCNN definition & character classifier
│   │   │   ├── full_page_ocr_service.py # Shirorekha-aware segmentation pipeline
│   │   │   ├── ocr_service.py     # OCR engine coordinator
│   │   │   ├── local_translation_service.py # Offline NLLB translation service
│   │   │   └── pdf_service.py     # PyMuPDF digital text extractor
│   │   └── tests/
│   │       ├── test_api.py        # API endpoint tests
│   │       └── test_services.py   # OCR, segmentation, & translation unit tests
│   └── models/
│       └── devanagari_cnn_best.pth # Trained CNN weights (excluded from git)
├── frontend/
│   ├── index.html                 # HTML shell
│   ├── package.json               # Node.js dependencies & scripts
│   ├── tsconfig.json              # TypeScript configuration
│   ├── vite.config.ts             # Vite configuration with API proxying
│   └── src/
│       ├── App.tsx                # Main single-page application UI
│       ├── App.css                # Component styling & responsive rules
│       └── main.tsx               # React application root
└── notebooks/
    ├── README.md                  # Model export & training documentation
    └── devanagari_cnn.ipynb       # Jupyter notebook for training DevanagariCNN
```

---

## 7. Model Attribution & Licensing

- **NLLB-200 Distilled 600M:** Released by Meta AI under **Creative Commons Non-Commercial (CC-BY-NC 4.0)**. Intended for non-commercial research, demonstration, and evaluation.
- **Devanagari Handwritten Character Dataset (DHCD):** Developed by the Department of Electronics and Computer Engineering, Tribhuvan University, Nepal.
