
import { useCallback, useState } from 'react'
import { AnimatePresence, motion } from 'motion/react'
import { useDropzone } from 'react-dropzone'
import {
  ArrowDown,
  ArrowDownToLine,
  ArrowLeftRight,
  ArrowRight,
  Check,
  ChevronDown,
  CircleAlert,
  Clock3,
  FileImage,
  FileText,
  Languages,
  LoaderCircle,
  ScanText,
  ShieldCheck,
  Sparkles,
  Upload,
  X,
} from 'lucide-react'
import { Toaster, toast } from 'sonner'
import './App.css'

type OCRPage = {
  page_number: number
  text: string
  extraction_method: 'embedded_text' | 'character_classifier' | 'unavailable'
  warning?: string | null
}

type OCRResponse = {
  success: boolean
  filename: string
  input_type: 'image' | 'pdf'
  page_count?: number | null
  extracted_text: string
  pages: OCRPage[]
  warnings: string[]
}

type TranslationResponse = {
  success: boolean
  original_text: string
  translated_text: string
  source_language?: string | null
  target_language: string
}

const MAX_FILE_SIZE = 10 * 1024 * 1024
const ACCEPTED_TYPES = {
  'image/png': ['.png'],
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/webp': ['.webp'],
  'application/pdf': ['.pdf'],
}

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`
}

function getErrorMessage(payload: unknown, fallback: string) {
  if (
    payload &&
    typeof payload === 'object' &&
    'detail' in payload &&
    typeof payload.detail === 'string'
  ) {
    return payload.detail
  }
  return fallback
}

export default function App() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [ocrResult, setOcrResult] = useState<OCRResponse | null>(null)
  const [sourceText, setSourceText] = useState('')
  const [translatedText, setTranslatedText] = useState('')
  const [isRecognizing, setIsRecognizing] = useState(false)
  const [isTranslating, setIsTranslating] = useState(false)
  const [sourceLanguage, setSourceLanguage] = useState('auto')
  const [activeTab, setActiveTab] = useState<'source' | 'translation'>('source')

  const clearFile = useCallback(() => {
    setFile(null)
    setOcrResult(null)
    setSourceText('')
    setTranslatedText('')
    setActiveTab('source')
    setPreviewUrl((current) => {
      if (current) URL.revokeObjectURL(current)
      return null
    })
  }, [])

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      const nextFile = acceptedFiles[0]
      if (!nextFile) return

      if (nextFile.size > MAX_FILE_SIZE) {
        toast.error('This file is too large', {
          description: 'Choose an image or PDF smaller than 10 MB.',
        })
        return
      }

      clearFile()
      setFile(nextFile)

      if (nextFile.type.startsWith('image/')) {
        setPreviewUrl(URL.createObjectURL(nextFile))
      }
    },
    [clearFile],
  )

  const {
    getRootProps,
    getInputProps,
    isDragActive,
    open: openFilePicker,
  } = useDropzone({
    onDrop,
    accept: ACCEPTED_TYPES,
    maxFiles: 1,
    maxSize: MAX_FILE_SIZE,
    noClick: true,
    noKeyboard: true,
    onDropRejected: (rejections) => {
      const reason = rejections[0]?.errors[0]?.code
      if (reason === 'file-too-large') {
        toast.error('File exceeds 10 MB')
      } else {
        toast.error('Unsupported file', {
          description: 'Upload a PNG, JPG, WEBP, or PDF file.',
        })
      }
    },
  })

  const recognizeDocument = async () => {
    if (!file) {
      toast.error('Choose a file first')
      return
    }

    setIsRecognizing(true)
    setOcrResult(null)
    setSourceText('')
    setTranslatedText('')

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await fetch('/api/ocr', {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(
          getErrorMessage(payload, `Recognition failed (${response.status})`),
        )
      }

      const result = payload as OCRResponse
      setOcrResult(result)
      setSourceText(result.extracted_text ?? '')
      setActiveTab('source')

      if (result.warnings?.length) {
        toast.warning('Recognition completed with limitations', {
          description: result.warnings[0],
        })
      } else {
        toast.success('File processed')
      }
    } catch (error) {
      toast.error('Could not process this file', {
        description:
          error instanceof Error
            ? error.message
            : 'Check that the backend is running and try again.',
      })
    } finally {
      setIsRecognizing(false)
    }
  }

  const translateText = async () => {
    const text = sourceText.trim()

    if (!text) {
      toast.error('There is no text to translate', {
        description: 'Recognize a file or enter some Devanagari text first.',
      })
      return
    }

    setIsTranslating(true)

    try {
      const response = await fetch('/api/translate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          source_language: sourceLanguage,
          target_language: 'en',
        }),
      })

      const payload = await response.json().catch(() => null)

      if (!response.ok) {
        throw new Error(
          getErrorMessage(payload, `Translation failed (${response.status})`),
        )
      }

      const result = payload as TranslationResponse
      setTranslatedText(result.translated_text)
      setActiveTab('translation')
      toast.success('Translation ready')
    } catch (error) {
      toast.error('Translation failed', {
        description:
          error instanceof Error
            ? error.message
            : 'Check your API configuration and try again.',
      })
    } finally {
      setIsTranslating(false)
    }
  }

  const downloadText = (text: string, suffix: string) => {
    if (!text.trim()) {
      toast.error('There is no text to download')
      return
    }

    const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.download = `lekha-${suffix}.txt`
    document.body.appendChild(link)
    link.click()
    link.remove()
    URL.revokeObjectURL(url)
    toast.success('Text file downloaded')
  }

  const isPdf = file?.type === 'application/pdf'
  const hasSourceText = sourceText.trim().length > 0

  return (
    <div className="lekha-app">
      <Toaster position="top-right" richColors closeButton />

      <header className="topbar">
        <div className="app-shell topbar-inner">
          <a className="brand" href="#home" aria-label="Lekha home">
            <span className="brand-mark devanagari">ल</span>
            <span className="brand-name">lekha<span>.</span></span>
          </a>

          <nav className="top-nav" aria-label="Main navigation">
            <a className="nav-link active" href="#workspace">Workspace</a>
            <a className="nav-link" href="#how-it-works">How it works</a>
          </nav>

          <div className="topbar-status">
            <span className="status-dot" />
            <span>Document studio</span>
          </div>
        </div>
      </header>

      <main id="home" className="app-shell main-content">
        <section className="hero">
          <motion.div
            className="hero-copy"
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
          >
            <div className="hero-kicker">
              <Sparkles size={14} />
              <span>YOUR WORDS, WITHOUT BORDERS</span>
            </div>
            <h1>
              Every script has
              <br />
              a <span className="hero-highlight">story to tell.</span>
            </h1>
            <p className="hero-description">
              Turn Devanagari documents into clear, editable text and translate
              them into English — all in one thoughtful workspace.
            </p>
            <div className="hero-trust">
              <span><ShieldCheck size={15} /> You stay in control</span>
              <span><Languages size={15} /> Devanagari to English</span>
            </div>
          </motion.div>

          <motion.div
            className="hero-art"
            aria-label="Decorative Devanagari typography"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ duration: 0.55, delay: 0.12 }}
          >
            <div className="art-orbit orbit-one" />
            <div className="art-orbit orbit-two" />
            <span className="art-glyph glyph-one devanagari">अ</span>
            <span className="art-glyph glyph-two devanagari">क</span>
            <span className="art-glyph glyph-three devanagari">ज्ञ</span>
            <span className="art-small-label">LANGUAGE · MEANING · CONNECTION</span>
            <span className="art-star star-one">✳</span>
            <span className="art-star star-two">✦</span>
          </motion.div>
        </section>

        <section id="workspace" className="workspace-section">
          <div className="section-heading">
            <div>
              <span className="eyebrow">THE WORKSPACE</span>
              <h2>Bring your document to life.</h2>
            </div>
            <span className="step-indicator"><span>01</span> Upload <i /> <span>02</span> Review <i /> <span>03</span> Translate</span>
          </div>

          <div className="workspace-grid">
            <section className="upload-panel panel">
              <div className="panel-heading">
                <div className="panel-icon upload-icon"><Upload size={18} /></div>
                <div>
                  <h3>Upload document</h3>
                  <p>Start with an image or a PDF.</p>
                </div>
              </div>

              <div
                {...getRootProps()}
                className={`dropzone ${isDragActive ? 'dropzone-active' : ''} ${file ? 'dropzone-has-file' : ''}`}
              >
                <input {...getInputProps()} aria-label="Choose document" />

                {file ? (
                  <div className="selected-file">
                    <div className="file-type-icon">
                      {isPdf ? <FileText size={25} /> : <FileImage size={25} />}
                    </div>
                    <div className="selected-file-details">
                      <strong title={file.name}>{file.name}</strong>
                      <span>{formatBytes(file.size)} · {isPdf ? 'PDF document' : 'Image file'}</span>
                    </div>
                    <button
                      className="icon-button remove-file"
                      type="button"
                      aria-label="Remove selected file"
                      onClick={(event) => {
                        event.stopPropagation()
                        clearFile()
                      }}
                    >
                      <X size={17} />
                    </button>
                  </div>
                ) : (
                  <>
                    <div className="dropzone-illustration">
                      <div className="upload-cloud"><Upload size={23} /></div>
                      <span className="floating-file"><FileText size={17} /></span>
                    </div>
                    <p className="dropzone-title">
                      {isDragActive ? 'Drop your file here' : 'Drop your file here'}
                    </p>
                    <p className="dropzone-subtitle">or choose one from your device</p>
                    <button
                      className="browse-button"
                      type="button"
                      onClick={(event) => {
                        event.stopPropagation()
                        openFilePicker()
                      }}
                    >
                      Browse files <ArrowRight size={15} />
                    </button>
                  </>
                )}
              </div>

              {previewUrl && (
                <div className="image-preview">
                  <img src={previewUrl} alt="Selected document preview" />
                </div>
              )}

              <div className="upload-guidelines">
                <div className="guideline-row">
                  <span><Check size={14} /> PNG, JPG, WEBP, PDF</span>
                  <span>Up to 10 MB</span>
                </div>
                <div className="guideline-row">
                  <span><Check size={14} /> Printed or handwritten input</span>
                  <span>One file at a time</span>
                </div>
              </div>

              <button
                className="primary-button recognize-button"
                type="button"
                onClick={recognizeDocument}
                disabled={!file || isRecognizing}
              >
                {isRecognizing ? (
                  <><LoaderCircle className="spin" size={18} /> Processing document…</>
                ) : (
                  <><ScanText size={18} /> Extract text <ArrowRight size={17} /></>
                )}
              </button>

              <p className="upload-footnote">
                <ShieldCheck size={14} /> Files are processed by your configured backend.
              </p>
            </section>

            <section className="results-panel panel">
              <div className="results-top">
                <div className="panel-heading">
                  <div className="panel-icon results-icon"><Languages size={18} /></div>
                  <div>
                    <h3>Your text workspace</h3>
                    <p>Review, edit and translate your text.</p>
                  </div>
                </div>
                <span className="editable-tag"><span /> Editable</span>
              </div>

              <div className="language-toolbar">
                <div className="language-label">
                  <span className="language-dot source-dot" />
                  <span>Source text</span>
                  <ChevronDown size={14} />
                </div>
                <ArrowLeftRight className="language-switch-icon" size={16} />
                <div className="language-label target-language">
                  <span className="language-dot target-dot" />
                  <span>English</span>
                  <ChevronDown size={14} />
                </div>
              </div>

              <div className="text-tabs" role="tablist" aria-label="Text view">
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === 'source'}
                  className={activeTab === 'source' ? 'text-tab selected' : 'text-tab'}
                  onClick={() => setActiveTab('source')}
                >
                  Devanagari text
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={activeTab === 'translation'}
                  className={activeTab === 'translation' ? 'text-tab selected' : 'text-tab'}
                  onClick={() => setActiveTab('translation')}
                  disabled={!translatedText}
                >
                  English translation
                </button>
              </div>

              <AnimatePresence mode="wait">
                {activeTab === 'source' ? (
                  <motion.div
                    key="source"
                    className="text-editor-wrap"
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.16 }}
                  >
                    <textarea
                      className="text-editor devanagari"
                      value={sourceText}
                      onChange={(event) => setSourceText(event.target.value)}
                      placeholder={'आपका निकाला गया टेक्स्ट यहाँ दिखाई देगा…\n\nया यहाँ देवनागरी टेक्स्ट टाइप करके अनुवाद करें।'}
                      aria-label="Editable Devanagari source text"
                      spellCheck={false}
                    />
                    <div className="editor-footer">
                      <span>{sourceText.length} characters</span>
                      <button
                        className="text-action"
                        type="button"
                        disabled={!sourceText.trim()}
                        onClick={() => {
                          setSourceText('')
                          setTranslatedText('')
                        }}
                      >
                        Clear text
                      </button>
                    </div>
                  </motion.div>
                ) : (
                  <motion.div
                    key="translation"
                    className="text-editor-wrap translation-wrap"
                    initial={{ opacity: 0, y: 4 }}
                    animate={{ opacity: 1, y: 0 }}
                    exit={{ opacity: 0, y: -4 }}
                    transition={{ duration: 0.16 }}
                  >
                    <div className="translation-output" aria-live="polite">
                      {translatedText}
                    </div>
                    <div className="editor-footer">
                      <span>{translatedText.length} characters</span>
                      <button
                        className="text-action"
                        type="button"
                        onClick={() => downloadText(translatedText, 'translation')}
                      >
                        <ArrowDownToLine size={14} /> Download
                      </button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {ocrResult?.warnings?.length ? (
                <div className="warning-notice">
                  <CircleAlert size={16} />
                  <div>
                    <strong>Processing note</strong>
                    <p>{ocrResult.warnings.join(' ')}</p>
                  </div>
                </div>
              ) : null}

              {ocrResult && (
                <div className="result-meta">
                  <Check size={14} />
                  <span>
                    {ocrResult.input_type === 'pdf'
                      ? `${ocrResult.page_count ?? ocrResult.pages.length} page(s) processed`
                      : 'Image processed'}
                  </span>
                  <span className="meta-separator">·</span>
                  <span>{ocrResult.filename}</span>
                </div>
              )}

              <div className="translate-controls">
                <label className="source-language-select">
                  <span>Source language</span>
                  <select
                    value={sourceLanguage}
                    onChange={(event) => setSourceLanguage(event.target.value)}
                  >
                    <option value="auto">Auto detect</option>
                    <option value="hi">Hindi</option>
                    <option value="mr">Marathi</option>
                    <option value="ne">Nepali</option>
                  </select>
                </label>
                <button
                  className="primary-button translate-button"
                  type="button"
                  onClick={translateText}
                  disabled={!hasSourceText || isTranslating}
                >
                  {isTranslating ? (
                    <><LoaderCircle className="spin" size={17} /> Translating…</>
                  ) : (
                    <><Languages size={17} /> Translate to English <ArrowRight size={16} /></>
                  )}
                </button>
              </div>

              <div className="result-bottom-actions">
                <span><Clock3 size={14} /> You can edit text before translating</span>
                <button
                  className="text-action"
                  type="button"
                  disabled={!hasSourceText}
                  onClick={() => downloadText(sourceText, 'source')}
                >
                  <ArrowDownToLine size={14} /> Export source
                </button>
              </div>
            </section>
          </div>
        </section>

        <section id="how-it-works" className="how-section">
          <div className="how-heading">
            <span className="eyebrow">A SIMPLER WAY TO READ</span>
            <h2>From script to understanding.</h2>
            <p>Three simple steps. You stay in control of every word.</p>
          </div>
          <div className="how-grid">
            <article className="how-card">
              <span className="how-number">01</span>
              <div className="how-icon upload-how"><Upload size={21} /></div>
              <h3>Bring it in</h3>
              <p>Upload a supported image or PDF containing Devanagari text.</p>
            </article>
            <article className="how-card">
              <span className="how-number">02</span>
              <div className="how-icon edit-how"><ScanText size={21} /></div>
              <h3>Make it yours</h3>
              <p>Review the extracted text and correct anything before translation.</p>
            </article>
            <article className="how-card">
              <span className="how-number">03</span>
              <div className="how-icon translate-how"><Languages size={21} /></div>
              <h3>Find the meaning</h3>
              <p>Translate your reviewed text into English and export the result.</p>
            </article>
          </div>
        </section>

        <footer className="footer">
          <a className="brand footer-brand" href="#home">
            <span className="brand-mark devanagari">ल</span>
            <span className="brand-name">lekha<span>.</span></span>
          </a>
          <p>Made for language, built for understanding.</p>
          <a className="back-to-top" href="#home">Back to top <ArrowDown size={14} /></a>
        </footer>
      </main>
    </div>
  )
}
