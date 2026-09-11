import { useState, useEffect, useRef } from 'react'
import './index.css'

const API_BASE = ''

export default function App() {
  const [question, setQuestion] = useState('')
  const [mode, setMode] = useState('hybrid_reranked') // 'text' | 'vision' | 'hybrid' | 'visual_reranked' | 'hybrid_reranked'
  const [result, setResult] = useState(null)
  const [comparison, setComparison] = useState(null)
  const [loading, setLoading] = useState(false)
  const [loadingStep, setLoadingStep] = useState(1)
  const [loadingAction, setLoadingAction] = useState(null)
  const [loadingMode, setLoadingMode] = useState(mode)
  const [documents, setDocuments] = useState([])
  const [selectedEvidence, setSelectedEvidence] = useState(null)
  
  const [health, setHealth] = useState(null)
  
  // Modals & Toasts
  const [docModal, setDocModal] = useState(null)
  const [lightboxImage, setLightboxImage] = useState(null)
  const [lightboxRotation, setLightboxRotation] = useState(0)
  const [toastMessage, setToastMessage] = useState('')
  // Inline Document Viewer State
  const [inlineZoom, setInlineZoom] = useState(1.0)
  const [inlineRotation, setInlineRotation] = useState(0)

  const zoomInInline = () => setInlineZoom(z => Math.min(Number((z + 0.25).toFixed(2)), 3.0))
  const zoomOutInline = () => setInlineZoom(z => Math.max(Number((z - 0.25).toFixed(2)), 0.5))
  const rotateAnticlockwiseInline = () => setInlineRotation(r => r - 90)
  const rotateClockwiseInline = () => setInlineRotation(r => r + 90)
  const resetInlineControls = () => {
    setInlineZoom(1.0)
    setInlineRotation(0)
  }

  const rotateAnticlockwise = () => setLightboxRotation((prev) => prev - 90)
  const rotateClockwise = () => setLightboxRotation((prev) => prev + 90)
  const resetRotation = () => setLightboxRotation(0)
  const closeLightbox = () => {
    setLightboxImage(null)
    setLightboxRotation(0)
  }

  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!lightboxImage) return
      if (e.key === 'Escape') closeLightbox()
      else if (e.key === 'ArrowLeft' || e.key === 'r' || e.key === 'R') rotateAnticlockwise()
      else if (e.key === 'ArrowRight') rotateClockwise()
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [lightboxImage])
  
  const searchInputRef = useRef(null)
  const toastTimeoutRef = useRef(null)

  useEffect(() => {
    fetch(`${API_BASE}/api/health`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setHealth(d))
      .catch(() => setHealth({ status: 'offline', text_index_loaded: false, visual_index_loaded: false, total_pages: 0 }))

    fetch(`${API_BASE}/api/documents`)
      .then(r => r.ok ? r.json() : null)
      .then(d => d && setDocuments(d.documents || []))
      .catch(() => {})

  }, [])

  // Keyboard accelerator (Cmd+K / Ctrl+K)
  useEffect(() => {
    const handleKeyDown = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault()
        if (searchInputRef.current) {
          searchInputRef.current.focus()
          searchInputRef.current.select()
        }
      }
      if (e.key === 'Escape') {
        setDocModal(null)
        setLightboxImage(null)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const triggerToast = (msg) => {
    if (toastTimeoutRef.current) clearTimeout(toastTimeoutRef.current)
    setToastMessage(msg)
    toastTimeoutRef.current = setTimeout(() => {
      setToastMessage('')
    }, 2800)
  }

  const abortControllerRef = useRef(null)
  const requestIdRef = useRef(null)

  const stopActiveRequest = () => {
    const requestId = requestIdRef.current
    if (abortControllerRef.current) abortControllerRef.current.abort()
    if (requestId) {
      fetch(`${API_BASE}/api/cancel/${encodeURIComponent(requestId)}`, { method: 'POST', keepalive: true }).catch(() => {})
    }
    requestIdRef.current = null
    setLoading(false)
    setLoadingAction(null)
    triggerToast('Request cancelled')
  }

  const createRequestId = () => {
    if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID()
    return `${Date.now()}-${Math.random().toString(36).slice(2)}`
  }

  const handleQuery = async (queryText = question, selectedMode = mode) => {
    const q = queryText.trim()
    if (!q) {
      if (searchInputRef.current) searchInputRef.current.focus()
      triggerToast('Please enter a search query')
      return
    }

    if (loading) return
    const controller = new AbortController()
    const requestId = createRequestId()
    abortControllerRef.current = controller
    requestIdRef.current = requestId

    setLoading(true)
    setLoadingAction('search')
    setLoadingMode(selectedMode)
    setLoadingStep(1)
    setResult(null)
    setComparison(null)
    setSelectedEvidence(null)

    const step1Timer = setTimeout(() => setLoadingStep(2), selectedMode === 'visual' ? 0 : 1200)
    const step2Timer = setTimeout(() => setLoadingStep(3), selectedMode === 'text' ? 1200 : 3200)

    try {
      const res = await fetch(`${API_BASE}/api/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, mode: selectedMode, top_k: 5, request_id: requestId }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || `Server error: HTTP ${res.status}`)
      }

      const data = await res.json()
      setResult(data)
      if (data.retrieved_pages?.length > 0) {
        setSelectedEvidence(data.retrieved_pages[0])
      }
      triggerToast(`Search finished in ${data.total_time_ms ? data.total_time_ms + 'ms' : 'real-time'}`)
    } catch (e) {
      if (e.name === 'AbortError') return
      setResult({ status: 'error', answer: `Connection error: ${e.message}` })
      triggerToast(`Error: ${e.message}`)
    } finally {
      clearTimeout(step1Timer)
      clearTimeout(step2Timer)
      setLoading(false)
      setLoadingAction(null)
      if (requestIdRef.current === requestId) requestIdRef.current = null
    }
  }

  const handleCompare = async (queryText = question) => {
    const q = queryText.trim()
    if (!q) {
      if (searchInputRef.current) searchInputRef.current.focus()
      triggerToast('Please enter a query to run Multi-Mode Comparison')
      return
    }

    if (loading) return
    const controller = new AbortController()
    const requestId = createRequestId()
    abortControllerRef.current = controller
    requestIdRef.current = requestId

    setLoading(true)
    setLoadingAction('compare')
    setLoadingMode('compare')
    setLoadingStep(1)
    setResult(null)
    setComparison(null)
    setSelectedEvidence(null)

    const step1Timer = setTimeout(() => setLoadingStep(2), 2000)
    const step2Timer = setTimeout(() => setLoadingStep(3), 5000)

    try {
      const res = await fetch(`${API_BASE}/api/compare`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: q, top_k: 3, request_id: requestId }),
        signal: controller.signal,
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        throw new Error(errorData.detail || `Server error: HTTP ${res.status}`)
      }

      const data = await res.json()
      setComparison(data)
      triggerToast(`Multi-Mode Benchmark finished in ${data.total_time_ms ? data.total_time_ms + 'ms' : 'completed'}`)
    } catch (e) {
      if (e.name === 'AbortError') return
      setComparison({ error: e.message })
      triggerToast(`Comparison Error: ${e.message}`)
    } finally {
      clearTimeout(step1Timer)
      clearTimeout(step2Timer)
      setLoading(false)
      setLoadingAction(null)
      if (requestIdRef.current === requestId) requestIdRef.current = null
    }
  }

  const showTextStage = loadingMode !== 'visual'
  const showVisualStage = loadingMode !== 'text'

  return (
    <div className="bg-surface-container-lowest text-on-surface antialiased relative min-h-screen selection:bg-primary-container selection:text-on-primary-container overflow-x-hidden text-sm">
      {/* Ambient Backdrop Glow */}
      <div className="fixed inset-0 pointer-events-none z-0 overflow-hidden">
        <div className="absolute top-1/4 left-1/2 -translate-x-1/2 w-[800px] h-[550px] ambient-glow-spin rounded-full bg-[radial-gradient(circle,rgba(160,120,255,0.12)_0%,rgba(76,215,246,0.05)_40%,transparent_75%)] blur-3xl"></div>
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_80%_60%_at_50%_-10%,rgba(160,120,255,0.12),rgba(15,19,28,0))]"></div>
      </div>

      {/* SINGLE SLEEK FULL-WIDTH COMPACT HEADER */}
      <header className="fixed top-0 left-0 right-0 z-50 bg-surface-container-lowest/90 backdrop-blur-xl border-b border-outline-variant/20 shadow-md">
        <div className="h-14 w-full px-4 sm:px-8 flex items-center justify-between gap-4">
          
          {/* Left Branding */}
          <div 
            className="flex items-center gap-2.5 cursor-pointer shrink-0"
            onClick={() => { setResult(null); setComparison(null); }}
          >
            <div className="relative flex items-center justify-center w-8 h-8 rounded-lg bg-surface-container-high shadow border border-primary/30 group hover:scale-105 transition-transform">
              <div className="absolute inset-0 rounded-lg bg-gradient-to-br from-primary/30 to-secondary/10"></div>
              <span className="relative font-display text-sm font-extrabold text-primary">S</span>
            </div>
            <div className="flex items-center gap-2">
              <span className="font-display text-base text-transparent bg-clip-text bg-gradient-to-r from-primary via-secondary to-tertiary font-extrabold tracking-wider">SUTRA</span>
              <span className="hidden sm:inline text-outline text-xs">•</span>
              <span className="hidden sm:inline text-xs text-on-surface-variant font-medium">
                Multimodal Document Intelligence RAG
              </span>
            </div>
          </div>

          {/* Center Badges & Dynamic Status */}
          <div className="hidden md:flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-surface-container-low border border-outline-variant/20">
              <span className={`w-1.5 h-1.5 rounded-full ${health?.status === 'ok' ? 'bg-tertiary pulse-dot-green' : 'bg-amber-500'}`}></span>
              <span className="text-tertiary font-medium">{health?.total_pages != null ? `${health.total_pages} Pages` : 'Index Offline'}</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-surface-container-low border border-outline-variant/20">
              <div className={`w-1.5 h-1.5 rounded-full ${health?.status === 'ok' ? 'bg-primary animate-pulse' : 'bg-outline'}`}></div>
              <span className="text-primary font-medium">{health?.configured_model ? `Groq: ${health.configured_model}` : 'Groq Ready'}</span>
            </div>
            <div className="flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-surface-container-low border border-outline-variant/20">
              <div className={`w-1.5 h-1.5 rounded-full ${health?.visual_index_loaded ? 'bg-secondary' : 'bg-outline'}`}></div>
              <span className="text-secondary font-medium">{health?.visual_index_loaded ? 'ColQwen2 VLM Active' : 'Visual Offline'}</span>
            </div>
          </div>

          <div className="flex items-center gap-1.5 shrink-0 text-xs">
            <span className={`w-1.5 h-1.5 rounded-full ${health?.llm_configured ? 'bg-tertiary' : 'bg-amber-500'}`}></span>
            <span className="text-on-surface-variant hidden sm:inline">
              {health?.llm_configured ? 'Groq Ready' : 'Groq Key Missing'}
            </span>
          </div>

        </div>
      </header>

      {/* MAIN WORKSPACE AREA */}
      <main className="w-full pt-20 pb-10 relative z-10 min-h-screen flex flex-col justify-between">
        <div className="w-full max-w-6xl mx-auto px-4 sm:px-6 space-y-4">
          
          {/* SEARCH CONSOLE (ALWAYS VISIBLE AT TOP) */}
          <section className="relative w-full max-w-4xl mx-auto stagger-1">
            <div className="relative w-full rounded-xl bg-surface-container/80 border border-outline-variant/30 backdrop-blur-xl shadow-md p-3 sm:p-3.5 space-y-2.5">
              <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2 w-full">
                <div className="relative flex-1 min-w-0 flex items-center bg-surface-container-lowest/90 rounded-lg border border-outline-variant/30 shadow-inner px-3 py-1.5 focus-within:border-primary/70 transition-all">
                  <span className="material-symbols-outlined text-outline text-[18px] mr-2 shrink-0">search</span>
                  <input 
                    ref={searchInputRef}
                    className="w-full bg-transparent text-xs sm:text-sm text-on-surface placeholder:text-outline/60 focus:outline-none"
                    placeholder="Ask a question about NIT Jamshedpur notices, fee deadlines, academic circulars..."
                    value={question}
                    onChange={e => setQuestion(e.target.value)}
                    onKeyDown={e => e.key === 'Enter' && handleQuery()}
                  />
                  <div className="flex items-center gap-1 shrink-0 ml-1">
                    {(question || result || comparison) && (
                      <button 
                        className="text-outline hover:text-on-surface p-0.5 rounded-full hover:bg-surface-container transition-colors"
                        onClick={() => {
                          setQuestion('')
                          setResult(null)
                          setComparison(null)
                          setSelectedEvidence(null)
                        }}
                        title="Clear Search & Reset View"
                      >
                        <span className="material-symbols-outlined text-[16px]">close</span>
                      </button>
                    )}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <button 
                    disabled={loading}
                    className={`flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-4 py-1.5 rounded-lg bg-primary-container text-white font-semibold text-xs shadow-sm transition-all ${loading ? 'opacity-50 cursor-not-allowed' : 'hover:brightness-110 active:scale-95'}`}
                    onClick={() => handleQuery()}
                  >
                    <span className="material-symbols-outlined text-[16px]">{loadingAction === 'search' ? 'sync' : 'manage_search'}</span>
                    <span>{loadingAction === 'search' ? 'Searching...' : 'Search'}</span>
                  </button>
                  <button 
                    disabled={loading}
                    className={`flex-1 sm:flex-none flex items-center justify-center gap-1.5 px-4 py-1.5 rounded-lg bg-gradient-to-r from-amber-600 to-amber-500 text-white font-semibold text-xs shadow-sm transition-all ${loading ? 'opacity-50 cursor-not-allowed' : 'hover:brightness-110 active:scale-95'}`}
                    onClick={() => handleCompare()}
                  >
                    <span className="material-symbols-outlined text-[16px]">{loadingAction === 'compare' ? 'sync' : 'electric_bolt'}</span>
                    <span>{loadingAction === 'compare' ? 'Comparing...' : 'Compare All'}</span>
                  </button>
                </div>
              </div>

              {/* Pipeline Selector Chips */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-1 border-t border-outline-variant/15 text-xs">
                <span className="text-[10px] text-outline uppercase tracking-wider font-mono select-none">Pipeline Mode:</span>
                <div className="flex flex-wrap items-center gap-1.5">
                  <button 
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md transition-all text-xs ${mode === 'text' ? 'bg-surface-container-high text-on-surface border border-tertiary/50 font-semibold shadow-sm' : 'bg-surface-container-low/50 text-on-surface-variant border border-outline-variant/10 hover:border-tertiary/30'}`}
                    onClick={() => { setMode('text'); triggerToast('Pipeline mode set to: TEXT BASELINE'); }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-tertiary"></span>
                    <span>Text Baseline (BM25)</span>
                  </button>
                  <button 
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md transition-all text-xs ${mode === 'visual' ? 'bg-surface-container-high text-on-surface border border-primary/50 font-semibold shadow-sm' : 'bg-surface-container-low/50 text-on-surface-variant border border-outline-variant/10 hover:border-primary-container/30'}`}
                    onClick={() => { setMode('visual'); triggerToast('Pipeline mode set to: VISUAL VLM'); }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-primary"></span>
                    <span>Visual VLM (ColQwen2)</span>
                  </button>
                  <button 
                    className={`flex items-center gap-1.5 px-2.5 py-1 rounded-md transition-all text-xs ${mode === 'hybrid_reranked' ? 'bg-surface-container-high text-on-surface border border-secondary/50 font-semibold shadow-sm' : 'bg-surface-container-low/50 text-on-surface-variant border border-outline-variant/10 hover:border-secondary/30'}`}
                    onClick={() => { setMode('hybrid_reranked'); triggerToast('Pipeline mode set to: HYBRID RERANKED'); }}
                  >
                    <span className="w-1.5 h-1.5 rounded-full bg-secondary"></span>
                    <span>Hybrid Reranked (ColQwen2 + RRF)</span>
                  </button>
                </div>
              </div>
            </div>
          </section>

          {/* LOADING / SCANNING STATE */}
          {loading ? (
            <div className="py-4 space-y-4 animate-fade-in-up">
              <section className="w-full rounded-xl bg-surface-container/70 border border-outline-variant/30 backdrop-blur-xl p-4 space-y-3">
                <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-2.5">
                  <div className="flex-1 min-w-0 flex items-center bg-surface-container-lowest rounded-lg px-3 py-1.5 shadow-inner border border-secondary/30">
                    <span className="material-symbols-outlined text-secondary mr-2 text-[18px] animate-pulse">saved_search</span>
                    <input 
                      className="w-full bg-transparent text-on-surface text-sm focus:outline-none cursor-not-allowed truncate font-medium"
                      readOnly 
                      value={question} 
                    />
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    <button onClick={stopActiveRequest} className="flex items-center gap-1.5 px-4 py-1.5 rounded-lg bg-error-container/30 text-error font-semibold text-xs border border-error/40 shadow-sm hover:brightness-110">
                      <span className="material-symbols-outlined text-[16px]">stop_circle</span>
                      <span>Stop</span>
                    </button>
                  </div>
                </div>
              </section>

              <div className="relative w-full rounded-xl bg-surface-container-low/90 border border-outline-variant/30 backdrop-blur-xl p-6 space-y-6">
                <div className="relative z-10 flex flex-col items-center text-center">
                  <div className="relative flex items-center justify-center w-28 h-28 mb-4">
                    <div className="absolute inset-0 rounded-full border border-primary/25 border-dashed animate-spin-slow"></div>
                    <div className="absolute inset-1 rounded-full bg-[conic-gradient(from_0deg,transparent_0_300deg,rgba(76,215,246,0.35)_355deg,rgba(160,120,255,0.7)_360deg)] animate-spin-slow pointer-events-none"></div>
                    <div className="relative flex items-center justify-center w-16 h-16 rounded-full bg-surface-container-high/90 border border-primary/40 backdrop-blur-2xl shadow animate-float-hologram">
                      <span className="material-symbols-outlined text-[28px] text-primary">document_scanner</span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-center justify-center mb-1">
                    <span className="material-symbols-outlined text-secondary animate-spin text-[18px]">sync</span>
                    <h2 className="text-lg text-on-surface font-extrabold tracking-tight">
                      Synthesizing grounded evidence<span className="animate-ellipsis text-secondary">...</span>
                    </h2>
                  </div>
                  <p className="max-w-md text-xs text-on-surface-variant">
                    Evaluating tri-mode RAG retrieval across text tokens and multi-vector document image patches indexed from NIT Jamshedpur Corpus.
                  </p>
                </div>

                <div className="relative z-10 w-full grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                  {showTextStage && <div className={`flex flex-col rounded-lg p-3 border transition-all ${loadingStep >= 1 ? 'bg-surface-container/90 border-tertiary/40' : 'bg-surface-container/40 border-outline-variant/20'}`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-tertiary font-bold">STAGE 01</span>
                      <span className="material-symbols-outlined text-tertiary text-[16px]">check_circle</span>
                    </div>
                    <h3 className="text-sm text-on-surface font-bold">Text Chunking</h3>
                    <p className="text-xs text-on-surface-variant mt-0.5">BM25 + all-MiniLM-L6-v2 dense vector search.</p>
                  </div>}
                  {showVisualStage && <div className={`flex flex-col rounded-lg p-3 border transition-all ${loadingStep >= 2 ? 'bg-surface-container/90 border-primary/40' : 'bg-surface-container/40 border-outline-variant/20'}`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-primary font-bold">STAGE 02</span>
                      <span className={`material-symbols-outlined text-[16px] ${loadingStep >= 2 ? 'text-primary animate-spin' : 'text-outline'}`}>
                        {loadingStep >= 2 ? 'sync' : 'hourglass_empty'}
                      </span>
                    </div>
                    <h3 className="text-sm text-on-surface font-bold">ColQwen2 Visual Patching</h3>
                    <p className="text-xs text-on-surface-variant mt-0.5">Multi-vector MaxSim interaction.</p>
                  </div>}
                  <div className={`flex flex-col rounded-lg p-3 border transition-all ${loadingStep >= 3 ? 'bg-surface-container/90 border-secondary/40' : 'bg-surface-container/40 border-outline-variant/20'}`}>
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-[10px] text-secondary font-bold">STAGE 03</span>
                      <span className={`material-symbols-outlined text-[16px] ${loadingStep >= 3 ? 'text-secondary animate-pulse' : 'text-outline'}`}>
                        {loadingStep >= 3 ? 'auto_awesome' : 'schedule'}
                      </span>
                    </div>
                    <h3 className="text-sm text-on-surface font-bold">Groq Answer Synthesis</h3>
                    <p className="text-xs text-on-surface-variant mt-0.5">Grounded answer generation via {health?.configured_model || 'Groq Qwen'}.</p>
                  </div>
                </div>
              </div>
            </div>
          ) : comparison ? (
            /* MULTI-MODE COMPARISON RESULTS VIEW */
            <div className="py-2 space-y-4 animate-fade-in-up">
              {comparison.error ? (
                <div className="p-4 rounded-xl bg-error-container/20 border border-error/40 text-error text-xs flex items-center gap-3">
                  <span className="material-symbols-outlined text-[24px]">warning</span>
                  <div>
                    <h3 className="font-bold text-sm">Multi-Mode Comparison Failed</h3>
                    <p className="text-on-surface-variant text-xs mt-0.5">{comparison.error}</p>
                  </div>
                </div>
              ) : (
                <>
                  {/* Benchmark Telemetry Banner */}
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3 rounded-xl bg-surface-container/90 border border-outline-variant/30 shadow-sm">
                    <div className="flex items-center gap-2.5 min-w-0">
                      <div className="w-8 h-8 rounded-lg bg-surface-container-high border border-outline-variant/30 flex items-center justify-center text-primary shrink-0">
                        <span className="material-symbols-outlined text-[18px]">view_column</span>
                      </div>
                      <div className="min-w-0">
                        <div className="flex items-center gap-2">
                          <h1 className="text-sm text-on-surface font-bold tracking-tight shrink-0">Multi-Mode Retrieval Evaluation</h1>
                          <span className="text-outline text-xs">•</span>
                          <span className="text-xs text-secondary font-medium truncate max-w-xs sm:max-w-md">"{comparison.question}"</span>
                        </div>
                      </div>
                    </div>
                    {comparison.total_time_ms != null && (
                      <div className="flex items-center gap-2 shrink-0 text-xs">
                        <div className="px-2.5 py-0.5 rounded-md bg-surface-container-low border border-outline-variant/30 flex items-center gap-1.5">
                          <span className="w-1.5 h-1.5 rounded-full bg-tertiary"></span>
                          <span className="text-outline">Total Latency:</span>
                          <span className="text-primary font-bold">{comparison.total_time_ms}ms</span>
                        </div>
                      </div>
                    )}
                  </div>

              {/* Multi-Column Evaluation Grid */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {/* COLUMN 1: TEXT BASELINE */}
                {(() => {
                  const modeData = comparison.modes?.text_baseline || comparison.modes?.text || {}
                  const topPage = modeData.retrieved_pages?.[0]
                  const modeTotalMs = modeData.retrieval_time_ms != null && modeData.generation_time_ms != null ? (modeData.retrieval_time_ms + modeData.generation_time_ms).toFixed(1) : null
                  return (
                    <div className="flex flex-col bg-surface-container-low/95 rounded-xl border border-outline-variant/30 shadow-md overflow-hidden">
                      <div className="h-1 w-full bg-tertiary"></div>
                      <div className="p-3.5 flex-1 flex flex-col justify-between space-y-3">
                        <div className="space-y-0.5">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-tertiary"></span>
                              <span className="text-sm text-on-surface font-bold">Text Baseline</span>
                            </div>
                            {modeTotalMs != null && (
                              <div className="flex flex-col items-end">
                                <span className="px-1.5 py-0.5 rounded bg-tertiary-container/30 text-tertiary text-[10px] font-bold">
                                  {modeTotalMs}ms
                                </span>
                                <span className="text-[9px] text-outline mt-0.5 font-mono">Retr: {modeData.retrieval_time_ms}ms</span>
                              </div>
                            )}
                          </div>
                          <p className="text-xs text-outline">BM25 + all-MiniLM-L6-v2 vector search.</p>
                        </div>
                        
                        <div className="p-2.5 rounded-lg bg-surface-container border border-outline-variant/20 space-y-0.5">
                          <div className="flex items-center justify-between text-[10px] text-outline">
                            <span>SYNTHESIZED ANSWER</span>
                            <span className="text-tertiary">Text Match</span>
                          </div>
                          <div className="text-xs text-on-surface leading-relaxed">
                            {modeData.answer || "No text answer returned."}
                          </div>
                        </div>

                        {topPage && (
                          <div className="px-2.5 py-1 rounded-lg bg-surface-container-highest border border-outline-variant/20 flex items-center justify-between text-xs">
                            <span className="text-[10px] text-outline">PRIMARY SOURCE:</span>
                            <span className="text-on-surface font-semibold truncate">
                              {topPage.notice_id} p.{topPage.page_number || topPage.page || 1}
                            </span>
                          </div>
                        )}

                        <div className="space-y-1 text-xs">
                          <span className="text-[10px] uppercase text-outline">Top Evidence</span>
                          {modeData.retrieved_pages?.slice(0, 3).map((page) => (
                            <div key={`${page.notice_id}_p${page.page_number || page.page || 1}`} className="p-1.5 rounded-lg bg-surface-container border border-outline-variant/10 flex items-center justify-between">
                              <span className="text-tertiary font-bold truncate">#{page.page_number || page.page || 1} {page.notice_id}</span>
                              <span className="text-outline text-[10px] shrink-0 ml-1">Score: {page.score != null ? Number(page.score).toFixed(4) : '-'}</span>
                            </div>
                          ))}
                        </div>

                        <div 
                          className="relative w-full aspect-[4/3] rounded-lg overflow-hidden bg-surface-container-lowest border border-outline-variant/20 flex items-center justify-center cursor-pointer group hover:border-tertiary transition-all"
                          onClick={() => topPage?.image_url && setLightboxImage({ url: topPage.image_url, title: `${topPage.notice_id} - Page ${topPage.page_number || topPage.page || 1}` })}
                        >
                          {topPage?.image_url ? (
                            <>
                              <img src={topPage.image_url} alt="Text Evidence" className="w-full h-full object-contain group-hover:scale-[1.02] transition-transform duration-200" onError={(e) => { e.target.style.display = 'none' }} />
                              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center pointer-events-none">
                                <span className="material-symbols-outlined text-white opacity-0 group-hover:opacity-100 text-2xl transition-opacity drop-shadow-md">zoom_in</span>
                              </div>
                            </>
                          ) : (
                            <div className="p-2 text-center text-xs text-outline">{topPage ? `${topPage.notice_id} p.${topPage.page_number || 1}` : 'No Image'}</div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })()}

                {/* COLUMN 2: VISUAL VLM (ColQwen2) */}
                {(() => {
                  const modeData = comparison.modes?.visual || comparison.modes?.visual_vlm || {}
                  const topPage = modeData.retrieved_pages?.[0]
                  const modeTotalMs = modeData.retrieval_time_ms != null && modeData.generation_time_ms != null ? (modeData.retrieval_time_ms + modeData.generation_time_ms).toFixed(1) : null
                  return (
                    <div className="flex flex-col bg-surface-container-low/95 rounded-xl border border-primary/40 shadow-md overflow-hidden">
                      <div className="h-1 w-full bg-primary-container"></div>
                      <div className="p-3.5 flex-1 flex flex-col justify-between space-y-3">
                        <div className="space-y-0.5">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                              <span className="text-sm text-on-surface font-bold">Visual VLM (ColQwen2)</span>
                            </div>
                            {modeTotalMs != null && (
                              <div className="flex flex-col items-end">
                                <span className="px-1.5 py-0.5 rounded bg-primary-container text-on-primary-container text-[10px] font-bold">
                                  {modeTotalMs}ms
                                </span>
                                <span className="text-[9px] text-outline mt-0.5 font-mono">Retr: {modeData.retrieval_time_ms}ms</span>
                              </div>
                            )}
                          </div>
                          <p className="text-xs text-outline">Multi-vector vision embeddings over raw page images.</p>
                        </div>

                        <div className="p-2.5 rounded-lg bg-surface-container border border-primary/30 space-y-0.5">
                          <div className="flex items-center justify-between text-[10px] text-outline">
                            <span>SYNTHESIZED ANSWER</span>
                            <span className="text-primary">Visual Grounding</span>
                          </div>
                          <div className="text-xs text-on-surface leading-relaxed">
                            {modeData.answer || "No visual answer returned."}
                          </div>
                        </div>

                        {topPage && (
                          <div className="px-2.5 py-1 rounded-lg bg-surface-container-highest border border-primary/30 flex items-center justify-between text-xs">
                            <span className="text-[10px] text-outline">TARGET NOTICE:</span>
                            <span className="text-primary font-semibold truncate">
                              {topPage.notice_id} p.{topPage.page_number || topPage.page || 1}
                            </span>
                          </div>
                        )}

                        <div className="space-y-1 text-xs">
                          <span className="text-[10px] uppercase text-outline">MaxSim Visual Alignment</span>
                          {modeData.retrieved_pages?.slice(0, 3).map((page) => (
                            <div key={`${page.notice_id}_p${page.page_number || page.page || 1}`} className="p-1.5 rounded-lg bg-surface-container border border-primary/30 flex items-center justify-between">
                              <span className="text-primary font-bold truncate">#{page.page_number || page.page || 1} {page.notice_id}</span>
                              <span className="text-primary font-bold text-[10px] shrink-0 ml-1">{page.score != null ? Number(page.score).toFixed(4) : '-'}</span>
                            </div>
                          ))}
                        </div>

                        <div 
                          className="relative w-full aspect-[4/3] rounded-lg overflow-hidden bg-surface-container-lowest border border-primary/30 flex items-center justify-center cursor-pointer group hover:border-primary transition-all"
                          onClick={() => topPage?.image_url && setLightboxImage({ url: topPage.image_url, title: `${topPage.notice_id} - Page ${topPage.page_number || topPage.page || 1}` })}
                        >
                          <div className="laser-scan text-primary pointer-events-none z-20"></div>
                          {topPage?.image_url ? (
                            <>
                              <img src={topPage.image_url} alt="Visual Evidence" className="w-full h-full object-contain group-hover:scale-[1.02] transition-transform duration-200" onError={(e) => { e.target.style.display = 'none' }} />
                              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center pointer-events-none z-30">
                                <span className="material-symbols-outlined text-white opacity-0 group-hover:opacity-100 text-2xl transition-opacity drop-shadow-md">zoom_in</span>
                              </div>
                            </>
                          ) : (
                            <div className="p-2 text-center text-xs text-primary">{topPage ? `${topPage.notice_id} p.${topPage.page_number || 1}` : 'No Image'}</div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })()}

                {/* COLUMN 3: HYBRID RERANKED */}
                {(() => {
                  const modeData = comparison.modes?.hybrid_reranked || comparison.modes?.hybrid_rrf_baseline || comparison.modes?.hybrid || {}
                  const topPage = modeData.retrieved_pages?.[0]
                  const modeTotalMs = modeData.retrieval_time_ms != null && modeData.generation_time_ms != null ? (modeData.retrieval_time_ms + modeData.generation_time_ms).toFixed(1) : null
                  return (
                    <div className="flex flex-col bg-surface-container-low/95 rounded-xl border border-outline-variant/30 shadow-md overflow-hidden">
                      <div className="h-1 w-full bg-secondary"></div>
                      <div className="p-3.5 flex-1 flex flex-col justify-between space-y-3">
                        <div className="space-y-0.5">
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-1.5">
                              <span className="w-2 h-2 rounded-full bg-secondary"></span>
                              <span className="text-sm text-on-surface font-bold">Hybrid Reranked RAG</span>
                            </div>
                            {modeTotalMs != null && (
                              <div className="flex flex-col items-end">
                                <span className="px-1.5 py-0.5 rounded bg-secondary-container/40 text-secondary text-[10px] font-bold">
                                  {modeTotalMs}ms
                                </span>
                                <span className="text-[9px] text-outline mt-0.5 font-mono">Retr: {modeData.retrieval_time_ms}ms</span>
                              </div>
                            )}
                          </div>
                          <p className="text-xs text-outline">Candidate union RRF + 216 DPI multi-scale visual reranking.</p>
                        </div>

                        <div className="p-2.5 rounded-lg bg-surface-container border border-secondary/25 space-y-0.5">
                          <div className="flex items-center justify-between text-[10px] text-outline">
                            <span>SYNTHESIZED ANSWER</span>
                            <span className="text-secondary">Verified Consensus</span>
                          </div>
                          <div className="text-xs text-on-surface leading-relaxed">
                            {modeData.answer || "No hybrid answer returned."}
                          </div>
                        </div>

                        {topPage && (
                          <div className="px-2.5 py-1 rounded-lg bg-surface-container-highest border border-secondary/25 flex items-center justify-between text-xs">
                            <span className="text-[10px] text-outline">PRIMARY RETRIEVAL:</span>
                            <span className="text-secondary font-semibold truncate">
                              {topPage.notice_id} p.{topPage.page_number || topPage.page || 1}
                            </span>
                          </div>
                        )}

                        <div className="space-y-1 text-xs">
                          <span className="text-[10px] uppercase text-outline">Candidate Fusion</span>
                          {modeData.retrieved_pages?.slice(0, 3).map((page) => (
                            <div key={`${page.notice_id}_p${page.page_number || page.page || 1}`} className="p-1.5 rounded-lg bg-surface-container border border-secondary/30 flex items-center justify-between">
                              <span className="text-secondary font-bold truncate">#{page.page_number || page.page || 1} {page.notice_id}</span>
                              <span className="text-secondary font-bold text-[10px] shrink-0 ml-1">{page.score != null ? Number(page.score).toFixed(4) : '-'}</span>
                            </div>
                          ))}
                        </div>

                        <div 
                          className="relative w-full aspect-[4/3] rounded-lg overflow-hidden bg-surface-container-lowest border border-secondary/30 flex items-center justify-center cursor-pointer group hover:border-secondary transition-all"
                          onClick={() => topPage?.image_url && setLightboxImage({ url: topPage.image_url, title: `${topPage.notice_id} - Page ${topPage.page_number || topPage.page || 1}` })}
                        >
                          <div className="laser-scan text-secondary pointer-events-none z-20"></div>
                          {topPage?.image_url ? (
                            <>
                              <img src={topPage.image_url} alt="Hybrid Evidence" className="w-full h-full object-contain group-hover:scale-[1.02] transition-transform duration-200" onError={(e) => { e.target.style.display = 'none' }} />
                              <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 transition-colors flex items-center justify-center pointer-events-none z-30">
                                <span className="material-symbols-outlined text-white opacity-0 group-hover:opacity-100 text-2xl transition-opacity drop-shadow-md">zoom_in</span>
                              </div>
                            </>
                          ) : (
                            <div className="p-2 text-center text-xs text-secondary">{topPage ? `${topPage.notice_id} p.${topPage.page_number || 1}` : 'No Image'}</div>
                          )}
                        </div>
                      </div>
                    </div>
                  )
                })()}
              </div>

              {/* Diagnostic Callout */}
              <div className="p-3 rounded-xl bg-surface-container-low/90 border border-outline-variant/30 shadow-sm text-xs">
                <div className="flex items-center gap-2">
                  <span className="material-symbols-outlined text-[18px] text-primary shrink-0">insights</span>
                  <p className="text-on-surface-variant">
                    The <strong className="text-tertiary font-medium">Text Baseline</strong> relies on keywords. The <strong className="text-primary font-medium">Visual VLM (ColQwen2)</strong> and <strong className="text-secondary font-medium">Hybrid Reranked RAG</strong> inspect spatial layout, stamps, and signature blocks.
                  </p>
                </div>
              </div>
                </>
              )}
            </div>
          ) : result ? (
            /* SINGLE MODE SEARCH RESULT VIEW (2-COLUMN SPLIT DASHBOARD WITH INTERACTIVE SIDE VIEWER) */
            <div className="w-full space-y-4 animate-fade-in-up">
              {(() => {
                const activeDoc = selectedEvidence || result.retrieved_pages?.[0]
                return (
                  <>
                    {/* SINGLE SEARCH TELEMETRY BANNER */}
                    <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-3.5 rounded-xl bg-surface-container/90 border border-outline-variant/30 backdrop-blur-xl shadow-md">
                      <div className="flex items-center gap-2.5 min-w-0">
                        <div className="w-8 h-8 rounded-lg bg-surface-container-high border border-outline-variant/30 flex items-center justify-center text-primary shrink-0">
                          <span className="material-symbols-outlined text-[20px]">auto_awesome</span>
                        </div>
                        <div className="min-w-0">
                          <div className="flex items-center gap-2 flex-wrap">
                            <h2 className="text-sm text-on-surface font-extrabold tracking-tight">Single Search Result</h2>
                            <span className="px-2 py-0.5 rounded-full bg-secondary-container/40 text-secondary border border-secondary/30 text-[10px] font-bold uppercase font-mono">
                              {mode.replace('_', ' ')}
                            </span>
                            <span className="text-outline text-xs">•</span>
                            <span className="text-xs text-on-surface-variant font-medium truncate max-w-md">"{question}"</span>
                          </div>
                        </div>
                      </div>
                      {result.total_time_ms != null && (
                        <div className="flex items-center gap-2 shrink-0 text-xs">
                          <div className="px-2.5 py-1 rounded-lg bg-surface-container-low border border-outline-variant/30 flex items-center gap-1.5 font-mono">
                            <span className="w-2 h-2 rounded-full bg-tertiary"></span>
                            <span className="text-outline text-[11px]">Latency:</span>
                            <span className="text-primary font-bold">{result.total_time_ms}ms</span>
                          </div>
                        </div>
                      )}
                    </div>

                    {/* 2-COLUMN SPLIT DASHBOARD */}
                    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5 items-start">
                      
                      {/* COLUMN 1 (LEFT ~7 COLS): INTERACTIVE DOCUMENT VIEWER */}
                      <div className="lg:col-span-7 flex flex-col gap-3 rounded-2xl bg-surface-container-low/95 border border-primary/30 p-4 shadow-xl backdrop-blur-md">
                        
                        {/* Viewer Header */}
                        <div className="flex items-center justify-between gap-2 pb-2.5 border-b border-outline-variant/20">
                          <div className="flex items-center gap-2 min-w-0">
                            <span className="material-symbols-outlined text-primary text-[20px]">description</span>
                            <div className="min-w-0">
                              <h3 className="text-xs sm:text-sm font-bold text-on-surface truncate">
                                {activeDoc?.notice_id || 'Document Evidence'}
                              </h3>
                              <p className="text-[10px] text-outline">
                                Page {activeDoc?.page_number || activeDoc?.page || 1} • {activeDoc?.notice_id}.pdf
                              </p>
                            </div>
                          </div>
                          {activeDoc?.score != null && (
                            <span className="px-2 py-0.5 rounded bg-primary-container/30 border border-primary/40 text-primary text-[10px] font-bold font-mono">
                              Match Score: {Number(activeDoc.score).toFixed(4)}
                            </span>
                          )}
                        </div>

                        {/* Interactive Controls Toolbar */}
                        <div className="flex flex-wrap items-center justify-between gap-2 p-1.5 rounded-lg bg-surface-container-lowest/80 border border-outline-variant/20 text-xs">
                          
                          {/* Zoom Buttons */}
                          <div className="flex items-center gap-1">
                            <button 
                              className="p-1 rounded hover:bg-surface-container-high text-on-surface active:scale-95 transition-all"
                              onClick={zoomOutInline}
                              title="Zoom Out (-)"
                            >
                              <span className="material-symbols-outlined text-[18px]">zoom_out</span>
                            </button>
                            <span className="text-[10px] font-mono text-outline px-1.5 py-0.5 rounded bg-surface-container/60 min-w-[42px] text-center">
                              {Math.round(inlineZoom * 100)}%
                            </span>
                            <button 
                              className="p-1 rounded hover:bg-surface-container-high text-on-surface active:scale-95 transition-all"
                              onClick={zoomInInline}
                              title="Zoom In (+)"
                            >
                              <span className="material-symbols-outlined text-[18px]">zoom_in</span>
                            </button>
                          </div>

                          {/* Rotate Buttons */}
                          <div className="flex items-center gap-1">
                            <button 
                              className="flex items-center gap-1 px-2 py-1 rounded bg-surface-container-high hover:bg-primary/20 hover:text-primary text-on-surface text-xs font-semibold border border-outline-variant/20 active:scale-95 transition-all shadow-sm"
                              onClick={rotateAnticlockwiseInline}
                              title="Rotate Anticlockwise 90° (Left)"
                            >
                              <span className="material-symbols-outlined text-[16px]">rotate_left</span>
                              <span className="hidden sm:inline">Rotate Left (-90°)</span>
                            </button>

                            <button 
                              className="flex items-center gap-1 px-2 py-1 rounded bg-surface-container-high hover:bg-primary/20 hover:text-primary text-on-surface text-xs font-semibold border border-outline-variant/20 active:scale-95 transition-all shadow-sm"
                              onClick={rotateClockwiseInline}
                              title="Rotate Clockwise 90° (Right)"
                            >
                              <span className="material-symbols-outlined text-[16px]">rotate_right</span>
                              <span className="hidden sm:inline">Rotate Right (+90°)</span>
                            </button>

                            <span className="text-[10px] font-mono text-outline px-1.5 py-0.5 rounded bg-surface-container/60">
                              {((inlineRotation % 360) + 360) % 360}°
                            </span>
                          </div>

                          {/* Reset & Fullscreen */}
                          <div className="flex items-center gap-1">
                            {(inlineZoom !== 1.0 || inlineRotation !== 0) && (
                              <button 
                                className="px-2 py-1 rounded bg-tertiary-container/30 hover:bg-tertiary-container/60 text-tertiary text-[11px] font-semibold border border-tertiary/30 active:scale-95 transition-all"
                                onClick={resetInlineControls}
                                title="Reset View"
                              >
                                Reset
                              </button>
                            )}

                            {activeDoc?.image_url && (
                              <button 
                                className="flex items-center gap-1 px-2.5 py-1 rounded bg-primary-container text-white text-xs font-semibold shadow-sm hover:brightness-110 active:scale-95 transition-all"
                                onClick={() => setLightboxImage({ url: activeDoc.image_url, title: `${activeDoc.notice_id} - Page ${activeDoc.page_number || 1}` })}
                                title="Open Fullscreen Lightbox Modal"
                              >
                                <span className="material-symbols-outlined text-[16px]">fullscreen</span>
                                <span>Fullscreen</span>
                              </button>
                            )}
                          </div>
                        </div>

                        {/* Dynamic Image Viewport Container */}
                        <div className="relative w-full h-[460px] rounded-xl overflow-auto bg-surface-container-lowest/90 border border-outline-variant/30 flex items-center justify-center p-3 shadow-inner select-none">
                          {activeDoc?.image_url ? (
                            <img 
                              src={activeDoc.image_url} 
                              alt="Document Evidence" 
                              style={{
                                transform: `scale(${inlineZoom}) rotate(${inlineRotation}deg)`,
                                transition: 'transform 0.25s cubic-bezier(0.4, 0, 0.2, 1)',
                                maxHeight: Math.abs(inlineRotation / 90) % 2 === 1 ? `${380 / inlineZoom}px` : '100%',
                                maxWidth: Math.abs(inlineRotation / 90) % 2 === 1 ? `${380 / inlineZoom}px` : '100%'
                              }}
                              className="object-contain rounded shadow-lg max-h-full max-w-full"
                              onError={(e) => { e.target.style.display = 'none' }}
                            />
                          ) : (
                            <div className="text-xs text-outline text-center p-4">No page image available</div>
                          )}
                        </div>

                        {/* Thumbnail Carousel Selector */}
                        <div className="space-y-1.5 pt-1">
                          <div className="text-[10px] text-outline font-semibold uppercase tracking-wider">
                            All Retrieved Evidence Pages ({result.retrieved_pages?.length || 0})
                          </div>
                          <div className="grid grid-cols-3 sm:grid-cols-5 gap-2">
                            {result.retrieved_pages?.map((page, idx) => {
                              const isSelected = (selectedEvidence?.notice_id === page.notice_id && (selectedEvidence?.page_number || selectedEvidence?.page) === (page.page_number || page.page)) || (!selectedEvidence && idx === 0)
                              return (
                                <button
                                  key={`${page.notice_id}_p${page.page_number || page.page || idx}`}
                                  className={`p-1.5 rounded-lg border text-left transition-all flex flex-col gap-1 ${isSelected ? 'bg-surface-container-high border-primary ring-1 ring-primary shadow-sm' : 'bg-surface-container-lowest/80 border-outline-variant/20 hover:border-primary/40'}`}
                                  onClick={() => {
                                    setSelectedEvidence(page)
                                    setInlineZoom(1.0)
                                    setInlineRotation(0)
                                  }}
                                >
                                  <div className="flex items-center justify-between text-[10px] font-bold truncate">
                                    <span className={isSelected ? 'text-primary' : 'text-on-surface'}>p.{page.page_number || page.page || 1}</span>
                                    <span className="text-outline text-[9px]">{page.score != null ? Number(page.score).toFixed(2) : '-'}</span>
                                  </div>
                                  {page.image_url ? (
                                    <img src={page.image_url} alt="Thumbnail" className="w-full h-14 object-contain rounded bg-surface-container-lowest" onError={(e) => { e.target.style.display = 'none' }} />
                                  ) : (
                                    <div className="w-full h-14 flex items-center justify-center text-[9px] text-outline bg-surface-container">No Img</div>
                                  )}
                                </button>
                              )
                            })}
                          </div>
                        </div>
                      </div>

                      {/* COLUMN 2 (RIGHT ~5 COLS): SYNTHESIZED ANSWER & METRICS */}
                      <div className="lg:col-span-5 flex flex-col gap-4">
                        
                        {/* Synthesized Answer Card */}
                        <div className="rounded-2xl bg-surface-container/90 border border-secondary/30 p-5 shadow-xl backdrop-blur-md space-y-3">
                          <div className="flex items-center justify-between pb-2 border-b border-outline-variant/20">
                            <div className="flex items-center gap-2">
                              <span className="material-symbols-outlined text-secondary text-[20px]">psychology</span>
                              <h3 className="text-sm font-extrabold text-on-surface tracking-tight">Synthesized Grounded Answer</h3>
                            </div>
                            <button 
                              className="flex items-center gap-1 px-2.5 py-1 rounded bg-surface-container-high hover:bg-surface-bright text-on-surface text-xs font-semibold border border-outline-variant/20 active:scale-95 transition-all"
                              onClick={() => {
                                navigator.clipboard.writeText(result.answer)
                                triggerToast('Answer copied to clipboard!')
                              }}
                              title="Copy Answer"
                            >
                              <span className="material-symbols-outlined text-[14px]">content_copy</span>
                              <span>Copy</span>
                            </button>
                          </div>

                          <div className="p-3.5 rounded-xl bg-surface-container-lowest/90 border border-outline-variant/30 space-y-1.5 shadow-inner">
                            <div className="text-[10px] text-secondary font-mono font-bold tracking-wider uppercase">ANSWER SYNTHESIS</div>
                            <div className="text-sm text-on-surface font-medium leading-relaxed whitespace-pre-wrap">
                              {result.answer || "No grounded answer generated."}
                            </div>
                          </div>

                          {/* Model info banner */}
                          <div className="flex items-center justify-between text-[11px] px-3 py-1.5 rounded-lg bg-surface-container-low border border-outline-variant/20 text-on-surface-variant">
                            <div className="flex items-center gap-1.5">
                              <span className="w-1.5 h-1.5 rounded-full bg-secondary"></span>
                              <span>Model: <strong className="text-on-surface">{health?.configured_model || 'qwen/qwen3.8-27b'}</strong></span>
                            </div>
                            <span>Grounding: <strong className="text-primary">Full Image</strong></span>
                          </div>
                        </div>

                        {/* Evidence Sources Breakdown List */}
                        <div className="rounded-2xl bg-surface-container/70 border border-outline-variant/25 p-4 shadow-md space-y-2.5">
                          <div className="flex items-center justify-between">
                            <h4 className="text-xs font-bold text-on-surface uppercase tracking-wider">Top Evidence Sources</h4>
                            <span className="text-[10px] text-outline font-mono">Ranked by MaxSim Score</span>
                          </div>

                          <div className="space-y-2">
                            {result.retrieved_pages?.map((page, idx) => {
                              const isSelected = (selectedEvidence?.notice_id === page.notice_id && (selectedEvidence?.page_number || selectedEvidence?.page) === (page.page_number || page.page)) || (!selectedEvidence && idx === 0)
                              return (
                                <div
                                  key={`src_${page.notice_id}_p${page.page_number || page.page || idx}`}
                                  className={`p-2.5 rounded-xl border transition-all cursor-pointer flex items-center justify-between gap-3 ${isSelected ? 'bg-surface-container-high border-primary shadow-sm' : 'bg-surface-container-low/60 border-outline-variant/15 hover:border-primary/30'}`}
                                  onClick={() => {
                                    setSelectedEvidence(page)
                                    setInlineZoom(1.0)
                                    setInlineRotation(0)
                                  }}
                                >
                                  <div className="flex items-center gap-2.5 min-w-0">
                                    <span className={`w-5 h-5 rounded-md flex items-center justify-center text-[10px] font-bold ${idx === 0 ? 'bg-primary-container text-white' : 'bg-surface-container-high text-on-surface-variant'}`}>
                                      #{idx + 1}
                                    </span>
                                    <div className="min-w-0">
                                      <div className="text-xs font-bold text-on-surface truncate">{page.notice_id}</div>
                                      <div className="text-[10px] text-outline">Page {page.page_number || page.page || 1} • {page.filename || `${page.notice_id}.pdf`}</div>
                                    </div>
                                  </div>

                                  {page.score != null && (
                                    <div className="flex flex-col items-end shrink-0">
                                      <span className="text-xs font-bold font-mono text-primary">{Number(page.score).toFixed(4)}</span>
                                      <span className="text-[9px] text-outline">similarity</span>
                                    </div>
                                  )}
                                </div>
                              )
                            })}
                          </div>
                        </div>

                        {/* Diagnostic Insight Callout */}
                        <div className="p-3 rounded-xl bg-surface-container-low/90 border border-outline-variant/20 text-xs flex items-center gap-2.5">
                          <span className="material-symbols-outlined text-primary text-[20px] shrink-0">insights</span>
                          <p className="text-on-surface-variant text-[11px] leading-snug">
                            Click any evidence thumbnail to display its full page document on the left viewer. Use <strong className="text-primary font-bold">Zoom</strong> and <strong className="text-primary font-bold">Rotate Left (-90°)</strong> controls to inspect timetable grids.
                          </p>
                        </div>

                      </div>
                    </div>
                  </>
                )
              })()}
            </div>
          ) : (
            /* LANDING CONTENT (HERO & CORPUS REPOSITORY) */
            <div className="w-full flex flex-col gap-5">
              <section className="flex flex-col items-center justify-center text-center py-1 stagger-2">
                <div className="max-w-xl flex flex-col items-center gap-1">
                  <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full bg-surface-container-high border border-primary/20 text-primary text-[10px] font-mono">
                    <span className="w-1.5 h-1.5 rounded-full bg-primary animate-pulse"></span>
                    Multimodal Document Intelligence RAG
                  </div>
                  <h1 className="text-base sm:text-lg text-on-surface font-extrabold tracking-tight">
                    Query Campus Intelligence &amp; Official Circulars
                  </h1>
                  <p className="text-xs text-on-surface-variant max-w-md">
                    Zero-shot QA across scanned administrative circulars, official signatures, and hostel allocations.
                  </p>
                </div>

                {/* Sample Prompt Chips */}
                <div className="mt-3 w-full max-w-2xl flex flex-col items-center stagger-3">
                  <div className="flex flex-wrap items-center justify-center gap-1.5">
                    {[
                      { icon: 'schedule', text: 'What is the last date for fee payment?' },
                      { icon: 'groups', text: 'Who is the President of the Student Council?' },
                      { icon: 'payments', text: 'What is the fellowship amount for JRF candidates?' },
                      { icon: 'draw', text: 'Who signed the student council notice?' }
                    ].map((p) => (
                      <button
                        key={p.text}
                        className="group flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-surface-container/50 border border-outline-variant/20 hover:border-primary/40 hover:bg-surface-container-high transition-all text-xs"
                        onClick={() => {
                          setQuestion(p.text)
                          if (searchInputRef.current) searchInputRef.current.focus()
                          triggerToast('Benchmark prompt selected!')
                        }}
                      >
                        <span className="material-symbols-outlined text-primary text-[14px]">{p.icon}</span>
                        <span className="text-on-surface-variant group-hover:text-on-surface font-medium">{p.text}</span>
                      </button>
                    ))}
                  </div>
                </div>
              </section>

              {/* BOTTOM SECTION: Indexed Corpus Repository Showcase */}
              <section className="flex flex-col gap-2.5 pb-4 stagger-4">
                <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 bg-surface-container/40 backdrop-blur-md px-3.5 py-2 rounded-xl border border-outline-variant/20 text-xs">
                  <div className="flex items-center gap-2.5">
                    <div className="w-6 h-6 rounded-md bg-surface-container-high flex items-center justify-center text-primary border border-primary/20 shrink-0">
                      <span className="material-symbols-outlined text-[16px]">folder_copy</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <h2 className="text-xs text-on-surface font-bold">Indexed Corpus Repository</h2>
                      <span className="px-1.5 py-0.2 rounded-full bg-primary-container/20 border border-primary/30 text-primary text-[9px] font-semibold">
                        {documents.length > 0 ? `${documents.length} Documents` : 'Corpus Repository'}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 text-[10px]">
                    <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-surface-container-low border border-outline-variant/15 text-on-surface-variant">
                      <span className="material-symbols-outlined text-secondary text-[13px]">layers</span>
                      <span>{health?.total_pages != null ? `${health.total_pages} Pages` : '25 Pages'}</span>
                    </div>
                    <div className="flex items-center gap-1 px-2 py-0.5 rounded bg-surface-container-low border border-tertiary/30 text-tertiary font-semibold">
                      <span className="w-1.5 h-1.5 rounded-full bg-tertiary"></span>
                      <span>Multi-Vector Index</span>
                    </div>
                  </div>
                </div>

                {/* Dynamic Document Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-2.5 stagger-5">
                  {documents.map((doc) => (
                    <div
                      key={doc.notice_id}
                      className="doc-card group relative flex flex-col justify-between p-3 rounded-xl bg-surface-container/40 border border-outline-variant/20 hover:border-primary/40 hover:bg-surface-container-high hover:-translate-y-0.5 transition-all shadow-sm cursor-pointer space-y-2"
                      onClick={() => setDocModal(doc)}
                    >
                      <div className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="material-symbols-outlined text-primary text-[15px] shrink-0">description</span>
                            <span className="font-mono text-[11px] font-semibold text-primary truncate">
                              {doc.notice_id}.pdf
                            </span>
                          </div>
                          <span className="px-1.5 py-0.2 rounded bg-surface-container-lowest text-[9px] text-on-surface-variant border border-outline-variant/20 shrink-0">
                            {doc.page_count} page{doc.page_count > 1 ? 's' : ''}
                          </span>
                        </div>
                        <h3 className="text-xs font-bold text-on-surface group-hover:text-primary transition-colors line-clamp-1">
                          {doc.original_filename || doc.canonical_filename || doc.notice_id}
                        </h3>
                        <p className="text-[11px] text-on-surface-variant line-clamp-2 leading-relaxed">
                          Official NIT Jamshedpur document page extract. Text availability: {doc.text_availability}.
                        </p>
                      </div>

                      <div className="pt-2 flex items-center justify-between border-t border-outline-variant/15 text-[10px]">
                        <span className="px-1.5 py-0.5 rounded bg-surface-container-high text-secondary font-medium border border-secondary/20">
                          {doc.text_availability === 'native' ? 'Native Text' : 'OCR Extracted'}
                        </span>
                        <span className="text-outline font-mono">Indexed</span>
                      </div>
                    </div>
                  ))}

                  {/* Corpus Index State Metric Card */}
                  <div className="relative flex flex-col justify-between p-3 rounded-xl bg-gradient-to-br from-surface-container/40 via-surface-container-low/60 to-primary-container/10 border border-outline-variant/20 shadow-sm space-y-2">
                    <div className="space-y-1">
                      <div className="flex items-center justify-between text-primary">
                        <div className="flex items-center gap-1">
                          <span className="material-symbols-outlined text-[15px] animate-spin" style={{ animationDuration: '9s' }}>hub</span>
                          <span className="text-[9px] uppercase font-semibold tracking-wider font-mono">Index State</span>
                        </div>
                        <span className="w-1.5 h-1.5 rounded-full bg-tertiary animate-pulse"></span>
                      </div>
                      <h4 className="text-xs font-bold text-on-surface">Embeddings Synced</h4>
                      <p className="text-[11px] text-on-surface-variant leading-relaxed">
                        Dual-vector indexes ready for Groq {health?.configured_model || 'Qwen'} synthesis.
                      </p>
                    </div>
                    <div className="pt-2 border-t border-outline-variant/15 flex items-center justify-between text-[10px] text-outline font-mono">
                      <span>Status</span>
                      <span className="text-on-surface font-semibold">{health?.status === 'ok' ? 'Ready' : 'Offline'}</span>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          )}

        </div>
      </main>

      {/* FULLSCREEN LIGHTBOX MODAL */}
      {lightboxImage && (
        <div 
          className="fixed inset-0 z-50 bg-black/90 backdrop-blur-md flex flex-col items-center justify-center p-3 sm:p-6 animate-fade-in"
          onClick={closeLightbox}
        >
          {/* Header Bar */}
          <div 
            className="w-full max-w-5xl flex items-center justify-between py-2 px-4 mb-3 bg-surface-container/95 border border-outline-variant/30 rounded-xl backdrop-blur-xl shadow-2xl gap-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <span className="material-symbols-outlined text-primary text-[20px] shrink-0">image</span>
              <h3 className="text-xs sm:text-sm font-bold text-on-surface truncate">{lightboxImage.title || 'Document Page Evidence'}</h3>
            </div>

            {/* Image Rotation Toolbar */}
            <div className="flex items-center gap-1.5 shrink-0 bg-surface-container-lowest/60 p-1 rounded-lg border border-outline-variant/20">
              <button 
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-surface-container-high hover:bg-primary/20 hover:text-primary text-on-surface text-xs font-semibold border border-outline-variant/20 transition-all active:scale-95 shadow-sm"
                onClick={rotateAnticlockwise}
                title="Rotate Anticlockwise 90° (R or Left Arrow)"
              >
                <span className="material-symbols-outlined text-[18px]">rotate_left</span>
                <span className="hidden sm:inline">Rotate Left (-90°)</span>
              </button>

              <button 
                className="flex items-center gap-1 px-2.5 py-1 rounded-md bg-surface-container-high hover:bg-primary/20 hover:text-primary text-on-surface text-xs font-semibold border border-outline-variant/20 transition-all active:scale-95 shadow-sm"
                onClick={rotateClockwise}
                title="Rotate Clockwise 90° (Right Arrow)"
              >
                <span className="material-symbols-outlined text-[18px]">rotate_right</span>
                <span className="hidden sm:inline">Rotate Right (+90°)</span>
              </button>

              {lightboxRotation !== 0 && (
                <button 
                  className="flex items-center gap-1 px-2 py-1 rounded-md bg-tertiary-container/40 hover:bg-tertiary-container/70 text-tertiary text-xs font-semibold border border-tertiary/40 transition-all active:scale-95 shadow-sm"
                  onClick={resetRotation}
                  title="Reset Image Rotation"
                >
                  <span className="material-symbols-outlined text-[16px]">restart_alt</span>
                  <span className="hidden sm:inline">Reset</span>
                </button>
              )}

              <span className="text-[10px] font-mono text-outline px-1.5 py-0.5 rounded bg-surface-container/80 ml-0.5">
                {((lightboxRotation % 360) + 360) % 360}°
              </span>
            </div>

            <button 
              className="flex items-center gap-1 px-3 py-1 rounded-lg bg-surface-container-high hover:bg-error/20 hover:text-error text-on-surface text-xs font-semibold border border-outline-variant/30 transition-all shrink-0"
              onClick={closeLightbox}
              title="Close Fullscreen View (Esc)"
            >
              <span className="material-symbols-outlined text-[18px]">close</span>
              <span>Close</span>
            </button>
          </div>

          {/* Image Canvas Container */}
          <div 
            className="relative max-w-5xl max-h-[85vh] w-full flex items-center justify-center p-4 rounded-2xl bg-surface-container-lowest/80 border border-outline-variant/20 shadow-2xl overflow-hidden"
            onClick={(e) => e.stopPropagation()}
          >
            <img 
              src={lightboxImage.url} 
              alt="Fullscreen Evidence" 
              style={{ 
                transform: `rotate(${lightboxRotation}deg)`,
                maxHeight: Math.abs(lightboxRotation / 90) % 2 === 1 ? '65vw' : '80vh',
                maxWidth: Math.abs(lightboxRotation / 90) % 2 === 1 ? '65vh' : '100%'
              }}
              className="object-contain rounded-lg shadow-lg select-none transition-transform duration-300 ease-in-out"
            />
          </div>
        </div>
      )}

      {/* DOCUMENT PREVIEW MODAL */}
      {docModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-md p-4 transition-all">
          <div className="relative w-full max-w-md rounded-xl bg-surface-container border border-primary/30 p-5 shadow-2xl space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-outline-variant/30">
              <div className="flex items-center gap-2">
                <span className="material-symbols-outlined text-primary text-[18px]">description</span>
                <span className="text-xs font-bold text-primary font-mono">{docModal.notice_id}.pdf</span>
              </div>
              <button className="p-1 rounded-lg text-outline hover:text-on-surface hover:bg-surface-container-high transition-colors" onClick={() => setDocModal(null)}>
                <span className="material-symbols-outlined text-[16px]">close</span>
              </button>
            </div>
            <div className="space-y-1.5 text-xs">
              <h3 className="text-sm text-on-surface font-semibold">{docModal.original_filename || docModal.canonical_filename || docModal.notice_id}</h3>
              <p className="text-on-surface-variant">Pages: {docModal.page_count} | Text: {docModal.text_availability}</p>
              <div className="p-2.5 rounded-lg bg-surface-container-lowest border border-outline-variant/30 flex items-center justify-between">
                <span className="text-on-surface-variant">Status: <span className="text-tertiary font-semibold">Indexed</span></span>
                <span className="text-on-surface-variant">Format: <span className="text-secondary font-semibold">PDF Page</span></span>
              </div>
            </div>
            <div className="flex items-center justify-end gap-2 pt-2 border-t border-outline-variant/20">
              <button className="px-3 py-1.5 rounded-lg bg-surface-container-high text-on-surface text-xs hover:bg-surface-bright transition-colors" onClick={() => setDocModal(null)}>
                Close
              </button>
              <button 
                className="px-3 py-1.5 rounded-lg bg-primary-container text-white text-xs font-semibold hover:brightness-110 shadow-sm transition-all flex items-center gap-1"
                onClick={() => {
                  setQuestion(`Summarize key details in ${docModal.notice_id}.pdf`)
                  setDocModal(null)
                  if (searchInputRef.current) searchInputRef.current.focus()
                  triggerToast(`Query formulated for ${docModal.notice_id}`)
                }}
              >
                <span className="material-symbols-outlined text-[14px]">query_stats</span>
                <span>Query This Notice</span>
              </button>
            </div>
          </div>
        </div>
      )}

      {/* TOAST NOTIFICATION */}
      {toastMessage && (
        <div className="fixed bottom-8 right-8 z-50 flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-surface-container-high border border-primary/40 text-on-surface shadow-lg text-xs">
          <span className="material-symbols-outlined text-primary text-[16px]">check_circle</span>
          <span>{toastMessage}</span>
        </div>
      )}

      {/* SLEEK COMPACT FOOTER */}
      <footer className="w-full bg-surface-container-lowest py-3 border-t border-outline-variant/20 mt-auto">
        <div className="w-full px-6 md:px-10 flex flex-col md:flex-row items-center justify-between gap-2 text-xs">
          <div className="flex flex-col md:flex-row items-center gap-2 text-center md:text-left">
            <span className="text-on-surface font-medium">SUTRA — NIT Jamshedpur Campus Information Intelligence</span>
            <span className="hidden md:inline text-outline">•</span>
            <span className="text-on-surface-variant">Visual/Hybrid RAG Benchmark</span>
            <span className="hidden md:inline text-outline">•</span>
            <span className="text-outline">Pranav Prakhar, Goutam Kumar Rajak, Angad Ram</span>
          </div>
          <div className="flex items-center gap-2 shrink-0 text-[10px] text-tertiary">
            <span className="w-1.5 h-1.5 rounded-full bg-tertiary"></span>
            <span>Status: {health?.status === 'ok' ? 'System Operational' : 'Ready'}</span>
          </div>
        </div>
      </footer>
    </div>
  )
}
