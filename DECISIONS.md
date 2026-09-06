# Architectural Decision Log — NIT Jamshedpur Visual/Hybrid RAG

This document records formal technical design decisions (Architectural Decision Records - ADRs) for the NIT Jamshedpur Document Intelligence RAG system, categorized by status: **[CONFIRMED]**, **[BASELINE]**, **[IMPROVED]**, or **[FUTURE]**.

---

### D1: Document Processing & Page Rendering

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED]** |
| **Chosen** | PyMuPDF (`fitz`) |
| **Alternatives** | `pdf2image` + Poppler, `pdfplumber` |
| **Reason** | PyMuPDF renders PDF pages to high-DPI PNG images and extracts text without requiring external Windows C++ binaries or Poppler installations. |
| **Evidence** | Successfully renders all 25 pages across 7 PDFs in `data/processed/pages/` reproducibly. |

---

### D2: OCR Text Extraction for Scanned Notices

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED]** |
| **Chosen** | EasyOCR + Custom Regex Dictionary Normalizer |
| **Alternatives** | Tesseract (`pytesseract`), PaddleOCR |
| **Reason** | Scanned cell borders caused dropped characters (`OINT SECRETARY`, `CE-PRESIDENT`). EasyOCR extracts bounding boxes, while custom regex normalization (`_clean_text`) maps typos to canonical headers. |
| **Evidence** | OCR text cleaning enabled Text RRF retrieval to achieve 100% Top-1 document recall across test notices. |

---

### D3: Text Retrieval Architecture

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED]** |
| **Chosen** | Hybrid Dense (`all-MiniLM-L6-v2`) + Lexical (BM25) via RRF ($k=60$) + N-Gram Phrase Boost |
| **Alternatives** | Dense only, BM25 only |
| **Reason** | Dense retrieval captures semantic query intent while BM25 accurately matches alphanumeric registration IDs (e.g. `2024UGCS050`). N-gram phrase matching prevents partial word matches from winning. |
| **Evidence** | Achieved 100% Top-1 Document Accuracy on student council and institutional queries. |

---

### D4: ColQwen2 VLM Multi-Vector Visual Retrieval

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED & UPGRADED]** |
| **Chosen** | `vidore/colqwen2-v1.0` (ColQwen2 VLM) multi-vector patch embeddings |
| **Alternatives** | Single-vector CLIP (`openai/clip-vit-base-patch32`), ResNet |
| **Reason** | Single-vector embeddings flatten pages and lose 2D cell geometry. ColQwen2 embeds high-resolution page images as $1000+$ multi-vector patch embeddings, preserving visual cell boundaries and multi-column table layout. |
| **Evidence** | ColQwen2 multi-vector retrieval correctly isolates complex table rows in `notice_2.pdf` that flat text vectors confused. |

---

### D5: GPU Hardware Acceleration & 4-Bit BitsAndBytes Quantization

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED & UPGRADED]** |
| **Chosen** | PyTorch 4-bit BitsAndBytes quantization (`load_in_4bit=True`, `bnb_4bit_compute_dtype=torch.bfloat16`) on `cuda:0` (NVIDIA RTX 3050 Laptop GPU) |
| **Alternatives** | CPU model execution |
| **Reason** | CPU execution caused 90–95% CPU load and ~45s query latency. 4-bit quantization reduces VRAM footprint to **~3.2 GB**, fitting inside RTX 3050 4GB VRAM. |
| **Evidence** | Reduced visual query latency from **~45 seconds (CPU)** to **100–250 ms (CUDA GPU)**. |

---

### D6: Hybrid RRF Fusion Ratio (70% Visual / 30% Text)

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED]** |
| **Chosen** | Reciprocal Rank Fusion ($0.70 \times \text{Visual RRF} + 0.30 \times \text{Text RRF}$) |
| **Alternatives** | 50/50 RRF, Score Sum |
| **Reason** | Text retrieval acts as a high-precision candidate anchor for proper nouns, while ColQwen2 VLM acts as a spatial layout anchor. |
| **Evidence** | 70/30 RRF maintains exact keyword precision while leveraging visual multi-vector geometry. |

---

### D7: Multimodal Answer Generation Provider

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED]** |
| **Chosen** | Primary Provider: **Groq Cloud API (`qwen/qwen3.8-27b`)** |
| **Alternatives** | xAI Grok (`grok-4.3`), Google Gemini (`gemini-3.8-flash`) |
| **Reason** | Groq API delivers ultra-fast streaming multimodal inference without consuming local GPU VRAM. |
| **Evidence** | Generates grounded, concise answers in ~1.2–2.5 seconds per API call. |

---

### D8: Streamlined Web Interface & Master Modes

| Field | Value |
|:---|:---|
| **Status** | **[CONFIRMED]** |
| **Chosen** | FastAPI Backend + React (Vite) Frontend with **3 Master Retrieval Modes**: `Text Baseline`, `Visual VLM`, `Hybrid RAG 70/30` |
| **Features** | Interactive single search, Compare All 3-column view, persistent API key banner, and source page evidence display. |
