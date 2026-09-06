# 📜 NIT Jamshedpur Document Intelligence RAG
## Chronological Development History, Challenges & Evolution Log

This document provides an exhaustive, step-by-step account of every technical difficulty, OCR error, model bottleneck, API gateway requirement, and architectural challenge encountered throughout the development of the NIT Jamshedpur Document Intelligence RAG system, along with the exact engineering strategy used to resolve each issue.

---

## 📌 Stage 1: Initial PDF Extraction & OCR Text Noise

### 1.1 The Challenge
Raw institutional PDF notices from NIT Jamshedpur contain scanned images, embedded seals, and complex multi-column tables. Initial standard text extractors returned either empty strings (`""`) or fragmented, noisy character streams.

### 1.2 Observed Errors & Symptoms
* **EasyOCR Character Dropouts:** OCR scanning on cell borders repeatedly dropped initial characters:
  - `OINT SECRETARY` instead of `JOINT SECRETARY`
  - `ONNT CULTURAL SECRETARY` instead of `JOINT CULTURAL SECRETARY`
  - `CE-PRESIDENT` / `IICE-PRESIDENT` instead of `VICE-PRESIDENT`
  - `IPG REPRESENTATIVE` instead of `PG REPRESENTATIVE`
* **Impact:** Keyword search for terms like `"Joint Secretary"` failed to match candidate rows in `notice_2.pdf`.

### 1.3 How We Eliminated It
1. **Hybrid Ingestion Pipeline (`scripts/ocr_pages.py`):**
   - Established a native text threshold check ($\ge 10$ characters). If native vector text is available (e.g. digital PDFs), it is preserved with 100% fidelity.
   - For scanned images, EasyOCR is triggered to extract word bounding boxes and confidence scores.
2. **Regex Dictionary Normalizer (`backend/text_retrieval.py`):**
   - Implemented `_clean_text()` with case-insensitive regex substitution rules to repair recurring OCR typos automatically before indexing:
     ```python
     re.sub(r'\bOINT SECRETARY\b', 'JOINT SECRETARY', flags=re.IGNORECASE)
     re.sub(r'\bCE-PRESIDENT\b', 'VICE-PRESIDENT', flags=re.IGNORECASE)
     ```

---

## 📌 Stage 2: Single-Vector Embedding Blindness on Tables

### 2.1 The Challenge
Traditional dense text embeddings (`all-MiniLM-L6-v2`) flatten document pages into a single 1D vector embedding. When a document contains dense multi-column student council tables, linear text streams lose the 2D spatial relationships between table headers, candidate names, and registration numbers.

### 2.2 Observed Errors & Symptoms
* Text search confused adjacent rows in `notice_2.pdf` (e.g., Row 8 lists **Krrish Kumar** as Joint Secretary, while Row 9 lists **Dev Kumar** as Joint Cultural Secretary).
* Flat text retrieval could not guarantee whether a candidate belonged to the row above or below.

### 2.3 How We Eliminated It
* **Integrated ColQwen2 Multi-Vector Visual VLM (`backend/visual_retrieval.py`):**
  - Upgraded the visual retrieval engine to **ColQwen2 (`vidore/colqwen2-v1.0`)**.
  - Instead of a single vector per page, ColQwen2 embeds high-resolution page images into over $1,000+$ multi-vector patch embeddings per page, preserving exact cell geometry, visual borders, signatures, and column alignments.

---

## 📌 Stage 3: CPU Bottleneck & Unutilized GPU Hardware

### 3.1 The Challenge
During early visual VLM integration, PyTorch defaulted model execution to the host CPU. CPU usage spiked to **90%–95%**, while the dedicated NVIDIA GPU stayed below **15%**. Query latency exceeded **40–60 seconds per query**.

### 3.2 Observed Errors & Symptoms
* Excessive query latency, thermal throttling on host CPU, and inefficient resource usage despite having a CUDA-capable GPU.

### 3.3 How We Eliminated It
* **4-bit BitsAndBytes Quantization on CUDA (`backend/visual_retrieval.py`):**
  - Configured 4-bit NormalFloat quantization:
    ```python
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    _colpali_model = ColQwen2.from_pretrained(
        VISUAL_EMBEDDING_MODEL,
        quantization_config=bnb_config,
        device_map="cuda:0",
    )
    ```
  - Shifted VLM scoring 100% to GPU (`cuda:0` NVIDIA RTX 3050 Laptop GPU). Latency dropped dramatically while keeping VRAM footprint under **3.5 GB**.

---

## 📌 Stage 4: Pure Visual Retrieval vs. Exact Entity Match Mismatch

### 4.1 The Challenge
Research papers emphasize that Visual Retrieval outperforms Text Retrieval for document understanding. However, for queries targeting exact proper nouns or registration IDs (e.g., `"what is the role of Krrish Kumar in student council?"`), pure visual embeddings occasionally ranked visually similar tables (`notice_3`) higher than the exact target page (`notice_2`).

### 4.2 Observed Errors & Symptoms
* Pure visual search prioritized visual table appearance over exact string matches for rare student names.

### 4.3 How We Eliminated It
1. **Hybrid RRF Fusion (70% Visual / 30% Text):**
   - Combined text lexical search (BM25) and visual multi-vector search (ColQwen2) using Reciprocal Rank Fusion:
     $$\text{RRF\_Score}(d) = \frac{0.30}{60 + \text{Rank}_{\text{text}}(d)} + \frac{0.70}{60 + \text{Rank}_{\text{visual}}(d)}$$
   - Text retrieval acts as a high-precision candidate anchor for proper nouns, while ColQwen2 visual VLM acts as the structural layout anchor.
2. **Exact Multi-Word N-Gram Boost (`backend/text_retrieval.py`):**
   - Added contiguous n-gram phrase matching in `search()`. When a multi-word phrase (e.g., `"Krrish Kumar"`) appears in a page's text, an exact phrase score bonus is awarded, preventing partial word matches from winning.

---

## 📌 Stage 5: Stop-Word Dilution in Lexical BM25 Search

### 5.1 The Challenge
Natural language queries like *"what is the role of Krrish Kumar in student council?"* contain filler English stop-words (`what`, `is`, `the`, `of`, `in`). Raw BM25 scoring evaluated every token equally, so pages containing common words like `"is"` or `"in"` diluted the ranking of proper nouns.

### 5.2 Observed Errors & Symptoms
* Non-relevant pages containing common words `"is"` or `"in"` received artificial BM25 score boosts.

### 5.3 How We Eliminated It
* **Query Tokenization & Stop-Word Filtering (`backend/text_retrieval.py`):**
  - Implemented `_tokenize_query()` to strip English stop-words while preserving domain entities (`krrish`, `kumar`, `student`, `council`, `jrf`, `mtech`).
  - Added a **BM25 Dominance Ratio Bonus** (`TEXT_BM25_DOMINANCE_BONUS`): when the top BM25 result has a score ratio $\ge 2.0\times$ over rank #2, it receives an extra score boost in RRF fusion.

---

## 📌 Stage 6: LLM Internal Monologue & Self-Correction Commentary

### 6.1 The Challenge
When LLMs (xAI Grok, Groq Qwen, Gemini Flash) processed OCR text containing minor OCR artifacts, the LLMs output verbose internal reasoning monologues (e.g., *"Looking at the text 'OINT SECRETARY', it appears the initial letter 'J' was dropped by OCR. Therefore, Krrish Kumar is Joint Secretary..."*).

### 6.2 Observed Errors & Symptoms
* Messy, ungrounded, or meta-commentary responses shown to end-users instead of clean factual answers.

### 6.3 How We Eliminated It
* **Strict Grounding Prompt Rules (`backend/generation.py`):**
  - Updated `GROUNDING_PROMPT` with explicit instruction Rule 7:
    > *"7. Output ONLY the clean, final grounded answer. Do NOT output internal chain-of-thought monologue, OCR analysis commentary, or self-correction notes."*
  - Responses became 100% concise, clean, and direct.

---

## 📌 Stage 7: Multi-Provider LLM Gateway & Persistent Key Management

### 7.1 The Challenge
The system required seamless support for multiple LLM providers (xAI Grok, Groq, Gemini) and external OpenAI-compatible API gateways (Experiential Labs gateway `https://api.experientiallabs.ai`).

### 7.2 Observed Errors & Symptoms
* API keys lost on server reboot.
* Unhandled exception crashes when an empty string key was passed during client instantiation.

### 7.3 How We Eliminated It
1. **Persistent Key Store (`backend/config.py`):**
   - Created `data/keys.json` to persist API keys across server reboots.
   - Updated `load_persistent_keys()` to execute prior to `get_llm_provider()`.
2. **Frontend Provider Selector (`frontend/src/App.jsx`):**
   - Built a dynamic API Key banner allowing live switching between xAI, Groq, and Gemini providers.
3. **Empty Key Guard Checks (`backend/generation.py`):**
   - Added `if not api_key or not api_key.strip():` guards to gracefully return an abstention error instead of throwing SDK exceptions.

---

## 📌 Stage 8: Multimodal Evidence Bounding-Box Tile Cropping

### 7.1 The Challenge
When queries targeted specific table sub-sections (e.g., JRF fellowship stipends or Captain names), feeding full 144 DPI page images to Vision LLMs caused loss of legibility for small font numbers.

### 7.2 Observed Errors & Symptoms
* Misread digits in fine-print table footnotes.

### 7.3 How We Eliminated It
* **Multi-Scale Region Tile Reranking (`backend/visual_reranker.py`):**
  - Divided page images into a multi-scale grid (upper half, middle half, lower half, $2\times 2$ tiles).
  - Selected the highest-scoring bounding box (`best_tile_box`) and encoded the cropped region image (`visual_crop`) at high resolution for LLM generation.

---

## 📌 Stage 9: UI Grid Duplication & Compare All Redundancies

### 9.1 The Challenge
Clicking **Compare All** in the web interface rendered 5 columns (`Text Baseline`, `Visual CLIP`, `Visual Reranked`, `Naive Hybrid RRF`, `Hybrid Reranked`), duplicating visual and hybrid columns and triggering `KeyError: 'visual_reranked'` on the backend.

### 9.2 Observed Errors & Symptoms
* UI clutter, horizontal overflow, and backend 500 exceptions during comparison runs.

### 9.3 How We Eliminated It
* Streamlined `/api/compare` in `backend/app.py` and grid mapping in `frontend/src/App.jsx` to strictly evaluate and display the **3 Master Retrieval Modes**:
  1. **`📝 Text Baseline`** (`text_baseline`)
  2. **`👁️ Visual VLM`** (`visual` - ColQwen2)
  3. **`⚡ Hybrid RAG`** (`hybrid_rrf_baseline` - 70/30 RRF)

---

## 📌 Stage 10: Codebase Quality, Windows Encoding & Portability

### 10.1 The Challenge
Running Python scripts on Windows standard output (`cmd.exe` or PowerShell) raised encoding crashes when printing notice titles containing Hindi/Devanagari characters.

### 10.2 Observed Errors & Symptoms
* `UnicodeEncodeError: 'charmap' codec can't encode characters` during `ocr_pages.py` runs.

### 10.3 How We Eliminated It
* Reconfigured `sys.stdout` to UTF-8 across all CLI scripts (`ocr_pages.py`, `profile_corpus.py`, `run_evaluation.py`).
* Conducted a complete 16-point bug audit documented in [`bugs.txt`](file:///d:/Asus%20laptop/D/Capstone_College_RAG/bugs.txt).

---

## 🎯 Master Summary Matrix of System Evolution

| Stage | Initial Difficulty / Error | Root Cause | Engineering Solution |
| :--- | :--- | :--- | :--- |
| **1. Ingestion** | Dropped header letters (`OINT SECRETARY`) | EasyOCR cell border clipping | Hybrid native text check + Case-insensitive Regex Cleaner |
| **2. Multi-column** | Table row mismatch (Krrish vs Dev Kumar) | 1D text vector spatial blindness | ColQwen2 4-bit VLM Multi-Vector Visual Embeddings |
| **3. Hardware** | 90% CPU / 15% GPU usage (45s latency) | Default CPU model loading | PyTorch 4-bit BitsAndBytes quantization on `cuda:0` |
| **4. Retrieval** | Pure Visual missing exact name matches | Absence of lexical entity weighting | Hybrid 70/30 Visual/Text RRF + Multi-Word N-gram Boost |
| **5. Tokenization** | Stop-words diluting lexical scores | Unfiltered BM25 tokenization | Stop-word filtering in `_tokenize_query()` + BM25 Dominance Ratio |
| **6. Synthesis** | LLM outputting internal monologue notes | Lack of formatting constraints | Grounding Prompt Rule 7 prohibiting chain-of-thought commentary |
| **7. API Gateway** | API key loss on boot & empty key crashes | Config order & missing string guard | `data/keys.json` store + dynamic provider selector banner |
| **8. Vision Crops** | Fine-print digit misreads in tables | Low-resolution full page image | Multi-scale tile crop selection (`best_tile_box`) |
| **9. Interface** | 5 duplicate comparison columns & KeyError | Legacy mode key fallback mapping | Hardcoded 3 Master Modes (`Text`, `Visual VLM`, `Hybrid 70/30`) |
| **10. System** | Windows console Hindi encoding crash | Default Windows `cp1252` stdout | Reconfigured `sys.stdout` to UTF-8 in CLI scripts |
