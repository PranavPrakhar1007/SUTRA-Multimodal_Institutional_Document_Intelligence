# SUTRA — Multimodal Institutional Document Intelligence

> **Natural language question answering over NIT Jamshedpur institutional documents with multi-stage visual verification, ColQwen2 VLM patch alignment, and grounded evidence display.**

---

### Team & Credits
- **Pranav Prakhar** — B.Tech CSE, Semester 5, NIT Jamshedpur
- **Goutam Kumar Rajak** — B.Tech CSE, Semester 5, NIT Jamshedpur
- **Angad Ram** — B.Tech CSE, Semester 5, NIT Jamshedpur

---

## 🏛️ Project Overview & Architecture

**SUTRA** (Multimodal Institutional Document Intelligence) solves complex structural, tabular, and signature-level question answering across scanned administrative notices, official office circulars, hostel allotment tables, and institute ordinances.

### Key Technical Highlights:
1. **100% CUDA GPU Acceleration**: Text embeddings (`all-MiniLM-L6-v2`) and Visual VLM (`vidore/colqwen2-v1.0`) run natively in CUDA GPU memory on NVIDIA GeForce RTX GPUs.
2. **ColQwen2 VLM Multi-Vector Patch Matching**: Employs `vidore/colqwen2-v1.0` (Qwen2-VL-2B-Instruct) multi-vector late-interaction patch matching, embedding raw document page images directly into fine-grained 128-d patch vectors.
3. **Streamlined 3 Retrieval Pipelines**:
   - **Text Baseline**: BM25 (Sparse) + Dense (`all-MiniLM-L6-v2`) combined via Reciprocal Rank Fusion on GPU.
   - **Visual VLM (ColQwen2)**: Direct OCR-free visual patch token alignment over raw A4 notice images, detecting stamps, signatures, and complex table rows.
   - **Hybrid RAG (70/30 Fusion)**: Weighted Reciprocal Rank Fusion ($0.7 \times \text{ColQwen2 Visual} + 0.3 \times \text{Text}$) for maximum accuracy.
4. **Flagship Multimodal Generation**: Powered by **`qwen/qwen3.8-27b`** via Groq LPU inference, establishing end-to-end architectural symmetry with the local ColQwen2 visual retriever.

---

## 📊 Master Benchmark Evaluation Results (23 Queries on CUDA GPU)

| Retrieval Mode | Top-1 Doc Accuracy | Hit@1 Page Accuracy | Hit@3 Page Accuracy | Hit@5 Page Accuracy | Page MRR | Average Latency |
|----------------|--------------------|---------------------|---------------------|---------------------|----------|-----------------|
| **Text Baseline (BM25 + Dense)** | **21/21 (100.0%)** | 20/21 (95.2%) | **21/21 (100.0%)** | **21/21 (100.0%)** | 0.9762 | **~14.2 ms** |
| **Legacy Visual Baseline (CLIP)** | 7/21 (33.3%) | 5/21 (23.8%) | 6/21 (28.6%) | 13/21 (61.9%) | 0.3829 | ~185.0 ms |
| **ColPali VLM (v1.2) Visual** | 20/21 (95.2%) | 20/21 (95.2%) | 21/21 (100.0%) | 21/21 (100.0%) | 0.9762 | ~310.0 ms |
| **ColQwen2 VLM (v1.0) Visual** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **1.0000** | ~231.4 ms |
| **Hybrid RAG (ColQwen2 70% / Text 30%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **1.0000** | ~324.1 ms |

---

## 📁 Repository Structure

```text
Capstone_College_RAG/
├── backend/                  # FastAPI application server & REST endpoints
│   ├── app.py                # Main REST API application (Health, Query, Compare, Key management)
│   ├── config.py             # System paths, constants & GPU model parameters (DOCUMENTS_DIR = data/documents)
│   ├── generation.py         # LLM answer synthesis (Groq Qwen 3.8 27B)
│   ├── hybrid_reranker.py    # Multi-stage Hybrid visual-text RRF re-ranking
│   ├── hybrid_retrieval.py   # Sparse (BM25) + Dense (MiniLM) + ColQwen2 Fusion
│   ├── text_retrieval.py     # Text indexing & sentence-transformers vector search
│   └── visual_retrieval.py   # Multi-vector ColQwen2 VLM index search
├── data/                     # Primary data workspace
│   ├── documents/            # 📥 INPUT PDF FOLDER: Place all external PDFs here!
│   ├── embeddings/           # Generated vector indices (text & ColQwen2 multi-vectors)
│   ├── evaluation/           # Evaluation queries dataset and results.json
│   └── processed/            # Extracted 216 DPI PNG page images, OCR text & metadata JSONs
├── frontend/                 # Sleek React + Vite Web UI
│   ├── src/
│   │   ├── App.jsx           # Single-page SUTRA document intelligence UI
│   │   └── index.css         # Modern design system, keyframes & glassmorphic styling
│   ├── package.json          # Node dependencies & scripts
│   └── vite.config.js        # Vite build & API proxy setup
├── scripts/                  # Data pipeline, OCR, index building, and evaluation scripts
│   ├── process_documents.py  # Render 216 DPI page images and extract text from data/documents/
│   ├── build_indices.py      # Build text & ColQwen2 visual embedding indices
│   ├── run_evaluation.py     # Benchmark retrieval precision & MRR metrics
│   └── profile_corpus.py     # Generate statistical profiles for corpus_profile.json
├── .gitignore                # Git exclusions (node_modules, API keys, caches)
├── DECISIONS.md              # Technical decision record & architecture rationale
├── HARDWARE.md               # Hardware specs & CUDA VRAM optimization details
├── README.md                 # Project documentation & Quickstart guide
└── requirements.txt          # Master Python dependency manifest
```

---

## 🚀 Complete Start-to-Finish Command Guide

Follow these exact step-by-step commands to clone, set up, ingest custom PDFs, build indices, run benchmarks, and launch SUTRA on any computer:

### 1. Step 1: Clone the Repository
```bash
git clone https://github.com/PranavPrakhar1007/SUTRA-Multimodal_Institutional_Document_Intelligence.git
cd SUTRA-Multimodal_Institutional_Document_Intelligence
```

---

### 2. Step 2: Create & Activate Python Virtual Environment
- **Windows (PowerShell)**:
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- **Linux / macOS**:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```

---

### 3. Step 3: Install Backend Python Dependencies & CUDA PyTorch
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
> **For NVIDIA GPU Users (CUDA Acceleration)**:
> Re-install the CUDA-enabled PyTorch build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

---

### 4. Step 4: Build & Install Frontend Web UI
```bash
cd frontend
npm install
npm run build
cd ..
```

---

### 5. Step 5: Place Your Custom PDFs in `data/documents/`
Copy or move any external PDF files you want SUTRA to index directly into the `data/documents/` subfolder:
- **Folder Path:** `data/documents/`
```bash
# Example: Place custom PDFs into data/documents/
# (e.g. data/documents/notice_1.pdf, data/documents/custom_circular.pdf)
```

---

### 6. Step 6: Process Documents & Rebuild Embedding Indices

Run the processing, OCR, and index-building scripts to extract 216 DPI page images, text chunks, and ColQwen2 multi-vectors for all PDFs in `data/documents/`:

> [!IMPORTANT]
> **Before running `build_indices.py`**: Make sure no background Uvicorn servers are running in other terminal windows, as ColQwen2 requires GPU VRAM to load.
> - On Windows PowerShell, you can stop all background Python servers with:
>   `Stop-Process -Name python -Force -ErrorAction SilentlyContinue`

```bash
# 1. Extract 216 DPI PNG page images & metadata JSONs:
python scripts/process_documents.py

# 2. Run OCR to extract text from scanned PDF pages:
python scripts/ocr_pages.py

# 3. Generate text vectors (MiniLM + BM25) and ColQwen2 visual multi-vectors:
python scripts/build_indices.py

# 4. Generate statistical corpus profile for health diagnostics:
python scripts/profile_corpus.py
```

---

### 7. Step 7: (Optional) Run Benchmark Evaluation Suite
To benchmark retrieval metrics (Hit@1, Hit@3, MRR, Latency) across the 28 evaluation queries:
```bash
python scripts/run_evaluation.py
```

---

### 8. Step 8: Configure Groq API Key & Launch SUTRA Server

Set your free Groq API Key (from [Groq Console](https://console.groq.com/keys)):

Edit the project `.env` file and set:
```dotenv
GROQ_API_KEY=gsk_your_actual_groq_api_key_here
GROQ_MODEL=qwen/qwen3.8-27b
```
The backend loads this file at startup. The key is never displayed or entered through the web UI.

#### Launch Production Server (Single Command):
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser to interact with SUTRA!

#### Option B: Development Mode (With Hot Reloading)
- **Terminal 1 (Backend Server)**:
  ```bash
  python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Terminal 2 (Frontend Dev Server)**:
  ```bash
  cd frontend
  npm run dev
  ```
Open **`http://localhost:5173`** in your web browser.

---

### 5. Re-Building Indices & Corpus Data (Optional)

Pre-indexed embeddings and corpus metadata are included in `data/`. If you want to re-process documents or re-generate indices from scratch:

```bash
# Profile the corpus documents
python scripts/profile_corpus.py

# Re-build text & visual ColQwen2 vector indices
python scripts/build_indices.py
```

---

### 6. Running Benchmark Evaluation Suite

To run full multi-query benchmarking across all 3 pipelines (Text, Visual, Hybrid) and compute Top-K accuracy & MRR metrics:

```bash
python scripts/run_evaluation.py
```

---

## 🛠️ Troubleshooting & FAQs

- **Issue**: `torch.cuda.is_available()` returns `False`.
  - **Solution**: Reinstall PyTorch with the CUDA wheel index URL: `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`.
- **Issue**: Port 8000 is already in use.
  - **Solution**: Specify a custom port: `python -m uvicorn backend.app:app --host 0.0.0.0 --port 8080`.
- **Issue**: `TesseractNotFoundError` or `pdf2image` error when running raw OCR re-indexing.
  - **Solution**: Install Tesseract OCR (`apt install tesseract-ocr` or Windows installer) and Poppler (`apt install poppler-utils` or Windows Poppler binary).

---

## 📄 License & Citation
Developed for Capstone Project at **National Institute of Technology (NIT) Jamshedpur**, Department of Computer Science & Engineering.

