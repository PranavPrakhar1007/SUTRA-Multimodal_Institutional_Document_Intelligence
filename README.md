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
│   ├── config.py             # System paths, constants & GPU model parameters
│   ├── generation.py         # LLM answer synthesis (Groq Qwen 3.8 27B)
│   ├── hybrid_reranker.py    # Multi-stage Hybrid visual-text RRF re-ranking
│   ├── hybrid_retrieval.py   # Sparse (BM25) + Dense (BGE) + ColQwen2 Fusion
│   ├── text_retrieval.py     # Text indexing & sentence-transformers vector search
│   └── visual_retrieval.py   # Multi-vector ColQwen2 VLM index search
├── frontend/                 # Sleek React + Vite Web UI
│   ├── src/
│   │   ├── App.jsx           # Single-page SUTRA document intelligence UI
│   │   └── index.css         # Modern design system, keyframes & glassmorphic styling
│   ├── package.json          # Node dependencies & scripts
│   └── vite.config.js        # Vite build & API proxy setup
├── data/                     # Indexed document vectors, text chunks & metadata
├── scripts/                  # Data pipeline, OCR, index building, and evaluation scripts
├── .gitignore                # Git exclusions (node_modules, API keys, caches)
├── CORPUS_MANIFEST.csv       # Corpus manifest of the 7 NIT Jamshedpur documents
├── DECISIONS.md              # Technical decision record & architecture rationale
├── HARDWARE.md               # Hardware specs & CUDA vRAM optimization details
├── README.md                 # Project documentation & Quickstart guide
└── requirements.txt          # Master Python dependency manifest
```

---

## 🚀 Complete Local Installation & Quickstart Guide

Follow these step-by-step instructions to clone, set up, and run SUTRA on a new local machine.

### 1. System Requirements & Prerequisites
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS
- **Python**: 3.10, 3.11, or 3.12 (3.11 recommended)
- **Node.js**: v18.0.0 or higher & `npm`
- **GPU (Recommended)**: NVIDIA GPU with CUDA 11.8 / 12.0+ (Automatic CPU fallback supported if GPU unavailable)
- **API Key**: Free Groq API Key from [Groq Console](https://console.groq.com/keys)

---

### 2. Step-by-Step Setup Instructions

#### Step 1: Clone the Repository
```bash
git clone https://github.com/PranavPrakhar1007/SUTRA-Multimodal_Institutional_Document_Intelligence.git
cd SUTRA-Multimodal_Institutional_Document_Intelligence
```

#### Step 2: Create and Activate a Python Virtual Environment
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

#### Step 3: Install Python Backend Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **NVIDIA GPU Users (Crucial for CUDA Acceleration)**:
> If PyTorch installs as CPU-only, reinstall the PyTorch CUDA build:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

#### Step 4: Build and Install Frontend Web UI
```bash
cd frontend
npm install
npm run build
cd ..
```

---

### 3. API Key & Environment Setup

You can set your Groq API Key in any of the following 3 ways:

1. **Via SUTRA Web UI Header (Recommended)**:
   - Simply launch the app, click the key icon in the top navigation bar, and enter your `gsk_...` key. It will persist automatically across server restarts in `data/keys.json`.
2. **Via `.env` File**:
   - Create a `.env` file in the project root directory:
     ```env
     GROQ_API_KEY=gsk_your_actual_groq_api_key_here
     ```
3. **Via Environment Variable**:
   - **Windows (PowerShell)**: `$env:GROQ_API_KEY="gsk_your_key_here"`
   - **Linux / macOS**: `export GROQ_API_KEY="gsk_your_key_here"`

---

### 4. Running the Application

#### Option A: Production Mode (Recommended — Single Server)
Serves both the backend API and the compiled React Web UI from FastAPI on port 8000:
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your web browser.

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

