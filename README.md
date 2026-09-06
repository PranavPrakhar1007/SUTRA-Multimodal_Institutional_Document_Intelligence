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
│   ├── generation.py         # Multi-provider LLM answer synthesis (Groq / xAI / Gemini)
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

## 🛠️ Prerequisites & One-Command Setup

### 1. Prerequisites
- **Python**: 3.10 or higher (Python 3.11 recommended)
- **Node.js**: 18.0 or higher & `npm`
- **GPU (Recommended)**: NVIDIA GPU with CUDA 11.8+ / 12.0+ (CPU fallback supported automatically)
- **GROQ API Key**: Free API key from [console.groq.com](https://console.groq.com/keys)

---

### 2. Quickstart Installation (One-Go Commands)

#### Step 1: Install Python Backend Dependencies in One Go
```bash
pip install -r requirements.txt
```

*(Note for CUDA GPU users: If PyTorch CUDA is needed, install via `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`)*

#### Step 2: Build & Install Frontend Dependencies in One Go
```bash
cd frontend && npm install && npm run build && cd ..
```

---

### 3. Launching the Application

#### Production Mode (Single-Server FastAPI Backend + Built Web UI)
```bash
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser.

#### Development Mode (With Hot Reloading)
- **Terminal 1 (Backend)**:
  ```bash
  python -m uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
  ```
- **Terminal 2 (Frontend Dev Server)**:
  ```bash
  cd frontend
  npm run dev
  ```
Open **`http://localhost:5173`** in your browser.

---

### 4. Running Benchmark Evaluation Suite

To evaluate all 3 retrieval pipelines across the 23 institutional test queries:
```bash
python scripts/run_evaluation.py
```

---

## 🗝️ API Key Configuration

1. Get a free API key from [Groq Console](https://console.groq.com/keys).
2. Enter your API key directly in the top header bar of the SUTRA Web UI (`gsk_...`), or pass it via request payload.
3. The API key is stored securely in your browser's local storage or server session (`data/keys.json` - excluded from git).

---

## 📄 License & Citation
Developed for Capstone Project at **National Institute of Technology (NIT) Jamshedpur**, Department of Computer Science & Engineering.
