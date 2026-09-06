# Hardware & Environment Specifications — NIT Jamshedpur Visual/Hybrid RAG

## System Environment

| Component | Specification / Value | Execution & Acceleration Context |
| :--- | :--- | :--- |
| **CPU** | AMD Ryzen 5 5600H | 6 cores / 12 threads, 3.3 GHz base |
| **System RAM** | 15.4 GB DDR4 | Host memory management |
| **GPU Hardware** | **NVIDIA GeForce RTX 3050 Laptop GPU** | **4.0 GB Dedicated VRAM (`cuda:0`)** |
| **OS** | Windows 11 | Host operating system |
| **Python** | 3.13+ | Runtime environment |
| **PyTorch** | CUDA 12.x Enabled (`cuda:0`) | 100% GPU accelerated tensor operations |
| **Node.js** | 18+ | Vite production bundler for React UI |
| **LLM/VLM Provider** | **Groq Cloud API** | **`qwen/qwen3.8-27b`** |

---

## Operational Model Footprint & GPU Allocation

| Component | Selected Model | Quantization / Precision | GPU VRAM / Host Footprint | Execution Device |
| :--- | :--- | :--- | :--- | :--- |
| **Text Embedder** | `sentence-transformers/all-MiniLM-L6-v2` | FP32 normalized | ~90 MB VRAM | **CUDA (`cuda:0`)** |
| **Lexical Engine** | BM25 (`rank_bm25`) + N-gram phrase boost | CPU In-Memory Index | ~1.5 MB RAM | Host CPU |
| **Visual VLM Embedder** | **`vidore/colqwen2-v1.0` (ColQwen2 VLM)** | **4-bit BitsAndBytes (`nf4` / `bfloat16`)** | **~3.2 GB VRAM** | **CUDA (`cuda:0`)** |
| **OCR Engine** | EasyOCR (`en`) | PyTorch FP32 | ~100 MB RAM | Host CPU |
| **Answer Generation** | **Groq `qwen/qwen3.8-27b`** | Cloud API Multi-Modal | 0 Local VRAM (API Payload) | External Groq API Gateway |

---

## Empirical Latency Profile (GPU Accelerated on RTX 3050)

| Phase / Mode | Latency (Measured) | Performance Characteristics |
| :--- | :--- | :--- |
| **Index & Model Cold-Start** | ~0.05 seconds | Loads pre-computed multi-vector PyTorch/NumPy arrays into VRAM |
| **Text Baseline Retrieval** | ~15–35 ms | CUDA GPU dense vector dot-product + BM25 Lexical N-gram boost |
| **Visual VLM Retrieval (`ColQwen2`)** | ~100–250 ms | CUDA 4-bit VLM multi-vector patch scoring ($1000+$ vectors per page) |
| **Hybrid RRF (70/30 RAG)** | ~120–280 ms | Reciprocal Rank Fusion of GPU Text + GPU Visual VLM scores |
| **VLM Answer Generation** | ~1.2–2.5 seconds | Groq API streaming round-trip & grounded answer synthesis |

---

## Hardware Acceleration Rationale

1. **4-Bit VLM Quantization on CUDA**: Loading `vidore/colqwen2-v1.0` using BitsAndBytes 4-bit NormalFloat (`nf4`) and `bfloat16` compute precision reduces VRAM requirements from 14GB+ to **~3.2 GB**, fitting comfortably inside the 4GB VRAM limit of the NVIDIA RTX 3050 Laptop GPU.
2. **Shift from CPU to 100% CUDA Acceleration**: In earlier iterations, CPU execution caused 90–95% CPU load and 45s latency. Transitioning vector calculations to CUDA (`cuda:0`) reduced retrieval latency from ~45 seconds down to **120–280 milliseconds** while freeing host CPU resources.
3. **Groq Vision LLM Synthesis**: Offloading final multimodal generation to Groq (`qwen/qwen3.8-27b`) enables state-of-the-art vision reasoning without exceeding local GPU memory boundaries.
