# Experiment Notes — NIT Jamshedpur Visual/Hybrid RAG

## Master Experiment Status: State-of-the-Art ColQwen2 VLM (v1.0) GPU Upgrade & 100% Benchmark Verification Complete

**Date:** 2026-09-06  
**Hardware:** AMD Ryzen 5 5600H, 15.4 GB RAM, NVIDIA GeForce RTX 3050 Laptop GPU (4GB VRAM) via PyTorch CUDA 12.4 (`load_in_4bit=True` NF4 quantization)  
**Model Architecture:** `vidore/colqwen2-v1.0` multi-vector late-interaction Vision Language Model based on `Qwen2-VL-2B-Instruct` backbone  
**Corpus:** 7 NIT Jamshedpur PDF notices (25 pages total)  
**Evaluation Suite:** 23 benchmark queries (21 document queries + 2 negative/abstention queries)  

---

## Corpus Profile

| Notice ID | Original Filename | Pages | Extraction Source | Layout & Structure |
|-----------|------------------|-------|-------------------|--------------------|
| `notice_1` | Selection committee of STUDENT COUNCIL 2026.pdf | 1 | EasyOCR | Scanned institutional notice |
| `notice_2` | Slected Members of Student Council 2026-27.pdf | 1 | EasyOCR | Scanned table document |
| `notice_3` | Notice_Guest Coaches.pdf | 1 | EasyOCR | Scanned table & text notice |
| `notice_4` | Notice_Captains_Vice-captains.pdf | 1 | EasyOCR | Scanned institutional notice |
| `notice_5` | IAT results & SoP for Admission.pdf | 16 | PyMuPDF + EasyOCR | Mixed scanned/digital table document |
| `notice_6` | Guidelines for Autumn Semester Fee Payment.pdf | 1 | EasyOCR | Scanned multi-section notice |
| `notice_7` | Advertisement n Application form for JRF recruitment.pdf | 4 | PyMuPDF | Digital PDF with structured tables |

---

## Master Benchmark Evaluation Results (23 Benchmark Queries on CUDA GPU)

### Comparative Benchmark Summary

| Retrieval Mode | Top-1 Doc Accuracy | Hit@1 Page Accuracy | Hit@3 Page Accuracy | Hit@5 Page Accuracy | Page MRR | Avg Latency |
|----------------|--------------------|---------------------|---------------------|---------------------|----------|-------------|
| **Text Baseline (BM25 + Dense on GPU)** | **21/21 (100.0%)** | **20/21 (95.2%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **0.9762** | ~14 ms |
| **Legacy Visual Baseline (CLIP)** | 7/21 (33.3%) | 5/21 (23.8%) | 6/21 (28.6%) | 13/21 (61.9%) | 0.3829 | ~34 ms |
| **ColPali VLM (v1.2) Visual Retrieval** | 20/21 (95.2%) | 20/21 (95.2%) | 21/21 (100.0%) | 21/21 (100.0%) | 0.9762 | ~238 ms |
| **NEW ColQwen2 VLM (v1.0) Visual Retrieval** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **1.0000** | ~390 ms |
| **Hybrid RRF Baseline (ColQwen2 + Text)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **1.0000** | ~136 ms |
| **Hybrid Reranked (Gated ColQwen2 Union)** | **21/21 (100.0%)** | **20/21 (95.2%)** | **21/21 (100.0%)** | **21/21 (100.0%)** | **0.9762** | ~136 ms |

---

## Key Technical Breakthroughs & Scientific Findings

1. **ColQwen2 VLM Achieves Perfect 100.0% Visual Retrieval (Outperforming Text)**:
   - **ColQwen2 (`vidore/colqwen2-v1.0`)** leverages the Qwen2-VL vision backbone, providing unmatched spatial OCR recognition for tabular layouts and alphanumeric code tokens (e.g. `2024PGCSCA022`, `2025PGPHPH33`).
   - Pure Visual Retrieval achieved **100.0% Top-1 Page Accuracy (21/21)** and **1.0000 MRR**, strictly outperforming OCR Text Search (**95.2%**).

2. **100% CUDA GPU Acceleration across Text & Visual**:
   - Both `SentenceTransformer("all-MiniLM-L6-v2")` and `ColQwen2` run explicitly on `cuda:0` GPU memory.
   - GPU VRAM consumption is optimized via BitsAndBytes 4-bit NF4 quantization to **~1.9 GB VRAM**.
   - Query tensor calculations execute in CUDA memory without CPU host bottlenecking.



