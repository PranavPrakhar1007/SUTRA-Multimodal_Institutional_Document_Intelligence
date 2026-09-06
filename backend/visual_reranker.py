"""Multi-scale ColPali VLM reranking for candidate document pages on GPU.

Evaluates high-resolution full page and region tile crops using ColPali multi-vector
late-interaction patch matching for precise visual document search.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backend.config import RAW_DIR, VISUAL_RERANK_DPI
from backend.path_utils import resolve_page_image
from backend.visual_retrieval import _get_colpali


class VisualReranker:
    def __init__(self, tile_grid: Tuple[int, int] = (2, 2), overlap: float = 0.2):
        self.tile_grid = tile_grid
        self.overlap = overlap

    def _extract_tiles(self, image: Image.Image) -> List[Tuple[str, Image.Image, Tuple[int, int, int, int]]]:
        """Return (label, image, box) using original image coordinates with overlap."""
        width, height = image.size
        tiles: List[Tuple[str, Image.Image, Tuple[int, int, int, int]]] = [
            ("full_page", image, (0, 0, width, height))
        ]

        rows, cols = self.tile_grid
        base_w = width / cols
        base_h = height / rows
        ov_w = int(base_w * self.overlap)
        ov_h = int(base_h * self.overlap)

        for r in range(rows):
            for c in range(cols):
                x0 = max(0, int(c * base_w) - ov_w)
                x1 = min(width, int((c + 1) * base_w) + ov_w)
                y0 = max(0, int(r * base_h) - ov_h)
                y1 = min(height, int((r + 1) * base_h) + ov_h)
                box = (x0, y0, x1, y1)
                tiles.append((f"grid_{r}_{c}", image.crop(box), box))

        bands = [
            ("upper_half", (0, 0, width, min(height, int(height * 0.55)))),
            ("middle_half", (0, int(height * 0.22), width, int(height * 0.78))),
            ("lower_half", (0, int(height * 0.45), width, height)),
        ]
        for label, box in bands:
            tiles.append((label, image.crop(box), box))

        return tiles

    def score_candidates(
        self,
        query: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        candidate_ids: Optional[Any] = None,
        visual_index: Optional[Any] = None,
    ) -> List[Dict[str, Any]]:
        cands = candidates if candidates is not None else (candidate_ids if isinstance(candidate_ids, list) else None)
        if cands is None and visual_index is not None:
            cands = visual_index.search(query, top_k=max(top_k * 2, 10))
        if not cands:
            return []

        import torch

        model, processor = _get_colpali()
        inputs_text = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            query_embeddings = model(**inputs_text).to(dtype=torch.float32)

        reranked: List[Dict[str, Any]] = []

        for candidate in cands:
            notice_id = candidate.get("notice_id", candidate.get("doc_id", ""))
            page_number = int(candidate.get("page_number", candidate.get("page", 1)))
            resolved = resolve_page_image(notice_id, page_number, candidate.get("image_path", ""))

            high_res = None
            pdf_path = RAW_DIR / f"{notice_id}.pdf"
            if pdf_path.exists():
                try:
                    import fitz
                    with fitz.open(str(pdf_path)) as doc:
                        if 1 <= page_number <= len(doc):
                            page_obj = doc[page_number - 1]
                            zoom = VISUAL_RERANK_DPI / 72.0
                            mat = fitz.Matrix(zoom, zoom)
                            pix = page_obj.get_pixmap(matrix=mat, alpha=False)
                            high_res = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except Exception as e:
                    print(f"PyMuPDF 216 DPI render error for {pdf_path} p{page_number}: {e}")

            if high_res is None and resolved is not None:
                try:
                    with Image.open(resolved) as src:
                        page_img = src.convert("RGB")
                        scale = VISUAL_RERANK_DPI / 144.0
                        new_size = (max(1, round(page_img.width * scale)), max(1, round(page_img.height * scale)))
                        high_res = page_img.resize(new_size, Image.Resampling.LANCZOS)
                except Exception:
                    high_res = None

            if high_res is None:
                entry = candidate.copy()
                entry["score"] = float(candidate.get("score", 0.0))
                entry["visual_rerank_score"] = float(candidate.get("score", 0.0))
                entry["retrieval_method"] = "visual_reranked"
                entry["visual_evidence_type"] = "stage1_fallback"
                reranked.append(entry)
                continue

            try:
                tiles = self._extract_tiles(high_res)
                tile_imgs = [t[1] for t in tiles]

                # Batched encoding for efficiency
                inputs_img = processor.process_images(tile_imgs).to(model.device)
                with torch.no_grad():
                    tile_embs = model(**inputs_img).to(device=model.device, dtype=torch.float32)

                sims_tensor = processor.score_multi_vector(query_embeddings, tile_embs)[0]
                sims = sims_tensor.cpu().numpy().tolist()

                full_page_sim = float(sims[0])
                local_start = 1
                local_end = len(sims)
                best_local_idx = int(local_start + np.argmax(sims[local_start:local_end])) if local_end > local_start else 0
                max_tile_sim = float(sims[best_local_idx]) if best_local_idx else full_page_sim
                best_label, _, best_box = tiles[best_local_idx]

                combined_score = 0.4 * full_page_sim + 0.6 * max_tile_sim

                entry = candidate.copy()
                entry["score"] = float(combined_score)
                entry["visual_rerank_score"] = float(combined_score)
                entry["full_page_sim"] = round(full_page_sim, 4)
                entry["max_tile_sim"] = round(max_tile_sim, 4)
                entry["best_tile_label"] = best_label
                entry["best_tile_index"] = best_local_idx
                entry["best_tile_box"] = list(best_box)
                entry["rerank_image_width"] = high_res.width
                entry["rerank_image_height"] = high_res.height
                entry["retrieval_method"] = "visual_reranked"
                entry["visual_evidence_type"] = "multi_scale_colpali_216dpi"
                reranked.append(entry)

            except Exception as exc:
                print(f"Visual rerank batch error for {notice_id} p{page_number}: {exc}")
                entry = candidate.copy()
                entry["score"] = float(candidate.get("score", 0.0))
                entry["retrieval_method"] = "visual_reranked"
                entry["visual_evidence_type"] = "stage1_fallback_error"
                reranked.append(entry)

        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for rank, entry in enumerate(reranked, start=1):
            entry["visual_rerank_rank"] = rank

        return reranked[:max(1, int(top_k))]
