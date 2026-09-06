"""Multi-scale ColPali VLM reranking for candidate document pages on GPU.

Evaluates high-resolution full page and region tile crops using ColPali multi-vector
late-interaction patch matching for precise visual document search.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backend.config import VISUAL_RERANK_DPI
from backend.path_utils import resolve_page_image
from backend.visual_retrieval import _get_colpali


class VisualReranker:
    def __init__(self, tile_grid: Tuple[int, int] = (2, 2)):
        self.tile_grid = tile_grid

    def _extract_tiles(self, image: Image.Image) -> List[Tuple[str, Image.Image, Tuple[int, int, int, int]]]:
        """Return (label, image, box) using original image coordinates."""
        width, height = image.size
        tiles: List[Tuple[str, Image.Image, Tuple[int, int, int, int]]] = [
            ("full_page", image, (0, 0, width, height))
        ]

        rows, cols = self.tile_grid
        for r in range(rows):
            for c in range(cols):
                x0 = (c * width) // cols
                x1 = ((c + 1) * width) // cols
                y0 = (r * height) // rows
                y1 = ((r + 1) * height) // rows
                box = (x0, y0, x1, y1)
                tiles.append((f"grid_{r}_{c}", image.crop(box), box))

        bands = [
            ("upper_half", (0, 0, width, height // 2)),
            ("middle_half", (0, height // 4, width, (3 * height) // 4)),
            ("lower_half", (0, height // 2, width, height)),
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
            notice_id = candidate.get("notice_id", "")
            page_number = int(candidate.get("page_number", candidate.get("page", 1)))
            resolved = resolve_page_image(notice_id, page_number, candidate.get("image_path", ""))

            if resolved is None:
                entry = candidate.copy()
                entry["visual_rerank_score"] = float(candidate.get("score", 0.0))
                entry["retrieval_method"] = "visual_reranked"
                entry["visual_evidence_type"] = "stage1_fallback"
                reranked.append(entry)
                continue

            try:
                with Image.open(resolved) as src:
                    page_img = src.convert("RGB")
                    if VISUAL_RERANK_DPI > 144:
                        scale = VISUAL_RERANK_DPI / 144.0
                        new_size = (max(1, round(page_img.width * scale)), max(1, round(page_img.height * scale)))
                        high_res = page_img.resize(new_size, Image.Resampling.LANCZOS)
                    else:
                        high_res = page_img

                tiles = self._extract_tiles(high_res)
                sims = []
                for _, tile_img, _ in tiles:
                    inputs_img = processor.process_images([tile_img]).to(model.device)
                    with torch.no_grad():
                        tile_emb = model(**inputs_img)[0].to(device=model.device, dtype=torch.float32)
                    score_val = processor.score_multi_vector(query_embeddings, [tile_emb])[0][0].cpu().item()
                    sims.append(float(score_val))

                full_page_sim = float(sims[0])
                local_start = 1
                local_end = len(sims)
                best_local_idx = int(local_start + np.argmax(sims[local_start:local_end])) if local_end > local_start else 0
                max_tile_sim = float(sims[best_local_idx]) if best_local_idx else full_page_sim
                best_label, _, best_box = tiles[best_local_idx]

                scale = high_res.width / page_img.width if page_img.width else 1.0
                original_box = tuple(int(round(v / scale)) for v in best_box) if scale != 1.0 else best_box

                combined_score = 0.4 * full_page_sim + 0.6 * max_tile_sim

                entry = candidate.copy()
                entry["score"] = float(combined_score)
                entry["visual_rerank_score"] = float(combined_score)
                entry["full_page_sim"] = round(full_page_sim, 4)
                entry["max_tile_sim"] = round(max_tile_sim, 4)
                entry["best_tile_label"] = best_label
                entry["best_tile_index"] = best_local_idx
                entry["best_tile_box"] = list(original_box)
                entry["rerank_image_width"] = high_res.width
                entry["rerank_image_height"] = high_res.height
                entry["retrieval_method"] = "visual_reranked"
                entry["visual_evidence_type"] = "multi_scale_colpali"
                reranked.append(entry)

            except Exception as exc:
                print(f"Visual rerank error for {resolved}: {exc}")
                entry = candidate.copy()
                entry["retrieval_method"] = "visual_reranked"
                entry["visual_evidence_type"] = "stage1_fallback_error"
                reranked.append(entry)

        reranked.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        for rank, entry in enumerate(reranked, start=1):
            entry["visual_rerank_rank"] = rank

        return reranked[:max(1, int(top_k))]
