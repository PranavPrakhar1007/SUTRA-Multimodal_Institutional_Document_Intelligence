"""Multi-scale ColPali VLM reranking for candidate document pages on GPU.

Evaluates high-resolution full page and region tile crops using ColPali multi-vector
late-interaction patch matching for precise visual document search.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from backend.config import DOCUMENTS_DIR, VISUAL_RERANK_DPI
from backend.path_utils import resolve_page_image
from backend.visual_retrieval import _get_colpali


class VisualReranker:
    def __init__(self, tile_grid: Tuple[int, int] = (2, 2), overlap: float = 0.2):
        self.tile_grid = tile_grid
        self.overlap = overlap
        self._render_cache: Dict[Tuple[str, int, int, int, int], Image.Image] = {}

    def _page_key(self, page_path: str, page_number: int, dpi: int, width: int, height: int) -> Tuple[str, int, int, int, int]:
        return (str(page_path), int(page_number), int(dpi), int(width), int(height))

    def _render_page_image(self, notice_id: str, page_number: int, resolved: Optional[str] = None) -> Optional[Image.Image]:
        """Return a cached high-resolution page render for the same PDF/image across repeated queries."""
        pdf_path = DOCUMENTS_DIR / f"{notice_id}.pdf"
        cache_key = None

        if pdf_path.exists():
            try:
                import os
                import pymupdf as fitz
                stat = pdf_path.stat()
                cache_key = self._page_key(str(pdf_path), page_number, VISUAL_RERANK_DPI, int(stat.st_size), int(stat.st_mtime_ns))
                if cache_key in self._render_cache:
                    return self._render_cache[cache_key]
                with fitz.open(str(pdf_path)) as doc:
                    if 1 <= page_number <= len(doc):
                        page_obj = doc[page_number - 1]
                        zoom = VISUAL_RERANK_DPI / 72.0
                        mat = fitz.Matrix(zoom, zoom)
                        pix = page_obj.get_pixmap(matrix=mat, alpha=False)
                        image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                        self._render_cache[cache_key] = image
                        return image
            except Exception:
                pass

        if resolved is not None:
            try:
                stat = Path(resolved).stat()
                cache_key = self._page_key(str(resolved), page_number, VISUAL_RERANK_DPI, int(stat.st_size), int(stat.st_mtime_ns))
                if cache_key in self._render_cache:
                    return self._render_cache[cache_key]
                with Image.open(resolved) as src:
                    page_img = src.convert("RGB")
                    scale = VISUAL_RERANK_DPI / 144.0
                    new_size = (max(1, round(page_img.width * scale)), max(1, round(page_img.height * scale)))
                    image = page_img.resize(new_size, Image.Resampling.LANCZOS)
                    self._render_cache[cache_key] = image
                    return image
            except Exception:
                return None
        return None

    def _extract_tiles(self, image: Image.Image) -> List[Tuple[str, Image.Image, Tuple[int, int, int, int]]]:
        """Return general-purpose region crops that preserve document structure better than fixed quadrants alone."""
        width, height = image.size
        rows, columns = self.tile_grid
        rows = max(1, int(rows))
        columns = max(1, int(columns))
        tiles: List[Tuple[str, Image.Image, Tuple[int, int, int, int]]] = [("full_page", image, (0, 0, width, height))]
        tile_width = width / columns
        tile_height = height / rows

        # Keep the first pass coarse, but also include horizontal and vertical bands so
        # table headers and row labels remain in focus on dense document pages.
        for row in range(rows):
            for column in range(columns):
                x0 = max(0, int((column - self.overlap / 2) * tile_width))
                y0 = max(0, int((row - self.overlap / 2) * tile_height))
                x1 = min(width, int((column + 1 + self.overlap / 2) * tile_width))
                y1 = min(height, int((row + 1 + self.overlap / 2) * tile_height))
                box = (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))
                label = f"grid_r{row + 1}_c{column + 1}"
                tiles.append((label, image.crop(box), box))

        if rows >= 2:
            band_h = max(1, height // 3)
            for index, y0 in enumerate((0, height // 2 - band_h // 2, max(0, height - band_h))):
                box = (0, y0, width, min(height, y0 + band_h))
                label = f"band_h{index + 1}"
                tiles.append((label, image.crop(box), box))

        if columns >= 2:
            band_w = max(1, width // 3)
            for index, x0 in enumerate((0, width // 2 - band_w // 2, max(0, width - band_w))):
                box = (x0, 0, min(width, x0 + band_w), height)
                label = f"band_v{index + 1}"
                tiles.append((label, image.crop(box), box))

        # Deduplicate exact same crop box while preserving the first-seen label.
        seen = set()
        unique_tiles: List[Tuple[str, Image.Image, Tuple[int, int, int, int]]] = []
        for label, tile_img, box in tiles:
            key = tuple(box)
            if key in seen:
                continue
            seen.add(key)
            unique_tiles.append((label, tile_img, box))
        return unique_tiles

    def score_candidates(
        self,
        query: str,
        candidates: Optional[List[Dict[str, Any]]] = None,
        top_k: int = 5,
        candidate_ids: Optional[Any] = None,
        visual_index: Optional[Any] = None,
        cancel_event: Optional[Any] = None,
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
            if cancel_event is not None and cancel_event.is_set():
                break
            notice_id = candidate.get("notice_id", candidate.get("doc_id", ""))
            page_number = int(candidate.get("page_number", candidate.get("page", 1)))
            resolved = resolve_page_image(notice_id, page_number, candidate.get("image_path", ""))
            high_res = self._render_page_image(notice_id, page_number, resolved)
            if high_res is None:
                try:
                    import pymupdf as fitz
                    pdf_path = DOCUMENTS_DIR / f"{notice_id}.pdf"
                    if pdf_path.exists():
                        with fitz.open(str(pdf_path)) as doc:
                            if 1 <= page_number <= len(doc):
                                page_obj = doc[page_number - 1]
                                zoom = VISUAL_RERANK_DPI / 72.0
                                mat = fitz.Matrix(zoom, zoom)
                                pix = page_obj.get_pixmap(matrix=mat, alpha=False)
                                high_res = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                except Exception as e:
                    print(f"PyMuPDF 216 DPI render error for {pdf_path} p{page_number}: {e}")

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
