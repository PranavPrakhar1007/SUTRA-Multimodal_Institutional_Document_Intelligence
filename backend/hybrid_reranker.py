"""Improved Hybrid retrieval using candidate union + ColPali VLM visual evidence.

Combines text (BM25 + Dense) candidate recall with ColPali VLM multi-vector
late-interaction patch matching scores for ultra-fast, high-precision retrieval on GPU.
"""

from typing import Any, Dict, List

from backend.config import HYBRID_CANDIDATE_UNION_SIZE
from backend.text_retrieval import TextIndex
from backend.visual_retrieval import VisualIndex


class HybridRerankedRetriever:
    def __init__(self, text_index: TextIndex, visual_index: VisualIndex):
        self.text_index = text_index
        self.visual_index = visual_index

    @staticmethod
    def _rank_score(rank: int | None, floor: float = 0.0, decay: float = 0.1) -> float:
        if rank is None:
            return floor
        return 1.0 / (10.0 + rank * decay)

    def search(
        self,
        query: str,
        top_k: int = 5,
        candidate_union_size: int | None = None,
    ) -> List[Dict[str, Any]]:
        union_size = HYBRID_CANDIDATE_UNION_SIZE if candidate_union_size is None else max(1, int(candidate_union_size))

        text_candidates = self.text_index.search(query, top_k=union_size)
        visual_candidates = self.visual_index.search(query, top_k=union_size)

        merged: Dict[str, Dict[str, Any]] = {}
        for rank, c in enumerate(text_candidates, start=1):
            key = f"{c['notice_id']}_p{c['page_number']}"
            entry = c.copy()
            entry["text_rank"] = rank
            entry["visual_rank"] = None
            entry["text_rank_score"] = self._rank_score(rank, floor=0.1, decay=0.1)
            entry["visual_rank_score"] = 0.0
            merged[key] = entry

        for rank, c in enumerate(visual_candidates, start=1):
            key = f"{c['notice_id']}_p{c['page_number']}"
            if key in merged:
                merged[key]["visual_rank"] = rank
                merged[key]["visual_rank_score"] = self._rank_score(rank, floor=0.1, decay=0.1)
                merged[key]["colpali_score"] = float(c.get("score", 0.0))
            else:
                entry = c.copy()
                entry["text_rank"] = None
                entry["visual_rank"] = rank
                entry["text_rank_score"] = 0.0
                entry["visual_rank_score"] = self._rank_score(rank, floor=0.1, decay=0.1)
                entry["colpali_score"] = float(c.get("score", 0.0))
                merged[key] = entry

        candidates = list(merged.values())
        
        # Sort candidate union by visual rank (ColPali VLM multi-vector match)
        visual_scored = sorted(
            candidates,
            key=lambda x: (x.get("visual_rank") if x.get("visual_rank") is not None else 999, -(x.get("colpali_score", 0.0)))
        )

        final_results: List[Dict[str, Any]] = []
        for visual_rank, entry in enumerate(visual_scored, start=1):
            text_score = float(entry.get("text_rank_score", 0.0))
            initial_visual_score = float(entry.get("visual_rank_score", 0.0))
            local_visual_rank_score = self._rank_score(visual_rank, floor=0.1, decay=0.1)

            # Text remains the anchor; ColPali VLM visual evidence contributes where available.
            if text_score > 0:
                combined = 0.75 * text_score + 0.20 * local_visual_rank_score + 0.05 * initial_visual_score
            else:
                # Visual-only candidates cannot overwhelm text-anchored candidates.
                combined = 0.35 * local_visual_rank_score + 0.15 * initial_visual_score

            result = entry.copy()
            result["score"] = float(combined)
            result["retrieval_method"] = "hybrid_reranked"
            result["fusion_details"] = {
                "text_rank": entry.get("text_rank"),
                "visual_rank": entry.get("visual_rank"),
                "text_rank_score": round(text_score, 4),
                "initial_visual_rank_score": round(initial_visual_score, 4),
                "visual_rerank_rank": visual_rank,
                "visual_rerank_rank_score": round(local_visual_rank_score, 4),
                "combined_score": round(combined, 4),
            }
            result["evidence_signal"] = round(combined, 4)
            final_results.append(result)

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[: max(1, int(top_k))]

