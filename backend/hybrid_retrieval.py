"""Naive weighted-RRF Hybrid baseline.

This module is intentionally preserved as a baseline so the improved Hybrid Reranked
pipeline can be compared against the simple fusion approach.
"""

from backend.config import HYBRID_RRF_K, HYBRID_RRF_TEXT_WEIGHT, HYBRID_RRF_VISUAL_WEIGHT
from backend.text_retrieval import TextIndex
from backend.visual_retrieval import VisualIndex


class HybridRetriever:
    def __init__(self, text_index: TextIndex, visual_index: VisualIndex):
        self.text_index = text_index
        self.visual_index = visual_index

    def search(
        self,
        query: str,
        top_k: int = 5,
        text_weight: float | None = None,
        visual_weight: float | None = None,
        rrf_k: int | None = None,
        candidate_k: int | None = None,
    ) -> list[dict]:
        text_weight = HYBRID_RRF_TEXT_WEIGHT if text_weight is None else float(text_weight)
        visual_weight = HYBRID_RRF_VISUAL_WEIGHT if visual_weight is None else float(visual_weight)
        rrf_k = HYBRID_RRF_K if rrf_k is None else int(rrf_k)
        candidate_k = max(int(top_k) * 2, 10) if candidate_k is None else int(candidate_k)

        if text_weight < 0 or visual_weight < 0 or text_weight + visual_weight <= 0:
            raise ValueError("Hybrid weights must be non-negative and not both zero")

        text_results = self.text_index.search(query, top_k=candidate_k)
        visual_results = self.visual_index.search(query, top_k=candidate_k)

        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}
        source_info: dict[str, dict] = {}

        for rank, result in enumerate(text_results, start=1):
            key = f"{result['notice_id']}_p{result['page_number']}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + text_weight / (rrf_k + rank)
            result_map.setdefault(key, result.copy())
            info = source_info.setdefault(key, {"text_rank": None, "visual_rank": None, "text_score": 0.0, "visual_score": 0.0})
            info["text_rank"] = rank
            info["text_score"] = float(result.get("score", 0.0))

        for rank, result in enumerate(visual_results, start=1):
            key = f"{result['notice_id']}_p{result['page_number']}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + visual_weight / (rrf_k + rank)
            result_map.setdefault(key, result.copy())
            info = source_info.setdefault(key, {"text_rank": None, "visual_rank": None, "text_score": 0.0, "visual_score": 0.0})
            info["visual_rank"] = rank
            info["visual_score"] = float(result.get("score", 0.0))

        ordered = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[: max(1, int(top_k))]
        results = []
        for key in ordered:
            entry = result_map[key].copy()
            entry["score"] = float(rrf_scores[key])
            entry["retrieval_method"] = "hybrid_rrf"
            entry["fusion_details"] = {
                **source_info[key],
                "text_weight": text_weight,
                "visual_weight": visual_weight,
                "rrf_k": rrf_k,
            }
            results.append(entry)
        return results
