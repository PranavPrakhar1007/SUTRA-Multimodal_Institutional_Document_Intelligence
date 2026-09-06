from typing import Any, Dict, List

from backend.config import HYBRID_CANDIDATE_UNION_SIZE, HYBRID_RRF_K
from backend.text_retrieval import TextIndex
from backend.visual_reranker import VisualReranker
from backend.visual_retrieval import VisualIndex


class HybridRerankedRetriever:
    def __init__(self, text_index: TextIndex, visual_index: VisualIndex):
        self.text_index = text_index
        self.visual_index = visual_index
        self.visual_reranker = VisualReranker()

    @staticmethod
    def _rank_score(rank: int | None, floor: float = 0.0, decay: float = 0.1) -> float:
        if rank is None or rank <= 0:
            return floor
        return 1.0 / (HYBRID_RRF_K + rank * decay)

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
            entry["text_rrf"] = self._rank_score(rank, floor=0.0)
            entry["visual_rrf"] = 0.0
            entry["colpali_score"] = 0.0
            merged[key] = entry

        for rank, c in enumerate(visual_candidates, start=1):
            key = f"{c['notice_id']}_p{c['page_number']}"
            if key in merged:
                merged[key]["visual_rank"] = rank
                merged[key]["visual_rrf"] = self._rank_score(rank, floor=0.0)
                merged[key]["colpali_score"] = float(c.get("score", 0.0))
            else:
                entry = c.copy()
                entry["text_rank"] = None
                entry["visual_rank"] = rank
                entry["text_rrf"] = 0.0
                entry["visual_rrf"] = self._rank_score(rank, floor=0.0)
                entry["colpali_score"] = float(c.get("score", 0.0))
                merged[key] = entry

        candidates = list(merged.values())
        if not candidates:
            return []

        # Invoke fine-grained multi-scale VisualReranker over candidate union
        try:
            reranked_visual = self.visual_reranker.score_candidates(
                query, candidates=candidates, top_k=len(candidates)
            )
            reranked_lookup = {
                f"{r['notice_id']}_p{r['page_number']}": r
                for r in reranked_visual
            }
        except Exception as exc:
            print(f"Hybrid visual reranking fallback ({exc})")
            reranked_lookup = {}

        final_results: List[Dict[str, Any]] = []
        for entry in candidates:
            key = f"{entry['notice_id']}_p{entry['page_number']}"
            v_info = reranked_lookup.get(key, {})

            text_rrf = float(entry.get("text_rrf", 0.0))
            visual_rrf = float(entry.get("visual_rrf", 0.0))
            v_score = float(v_info.get("visual_rerank_score", entry.get("colpali_score", 0.0)))

            # Normalize v_score (typical ColPali scores range ~0..25)
            v_norm = min(1.0, max(0.0, v_score / 25.0)) if v_score > 0 else 0.0

            # Score fusion: text anchor + visual RRF + fine-grained visual score
            if text_rrf > 0 and visual_rrf > 0:
                combined = 0.50 * text_rrf + 0.35 * visual_rrf + 0.15 * v_norm
            elif text_rrf > 0:
                combined = 0.85 * text_rrf
            else:
                combined = 0.70 * visual_rrf + 0.30 * v_norm

            result = entry.copy()
            if v_info:
                for k in ("best_tile_box", "best_tile_label", "visual_evidence_type", "full_page_sim", "max_tile_sim"):
                    if k in v_info:
                        result[k] = v_info[k]

            result["score"] = float(combined)
            result["retrieval_method"] = "hybrid_reranked"
            result["fusion_details"] = {
                "text_rank": entry.get("text_rank"),
                "visual_rank": entry.get("visual_rank"),
                "text_rrf": round(text_rrf, 4),
                "visual_rrf": round(visual_rrf, 4),
                "visual_rerank_score": round(v_score, 4),
                "combined_score": round(combined, 4),
            }
            result["evidence_signal"] = round(combined, 4)
            final_results.append(result)

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[: max(1, int(top_k))]

