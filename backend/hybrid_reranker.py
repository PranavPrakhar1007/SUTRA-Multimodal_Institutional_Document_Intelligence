from typing import Any, Dict, List

from backend.config import HYBRID_CANDIDATE_UNION_SIZE, HYBRID_RRF_K, HYBRID_VISUAL_RERANK_SIZE
from backend.text_retrieval import TextIndex
from backend.visual_reranker import VisualReranker
from backend.visual_retrieval import VisualIndex
from backend.query_constraints import constrain_candidates


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
        cancel_event: Any | None = None,
    ) -> List[Dict[str, Any]]:
        union_size = HYBRID_CANDIDATE_UNION_SIZE if candidate_union_size is None else max(1, int(candidate_union_size))

        text_candidates = self.text_index.search(query, top_k=union_size)
        if cancel_event is not None and cancel_event.is_set():
            return []
        visual_candidates = self.visual_index.search(query, top_k=union_size)
        if cancel_event is not None and cancel_event.is_set():
            return []

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

        candidates, constraints = constrain_candidates(candidates, query)

        # Sort candidates by preliminary combined score before fine-grained visual reranking
        if constraints.is_structured_lookup:
            candidates.sort(
                key=lambda x: (
                    x.get("visual_rank") is not None,
                    -int(x.get("visual_rank") or 10**9),
                    float(x.get("colpali_score", 0.0)),
                ),
                reverse=True,
            )
        else:
            candidates.sort(
                key=lambda x: 0.5 * float(x.get("text_rrf", 0.0)) + 0.5 * float(x.get("visual_rrf", 0.0)) + 0.1 * float(x.get("colpali_score", 0.0)),
                reverse=True,
            )

        # Rerank the complete candidate union.  Structured table questions are
        # often absent from OCR, so truncating before visual reranking can
        # permanently discard the only page containing the answer.
        top_candidates_to_rerank = candidates[:HYBRID_VISUAL_RERANK_SIZE]
        try:
            reranked_visual = self.visual_reranker.score_candidates(
                query,
                candidates=top_candidates_to_rerank,
                top_k=len(top_candidates_to_rerank),
                cancel_event=cancel_event,
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

            # Score fusion with visual dominance scaling when visual MaxSim alignment is strong (>10.0)
            if v_score > 10.0:
                combined = 0.60 * v_norm + 0.25 * visual_rrf + 0.15 * text_rrf
            elif text_rrf > 0 and visual_rrf > 0:
                combined = 0.40 * text_rrf + 0.40 * visual_rrf + 0.20 * v_norm
            elif text_rrf > 0:
                combined = 0.70 * text_rrf + 0.30 * v_norm
            else:
                combined = 0.60 * visual_rrf + 0.40 * v_norm

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
                "constraint_matches": bool(constraints.has_explicit_constraints),
            }
            result["evidence_signal"] = round(combined, 4)
            final_results.append(result)

        final_results.sort(key=lambda x: x["score"], reverse=True)
        return final_results[: max(1, int(top_k))]

