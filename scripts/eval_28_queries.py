import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.config import EVALUATION_DIR, EMBEDDINGS_DIR, HYBRID_RRF_TEXT_WEIGHT, HYBRID_RRF_VISUAL_WEIGHT
from backend.hybrid_retrieval import HybridRetriever
from backend.text_retrieval import TextIndex
from backend.visual_retrieval import VisualIndex

def norm_doc(value) -> str:
    value = str(value or "").strip().lower()
    return value[:-4] if value.endswith(".pdf") else value

def page_key(result: dict) -> tuple[str, int]:
    return norm_doc(result.get("notice_id", result.get("doc_id", ""))), int(result.get("page_number", result.get("page", 1)))

def main():
    queries_path = EVALUATION_DIR / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} evaluation queries from queries.json.")

    text = TextIndex()
    visual = VisualIndex()
    text.load(EMBEDDINGS_DIR / "text")
    visual.load(EMBEDDINGS_DIR / "visual")

    hybrid_rrf = HybridRetriever(text, visual)

    modes = ["text_baseline", "visual_colpali_vlm", "hybrid_rrf_baseline"]
    page_count = len(text.entries)

    # Warmup
    text.search("warmup query", top_k=1)
    visual.search("warmup query", top_k=1)
    hybrid_rrf.search("warmup query", top_k=1)

    all_results = []
    summary_metrics = {
        m: {
            "positive_query_count": 0,
            "negative_query_count": 0,
            "doc_top1_hits": 0,
            "page_hit_at_1_count": 0,
            "page_hit_at_3_count": 0,
            "page_hit_at_5_count": 0,
            "mrr_sum": 0.0,
            "total_latency_ms": 0.0,
        }
        for m in modes
    }

    pos_count = 0
    neg_count = 0

    for q in queries:
        qid = q["query_id"]
        question = q["question"]
        is_negative = not q.get("expected_document")
        expected_doc = norm_doc(q.get("expected_document"))
        expected_page = int(q.get("expected_page", 0)) if q.get("expected_page") is not None else 0

        if is_negative:
            neg_count += 1
        else:
            pos_count += 1

        q_out = {
            "query_id": qid,
            "question": question,
            "expected_document": f"{expected_doc}.pdf" if expected_doc else None,
            "expected_page": expected_page or None,
            "is_negative": is_negative,
            "modes": {},
        }

        for mode in modes:
            start = time.perf_counter()
            if mode == "text_baseline":
                retrieved = text.search(question, top_k=page_count)
            elif mode == "visual_colpali_vlm":
                retrieved = visual.search(question, top_k=page_count)
            elif mode == "hybrid_rrf_baseline":
                retrieved = hybrid_rrf.search(
                    question,
                    top_k=page_count,
                    text_weight=HYBRID_RRF_TEXT_WEIGHT,
                    visual_weight=HYBRID_RRF_VISUAL_WEIGHT,
                )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            top_result = retrieved[0] if retrieved else None

            if is_negative:
                q_out["modes"][mode] = {
                    "top_result": top_result,
                    "retrieval_time_ms": latency_ms,
                }
                summary_metrics[mode]["negative_query_count"] += 1
                summary_metrics[mode]["total_latency_ms"] += latency_ms
            else:
                doc_rank = None
                page_rank = None
                for rank, res in enumerate(retrieved, start=1):
                    d, p = page_key(res)
                    if d == expected_doc and doc_rank is None:
                        doc_rank = rank
                    if d == expected_doc and p == expected_page and page_rank is None:
                        page_rank = rank

                hit1 = page_rank == 1
                hit3 = page_rank is not None and page_rank <= 3
                hit5 = page_rank is not None and page_rank <= 5
                mrr = 1.0 / page_rank if page_rank else 0.0

                q_out["modes"][mode] = {
                    "top_result": top_result,
                    "correct_document_retrieved": doc_rank == 1,
                    "correct_page_retrieved": hit1,
                    "doc_rank": doc_rank,
                    "page_rank": page_rank,
                    "hit1_page": hit1,
                    "hit3_page": hit3,
                    "hit5_page": hit5,
                    "mrr_page": round(mrr, 4),
                    "retrieval_time_ms": latency_ms,
                }

                m_data = summary_metrics[mode]
                m_data["positive_query_count"] += 1
                if doc_rank == 1:
                    m_data["doc_top1_hits"] += 1
                if hit1:
                    m_data["page_hit_at_1_count"] += 1
                if hit3:
                    m_data["page_hit_at_3_count"] += 1
                if hit5:
                    m_data["page_hit_at_5_count"] += 1
                m_data["mrr_sum"] += mrr
                m_data["total_latency_ms"] += latency_ms

        all_results.append(q_out)

    final_summary = {}
    for mode, m_data in summary_metrics.items():
        n_pos = m_data["positive_query_count"]
        n_total = n_pos + m_data["negative_query_count"]
        final_summary[mode] = {
            "positive_query_count": n_pos,
            "negative_query_count": m_data["negative_query_count"],
            "doc_top1_accuracy": round(m_data["doc_top1_hits"] / n_pos, 4) if n_pos > 0 else 0.0,
            "page_hit_at_1": round(m_data["page_hit_at_1_count"] / n_pos, 4) if n_pos > 0 else 0.0,
            "page_hit_at_3": round(m_data["page_hit_at_3_count"] / n_pos, 4) if n_pos > 0 else 0.0,
            "page_hit_at_5": round(m_data["page_hit_at_5_count"] / n_pos, 4) if n_pos > 0 else 0.0,
            "page_mrr": round(m_data["mrr_sum"] / n_pos, 4) if n_pos > 0 else 0.0,
            "avg_retrieval_latency_ms": round(m_data["total_latency_ms"] / n_total, 2) if n_total > 0 else 0.0,
        }

    output = {
        "benchmark_version": "2026-09-07-colqwen2-28queries",
        "evaluation_scope": "retrieval_only",
        "total_queries": len(queries),
        "positive_queries": pos_count,
        "negative_queries": neg_count,
        "corpus_pages": page_count,
        "summary_metrics": final_summary,
        "results": all_results,
    }

    res_file = EVALUATION_DIR / "results.json"
    res_file.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print("SUCCESS: Wrote updated evaluation results to data/evaluation/results.json")

if __name__ == "__main__":
    main()
