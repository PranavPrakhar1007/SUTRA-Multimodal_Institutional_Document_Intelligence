"""Deterministic retrieval benchmark for the NIT Jamshedpur document corpus.

Metrics are separated correctly:
- Positive queries: document/page retrieval metrics only.
- Negative queries: score/rejection signals only; no fake page accuracy.
- MRR is computed over the full evaluated ranking, not mixed with negative queries.

This script evaluates retrieval only. Generated-answer correctness requires a separate
end-to-end evaluation because answer generation depends on the configured API provider.
"""

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.config import (
    EVALUATION_DIR,
    EMBEDDINGS_DIR,
    HYBRID_CANDIDATE_UNION_SIZE,
    HYBRID_RRF_TEXT_WEIGHT,
    HYBRID_RRF_VISUAL_WEIGHT,
    PROJECT_ROOT,
)
from backend.hybrid_reranker import HybridRerankedRetriever
from backend.hybrid_retrieval import HybridRetriever
from backend.text_retrieval import TextIndex
from backend.visual_reranker import VisualReranker
from backend.visual_retrieval import VisualIndex


def norm_doc(value) -> str:
    value = str(value or "").strip().lower()
    return value[:-4] if value.endswith(".pdf") else value


def page_key(result: dict) -> tuple[str, int]:
    return norm_doc(result.get("notice_id", result.get("doc_id", ""))), int(result.get("page_number", result.get("page", 1)))


def evaluate_positive(results: list[dict], expected_doc: str, expected_page: int) -> dict:
    doc_rank = None
    page_rank = None
    for rank, result in enumerate(results, start=1):
        doc, page = page_key(result)
        if doc == expected_doc and doc_rank is None:
            doc_rank = rank
        if doc == expected_doc and page == expected_page and page_rank is None:
            page_rank = rank

    return {
        "doc_rank": doc_rank,
        "page_rank": page_rank,
        "hit1_page": page_rank == 1,
        "hit3_page": page_rank is not None and page_rank <= 3,
        "hit5_page": page_rank is not None and page_rank <= 5,
        "mrr_page": 1.0 / page_rank if page_rank else 0.0,
    }


def main():
    queries_path = EVALUATION_DIR / "queries.json"
    queries = json.loads(queries_path.read_text(encoding="utf-8"))
    print(f"Loaded {len(queries)} queries")

    text = TextIndex()
    visual = VisualIndex()
    text.load(EMBEDDINGS_DIR / "text")
    visual.load(EMBEDDINGS_DIR / "visual")

    visual_reranker = VisualReranker()
    hybrid_rrf = HybridRetriever(text, visual)
    hybrid_reranked = HybridRerankedRetriever(text, visual)

    modes = ["text_baseline", "visual_vlm", "visual_reranked", "hybrid_rrf_baseline", "hybrid_reranked"]

    # Warm all models before timing retrieval.
    init_start = time.perf_counter()
    text.search("warmup query", top_k=1)
    visual.search("warmup query", top_k=1)
    visual_reranker.score_candidates("warmup query", candidate_ids=None, top_k=1)
    hybrid_rrf.search("warmup query", top_k=1)
    hybrid_reranked.search("warmup query", top_k=1)
    warmup_init_ms = round((time.perf_counter() - init_start) * 1000, 2)

    page_count = len(text.entries)
    all_results = []

    for q in queries:
        qid = q["query_id"]
        question = q["question"]
        is_negative = not q.get("expected_document")
        expected_doc = norm_doc(q.get("expected_document"))
        expected_page = int(q.get("expected_page", 0)) if q.get("expected_page") is not None else 0

        q_out = {
            "query_id": qid,
            "question": question,
            "expected_document": f"{expected_doc}.pdf" if expected_doc else None,
            "expected_page": expected_page or None,
            "query_type": q.get("query_type", ""),
            "document_condition": q.get("document_condition", ""),
            "is_negative": is_negative,
            "modes": {},
        }

        for mode in modes:
            start = time.perf_counter()
            if mode == "text_baseline":
                retrieved = text.search(question, top_k=page_count)
            elif mode == "visual_vlm":
                retrieved = visual.search(question, top_k=page_count)
            elif mode == "visual_reranked":
                stage1 = visual.search(question, top_k=HYBRID_CANDIDATE_UNION_SIZE)
                retrieved = visual_reranker.score_candidates(question, stage1, top_k=page_count)
            elif mode == "hybrid_rrf_baseline":
                retrieved = hybrid_rrf.search(
                    question,
                    top_k=page_count,
                    text_weight=HYBRID_RRF_TEXT_WEIGHT,
                    visual_weight=HYBRID_RRF_VISUAL_WEIGHT,
                )
            else:
                retrieved = hybrid_reranked.search(
                    question,
                    top_k=page_count,
                    candidate_union_size=HYBRID_CANDIDATE_UNION_SIZE,
                )
            latency_ms = round((time.perf_counter() - start) * 1000, 2)

            top = retrieved[0] if retrieved else {}
            item = {
                "top_result": {
                    "notice_id": norm_doc(top.get("notice_id", top.get("doc_id", ""))) if top else None,
                    "page_number": int(top.get("page_number", top.get("page", 0))) if top else None,
                    "score": round(float(top.get("score", 0.0)), 6) if top else 0.0,
                },
                "retrieval_time_ms": latency_ms,
            }

            if is_negative:
                item.update({
                    "negative_query": True,
                    "top_score": item["top_result"]["score"],
                    "rejection_signal": item["top_result"]["score"] <= 0.0,
                })
            else:
                item.update(evaluate_positive(retrieved, expected_doc, expected_page))

            q_out["modes"][mode] = item

        all_results.append(q_out)

    positive = [r for r in all_results if not r["is_negative"]]
    negative = [r for r in all_results if r["is_negative"]]
    summary = {}

    for mode in modes:
        n = len(positive)
        doc_top1 = sum(1 for r in positive if r["modes"][mode]["doc_rank"] == 1)
        hit1 = sum(1 for r in positive if r["modes"][mode]["hit1_page"])
        hit3 = sum(1 for r in positive if r["modes"][mode]["hit3_page"])
        hit5 = sum(1 for r in positive if r["modes"][mode]["hit5_page"])
        mrr = sum(r["modes"][mode]["mrr_page"] for r in positive) / n if n else 0.0
        latencies = [r["modes"][mode]["retrieval_time_ms"] for r in all_results]
        negative_zero = sum(1 for r in negative if r["modes"][mode]["rejection_signal"])

        summary[mode] = {
            "positive_query_count": n,
            "negative_query_count": len(negative),
            "doc_top1_accuracy": round(doc_top1 / n, 4) if n else 0.0,
            "page_hit_at_1": round(hit1 / n, 4) if n else 0.0,
            "page_hit_at_3": round(hit3 / n, 4) if n else 0.0,
            "page_hit_at_5": round(hit5 / n, 4) if n else 0.0,
            "page_mrr": round(mrr, 4),
            "avg_retrieval_latency_ms": round(sum(latencies) / len(latencies), 2) if latencies else 0.0,
            "negative_zero_score_rate": round(negative_zero / len(negative), 4) if negative else 0.0,
        }

    output = {
        "benchmark_version": "2026-09-06-retrieval-v2",
        "evaluation_scope": "retrieval_only",
        "total_queries": len(all_results),
        "positive_queries": len(positive),
        "negative_queries": len(negative),
        "corpus_pages": page_count,
        "initialization_warmup_ms": warmup_init_ms,
        "summary_metrics": summary,
        "results": all_results,
    }

    out_path = EVALUATION_DIR / "results.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    # Also save to results/evaluation_results.json for top-level access
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / "evaluation_results.json").write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nBENCHMARK SUMMARY — POSITIVE QUERIES ONLY")
    print("=" * 105)
    print(f"{'Mode':<24} | {'Doc@1':<8} | {'Page@1':<8} | {'Page@3':<8} | {'Page@5':<8} | {'MRR':<8} | {'Avg ms':<10}")
    print("-" * 105)
    for mode in modes:
        s = summary[mode]
        print(
            f"{mode:<24} | {s['doc_top1_accuracy']*100:6.1f}% | {s['page_hit_at_1']*100:6.1f}% | "
            f"{s['page_hit_at_3']*100:6.1f}% | {s['page_hit_at_5']*100:6.1f}% | "
            f"{s['page_mrr']:.4f} | {s['avg_retrieval_latency_ms']:8.2f}"
        )
    print("=" * 105)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
