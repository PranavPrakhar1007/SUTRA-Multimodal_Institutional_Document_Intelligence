"""Rebuild text and visual retrieval indices from the processed corpus."""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.config import EMBEDDINGS_DIR, PROCESSED_DIR
from backend.hybrid_reranker import HybridRerankedRetriever
from backend.hybrid_retrieval import HybridRetriever
from backend.text_retrieval import TextIndex
from backend.visual_reranker import VisualReranker
from backend.visual_retrieval import VisualIndex


def main():
    start = time.perf_counter()

    print("--- Building Text Index ---")
    text = TextIndex()
    text.build_from_processed(PROCESSED_DIR)
    text_dir = EMBEDDINGS_DIR / "text"
    text.save(text_dir)

    print("\n--- Building Visual Index ---")
    visual = VisualIndex()
    visual.build_from_processed(PROCESSED_DIR)
    visual_dir = EMBEDDINGS_DIR / "visual"
    visual.save(visual_dir)

    print("\n--- Verifying Index Persistence ---")
    text_loaded = TextIndex()
    text_loaded.load(text_dir)
    visual_loaded = VisualIndex()
    visual_loaded.load(visual_dir)
    print(f"Verified disk load: {len(text_loaded.entries)} text pages, {len(visual_loaded.entries)} visual pages.")

    print("\n--- Smoke Testing All 5 Retrieval Modes ---")
    query = "What is the fee payment method?"
    hybrid_baseline = HybridRetriever(text_loaded, visual_loaded)
    hybrid_reranked = HybridRerankedRetriever(text_loaded, visual_loaded)
    visual_reranker = VisualReranker()

    for label, results in (
        ("Text Baseline", text_loaded.search(query, top_k=3)),
        ("Visual VLM", visual_loaded.search(query, top_k=3)),
        ("Hybrid RRF Baseline", hybrid_baseline.search(query, top_k=3)),
        ("Visual Reranked", visual_reranker.score_candidates(query, candidate_ids=None, top_k=3)),
        ("Hybrid Reranked", hybrid_reranked.search(query, top_k=3)),
    ):
        print(f"\nMode: {label}")
        for r in results:
            nid = r.get("notice_id", r.get("doc_id", "unknown"))
            pnum = r.get("page_number", r.get("page", 1))
            score = r.get("score", 0.0)
            print(f"  {nid} p{pnum} score={score:.4f}")

    print(f"\nDone in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
