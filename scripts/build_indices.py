"""Rebuild text and visual retrieval indices from the processed corpus."""

import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.config import EMBEDDINGS_DIR, PROCESSED_DIR
from backend.hybrid_retrieval import HybridRetriever
from backend.text_retrieval import TextIndex
from backend.visual_retrieval import VisualIndex


def main():
    start = time.perf_counter()

    print("--- Building Text Index ---")
    text = TextIndex()
    text.build_from_processed(PROCESSED_DIR)
    text.save(EMBEDDINGS_DIR / "text")

    print("\n--- Building Visual Index ---")
    visual = VisualIndex()
    visual.build_from_processed(PROCESSED_DIR)
    visual.save(EMBEDDINGS_DIR / "visual")

    print("\n--- Smoke Test ---")
    query = "What is the fee payment method?"
    for label, results in (
        ("Text", text.search(query, top_k=3)),
        ("Visual", visual.search(query, top_k=3)),
        ("Hybrid RRF", HybridRetriever(text, visual).search(query, top_k=3)),
    ):
        print(label)
        for r in results:
            print(f"  {r['notice_id']} p{r['page_number']} score={r['score']:.4f}")

    print(f"\nDone in {time.perf_counter() - start:.1f}s")


if __name__ == "__main__":
    main()
