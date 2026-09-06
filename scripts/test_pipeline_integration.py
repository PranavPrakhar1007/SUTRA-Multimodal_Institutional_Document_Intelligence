"""Automated integration test for SUTRA RAG retrieval and generation pipelines."""

import sys
import unittest
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.config import EMBEDDINGS_DIR, PROCESSED_DIR
from backend.generation import generate_answer
from backend.hybrid_reranker import HybridRerankedRetriever
from backend.hybrid_retrieval import HybridRetriever
from backend.text_retrieval import TextIndex
from backend.visual_reranker import VisualReranker
from backend.visual_retrieval import VisualIndex


class TestPipelineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text_index = TextIndex()
        cls.visual_index = VisualIndex()
        
        text_dir = EMBEDDINGS_DIR / "text"
        visual_dir = EMBEDDINGS_DIR / "visual"
        
        if (text_dir / "dense_embeddings.npy").exists():
            cls.text_index.load(text_dir)
        else:
            cls.text_index.build_from_processed(PROCESSED_DIR)

        if (visual_dir / "colpali_embeddings.npz").exists():
            cls.visual_index.load(visual_dir)
        else:
            cls.visual_index.build_from_processed(PROCESSED_DIR)

        cls.hybrid_baseline = HybridRetriever(cls.text_index, cls.visual_index)
        cls.visual_reranker = VisualReranker()
        cls.hybrid_reranked = HybridRerankedRetriever(cls.text_index, cls.visual_index)

    def test_text_retrieval(self):
        query = "What is the fee payment method?"
        results = self.text_index.search(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("notice_id", results[0])
        self.assertEqual(results[0]["retrieval_method"], "text_rrf")

    def test_visual_retrieval(self):
        query = "What is the fee payment method?"
        results = self.visual_index.search(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertIn("notice_id", results[0])

    def test_visual_reranker(self):
        query = "Who is the Vice-President of Student Council?"
        stage1 = self.visual_index.search(query, top_k=5)
        reranked = self.visual_reranker.score_candidates(query, candidates=stage1, top_k=3)
        self.assertGreater(len(reranked), 0)
        self.assertEqual(reranked[0]["retrieval_method"], "visual_reranked")

    def test_hybrid_reranked(self):
        query = "What is the fellowship amount for JRF candidates?"
        results = self.hybrid_reranked.search(query, top_k=3)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0]["retrieval_method"], "hybrid_reranked")
        self.assertIn("fusion_details", results[0])

    def test_generation_abstention(self):
        empty_result = {"notice_id": "notice_1", "page_number": 1, "text": ""}
        ans = generate_answer("What is the speed of light in vacuum?", empty_result, mode="text")
        self.assertIn("status", ans)
        self.assertIn(ans["status"], ("abstain", "abstained", "error"))


if __name__ == "__main__":
    unittest.main()
