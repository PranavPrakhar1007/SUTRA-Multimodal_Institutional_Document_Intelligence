"""Text retrieval using dense embeddings + BM25 + RRF."""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np

from backend.config import (
    METADATA_DIR,
    TEXT_BM25_DOMINANCE_BONUS,
    TEXT_BM25_DOMINANCE_MIN_SCORE,
    TEXT_BM25_DOMINANCE_RATIO,
    TEXT_DIR,
    TEXT_EMBEDDING_MODEL,
)
from backend.path_utils import resolve_page_image

_sentence_model = None


def _get_sentence_model():
    global _sentence_model
    if _sentence_model is None:
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading text embedding model: {TEXT_EMBEDDING_MODEL} on {device}...")
        _sentence_model = SentenceTransformer(TEXT_EMBEDDING_MODEL, device=device)
        print(f"Text embedding model loaded on {device}.")
    return _sentence_model


class TextIndex:
    def __init__(self):
        self.entries: list[dict] = []
        self.dense_embeddings: Optional[np.ndarray] = None
        self.bm25 = None
        self._tokenized_corpus: list[list[str]] = []

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", (text or "").lower())

    @classmethod
    def _clean_text(cls, text: str) -> str:
        if not text:
            return ""
        lines = []
        for line in text.split("\n"):
            line_clean = line.strip()
            line_clean = re.sub(r'^[I|lJ]+(?=[A-Za-z]{2,})', '', line_clean)
            line_clean = re.sub(r'\bIPG REPRESENTATIVE\b', 'PG REPRESENTATIVE', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bIPRESIDENT\b', 'PRESIDENT', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bIICE-PRESIDENT\b', 'VICE-PRESIDENT', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bCE-PRESIDENT\b', 'VICE-PRESIDENT', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bITECHNICAL\b', 'TECHNICAL', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bICLUB\b', 'CLUB', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bJJONNT\b', 'JOINT', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bOINT CULTURAL SECRETARY\b', 'JOINT CULTURAL SECRETARY', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bONNT CULTURAL SECRETARY\b', 'JOINT CULTURAL SECRETARY', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bOINT TECHNICAL SECRETARY\b', 'JOINT TECHNICAL SECRETARY', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bOINT SPORTS SECRETARY\b', 'JOINT SPORTS SECRETARY', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bOINT ALUMNI SECRETARY\b', 'JOINT ALUMNI SECRETARY', line_clean, flags=re.IGNORECASE)
            line_clean = re.sub(r'\bOINT SECRETARY\b', 'JOINT SECRETARY', line_clean, flags=re.IGNORECASE)
            lines.append(line_clean)
        return "\n".join(lines)


    @classmethod
    def _tokenize_query(cls, text: str) -> list[str]:
        tokens = cls._tokenize(text)
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "shall",
            "should", "can", "could", "may", "might", "must", "of", "for", "to",
            "in", "on", "at", "by", "from", "with", "about", "against", "between",
            "into", "through", "during", "before", "after", "above", "below",
            "up", "down", "out", "off", "over", "under", "again", "further",
            "then", "once", "who", "whom", "this", "that", "these", "those",
            "am", "what", "which", "whose", "where", "when", "why", "how"
        }
        filtered = [t for t in tokens if t not in stop_words and len(t) > 1]
        return filtered if filtered else tokens

    def build_from_processed(self, processed_dir: Path):
        del processed_dir
        self.entries = []
        texts = []

        metadata_files = sorted(METADATA_DIR.glob("notice_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        for meta_file in metadata_files:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            notice_id = meta["notice_id"]
            for page_info in meta.get("pages", []):
                page_num = int(page_info["page_number"])
                path = TEXT_DIR / notice_id / f"page_{page_num}.txt"
                text = self._clean_text(path.read_text(encoding="utf-8").strip()) if path.exists() else ""
                image_path = resolve_page_image(notice_id, page_num, page_info.get("image_path", ""))
                entry = {
                    "notice_id": notice_id,
                    "doc_id": notice_id,
                    "filename": f"{notice_id}.pdf",
                    "file_name": f"{notice_id}.pdf",
                    "page_number": page_num,
                    "page": page_num,
                    "text": text,
                    "has_text": len(text) > 10,
                    "text_length": len(text),
                    "image_path": str(image_path) if image_path else "",
                }
                self.entries.append(entry)
                texts.append(text)

        print(f"Text index: {len(self.entries)} pages loaded")
        print(f"  Pages with text: {sum(1 for e in self.entries if e['has_text'])}")
        print(f"  Pages without text: {sum(1 for e in self.entries if not e['has_text'])}")

        model = _get_sentence_model()
        embedding_inputs = [t if t else "" for t in texts]
        self.dense_embeddings = np.asarray(
            model.encode(embedding_inputs, show_progress_bar=True, normalize_embeddings=True), dtype=np.float32
        )

        from rank_bm25 import BM25Okapi
        self._tokenized_corpus = [self._tokenize(t) for t in texts]
        # BM25 accepts empty token lists; unlike the old implementation, do not inject synthetic words.
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"Dense embeddings: {self.dense_embeddings.shape}")
        print("BM25 index built.")

    def search_dense(self, query: str, top_k: int = 5) -> list[dict]:
        if self.dense_embeddings is None or not self.entries:
            return []
        model = _get_sentence_model()
        q = model.encode([query], normalize_embeddings=True)
        sims = np.dot(self.dense_embeddings, q.T).flatten()
        top_k = min(max(1, int(top_k)), len(self.entries))
        idxs = np.argsort(-sims, kind="stable")[:top_k]
        return [dict(self.entries[int(i)], score=float(sims[i]), retrieval_method="dense") for i in idxs]

    def search_bm25(self, query: str, top_k: int = 5) -> list[dict]:
        if self.bm25 is None or not self.entries:
            return []
        scores = self.bm25.get_scores(self._tokenize_query(query))
        top_k = min(max(1, int(top_k)), len(self.entries))
        idxs = np.argsort(-scores, kind="stable")[:top_k]
        return [dict(self.entries[int(i)], score=float(scores[i]), retrieval_method="bm25") for i in idxs if scores[i] > 0]

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        top_k = min(max(1, int(top_k)), len(self.entries)) if self.entries else 0
        if top_k == 0:
            return []

        dense_results = self.search_dense(query, top_k=min(len(self.entries), top_k * 2))
        bm25_results = self.search_bm25(query, top_k=min(len(self.entries), top_k * 2))
        k = 60
        rrf_scores: dict[str, float] = {}
        result_map: dict[str, dict] = {}

        for rank, result in enumerate(dense_results, start=1):
            key = f"{result['notice_id']}_p{result['page_number']}"
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (k + rank)
            result_map.setdefault(key, result)

        dominant_key = None
        if len(bm25_results) >= 2:
            top1 = bm25_results[0]["score"]
            top2 = bm25_results[1]["score"]
            if top2 > 0 and top1 / top2 >= TEXT_BM25_DOMINANCE_RATIO and top1 > TEXT_BM25_DOMINANCE_MIN_SCORE:
                dominant_key = f"{bm25_results[0]['notice_id']}_p{bm25_results[0]['page_number']}"

        for rank, result in enumerate(bm25_results, start=1):
            key = f"{result['notice_id']}_p{result['page_number']}"
            score = 1.0 / (k + rank)
            if key == dominant_key:
                score += TEXT_BM25_DOMINANCE_BONUS
            rrf_scores[key] = rrf_scores.get(key, 0.0) + score
            result_map.setdefault(key, result)

        query_words = self._tokenize_query(query)
        if len(query_words) >= 2:
            for key, result in result_map.items():
                page_text_lower = (result.get("text") or "").lower()
                for n in range(2, min(4, len(query_words) + 1)):
                    for i in range(len(query_words) - n + 1):
                        phrase = " ".join(query_words[i:i+n])
                        if phrase in page_text_lower:
                            rrf_scores[key] += 0.04 * n

        ordered = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        out = []
        for key in ordered:
            entry = result_map[key].copy()
            entry["score"] = float(rrf_scores[key])
            entry["retrieval_method"] = "text_rrf"
            entry["text_bm25_dominant"] = key == dominant_key
            out.append(entry)
        return out

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "dense_embeddings.npy", self.dense_embeddings)
        entries = []
        for e in self.entries:
            entries.append({
                "notice_id": e["notice_id"],
                "page_number": e["page_number"],
                "has_text": e["has_text"],
                "text_length": e["text_length"],
                "relative_image_path": self._relative_image_path(e.get("image_path", "")),
            })
        (path / "text_entries.json").write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")
        (path / "index_meta.json").write_text(json.dumps({
            "index_type": "text_rrf",
            "model": TEXT_EMBEDDING_MODEL,
            "entry_count": len(entries),
            "embedding_dimension": int(self.dense_embeddings.shape[1]) if self.dense_embeddings is not None and self.dense_embeddings.size else 0,
        }, indent=2), encoding="utf-8")

    @staticmethod
    def _relative_image_path(image_path: str) -> str:
        if not image_path:
            return ""
        from backend.path_utils import relative_to_project
        p = Path(image_path)
        return relative_to_project(p) if p.exists() else image_path.replace("\\", "/")

    def load(self, path: Path):
        meta_file = path / "index_meta.json"
        if meta_file.exists():
            try:
                meta = json.loads(meta_file.read_text(encoding="utf-8"))
                saved_model = meta.get("model")
                if saved_model and saved_model != TEXT_EMBEDDING_MODEL:
                    print(f"WARNING: Text index model mismatch (saved: '{saved_model}', configured: '{TEXT_EMBEDDING_MODEL}'). Rebuilding index...")
                    self.build_from_processed(PROCESSED_DIR)
                    self.save(path)
                    return
            except Exception as e:
                print(f"Error reading text index metadata: {e}")

        self.dense_embeddings = np.load(path / "dense_embeddings.npy")
        raw = json.loads((path / "text_entries.json").read_text(encoding="utf-8"))
        if len(raw) != len(self.dense_embeddings):
            raise ValueError(f"Text index mismatch: {len(raw)} entries vs {len(self.dense_embeddings)} embeddings")

        self.entries = []
        texts = []
        for e in raw:
            nid = e.get("notice_id", "")
            pnum = int(e.get("page_number", 1))
            text_path = TEXT_DIR / nid / f"page_{pnum}.txt"
            text = self._clean_text(text_path.read_text(encoding="utf-8").strip()) if text_path.exists() else ""
            image_path = resolve_page_image(nid, pnum, e.get("relative_image_path", e.get("image_path", "")))
            entry = {
                "notice_id": nid,
                "doc_id": nid,
                "filename": f"{nid}.pdf",
                "file_name": f"{nid}.pdf",
                "page_number": pnum,
                "page": pnum,
                "text": text,
                "has_text": bool(e.get("has_text", len(text) > 10)),
                "text_length": len(text),
                "image_path": str(image_path) if image_path else "",
            }
            self.entries.append(entry)
            texts.append(text)

        from rank_bm25 import BM25Okapi
        self._tokenized_corpus = [self._tokenize(t) for t in texts]
        self.bm25 = BM25Okapi(self._tokenized_corpus)
        print(f"Text index loaded from {path}: {len(self.entries)} pages")
