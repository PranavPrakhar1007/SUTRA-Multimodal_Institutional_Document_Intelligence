"""GPU-accelerated ColPali VLM visual retrieval module.

Uses 4-bit quantized ColPali (vidore/colpali-v1.2) on NVIDIA CUDA GPUs for
high-resolution multi-vector visual retrieval over scanned institutional documents.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from backend.config import METADATA_DIR, PAGES_DIR, PROCESSED_DIR, VISUAL_EMBEDDING_MODEL
from backend.path_utils import relative_to_project, resolve_page_image

_colpali_model = None
_colpali_processor = None


def _get_colpali():
    """Lazy-load 4-bit quantized ColPali / ColQwen2 VLM on CUDA GPU."""
    global _colpali_model, _colpali_processor
    if _colpali_model is None:
        import torch
        from transformers import BitsAndBytesConfig

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Loading Visual VLM ({VISUAL_EMBEDDING_MODEL}) on {device}...")

        if "colqwen" in VISUAL_EMBEDDING_MODEL.lower():
            from colpali_engine.models import ColQwen2, ColQwen2Processor
            model_cls = ColQwen2
            proc_cls = ColQwen2Processor
        else:
            from colpali_engine.models import ColPali, ColPaliProcessor
            model_cls = ColPali
            proc_cls = ColPaliProcessor

        if torch.cuda.is_available():
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            _colpali_model = model_cls.from_pretrained(
                VISUAL_EMBEDDING_MODEL,
                quantization_config=bnb_config,
                device_map=device,
            )
        else:
            _colpali_model = model_cls.from_pretrained(VISUAL_EMBEDDING_MODEL, device_map=device)

        _colpali_processor = proc_cls.from_pretrained(VISUAL_EMBEDDING_MODEL)
        _colpali_model.eval()
        print(f"Visual VLM ({VISUAL_EMBEDDING_MODEL}) loaded successfully on {device}.")
    return _colpali_model, _colpali_processor


class VisualIndex:
    """ColPali multi-vector visual document index."""

    def __init__(self):
        self.entries: list[dict] = []
        self.image_embeddings: list[np.ndarray] = []

    def _encode_images(self, PIL_images: list[Image.Image]) -> list[np.ndarray]:
        import torch

        model, processor = _get_colpali()
        inputs = processor.process_images(PIL_images).to(model.device)
        with torch.no_grad():
            embeddings = model(**inputs)
        # Returns float32 numpy arrays per page image (num_patches, dim)
        return [emb.cpu().to(torch.float32).numpy() for emb in embeddings]

    def build_from_processed(self, processed_dir: Path):
        """Build portable ColPali multi-vector visual index from page images."""
        self._cuda_img_tensors = None
        metadata_files = sorted(
            METADATA_DIR.glob("notice_*.json"),
            key=lambda p: int(p.stem.split("_")[1]),
        )

        self.entries = []
        self.image_embeddings = []

        for meta_file in metadata_files:
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)

            notice_id = meta["notice_id"]
            for page_info in meta.get("pages", []):
                page_num = int(page_info["page_number"])
                image_path = PAGES_DIR / notice_id / f"page_{page_num}.png"
                if not image_path.exists():
                    print(f"WARNING: image missing: {image_path}")
                    continue

                entry = {
                    "notice_id": notice_id,
                    "doc_id": notice_id,
                    "filename": f"{notice_id}.pdf",
                    "file_name": f"{notice_id}.pdf",
                    "page_number": page_num,
                    "page": page_num,
                    "relative_image_path": relative_to_project(image_path),
                    "image_width": page_info.get("image_width"),
                    "image_height": page_info.get("image_height"),
                }
                self.entries.append(entry)

                with Image.open(image_path) as img:
                    rgb_img = img.convert("RGB")
                    emb = self._encode_images([rgb_img])[0]
                    self.image_embeddings.append(emb)

                print(f"  ColPali embedded: {notice_id} p{page_num} shape={emb.shape}")

        print(f"ColPali Visual Index: {len(self.entries)} page images embedded successfully.")

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        import torch

        if not self.image_embeddings or not self.entries:
            return []

        top_k = max(1, min(int(top_k), len(self.entries)))
        model, processor = _get_colpali()

        inputs_text = processor.process_queries([query]).to(model.device)
        with torch.no_grad():
            query_embeddings = model(**inputs_text)

        # Ensure float32 precision for score_multi_vector calculation
        query_embeddings = query_embeddings.to(dtype=torch.float32)

        if not hasattr(self, "_cuda_img_tensors") or self._cuda_img_tensors is None or len(self._cuda_img_tensors) != len(self.image_embeddings):
            self._cuda_img_tensors = [
                torch.from_numpy(emb).to(device=model.device, dtype=torch.float32)
                for emb in self.image_embeddings
            ]

        scores_tensor = processor.score_multi_vector(query_embeddings, self._cuda_img_tensors)[0]
        scores = scores_tensor.cpu().numpy().tolist()

        top_indices = np.argsort(-np.array(scores), kind="stable")[:top_k]

        results = []
        for idx in top_indices:
            entry = self.entries[int(idx)].copy()
            image_path = resolve_page_image(
                entry["notice_id"], entry["page_number"], entry.get("relative_image_path", "")
            )
            entry["image_path"] = str(image_path) if image_path else ""
            entry["score"] = float(scores[idx])
            entry["retrieval_method"] = "colqwen2_vlm" if "colqwen" in VISUAL_EMBEDDING_MODEL.lower() else "colpali_vlm"
            results.append(entry)
        return results

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        # Save multi-vector embeddings as npz archive
        np.savez_compressed(
            path / "colpali_embeddings.npz",
            **{f"emb_{i}": emb for i, emb in enumerate(self.image_embeddings)},
        )
        with open(path / "visual_entries.json", "w", encoding="utf-8") as f:
            json.dump(self.entries, f, indent=2, ensure_ascii=False)
        with open(path / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "index_type": "colqwen2_vlm" if "colqwen" in VISUAL_EMBEDDING_MODEL.lower() else "colpali_vlm",
                    "model": VISUAL_EMBEDDING_MODEL,
                    "entry_count": len(self.entries),
                },
                f,
                indent=2,
            )
        print(f"Visual index ({VISUAL_EMBEDDING_MODEL}) saved to {path}")

    def load(self, path: Path):
        self._cuda_img_tensors = None
        meta_file = path / "index_meta.json"
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                    saved_model = meta.get("model")
                    if saved_model != VISUAL_EMBEDDING_MODEL:
                        print(f"WARNING: Visual index model mismatch (saved: '{saved_model}', configured: '{VISUAL_EMBEDDING_MODEL}'). Rebuilding index...")
                        self.build_from_processed(PROCESSED_DIR)
                        self.save(path)
                        return
            except Exception as e:
                print(f"Error reading index_meta.json: {e}. Invalidating index & rebuilding...")
                self.build_from_processed(PROCESSED_DIR)
                self.save(path)
                return
        else:
            print("WARNING: index_meta.json missing. Rebuilding visual index...")
            self.build_from_processed(PROCESSED_DIR)
            self.save(path)
            return

        npz_file = path / "colpali_embeddings.npz"
        entries_file = path / "visual_entries.json"

        if not npz_file.exists() or not entries_file.exists():
            print("WARNING: Missing visual embeddings NPZ or visual_entries.json. Rebuilding index...")
            self.build_from_processed(PROCESSED_DIR)
            self.save(path)
            return

        try:
            with open(entries_file, "r", encoding="utf-8") as f:
                raw_entries = json.load(f)

            with np.load(npz_file) as archive:
                self.image_embeddings = [archive[f"emb_{i}"] for i in range(len(raw_entries))]

            self.entries = []
            for e in raw_entries:
                nid = e.get("notice_id", "")
                pnum = int(e.get("page_number", 1))
                img_path = resolve_page_image(nid, pnum, e.get("relative_image_path", ""))
                entry = e.copy()
                if img_path:
                    entry["relative_image_path"] = relative_to_project(img_path)
                self.entries.append(entry)

            if len(self.entries) != len(self.image_embeddings) or len(self.entries) == 0:
                raise ValueError(f"Entry/embedding count mismatch: {len(self.entries)} vs {len(self.image_embeddings)}")

        except Exception as exc:
            print(f"WARNING: Error loading visual index archive ({exc}). Rebuilding index...")
            self.build_from_processed(PROCESSED_DIR)
            self.save(path)
            return

        print(f"Visual index ({VISUAL_EMBEDDING_MODEL}) loaded from {path}: {len(self.entries)} pages")
