"""OCR enhancement that preserves native text and records OCR metadata."""

import json
import sys
import time
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.config import METADATA_DIR, PAGES_DIR, TEXT_DIR

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

NATIVE_TEXT_THRESHOLD = 10


def run_ocr():
    print("=" * 60)
    print("OCR Enhancement")
    print("=" * 60)
    start = time.perf_counter()

    import easyocr

    reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    total_ocr = 0
    total_preserved = 0
    total_failed = 0

    doc_dirs = sorted([p for p in PAGES_DIR.iterdir() if p.is_dir()], key=lambda p: p.name)
    for notice_dir in doc_dirs:
        notice_id = notice_dir.name
        text_dir = TEXT_DIR / notice_id
        text_dir.mkdir(parents=True, exist_ok=True)

        meta_path = METADATA_DIR / f"{notice_id}.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {"notice_id": notice_id, "pages": []}
        page_meta = {int(p["page_number"]): p for p in meta.get("pages", [])}

        for image_path in sorted(notice_dir.glob("page_*.png"), key=lambda p: int(p.stem.split("_")[1])):
            page_num = int(image_path.stem.split("_")[1])
            text_file = text_dir / f"page_{page_num}.txt"
            existing = text_file.read_text(encoding="utf-8").strip() if text_file.exists() else ""

            page_record = page_meta.get(page_num, {"page_number": page_num})

            if len(existing) >= NATIVE_TEXT_THRESHOLD:
                page_record["text_source"] = page_record.get("text_source", "native")
                page_record["has_text"] = True
                page_record["text_length"] = len(existing)
                total_preserved += 1
                page_meta[page_num] = page_record
                continue

            try:
                results = reader.readtext(str(image_path), detail=1, paragraph=False)
                lines = []
                regions = []
                for item in results:
                    if len(item) >= 3:
                        box, text, confidence = item
                        text = str(text).strip()
                        if text:
                            lines.append(text)
                            regions.append({
                                "box": [[float(x), float(y)] for x, y in box],
                                "text": text,
                                "confidence": round(float(confidence), 4),
                            })

                ocr_text = "\n".join(lines).strip()
                if ocr_text:
                    text_file.write_text(ocr_text, encoding="utf-8")
                    page_record["text_source"] = "ocr"
                    page_record["has_text"] = len(ocr_text) >= NATIVE_TEXT_THRESHOLD
                    page_record["text_length"] = len(ocr_text)
                    page_record["word_count"] = len(ocr_text.split())
                    page_record["ocr_regions"] = regions
                    total_ocr += 1
                else:
                    page_record["text_source"] = "none"
                    page_record["has_text"] = False
                    page_record["text_length"] = 0
                    page_record["ocr_regions"] = []
                    total_failed += 1
            except Exception as exc:
                print(f"ERROR {notice_id}/page_{page_num}: {exc}")
                page_record["text_source"] = "ocr_error"
                page_record["ocr_error"] = str(exc)
                total_failed += 1

            page_meta[page_num] = page_record

        meta["pages"] = [page_meta[p] for p in sorted(page_meta)]
        total_pages = int(meta.get("page_count", len(meta["pages"])))
        text_pages = sum(1 for p in meta["pages"] if p.get("has_text"))
        meta["total_text_pages"] = text_pages
        meta["text_availability"] = "full" if text_pages == total_pages else ("partial" if text_pages else "none")
        meta["ocr_applied"] = True
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # Rebuild the combined metadata so there is no stale all_documents.json after OCR.
    combined = []
    for meta_file in sorted([f for f in METADATA_DIR.glob("*.json") if f.name != "all_documents.json"]):
        combined.append(json.loads(meta_file.read_text(encoding="utf-8")))
    combined_path = TEXT_DIR.parent / "all_documents.json"
    combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")

    elapsed = time.perf_counter() - start
    print(f"OCR complete: {total_ocr} OCR'd, {total_preserved} native-text pages preserved, {total_failed} failed")
    print(f"Time: {elapsed:.1f}s")
    print("Next: python scripts/build_indices.py")


if __name__ == "__main__":
    run_ocr()
