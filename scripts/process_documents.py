"""
Phase 4 — Document Processing

For every canonical PDF:
1. Render each page as a PNG image (for Visual RAG path)
2. Extract text per page (where available)
3. Generate metadata JSON

Uses PyMuPDF (fitz) for both rendering and text extraction.
"""

import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.path_utils import relative_to_project
import pymupdf as fitz  # PyMuPDF


def process_document(pdf_path: Path, output_dir: Path, notice_id: str) -> dict:
    """Process a single PDF document."""
    pages_dir = output_dir / "pages" / notice_id
    text_dir = output_dir / "text" / notice_id
    pages_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    doc = fitz.open(str(pdf_path))
    metadata = {
        "notice_id": notice_id,
        "canonical_filename": pdf_path.name,
        "page_count": len(doc),
        "pages": [],
        "processing_status": "success",
    }

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_info = {
            "page_number": page_num + 1,
            "width": round(page.rect.width, 1),
            "height": round(page.rect.height, 1),
        }

        # --- Render page as image ---
        # Use 2x zoom for good quality (default 72 DPI → 144 DPI)
        zoom = 2.0
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)

        image_filename = f"page_{page_num + 1}.png"
        image_path = pages_dir / image_filename
        pix.save(str(image_path))

        page_info["image_path"] = relative_to_project(image_path)
        page_info["image_size_bytes"] = image_path.stat().st_size
        page_info["image_width"] = pix.width
        page_info["image_height"] = pix.height

        # --- Extract text ---
        text = page.get_text("text")
        text_filename = f"page_{page_num + 1}.txt"
        text_path = text_dir / text_filename

        with open(text_path, "w", encoding="utf-8") as f:
            f.write(text)

        page_info["text_path"] = relative_to_project(text_path)
        page_info["text_length"] = len(text)
        page_info["has_text"] = len(text.strip()) > 10
        page_info["word_count"] = len(text.split()) if text.strip() else 0

        # Text preview
        page_info["text_preview"] = text[:200].replace("\n", " ").strip() if text else ""

        metadata["pages"].append(page_info)

    doc.close()

    # Summary
    text_pages = sum(1 for p in metadata["pages"] if p["has_text"])
    metadata["text_availability"] = (
        "full" if text_pages == metadata["page_count"] else
        "partial" if text_pages > 0 else
        "none"
    )
    metadata["total_text_pages"] = text_pages

    return metadata


def main():
    from backend.config import DOCUMENTS_DIR, PROCESSED_DIR

    raw_dir = DOCUMENTS_DIR
    output_dir = PROCESSED_DIR
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(list(raw_dir.glob("*.pdf")))

    if not pdfs:
        print(f"ERROR: No PDF documents found in {raw_dir}")
        return

    print(f"Processing {len(pdfs)} documents...\n")

    all_metadata = []

    for pdf in pdfs:
        notice_id = pdf.stem
        print(f"Processing {notice_id} ({pdf.name})...")

        try:
            metadata = process_document(pdf, output_dir, notice_id)

            # Save per-document metadata
            meta_path = metadata_dir / f"{notice_id}.json"
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            all_metadata.append(metadata)

            print(f"  Pages: {metadata['page_count']}")
            print(f"  Text availability: {metadata['text_availability']}")
            print(f"  Images saved to: data/processed/pages/{notice_id}/")
            print(f"  Text saved to: data/processed/text/{notice_id}/")
            print()

        except Exception as e:
            print(f"  ERROR processing {notice_id}: {e}")
            all_metadata.append({
                "notice_id": notice_id,
                "processing_status": "error",
                "error": str(e),
            })

    # Save combined metadata
    combined_path = output_dir / "all_documents.json"
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, indent=2, ensure_ascii=False)

    total_pages = sum(m.get("page_count", 0) for m in all_metadata)
    total_images = sum(len(m.get("pages", [])) for m in all_metadata)

    print("=" * 60)
    print(f"Document Processing Complete")
    print(f"  Total documents: {len(pdfs)}")
    print(f"  Total pages: {total_pages}")
    print(f"  Total page images generated: {total_images}")
    print(f"  Metadata saved to: {metadata_dir}")
    print(f"  Combined metadata: {combined_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
