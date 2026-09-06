"""
Phase 2 — Corpus Profiling

Inspect every canonical PDF and record observed characteristics:
- page count
- text availability (per page)
- table detection
- image presence
- visual complexity estimate
- document condition

Uses pdfplumber (already installed).
Outputs: data/corpus_profile.json
"""

import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import pdfplumber


def profile_page(page, page_num: int) -> dict:
    """Profile a single page of a PDF."""
    profile = {
        "page_number": page_num,
        "width": round(page.width, 1),
        "height": round(page.height, 1),
    }

    # --- Text extraction ---
    text = page.extract_text() or ""
    profile["text_length"] = len(text)
    profile["has_text"] = len(text.strip()) > 10
    profile["word_count"] = len(text.split()) if text.strip() else 0

    # --- Table detection ---
    tables = page.find_tables()
    profile["table_count"] = len(tables)
    profile["has_tables"] = len(tables) > 0

    # If tables exist, record dimensions
    if tables:
        profile["table_details"] = []
        for t_idx, table in enumerate(tables):
            rows = table.extract()
            if rows:
                profile["table_details"].append({
                    "table_index": t_idx,
                    "rows": len(rows),
                    "columns": len(rows[0]) if rows else 0,
                })

    # --- Image detection ---
    images = page.images
    profile["image_count"] = len(images)
    profile["has_images"] = len(images) > 0

    # --- Lines/rects (visual structure indicators) ---
    lines = page.lines
    rects = page.rects
    profile["line_count"] = len(lines)
    profile["rect_count"] = len(rects)
    profile["has_visual_structure"] = len(lines) > 5 or len(rects) > 3

    # --- Estimate if page might be scanned ---
    # Heuristic: if page has images but very little text, it might be scanned
    if len(images) > 0 and len(text.strip()) < 20:
        profile["scan_status"] = "likely_scanned"
    elif len(text.strip()) < 5 and len(images) == 0:
        profile["scan_status"] = "possibly_blank_or_problematic"
    else:
        profile["scan_status"] = "digital_text"

    # Store first 200 chars of text for inspection
    profile["text_preview"] = text[:200].replace("\n", " ").strip() if text else ""

    return profile


def profile_document(pdf_path: Path, notice_id: str) -> dict:
    """Profile an entire PDF document."""
    doc_profile = {
        "notice_id": notice_id,
        "canonical_filename": pdf_path.name,
        "file_size_bytes": pdf_path.stat().st_size,
        "pages": [],
        "summary": {},
    }

    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            doc_profile["page_count"] = len(pdf.pages)

            total_text_len = 0
            total_tables = 0
            total_images = 0
            scanned_pages = 0
            text_pages = 0
            pages_with_tables = 0
            pages_with_visual_structure = 0

            for page_num, page in enumerate(pdf.pages, 1):
                pp = profile_page(page, page_num)
                doc_profile["pages"].append(pp)

                total_text_len += pp["text_length"]
                total_tables += pp["table_count"]
                total_images += pp["image_count"]
                if pp["scan_status"] == "likely_scanned":
                    scanned_pages += 1
                if pp["has_text"]:
                    text_pages += 1
                if pp["has_tables"]:
                    pages_with_tables += 1
                if pp["has_visual_structure"]:
                    pages_with_visual_structure += 1

            # --- Summary ---
            doc_profile["summary"] = {
                "page_count": len(pdf.pages),
                "total_text_characters": total_text_len,
                "total_tables": total_tables,
                "total_images": total_images,
                "text_pages": text_pages,
                "scanned_pages": scanned_pages,
                "pages_with_tables": pages_with_tables,
                "pages_with_visual_structure": pages_with_visual_structure,
                "text_availability": "full" if text_pages == len(pdf.pages) else
                                     "partial" if text_pages > 0 else "none",
                "has_tables": total_tables > 0,
                "has_scanned_content": scanned_pages > 0,
                "visual_complexity": "high" if pages_with_visual_structure > len(pdf.pages) * 0.5 else
                                     "medium" if pages_with_visual_structure > 0 else "low",
            }

            # Determine document condition
            if scanned_pages > 0:
                condition = "mixed_scan_digital" if text_pages > 0 else "scanned"
            elif total_tables > 0 and pages_with_visual_structure > 0:
                condition = "structured_with_tables"
            elif pages_with_visual_structure > 0:
                condition = "visually_structured"
            else:
                condition = "clean_text"

            doc_profile["summary"]["document_condition"] = condition
            doc_profile["processing_status"] = "profiled"

    except Exception as e:
        doc_profile["processing_status"] = "error"
        doc_profile["error"] = str(e)

    return doc_profile


def main():
    project_root = Path(__file__).resolve().parent.parent
    raw_dir = project_root / "data" / "raw"

    pdfs = sorted(raw_dir.glob("notice_*.pdf"), key=lambda p: int(p.stem.split("_")[1]))

    if not pdfs:
        print("ERROR: No canonical PDFs found in data/raw/")
        return

    print(f"Profiling {len(pdfs)} documents...\n")

    corpus_profile = {
        "total_documents": len(pdfs),
        "documents": [],
        "corpus_summary": {},
    }

    for pdf in pdfs:
        notice_id = pdf.stem
        print(f"Profiling {notice_id} ({pdf.name})...")
        profile = profile_document(pdf, notice_id)
        corpus_profile["documents"].append(profile)

        s = profile.get("summary", {})
        print(f"  Pages: {s.get('page_count', '?')}")
        print(f"  Text availability: {s.get('text_availability', '?')}")
        print(f"  Tables: {s.get('total_tables', 0)}")
        print(f"  Images: {s.get('total_images', 0)}")
        print(f"  Document condition: {s.get('document_condition', '?')}")
        print(f"  Visual complexity: {s.get('visual_complexity', '?')}")
        print()

    # --- Corpus-level summary ---
    total_pages = sum(d.get("summary", {}).get("page_count", 0) for d in corpus_profile["documents"])
    docs_with_tables = sum(1 for d in corpus_profile["documents"] if d.get("summary", {}).get("has_tables", False))
    docs_scanned = sum(1 for d in corpus_profile["documents"] if d.get("summary", {}).get("has_scanned_content", False))

    conditions = [d.get("summary", {}).get("document_condition", "unknown") for d in corpus_profile["documents"]]

    corpus_profile["corpus_summary"] = {
        "total_documents": len(pdfs),
        "total_pages": total_pages,
        "documents_with_tables": docs_with_tables,
        "documents_with_scanned_content": docs_scanned,
        "document_conditions": conditions,
    }

    # --- Save ---
    output_path = project_root / "data" / "corpus_profile.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus_profile, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Corpus Profile Complete")
    print(f"  Total documents: {len(pdfs)}")
    print(f"  Total pages: {total_pages}")
    print(f"  Documents with tables: {docs_with_tables}")
    print(f"  Documents with scanned content: {docs_scanned}")
    print(f"  Saved to: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
