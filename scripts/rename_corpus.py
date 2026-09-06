"""
Phase 1 — Corpus Registration

[CONFIRMED] Use the real NIT Jamshedpur PDFs from the Sample folder.
Rename them to canonical notice_1.pdf … notice_N.pdf.
Preserve original filename mapping in CORPUS_MANIFEST.csv.

Deterministic ordering: alphabetical sort on original filenames.
Files are COPIED (not moved) to preserve originals.
"""

import csv
import shutil
from pathlib import Path


def main():
    project_root = Path(__file__).resolve().parent.parent

    # --- Locate the Sample folder (may have trailing space) ---
    sample_candidates = list(project_root.glob("Sample*"))
    sample_dir = None
    for candidate in sample_candidates:
        if candidate.is_dir():
            sample_dir = candidate
            break

    if sample_dir is None:
        print("ERROR: Could not find 'Sample' directory in project root.")
        print(f"Project root: {project_root}")
        print(f"Contents: {list(project_root.iterdir())}")
        return

    print(f"Found sample directory: '{sample_dir.name}' at {sample_dir}")

    # --- List all PDFs ---
    pdfs = sorted(sample_dir.glob("*.pdf"), key=lambda p: p.name.lower())
    print(f"\nFound {len(pdfs)} PDF files:")
    for i, pdf in enumerate(pdfs, 1):
        print(f"  {i}. {pdf.name} ({pdf.stat().st_size:,} bytes)")

    if len(pdfs) == 0:
        print("ERROR: No PDF files found in Sample directory.")
        return

    # --- Create data/raw/ directory ---
    raw_dir = project_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    # --- Copy and rename ---
    manifest_rows = []
    for i, pdf in enumerate(pdfs, 1):
        canonical_name = f"notice_{i}.pdf"
        dest = raw_dir / canonical_name

        shutil.copy2(str(pdf), str(dest))
        print(f"\n  Copied: {pdf.name}")
        print(f"      -> {canonical_name}")

        # Verify file integrity
        assert dest.stat().st_size == pdf.stat().st_size, \
            f"Size mismatch for {canonical_name}!"

        manifest_rows.append({
            "notice_id": f"notice_{i}",
            "original_filename": pdf.name,
            "canonical_filename": canonical_name,
            "size_bytes": pdf.stat().st_size,
        })

    # --- Write CORPUS_MANIFEST.csv ---
    manifest_path = project_root / "CORPUS_MANIFEST.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "notice_id", "original_filename", "canonical_filename", "size_bytes"
        ])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"\n{'='*60}")
    print(f"Corpus registration complete.")
    print(f"  Total documents: {len(pdfs)}")
    print(f"  Canonical location: {raw_dir}")
    print(f"  Manifest: {manifest_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
