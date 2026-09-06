"""Path utilities for portable corpus/index artifacts."""

from pathlib import Path
from typing import Optional

from backend.config import PROJECT_ROOT, PAGES_DIR


def normalize_relative_path(value: str) -> str:
    """Normalize Windows/POSIX separators and remove a leading ./ ."""
    if not value:
        return ""
    return str(value).replace("\\", "/").lstrip("./")


def resolve_project_path(value: str) -> Optional[Path]:
    """Resolve an artifact path stored as either absolute or project-relative."""
    if not value:
        return None

    raw = Path(value)
    if raw.is_absolute() and raw.exists():
        return raw

    relative = normalize_relative_path(value)
    if relative:
        candidate = PROJECT_ROOT / relative
        if candidate.exists():
            return candidate

    return None


def resolve_page_image(notice_id: str, page_number: int, stored_path: str = "") -> Optional[Path]:
    """Resolve a page image from a stored path, falling back to canonical layout."""
    resolved = resolve_project_path(stored_path)
    if resolved is not None:
        return resolved

    candidate = PAGES_DIR / notice_id / f"page_{page_number}.png"
    return candidate if candidate.exists() else None


def relative_to_project(path: Path) -> str:
    """Return a portable project-relative POSIX path."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.as_posix().replace("\\", "/")
