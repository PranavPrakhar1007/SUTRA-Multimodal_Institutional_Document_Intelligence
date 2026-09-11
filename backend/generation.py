"""Grounded answer generation for Text, Visual and Hybrid retrieval modes."""

import base64
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image

import backend.config as cfg
from backend.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_TEMPERATURE,
    GROQ_BASE_URL,
    GROQ_MODEL,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
)


ABSTENTION_TEXT = "I could not find sufficient evidence in the available documents to answer this question."

GROUNDING_PROMPT = """You are an assistant that answers questions about NIT Jamshedpur institutional documents.

Rules:
1. Use ONLY the supplied document evidence. Do not use general knowledge or infer facts that are not visible/supported.
2. If the supplied evidence does not contain enough information, reply exactly with an insufficiency statement instead of guessing.
3. For timetables, class schedules, and tables:
   - Carefully read the column headers from left to right. Pay close attention to the exact time slots (e.g., 2:00 PM – 2:55 PM vs 3:00 PM – 3:55 PM).
   - Carefully read the row headers from top to bottom (e.g., Monday, Tuesday).
   - To find a subject at a specific time, trace a straight vertical line down from the exact time column, and a straight horizontal line across from the exact day row.
   - Do NOT mix up adjacent columns (e.g., do not confuse the 2:00 PM column with the 3:00 PM column).
    - If the exact row/column intersection is visibly empty or white, state that no subject is scheduled. Never copy a value from a neighboring cell or from a subject legend below the grid.
4. Preserve exact names, numbers, dates, units, identifiers, and codes.
5. Keep the final answer concise and direct (state the exact subject name).
6. Output ONLY the clean, final grounded answer. Do NOT output internal chain-of-thought monologue or OCR commentary.

The source page shown is an official NIT Jamshedpur document."""



def _encode_image_base64(
    image_path: str,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    scale: float = 1.0,
    rotate_degrees: int = 0,
    focus_fraction: Optional[Tuple[float, float, float, float]] = None,
) -> str:
    """Encode the full image or an exact crop using original image coordinates."""
    with Image.open(image_path) as src:
        image = src.convert("RGB")
        if crop_box is not None:
            x0, y0, x1, y1 = [int(v) for v in crop_box]
            x0 = max(0, min(x0, image.width))
            x1 = max(x0 + 1, min(x1, image.width))
            y0 = max(0, min(y0, image.height))
            y1 = max(y0 + 1, min(y1, image.height))
            image = image.crop((x0, y0, x1, y1))
        if rotate_degrees:
            image = image.rotate(int(rotate_degrees), expand=True)
        if focus_fraction is not None:
            fx0, fy0, fx1, fy1 = focus_fraction
            image = image.crop((
                round(max(0.0, min(1.0, fx0)) * image.width),
                round(max(0.0, min(1.0, fy0)) * image.height),
                round(max(0.0, min(1.0, fx1)) * image.width),
                round(max(0.0, min(1.0, fy1)) * image.height),
            ))
        if scale != 1.0:
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        buffer = BytesIO()
        image.save(buffer, format="PNG", optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_prompt(question: str, notice_id: str, page_number: int, text_context: Optional[str] = None) -> str:
    prompt = f"{GROUNDING_PROMPT}\n\nDocument: {notice_id} (Page {page_number})\n\n"
    if text_context:
        prompt += f"--- DOCUMENT TEXT ---\n{text_context[:8000]}\n--- END DOCUMENT TEXT ---\n\n"
    prompt += f"Question: {question}\n\n"
    if text_context:
        prompt += "Answer ONLY from the supplied document text and any supplied image evidence."
    else:
        prompt += "Answer ONLY from the supplied document image evidence."
    return prompt


def _error_result(answer: str, notice_id: str, page_number: int, method: str, error: Optional[str] = None) -> dict:
    data = {
        "answer": answer,
        "source": {"notice_id": notice_id, "page_number": page_number},
        "generation_method": method,
        "status": "error",
    }
    if error:
        data["error"] = error
    return data


_openai_clients = {}


def _get_openai_client(api_key: str, base_url: str, timeout: float):
    cache_key = (api_key, base_url)
    if cache_key not in _openai_clients:
        from openai import OpenAI
        _openai_clients[cache_key] = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    return _openai_clients[cache_key]


def _generate_openai_compatible(
    question: str,
    notice_id: str,
    page_number: int,
    api_key: str,
    base_url: str,
    model: str,
    provider_name: str,
    image_path: Optional[str] = None,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    text_context: Optional[str] = None,
    rotate_degrees: int = 0,
    focus_fraction: Optional[Tuple[float, float, float, float]] = None,
) -> dict:
    """Call an OpenAI-compatible provider with text-only or multimodal evidence."""
    if not api_key or not api_key.strip():
        return _error_result(f"{provider_name} API key not configured.", notice_id, page_number, "error_no_api_key")

    try:
        client = _get_openai_client(api_key, base_url, OPENAI_REQUEST_TIMEOUT_SECONDS)
        prompt = _build_prompt(question, notice_id, page_number, text_context)

        if image_path:
            image_b64 = _encode_image_base64(
                image_path,
                crop_box=crop_box,
                scale=1.0,
                rotate_degrees=rotate_degrees,
                focus_fraction=focus_fraction,
            )
            content = [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": prompt},
            ]
            if crop_box:
                method = f"{provider_name}_visual_crop" if not text_context else f"{provider_name}_hybrid_crop"
            else:
                method = f"{provider_name}_visual" if not text_context else f"{provider_name}_hybrid"
        else:
            content = [{"type": "text", "text": prompt}]
            method = f"{provider_name}_text"

        response = None
        for attempt in range(2):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": content}],
                    max_tokens=GENERATION_MAX_TOKENS,
                    temperature=GENERATION_TEMPERATURE,
                )
                break
            except Exception:
                if attempt == 1:
                    raise
        answer = (response.choices[0].message.content or "").strip()
        status = "success"
        if not answer or answer == ABSTENTION_TEXT:
            answer = ABSTENTION_TEXT
            status = "abstained"

        source = {"notice_id": notice_id, "page_number": page_number}
        if image_path:
            source["evidence_type"] = "visual_crop" if crop_box else "page_image"
        if crop_box:
            source["evidence_box"] = [int(v) for v in crop_box]

        return {
            "answer": answer,
            "source": source,
            "generation_method": method,
            "model": model,
            "status": status,
        }
    except Exception as exc:
        return _error_result(f"Error generating answer via {provider_name}.", notice_id, page_number, f"{provider_name}_error", str(exc))


def _provider_result(question, notice_id, page_number, image_path=None, crop_box=None, text_context=None, rotate_degrees=0, focus_fraction=None):
    """Route to Groq provider."""
    return _generate_openai_compatible(
        question, notice_id, page_number, cfg.GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, "groq",
        image_path=image_path, crop_box=crop_box, text_context=text_context,
        rotate_degrees=rotate_degrees,
        focus_fraction=focus_fraction,
    )


def _structured_visual_query(question: str) -> bool:
    """Identify queries where page geometry and orientation carry the answer."""
    import re

    return bool(re.search(
        r"\b(table|timetable|time table|schedule|time slot|subject|semester|column|row|\d{1,2}\s*[-:]\s*\d{1,2})\b",
        question,
        flags=re.IGNORECASE,
    ))


def _schedule_visual_query(question: str) -> bool:
    import re

    return bool(re.search(
        r"\b(timetable|time table|time slot|which subject|\bsubject\b).*(\b(day|monday|tuesday|wednesday|thursday|friday)\b|\b\d{1,2}\s*[-:]\s*\d{1,2})",
        question,
        flags=re.IGNORECASE,
    ))


def _resolve_image(retrieved_result: dict):
    from backend.path_utils import resolve_page_image

    notice_id = retrieved_result.get("notice_id", "unknown")
    page_number = int(retrieved_result.get("page_number", retrieved_result.get("page", 1)))
    return resolve_page_image(notice_id, page_number, retrieved_result.get("image_path", ""))


def _resolve_text(retrieved_result: dict) -> str:
    from backend.config import TEXT_DIR

    text = retrieved_result.get("text", "") or ""
    if text.strip():
        txt = text.strip()
    else:
        notice_id = retrieved_result.get("notice_id", "unknown")
        page_number = int(retrieved_result.get("page_number", 1))
        path = TEXT_DIR / notice_id / f"page_{page_number}.txt"
        txt = path.read_text(encoding="utf-8").strip() if path.exists() else ""

    # Filter out noisy OCR text containing Mostly non-alphabetical garbage
    if txt:
        alpha_count = sum(1 for c in txt if c.isalpha())
        if alpha_count < 15 or (alpha_count / max(1, len(txt))) < 0.25:
            return ""
    return txt


def generate_answer(question: str, retrieved_result: dict, mode: str = "visual", project_root: Optional[Path] = None) -> dict:
    """Generate from the evidence selected by retrieval/reranking."""
    del project_root  # kept for API compatibility

    notice_id = retrieved_result.get("notice_id", "unknown")
    page_number = int(retrieved_result.get("page_number", retrieved_result.get("page", 1)))
    mode = mode.lower()
    image_path = _resolve_image(retrieved_result)
    text_context = _resolve_text(retrieved_result)
    rotate_degrees = 90 if _structured_visual_query(question) else 0
    # After orientation correction, keep the schedule grid and its headers while
    # excluding a neighboring subject legend that can confuse cell lookups.
    focus_fraction = (0.0, 0.33, 1.0, 0.52) if _schedule_visual_query(question) else None

    # Reranking boxes are measured on the high-resolution render, while the
    # API page image is normally 144 DPI. Convert coordinates before cropping;
    # using the raw box silently crops the wrong cell (or a tiny corner).
    crop_box = None
    if image_path and retrieved_result.get("best_tile_box"):
        try:
            with Image.open(image_path) as image:
                rw = float(retrieved_result.get("rerank_image_width", image.width))
                rh = float(retrieved_result.get("rerank_image_height", image.height))
                x0, y0, x1, y1 = retrieved_result["best_tile_box"]
                crop_box = (
                    round(x0 * image.width / rw), round(y0 * image.height / rh),
                    round(x1 * image.width / rw), round(y1 * image.height / rh),
                )
        except (OSError, TypeError, ValueError, ZeroDivisionError):
            crop_box = None

    if mode in ("text", "text_baseline"):
        if len(text_context) <= 10:
            return {"answer": ABSTENTION_TEXT, "source": {"notice_id": notice_id, "page_number": page_number}, "generation_method": "abstention_no_text", "status": "abstain"}
        return _provider_result(question, notice_id, page_number, text_context=text_context)

    if mode in ("hybrid", "hybrid_rrf_baseline", "hybrid_reranked"):
        if not image_path and not text_context:
            return {"answer": ABSTENTION_TEXT, "source": {"notice_id": notice_id, "page_number": page_number}, "generation_method": "abstention_no_evidence", "status": "abstain"}
        # Send full page image to VLM so top time headers (2-3pm vs 3-4pm) are fully visible
        return _provider_result(
            question,
            notice_id,
            page_number,
            image_path=str(image_path) if image_path else None,
            crop_box=crop_box,
            text_context=text_context or None,
            rotate_degrees=rotate_degrees,
            focus_fraction=focus_fraction,
        )

    if mode in ("visual", "vision", "visual_colqwen2", "visual_clip_baseline", "visual_reranked"):
        if not image_path:
            return {"answer": "Page image not found for visual generation.", "source": {"notice_id": notice_id, "page_number": page_number}, "generation_method": "error_no_image", "status": "error"}
        return _provider_result(
            question,
            notice_id,
            page_number,
            image_path=str(image_path),
            crop_box=crop_box,
            rotate_degrees=rotate_degrees,
            focus_fraction=focus_fraction,
        )

    # Unknown mode fallback to text-only if text exists, else error
    if text_context:
        return _provider_result(question, notice_id, page_number, text_context=text_context)
    return {"answer": ABSTENTION_TEXT, "source": {"notice_id": notice_id, "page_number": page_number}, "generation_method": "error_unknown_mode", "status": "error"}
