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
    GEMINI_MODEL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
    XAI_BASE_URL,
    XAI_MODEL,
)


ABSTENTION_TEXT = "I could not find sufficient evidence in the available documents to answer this question."

GROUNDING_PROMPT = """You are an assistant that answers questions about NIT Jamshedpur institutional documents.

Rules:
1. Use ONLY the supplied document evidence. Do not use general knowledge or infer facts that are not visible/supported.
2. If the supplied evidence does not contain enough information, reply exactly or nearly exactly with an insufficiency statement instead of guessing.
3. For tables, carefully inspect row labels, column headers, the corresponding cell, footnotes, and nearby context.
4. Preserve exact names, numbers, dates, units, identifiers and codes.
5. For counting questions, count only visible entries supported by the evidence.
6. Keep the final answer concise but complete.
7. Output ONLY the clean, final grounded answer. Do NOT output internal chain-of-thought monologue, OCR analysis commentary, or self-correction notes.

The source page shown is an official NIT Jamshedpur document."""



def _encode_image_base64(image_path: str, crop_box: Optional[Tuple[int, int, int, int]] = None, scale: float = 1.0) -> str:
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
) -> dict:
    """Call an OpenAI-compatible provider with text-only or multimodal evidence."""
    if not api_key or not api_key.strip():
        return _error_result(f"{provider_name} API key not configured.", notice_id, page_number, "error_no_api_key")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=base_url, timeout=OPENAI_REQUEST_TIMEOUT_SECONDS)
        prompt = _build_prompt(question, notice_id, page_number, text_context)

        if image_path:
            image_b64 = _encode_image_base64(image_path, crop_box=crop_box, scale=1.0)
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

        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            max_tokens=GENERATION_MAX_TOKENS,
            temperature=GENERATION_TEMPERATURE,
        )
        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = ABSTENTION_TEXT

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
            "status": "success",
        }
    except Exception as exc:
        return _error_result(f"Error generating answer via {provider_name}.", notice_id, page_number, f"{provider_name}_error", str(exc))


def _generate_gemini(
    question: str,
    notice_id: str,
    page_number: int,
    image_path: Optional[str] = None,
    crop_box: Optional[Tuple[int, int, int, int]] = None,
    text_context: Optional[str] = None,
) -> dict:
    if not cfg.GEMINI_API_KEY or not cfg.GEMINI_API_KEY.strip():
        return _error_result("Gemini API key not configured.", notice_id, page_number, "error_no_api_key")

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=cfg.GEMINI_API_KEY)
        prompt = _build_prompt(question, notice_id, page_number, text_context)
        parts = []

        if image_path:
            with Image.open(image_path) as src:
                image = src.convert("RGB")
                if crop_box:
                    x0, y0, x1, y1 = [int(v) for v in crop_box]
                    x0 = max(0, min(x0, image.width))
                    x1 = max(x0 + 1, min(x1, image.width))
                    y0 = max(0, min(y0, image.height))
                    y1 = max(y0 + 1, min(y1, image.height))
                    image = image.crop((x0, y0, x1, y1))
                buffer = BytesIO()
                image.save(buffer, format="PNG")
                parts.append(types.Part.from_bytes(data=buffer.getvalue(), mime_type="image/png"))

        parts.append(types.Part.from_text(text=prompt))
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Content(role="user", parts=parts)],
        )
        answer = (getattr(response, "text", "") or "").strip() or ABSTENTION_TEXT

        source = {"notice_id": notice_id, "page_number": page_number}
        if image_path:
            source["evidence_type"] = "visual_crop" if crop_box else "page_image"
        if crop_box:
            source["evidence_box"] = [int(v) for v in crop_box]

        return {
            "answer": answer,
            "source": source,
            "generation_method": "gemini_hybrid" if image_path and text_context else ("gemini_visual" if image_path else "gemini_text"),
            "model": GEMINI_MODEL,
            "status": "success",
        }
    except Exception as exc:
        return _error_result("Error generating answer via Gemini.", notice_id, page_number, "gemini_error", str(exc))


def _provider_result(question, notice_id, page_number, image_path=None, crop_box=None, text_context=None):
    """Route to the configured provider. Visual requests never fall back to text-only models."""
    provider = cfg.LLM_PROVIDER

    if provider == "xai":
        return _generate_openai_compatible(
            question, notice_id, page_number, cfg.XAI_API_KEY, XAI_BASE_URL, XAI_MODEL, "xai",
            image_path=image_path, crop_box=crop_box, text_context=text_context,
        )
    if provider == "groq":
        return _generate_openai_compatible(
            question, notice_id, page_number, cfg.GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, "groq",
            image_path=image_path, crop_box=crop_box, text_context=text_context,
        )
    if provider == "gemini":
        return _generate_gemini(
            question, notice_id, page_number, image_path=image_path, crop_box=crop_box, text_context=text_context
        )

    return _error_result(
        "No multimodal/text provider is configured. Set an xAI, Groq, or Gemini API key.",
        notice_id,
        page_number,
        "error_no_provider",
    )


def _resolve_image(retrieved_result: dict):
    from backend.config import PAGES_DIR
    from backend.path_utils import resolve_page_image

    notice_id = retrieved_result.get("notice_id", "unknown")
    page_number = int(retrieved_result.get("page_number", retrieved_result.get("page", 1)))
    return resolve_page_image(notice_id, page_number, retrieved_result.get("image_path", ""))


def _resolve_text(retrieved_result: dict) -> str:
    from backend.config import TEXT_DIR

    text = retrieved_result.get("text", "") or ""
    if text.strip():
        return text.strip()
    notice_id = retrieved_result.get("notice_id", "unknown")
    page_number = int(retrieved_result.get("page_number", 1))
    path = TEXT_DIR / notice_id / f"page_{page_number}.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def generate_answer(question: str, retrieved_result: dict, mode: str = "visual", project_root: Optional[Path] = None) -> dict:
    """Generate from the evidence selected by retrieval/reranking."""
    del project_root  # kept for API compatibility

    notice_id = retrieved_result.get("notice_id", "unknown")
    page_number = int(retrieved_result.get("page_number", retrieved_result.get("page", 1)))
    mode = mode.lower()
    image_path = _resolve_image(retrieved_result)
    text_context = _resolve_text(retrieved_result)

    if mode in ("text", "text_baseline"):
        if len(text_context) <= 10:
            return {"answer": ABSTENTION_TEXT, "source": {"notice_id": notice_id, "page_number": page_number}, "generation_method": "abstention_no_text", "status": "abstain"}
        return _provider_result(question, notice_id, page_number, text_context=text_context)

    if mode in ("hybrid", "hybrid_rrf_baseline", "hybrid_reranked"):
        if not image_path and not text_context:
            return {"answer": ABSTENTION_TEXT, "source": {"notice_id": notice_id, "page_number": page_number}, "generation_method": "abstention_no_evidence", "status": "abstain"}
        crop = retrieved_result.get("best_tile_box")
        crop_box = tuple(crop) if crop and len(crop) == 4 else None
        return _provider_result(
            question,
            notice_id,
            page_number,
            image_path=str(image_path) if image_path else None,
            crop_box=crop_box,
            text_context=text_context or None,
        )

    # Visual modes: never call a text-only model and never silently inject OCR text.
    if not image_path:
        return {"answer": "Page image not found for visual generation.", "source": {"notice_id": notice_id, "page_number": page_number}, "generation_method": "error_no_image", "status": "error"}

    crop = retrieved_result.get("best_tile_box") if mode in ("visual", "visual_reranked") else None
    crop_box = tuple(crop) if crop and len(crop) == 4 else None
    return _provider_result(question, notice_id, page_number, image_path=str(image_path), crop_box=crop_box)
