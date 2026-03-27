"""
Frontier model judge / verifier.

After retrieval produces a shortlist, the judge evaluates each candidate
against the user's actual request and returns structured verdicts:
  - exact_match: truly satisfies all criteria
  - close_alternative: somewhat related but with notable differences
  - reject: clearly wrong

Also produces an honest overall inventory summary for truthful messaging.

For image search, optionally receives the query image + candidate images
for true multimodal visual evaluation.
"""

import base64
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from app.services.retriever import RetrievalResult


# ── Error type ────────────────────────────────────────────────────────────────

class JudgeUnavailableError(Exception):
    """Raised when the judge cannot evaluate candidates (no API key or provider error)."""
    pass


# ── Data types ────────────────────────────────────────────────────────────────


@dataclass
class CandidateJudgment:
    product_id: int
    verdict: str   # "exact_match" | "close_alternative" | "reject"
    reason: str    # short reason (≤ 15 words)


@dataclass
class JudgeResult:
    judgments: list[CandidateJudgment] = field(default_factory=list)
    overall_summary: str = ""

    def exact_matches(self) -> list[CandidateJudgment]:
        return [j for j in self.judgments if j.verdict == "exact_match"]

    def close_alternatives(self) -> list[CandidateJudgment]:
        return [j for j in self.judgments if j.verdict == "close_alternative"]

    def accepted(self) -> list[CandidateJudgment]:
        return [j for j in self.judgments if j.verdict != "reject"]

    def rejected(self) -> list[CandidateJudgment]:
        return [j for j in self.judgments if j.verdict == "reject"]


# ── Prompts ───────────────────────────────────────────────────────────────────

_TEXT_JUDGE_SYSTEM = """\
You are a strict product match judge for a fashion shopping assistant.

Given a user's request and candidate products from a catalog, evaluate each candidate.

Verdicts:
- "exact_match": truly satisfies ALL the user's criteria (right category, color, price range — both above any minimum AND below any maximum — style, gender, brand, plain vs patterned, etc.)
- "close_alternative": same general type but a notable difference exists (slightly wrong color, patterned instead of plain, slightly different subcategory, minor price difference, etc.)
- "reject": clearly wrong — completely different category, violates hard constraints (wrong brand when brand was specified, price outside stated range), obviously unrelated

Rules:
- Be strict. 1 great match beats 5 weak ones.
- If user said "plain" or "solid", reject striped/patterned/printed items.
- If user gave a specific color (e.g. "white"), reject items with clearly different colors (e.g. blue, red).
- If user gave a maximum price (e.g. "under $80"), REJECT items above that price — no exceptions.
- If user gave a minimum price (e.g. "over $100", "above $50"), REJECT items at or below that price.
- If user named a specific brand (e.g. "Lee Cooper", "Nike"), REJECT items from other brands unless truly nothing from the requested brand exists.
- If the query is vague/broad (no specific constraints), use "close_alternative" generously.
- Always include an honest overall_summary stating: how many exact matches found, and if none, why (wrong price range, wrong brand, wrong category, etc.).

Output ONLY valid JSON — no markdown fences, no extra text:
{
  "judgments": [
    {"product_id": <int>, "verdict": "exact_match|close_alternative|reject", "reason": "<brief reason, max 15 words>"}
  ],
  "overall_summary": "<honest 1-2 sentence summary: what was found, what was unavailable, why>"
}
"""

_IMAGE_JUDGE_SYSTEM = """\
You are a strict visual product match judge for a fashion shopping assistant.

You will see a query image uploaded by the user, followed by candidate product images.
Evaluate each candidate's visual similarity to the query image.

Verdicts:
- "exact_match": visually very similar — same garment type, very similar dominant color, similar style/silhouette
- "close_alternative": same general category but a notable visual difference (different color, patterned vs plain, different fit/silhouette)
- "reject": clearly different — wrong garment type, completely different colors/style, obvious visual mismatch

Rules:
- A striped shirt is NOT a close match for a plain white shirt.
- A bag is NOT a match for a shirt regardless of color.
- Pay attention to: garment type, dominant color, plain vs patterned, overall silhouette.
- If nothing in the catalog is visually close, say so honestly in overall_summary.

Output ONLY valid JSON — no markdown fences, no extra text:
{
  "judgments": [
    {"product_id": <int>, "verdict": "exact_match|close_alternative|reject", "reason": "<brief visual reason, max 15 words>"}
  ],
  "overall_summary": "<honest 1-2 sentence summary of visual similarity found in this catalog>"
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────

def _format_candidates_text(candidates: list[RetrievalResult]) -> str:
    lines = []
    for p in candidates:
        line = (
            f"[ID={p.id}] {p.product_name} | "
            f"Category: {p.category} | "
            f"Color: {p.base_color or 'N/A'} | "
            f"Usage: {p.usage or 'N/A'} | "
            f"Gender: {p.gender} | "
            f"Price: ${p.price:.2f}"
        )
        if p.description:
            desc_snippet = p.description[:100].replace("\n", " ").strip()
            line += f" | Desc: {desc_snippet}"
        lines.append(line)
    return "\n".join(lines)


def _parse_judge_json(raw: str, candidates: list[RetrievalResult]) -> JudgeResult:
    """Parse JSON from judge response; graceful fallback on parse error."""
    valid_ids = {p.id for p in candidates}
    try:
        clean = re.sub(r"```[a-z]*\n?", "", raw).strip().rstrip("```").strip()
        data = json.loads(clean)
        judgments = []
        for j in data.get("judgments", []):
            pid = int(j.get("product_id", -1))
            if pid not in valid_ids:
                continue
            verdict = j.get("verdict", "close_alternative")
            if verdict not in ("exact_match", "close_alternative", "reject"):
                verdict = "close_alternative"
            judgments.append(CandidateJudgment(
                product_id=pid,
                verdict=verdict,
                reason=str(j.get("reason", "")).strip(),
            ))
        return JudgeResult(
            judgments=judgments,
            overall_summary=str(data.get("overall_summary", "")).strip(),
        )
    except Exception:
        # Fallback: mark all as close_alternative so nothing is silently dropped
        return JudgeResult(
            judgments=[
                CandidateJudgment(product_id=p.id, verdict="close_alternative", reason="")
                for p in candidates
            ],
            overall_summary="Found some candidates; quality check unavailable.",
        )


def _load_image_b64(image_path: str, dataset_root: str, thumbnail_dir: str = "") -> Optional[str]:
    """Load a product image as base64. Tries thumbnail first, then original."""
    pid = Path(image_path).stem
    if thumbnail_dir:
        thumb = Path(thumbnail_dir) / f"{pid}.jpg"
        if thumb.exists():
            return base64.b64encode(thumb.read_bytes()).decode()
    full = Path(dataset_root) / image_path
    if full.exists():
        return base64.b64encode(full.read_bytes()).decode()
    return None


# ── Public API ─────────────────────────────────────────────────────────────────

def judge_text_candidates(
    user_ask: str,
    search_state_summary: str,
    candidates: list[RetrievalResult],
    api_key: str = "",
) -> JudgeResult:
    """
    Judge a text-search shortlist against the user's request.

    Returns JudgeResult on success.
    Returns empty JudgeResult (no judgments) when candidates list is empty.
    Raises JudgeUnavailableError when the LLM cannot be called — no score-based fallback.
    """
    if not candidates:
        print("[JUDGE] text: no candidates", flush=True)
        return JudgeResult(overall_summary="No products found for that request.")

    if not api_key:
        print(f"[JUDGE] text: unavailable reason=no_api_key candidates={len(candidates)}", flush=True)
        raise JudgeUnavailableError("no_api_key")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        products_text = _format_candidates_text(candidates)
        user_content = (
            f'User request: "{user_ask}"\n'
            f"Active search context: {search_state_summary}\n\n"
            f"Candidates to evaluate ({len(candidates)} items):\n{products_text}"
        )
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=_TEXT_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_content}],
        )
        result = _parse_judge_json(resp.content[0].text.strip(), candidates)
        print(
            f"[JUDGE] text: source=LLM "
            f"exact={len(result.exact_matches())} "
            f"close={len(result.close_alternatives())} "
            f"rejected={len(result.rejected())}",
            flush=True,
        )
        return result
    except JudgeUnavailableError:
        raise
    except Exception as exc:
        print(f"[JUDGE] text: unavailable reason=provider_error exc={exc!r}", flush=True)
        raise JudgeUnavailableError(f"provider_error: {exc}") from exc


def judge_image_candidates(
    user_text: str,
    query_image_b64: str,
    candidates: list[RetrievalResult],
    api_key: str = "",
    dataset_root: str = "",
    thumbnail_dir: str = "",
    max_visual: int = 12,
) -> JudgeResult:
    """
    Judge image-search candidates using multimodal vision.
    Sends query image + candidate images to the model.

    Returns JudgeResult on success.
    Returns empty JudgeResult when candidates list is empty.
    Raises JudgeUnavailableError when the LLM cannot be called — no score-based fallback.
    """
    if not candidates:
        print("[JUDGE] image: no candidates", flush=True)
        return JudgeResult(overall_summary="No visually similar products found.")

    if not api_key:
        print(f"[JUDGE] image: unavailable reason=no_api_key candidates={len(candidates)}", flush=True)
        raise JudgeUnavailableError("no_api_key")

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)

        visual_batch = candidates[:max_visual]
        text_only_batch = candidates[max_visual:]

        content: list = []

        # Query image
        label = "Query image"
        if user_text.strip():
            label += f' — user also said: "{user_text.strip()}"'
        content.append({"type": "text", "text": label + ":\n"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": query_image_b64},
        })

        content.append({"type": "text", "text": "\nCandidate products from catalog:\n"})

        images_loaded = 0
        text_only_fallback: list[RetrievalResult] = []

        for p in visual_batch:
            img_b64 = _load_image_b64(p.image_path, dataset_root, thumbnail_dir)
            if img_b64:
                content.append({
                    "type": "text",
                    "text": f"\n[ID={p.id}] {p.product_name} | {p.category} | {p.base_color or 'N/A'} | ${p.price:.2f}:\n",
                })
                content.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": img_b64},
                })
                images_loaded += 1
            else:
                text_only_fallback.append(p)

        # Append text-only candidates
        remaining = text_only_fallback + text_only_batch
        if remaining:
            content.append({
                "type": "text",
                "text": "\nAdditional candidates (no image available — metadata only):\n"
                        + _format_candidates_text(remaining),
            })

        if images_loaded == 0:
            # No product images loadable — fall back to text-only LLM judge
            return judge_text_candidates(
                user_text or "visually similar items",
                "image search — no candidate images available",
                candidates,
                api_key,
            )

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1500,
            system=_IMAGE_JUDGE_SYSTEM,
            messages=[{"role": "user", "content": content}],
        )
        result = _parse_judge_json(resp.content[0].text.strip(), candidates)
        print(
            f"[JUDGE] image: source=LLM images_loaded={images_loaded} "
            f"exact={len(result.exact_matches())} "
            f"close={len(result.close_alternatives())} "
            f"rejected={len(result.rejected())}",
            flush=True,
        )
        return result

    except JudgeUnavailableError:
        raise
    except Exception as exc:
        print(f"[JUDGE] image: unavailable reason=provider_error exc={exc!r}", flush=True)
        raise JudgeUnavailableError(f"provider_error: {exc}") from exc
