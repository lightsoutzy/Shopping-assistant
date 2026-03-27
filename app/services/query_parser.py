"""
Extracts structured shopping constraints from a natural language query.
Uses Claude when an API key is available; falls back to keyword extraction otherwise.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ParsedQuery:
    category: Optional[str] = None    # one of the 8 demo categories
    color: Optional[str] = None
    gender: Optional[str] = None      # Men | Women | Unisex
    usage: Optional[str] = None
    season: Optional[str] = None
    max_price: Optional[float] = None
    min_price: Optional[float] = None  # "over $100", "above $50"
    brand: Optional[str] = None        # specific brand name, e.g. "Lee Cooper"
    requested_count: Optional[int] = None  # "give me 3", "show me two options"
    plain_only: bool = False           # user wants plain/solid items, not patterned
    free_text: str = ""


# ── LLM parser ────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are a shopping assistant parser. Extract constraints from the user query and return JSON only.

JSON fields (all nullable):
  category       : one of [t-shirt, shirt, shoes, sneakers, jacket, hoodie, shorts, bag] or null
  color          : lowercase color name or null
  gender         : one of [Men, Women, Unisex] or null
  usage          : one of [Casual, Sports, Formal, Smart Casual, Travel, Party] or null
  season         : one of [Summer, Winter, Fall, Spring] or null
  max_price      : number (USD) or null — from "under/below/less than $N"
  min_price      : number (USD) or null — from "over/above/more than/at least $N"
  brand          : exact brand name string if user names a specific brand (e.g. "Lee Cooper", "Nike"), else null
  requested_count: integer or null — extract when user asks for a specific number of items
                   e.g. "give me 3" → 3, "show me two options" → 2, "one jacket" → 1
  plain_only     : true if user explicitly wants plain/solid/unpatterned items, else false
  free_text      : the original query, unchanged

Brand detection: if the entire query (or a significant part) is a brand/company name with no other
shopping keywords, extract it as brand. E.g. "Lee Cooper" → brand="Lee Cooper", "Lotto shoes" → brand="Lotto".

Return valid JSON only. No explanation, no markdown.
"""


def parse_with_llm(text: str, api_key: str) -> ParsedQuery:
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    msg = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        system=_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": text}],
    )
    raw = msg.content[0].text.strip()
    data = json.loads(raw)
    rc = data.get("requested_count")
    return ParsedQuery(
        category=data.get("category"),
        color=data.get("color"),
        gender=data.get("gender"),
        usage=data.get("usage"),
        season=data.get("season"),
        max_price=data.get("max_price"),
        min_price=data.get("min_price"),
        brand=data.get("brand") or None,
        requested_count=int(rc) if rc is not None else None,
        plain_only=bool(data.get("plain_only", False)),
        free_text=data.get("free_text", text),
    )


# ── Keyword fallback ───────────────────────────────────────────────────────────

_PLAIN_WORDS = {"plain", "solid", "simple", "minimal", "minimalist", "unpatterned",
                "no pattern", "no print", "no graphic", "block color", "block colour"}

_CATEGORIES = ["t-shirt", "shirt", "shoes", "sneakers", "jacket", "hoodie", "shorts", "bag"]
_CATEGORY_ALIASES = {
    "tshirt": "t-shirt", "t shirt": "t-shirt", "sneaker": "sneakers",
    "handbag": "bag", "backpack": "bag", "sweatshirt": "hoodie",
    "sweater": "hoodie",
}
_COLORS = [
    "black", "white", "blue", "red", "green", "grey", "gray", "brown",
    "pink", "yellow", "navy", "beige", "orange", "purple",
]
_USAGES = {
    "sport": "Sports", "sports": "Sports", "casual": "Casual",
    "formal": "Formal", "smart casual": "Smart Casual",
    "travel": "Travel", "party": "Party",
}
_SEASONS = {
    "summer": "Summer", "winter": "Winter",
    "fall": "Fall", "autumn": "Fall", "spring": "Spring",
}
_GENDERS = {
    "men": "Men", "man": "Men", "male": "Men", "boys": "Men",
    "women": "Women", "woman": "Women", "female": "Women", "girls": "Women",
    "unisex": "Unisex",
}
_PRICE_RE = re.compile(r"\$?\b(\d+(?:\.\d+)?)\b", re.IGNORECASE)
_UNDER_RE = re.compile(
    r"(?:under|less than|below|max|maximum|no more than)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_OVER_RE = re.compile(
    r"(?:over|above|more than|at least|minimum|min)\s*\$?\s*(\d+(?:\.\d+)?)",
    re.IGNORECASE,
)
_COUNT_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10}
_COUNT_RE = re.compile(
    r"\b(?:show|give|find|get|fetch|list|display|recommend)\s+(?:me\s+)?(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b"
    r"|\bi\s+(?:want|need)\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\b",
    re.IGNORECASE,
)


def parse_with_keywords(text: str) -> ParsedQuery:
    lower = text.lower()
    result = ParsedQuery(free_text=text)

    # category — check aliases first to avoid "shirt" matching inside "tshirt"
    for alias, cat in _CATEGORY_ALIASES.items():
        if alias in lower:
            result.category = cat
            break
    if result.category is None:
        for cat in _CATEGORIES:
            if cat in lower:
                result.category = cat
                break

    # color
    for color in _COLORS:
        if re.search(rf"\b{color}\b", lower):
            result.color = color
            break

    # gender
    for kw, gender in _GENDERS.items():
        if re.search(rf"\b{kw}\b", lower):
            result.gender = gender
            break

    # usage
    for kw, usage in _USAGES.items():
        if kw in lower:
            result.usage = usage
            break

    # season
    for kw, season in _SEASONS.items():
        if kw in lower:
            result.season = season
            break

    # max price (under/below/less than)
    m = _UNDER_RE.search(text)
    if m:
        result.max_price = float(m.group(1))

    # min price (over/above/more than)
    m2 = _OVER_RE.search(text)
    if m2:
        result.min_price = float(m2.group(1))

    # requested count ("give me 3", "show me two options")
    m3 = _COUNT_RE.search(text)
    if m3:
        raw = (m3.group(1) or m3.group(2) or "").lower()
        result.requested_count = _COUNT_WORDS.get(raw) or (int(raw) if raw.isdigit() else None)

    # plain/solid preference
    result.plain_only = any(w in lower for w in _PLAIN_WORDS)

    return result


# ── Public entry point ────────────────────────────────────────────────────────

def parse_query(text: str, api_key: str = "") -> ParsedQuery:
    """Parse query using LLM if key is available, otherwise keyword fallback."""
    if api_key:
        try:
            return parse_with_llm(text, api_key)
        except Exception:
            pass
    return parse_with_keywords(text)
