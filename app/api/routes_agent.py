"""
POST /agent — unified endpoint for all user interactions.

Orchestration flow:
  1. rate limit check
  2. planner  → determines action
  3. non-retrieval actions handled directly (state questions, capabilities, etc.)
  4. retrieval actions: retrieve shortlist → judge → compose response
"""

import base64
import io
import logging
import re
import uuid

from fastapi import APIRouter, HTTPException, Request

logger = logging.getLogger("agent")

from app import config
from app.schemas import (
    AcceptedProduct,
    AgentRequest,
    AgentResponse,
    ProductItem,
    SearchState,
    _state_summary,
)
from app.services import query_parser
from app.services.judge import judge_image_candidates, judge_text_candidates
from app.services.planner import plan as make_plan
from app.services.rate_limiter import check_rate_limit, record_llm_call
from app.services.judge import JudgeUnavailableError
from app.services.orchestrator import LLM_UNAVAILABLE_MSG, LLMUnavailableError, ToolBundle, compose_turn
from app.services.response_generator import (
    generate_chat,
    generate_clarification,
    generate_scope_refusal,
)

router = APIRouter()

# ── Constants ─────────────────────────────────────────────────────────────────

_SHORTLIST_SIZE = 30   # candidates fetched from retrieval before judging
_MAX_SHOWN = 5         # max products shown per response
_NEXT_BATCH_SIZE = 3   # products shown for "show me another"

# ── Small helpers ─────────────────────────────────────────────────────────────

_BEST_ONE_MSG_RE = re.compile(
    r"\bbest (one|option|item|choice)\b"
    r"|\bif you (had to|have to) (recommend|pick|choose)\b"
    r"|\bwhich (one|would you) (recommend|pick|choose)\b"
    r"|\bwhich (is|are) (best|closest|most versatile)\b"
    r"|\bnarrow (it |this )?down\b"
    r"|\bamong these\b",
    re.IGNORECASE,
)

# Catalog categories supported for bundle decomposition
_BUNDLE_CAT_MAP: dict[str, str] = {
    "hoodie": "hoodie", "sweatshirt": "hoodie",
    "jacket": "jacket", "outerwear": "jacket",
    "shirt": "shirt", "t-shirt": "shirt", "tshirt": "shirt", "top": "shirt",
    "shoe": "shoes", "shoes": "shoes",
    "sneaker": "sneakers", "sneakers": "sneakers", "boot": "shoes",
    "pant": "shorts", "trouser": "shorts", "jean": "shorts", "short": "shorts",
    "bag": "bag", "backpack": "bag", "handbag": "bag",
}

_COLORS = [
    "white", "black", "blue", "red", "green", "grey", "gray", "brown",
    "pink", "yellow", "navy", "beige", "orange", "purple",
]
_USAGES = {
    "casual": "Casual", "sport": "Sports", "sports": "Sports",
    "formal": "Formal", "smart casual": "Smart Casual", "travel": "Travel",
}


def _product_item_from_row(row, verdict: str, reason: str) -> ProductItem:
    return ProductItem(
        id=int(row["id"]),
        product_name=str(row["product_name"]),
        brand=row.get("brand") or None,
        category=str(row["category"]),
        base_color=row.get("base_color") or None,
        usage=row.get("usage") or None,
        season=row.get("season") or None,
        gender=str(row.get("gender", "Unisex")),
        price=float(row.get("price", 0)),
        image_path=str(row.get("image_path", "")),
        reason=reason,
        match_tier="perfect" if verdict == "exact_match" else "close",
        match_score=90 if verdict == "exact_match" else 70,
    )


def _build_free_text(state: SearchState, extra: str = "") -> str:
    """Build a TF-IDF query string from current state + extra terms."""
    parts = []
    if state.plain_only:
        parts.append("plain solid")
    if state.color:
        parts.append(state.color)
    if state.category:
        parts.append(state.category)
    if state.usage:
        parts.append(state.usage)
    if state.brand:
        parts.append(state.brand)
    if extra:
        parts.append(extra)
    return " ".join(parts) if parts else extra or "fashion item"


def _filter_active_by_color_or_usage(active: list[dict], question: str) -> tuple[list[dict], str]:

    """Quick filter for followup_on_results: colour/usage hard filter."""
    q = question.lower()
    filtered = list(active)
    desc = ""

    for color in _COLORS:
        if re.search(rf"\b{color}\b", q):
            new_f = [p for p in filtered if color in (p.get("base_color") or "").lower()]
            if new_f:
                filtered = new_f
                desc = f"color: {color}"
            else:
                filtered = []
                desc = f"color: {color}"
            break

    # --- brand filter (only check brands present in active results) -----------
    if filtered and not desc:
        known_brands = {
            (p.get("brand") or "").lower(): p.get("brand", "")
            for p in active
            if p.get("brand")
        }
        for brand_lower, brand_display in known_brands.items():
            if brand_lower and re.search(rf"\b{re.escape(brand_lower)}\b", q):
                new_f = [p for p in filtered if brand_lower in (p.get("brand") or "").lower()]
                if new_f:
                    filtered = new_f
                    desc = f"brand: {brand_display}"
                else:
                    filtered = []
                    desc = f"brand: {brand_display}"
                break

    if filtered and not desc:
        for kw, usage_val in _USAGES.items():
            if kw in q:
                new_f = [p for p in filtered if usage_val.lower() in (p.get("usage") or "").lower()]
                if new_f:
                    filtered = new_f
                    desc = f"usage: {usage_val}"
                else:
                    filtered = []
                    desc = f"usage: {usage_val}"
                break

    # Price sort (not a filter)
    if any(w in q for w in ["cheapest", "least expensive", "lowest price"]):
        filtered = sorted(filtered, key=lambda p: p.get("price", 999))
    elif any(w in q for w in ["most expensive", "priciest", "highest price"]):
        filtered = sorted(filtered, key=lambda p: -p.get("price", 0))

    return filtered, desc


def _filter_active_products(
    active_products: list[dict],
    question: str,
) -> tuple[list[dict], str, bool]:
    """
    Backward-compatible 3-return-value wrapper used by existing tests.
    Returns (filtered, description, has_hard_filter).
    """
    q = question.lower()
    filtered, desc = _filter_active_by_color_or_usage(active_products, question)

    # has_hard_filter = True when a colour, brand, or usage filter was attempted
    known_brands = {(p.get("brand") or "").lower() for p in active_products if p.get("brand")}
    has_hard_filter = (
        any(re.search(rf"\b{c}\b", q) for c in _COLORS)
        or any(kw in q for kw in _USAGES)
        or any(b and re.search(rf"\b{re.escape(b)}\b", q) for b in known_brands)
    )
    return filtered, desc, has_hard_filter


# ── LLM error response helper ─────────────────────────────────────────────────

def _llm_error_response(
    intent: str,
    session_id: str,
    reason: str,
    search_state=None,
) -> AgentResponse:
    """Return a clean error response when the LLM path is unavailable.  No products, no fake text."""
    print(f"[AGENT] LLM unavailable intent={intent!r} reason={reason!r}", flush=True)
    return AgentResponse(
        intent=intent,
        message=LLM_UNAVAILABLE_MSG,
        products=[],
        session_id=session_id,
        context_action="keep",
        search_state=search_state or SearchState(),
    )


# ── Non-retrieval action handlers ─────────────────────────────────────────────

def _handle_ask_current_state(req: AgentRequest, session_id: str) -> AgentResponse:
    ss = req.search_state
    parts = []
    if ss.category:
        parts.append(f"looking for **{ss.category}**")
    if ss.color:
        parts.append(f"in **{ss.color}**")
    if ss.max_price:
        parts.append(f"under **${ss.max_price:.0f}**")
    if ss.gender:
        parts.append(f"for **{ss.gender}**")
    if ss.usage:
        parts.append(f"({ss.usage} style)")
    if ss.plain_only:
        parts.append("*(plain/solid only)*")
    if ss.image_active and ss.image_summary:
        parts.append(f"*(image search: {ss.image_summary})*")

    if parts:
        msg = "Your current search is: " + ", ".join(parts) + "."
        if ss.result_count:
            had = ss.had_perfect_matches
            quality = " exact matches" if had else " close alternatives"
            msg += f" Last search returned **{ss.result_count}**{quality}."
    else:
        msg = "I don't have an active search yet. What would you like to find?"

    return AgentResponse(
        intent="ask_current_state",
        message=msg,
        session_id=session_id,
        context_action="keep",
        search_state=ss,
    )


def _handle_explain_match_quality(
    req: AgentRequest, session_id: str, api_key: str
) -> AgentResponse:
    ss = req.search_state
    had_perfect = ss.had_perfect_matches

    # Build a structured context for the LLM to explain quality naturally
    if had_perfect is None:
        context = "The user is asking about match quality but no search has been run yet."
    elif had_perfect:
        context = (
            f"The {ss.result_count} item(s) currently shown are exact matches "
            f"that satisfy all the user's criteria."
        )
    else:
        rejected = ss.rejected_summary or "The catalog may not have an exact fit."
        context = (
            f"The search found no exact matches. {ss.result_count} close alternative(s) are shown. "
            f"Reason from catalog assessment: {rejected}"
        )

    record_llm_call(session_id)
    msg = generate_chat(
        f"The user asked: \"{req.text}\"\n\n"
        f"Explain the search quality naturally in 1-2 sentences: {context}\n"
        f"Be honest and specific. Don't use generic phrases.",
        api_key,
    )
    print("[AGENT]   → explain_match_quality via LLM", flush=True)

    return AgentResponse(
        intent="explain_match_quality",
        message=msg,
        session_id=session_id,
        context_action="keep",
        search_state=ss,
    )


def _handle_next_result(req: AgentRequest, state, session_id: str, api_key: str) -> AgentResponse:
    ss = req.search_state
    pool = ss.accepted_pool
    shown_ids = set(ss.shown_product_ids)

    unshown = [p for p in pool if p.product_id not in shown_ids]

    if not unshown:
        record_llm_call(session_id)
        bundle = ToolBundle(
            user_message=req.text,
            action="next_result",
            chat_history=req.chat_history,
            search_state_summary=_state_summary(ss),
            active_products=req.active_products,
            shown_products=[],
            n_exact=0,
            n_close=0,
            filter_context="All results from this search have already been shown.",
        )
        try:
            msg = compose_turn(bundle, api_key, config.ORCHESTRATOR_TEMPERATURE)
        except (LLMUnavailableError, JudgeUnavailableError) as exc:
            return _llm_error_response("next_result", session_id, str(exc), ss)
        return AgentResponse(
            intent="next_result",
            message=msg,
            session_id=session_id,
            context_action="keep",
            search_state=ss,
        )

    next_batch = unshown[:_NEXT_BATCH_SIZE]
    df = state.catalog.df
    product_items: list[ProductItem] = []

    for pool_item in next_batch:
        rows = df[df["id"] == pool_item.product_id]
        if rows.empty:
            continue
        row = rows.iloc[0]
        product_items.append(_product_item_from_row(row, pool_item.verdict, pool_item.reason))

    if not product_items:
        return AgentResponse(
            intent="next_result",
            message="Couldn't load the next items. Try a new search.",
            session_id=session_id,
            context_action="keep",
            search_state=ss,
        )

    new_shown = list(shown_ids) + [p.id for p in product_items]
    new_state = ss.model_copy(update={"shown_product_ids": new_shown})

    n_exact = sum(1 for p in product_items if p.match_tier == "perfect")
    n_close = len(product_items) - n_exact
    record_llm_call(session_id)
    bundle = ToolBundle(
        user_message=req.text,
        action="next_result",
        chat_history=req.chat_history,
        search_state_summary=_state_summary(ss),
        active_products=req.active_products,
        shown_products=product_items,
        n_exact=n_exact,
        n_close=n_close,
    )
    try:
        msg = compose_turn(bundle, api_key, config.ORCHESTRATOR_TEMPERATURE)
    except (LLMUnavailableError, JudgeUnavailableError) as exc:
        return _llm_error_response("next_result", session_id, str(exc), ss)
    print(f"[AGENT]   → next_result: showing {len(product_items)} from pool", flush=True)
    return AgentResponse(
        intent="next_result",
        message=msg,
        products=product_items,
        session_id=session_id,
        context_action="keep",
        search_state=new_state,
    )


def _handle_followup(req: AgentRequest, session_id: str, api_key: str) -> AgentResponse:
    active = req.active_products
    ss = req.search_state   # preserved and returned on every path

    if not active:
        record_llm_call(session_id)
        no_context_msg = generate_chat(
            f"The user asked a follow-up question (\"{req.text}\") but there are no products "
            "currently in context. Tell them briefly and invite them to start a search.",
            api_key,
        )
        return AgentResponse(
            intent="followup_on_results",
            message=no_context_msg,
            session_id=session_id,
            context_action="keep",
            search_state=ss,
        )

    filtered, filter_desc = _filter_active_by_color_or_usage(active, req.text)

    if filter_desc and not filtered:
        # No items in active results match the requested filter
        record_llm_call(session_id)
        bundle = ToolBundle(
            user_message=req.text,
            action="followup_on_results",
            chat_history=req.chat_history,
            search_state_summary=_state_summary(ss),
            active_products=active,
            shown_products=[],
            n_exact=0,
            n_close=0,
            filter_context=(
                f"The user asked for {filter_desc} but NONE of the currently shown products "
                f"match that filter. Mention what IS available and offer to search the full catalog."
            ),
        )
        print(f"[AGENT]   → followup filter-no-match: filter={filter_desc!r}", flush=True)
        try:
            msg = compose_turn(bundle, api_key, config.ORCHESTRATOR_TEMPERATURE)
        except (LLMUnavailableError, JudgeUnavailableError) as exc:
            return _llm_error_response("followup_on_results", session_id, str(exc), ss)
        return AgentResponse(
            intent="followup_on_results",
            message=msg,
            products=[],
            session_id=session_id,
            context_action="keep",
            search_state=ss,
        )

    products_to_show = filtered if filtered else active

    # ── "Best one" / "which one" handling ────────────────────────────────────
    # When the user asks which single item is best, pick the top-scored one and
    # tell the orchestrator to recommend it — do NOT expand the search universe.
    is_best_one = bool(_BEST_ONE_MSG_RE.search(req.text))
    filter_context = f"User applied filter: {filter_desc}" if filter_desc else ""

    if is_best_one and products_to_show:
        # Sort by match_score desc, pick the single best item
        sorted_items = sorted(products_to_show, key=lambda p: p.get("match_score", 0), reverse=True)
        products_to_show = sorted_items[:1]
        filter_context = (
            "User asked which single item from the shown results is best. "
            "Recommend ONLY the one item shown below. Do not expand the search or mention other items."
        )
        print(f"[AGENT]   → followup best-one: picked product_id={products_to_show[0].get('id')}", flush=True)

    try:
        product_items = [ProductItem(**p) for p in products_to_show]
    except Exception:
        product_items = []

    n_exact = 1 if is_best_one and product_items else 0
    n_close = 0 if is_best_one else len(product_items)

    record_llm_call(session_id)
    bundle = ToolBundle(
        user_message=req.text,
        action="followup_on_results",
        chat_history=req.chat_history,
        search_state_summary=_state_summary(ss),
        active_products=active,
        shown_products=product_items,
        n_exact=n_exact,
        n_close=n_close,
        filter_context=filter_context,
    )
    try:
        message = compose_turn(bundle, api_key, config.ORCHESTRATOR_TEMPERATURE)
    except (LLMUnavailableError, JudgeUnavailableError) as exc:
        return _llm_error_response("followup_on_results", session_id, str(exc), ss)

    if is_best_one:
        new_action = "replace"
    else:
        new_action = "replace" if (filter_desc and filtered) else "keep"

    return AgentResponse(
        intent="followup_on_results",
        message=message,
        products=product_items,
        session_id=session_id,
        context_action=new_action,
        search_state=ss,
    )


# ── Retrieval + judge pipeline ────────────────────────────────────────────────

def _build_search_state_from_constraints(
    constraints: query_parser.ParsedQuery,
    req: AgentRequest,
    base_state: SearchState,
) -> SearchState:
    """Merge parsed constraints with the base (existing) state."""
    return SearchState(
        category=constraints.category or base_state.category,
        color=constraints.color or base_state.color,
        max_price=constraints.max_price or req.max_price or base_state.max_price,
        min_price=constraints.min_price or base_state.min_price,
        brand=constraints.brand or base_state.brand,
        gender=constraints.gender or base_state.gender,
        usage=constraints.usage or base_state.usage,
        plain_only=constraints.plain_only or base_state.plain_only,
    )


def _run_text_search(
    req: AgentRequest,
    state,
    session_id: str,
    api_key: str,
    action: str,
) -> AgentResponse:
    """
    For actions: new_search, refine_search
    Parse → retrieve shortlist → judge → compose.
    """
    # For refine_search, inherit existing state; for new_search, start fresh
    base_state = req.search_state if action == "refine_search" else SearchState()

    record_llm_call(session_id)
    constraints = query_parser.parse_query(req.text, api_key)
    merged_state = _build_search_state_from_constraints(constraints, req, base_state)

    # Build retrieval query
    retrieval_text = constraints.free_text or _build_free_text(merged_state)
    if action == "refine_search" and not constraints.free_text.strip():
        retrieval_text = _build_free_text(merged_state)

    print(
        f"[AGENT]   → constraints: category={constraints.category!r} color={constraints.color!r} "
        f"brand={constraints.brand!r} min_price={constraints.min_price} "
        f"max_price={constraints.max_price} count={constraints.requested_count} "
        f"image_this_turn={bool(req.image_b64)}",
        flush=True,
    )

    retriever = state.retriever
    shortlist = retriever.search(
        query=retrieval_text,
        category=merged_state.category or req.category,
        brand=merged_state.brand,
        color=merged_state.color,
        gender=merged_state.gender,
        usage=merged_state.usage,
        max_price=merged_state.max_price,
        min_price=merged_state.min_price,
        top_k=_SHORTLIST_SIZE,
    )

    print(
        f"[AGENT]   → retrieved {len(shortlist)} candidates "
        f"(category={merged_state.category!r} color={merged_state.color!r} "
        f"brand={merged_state.brand!r} min_price={merged_state.min_price} "
        f"max_price={merged_state.max_price})",
        flush=True,
    )

    state_summary = _state_summary(merged_state)
    record_llm_call(session_id)
    try:
        judge_result = judge_text_candidates(
            user_ask=req.text,
            search_state_summary=state_summary,
            candidates=shortlist,
            api_key=api_key,
        )
    except JudgeUnavailableError as exc:
        return _llm_error_response(action, session_id, str(exc), req.search_state)

    exact_js = judge_result.exact_matches()
    close_js = judge_result.close_alternatives()
    n_rejected = len(judge_result.rejected())

    print(
        f"[AGENT]   → judge: exact={len(exact_js)} close={len(close_js)} rejected={n_rejected}",
        flush=True,
    )
    if exact_js:
        print(f"[AGENT]   → accepted exact: {[j.product_id for j in exact_js[:3]]}", flush=True)
    if close_js:
        print(f"[AGENT]   → accepted close: {[j.product_id for j in close_js[:3]]}", flush=True)

    # Select which judgments to show — respect requested_count if specified
    max_to_show = constraints.requested_count or _MAX_SHOWN
    if exact_js:
        shown_js = exact_js[:max_to_show]
    elif close_js:
        shown_js = close_js[:max_to_show]
    else:
        shown_js = []

    # Build ProductItem list
    shortlist_by_id = {p.id: p for p in shortlist}
    df = state.catalog.df
    product_items: list[ProductItem] = []

    for j in shown_js:
        p = shortlist_by_id.get(j.product_id)
        if p:
            product_items.append(_product_item_from_row(
                {
                    "id": p.id, "product_name": p.product_name, "brand": p.brand,
                    "category": p.category, "base_color": p.base_color,
                    "usage": p.usage, "season": p.season, "gender": p.gender,
                    "price": p.price, "image_path": p.image_path,
                },
                j.verdict, j.reason,
            ))

    # Build accepted pool for next_result
    all_accepted = exact_js + close_js
    accepted_pool = [
        AcceptedProduct(product_id=j.product_id, verdict=j.verdict, reason=j.reason)
        for j in all_accepted
    ]
    shown_ids = [p.id for p in product_items]

    had_perfect = len(exact_js) > 0
    new_state = SearchState(
        category=merged_state.category,
        color=merged_state.color,
        max_price=merged_state.max_price,
        min_price=merged_state.min_price,
        brand=merged_state.brand,
        gender=merged_state.gender,
        usage=merged_state.usage,
        plain_only=merged_state.plain_only,
        last_action=action,
        last_intent_action=action,
        last_query_summary=state_summary,
        result_count=len(product_items),
        had_perfect_matches=had_perfect,
        rejected_summary=judge_result.overall_summary,
        accepted_pool=accepted_pool,
        shown_product_ids=shown_ids,
        # Text search clears any sticky image context
        image_active=False,
        image_summary="",
    )

    record_llm_call(session_id)
    bundle = ToolBundle(
        user_message=req.text,
        action=action,
        chat_history=req.chat_history,
        search_state_summary=state_summary,
        active_products=req.active_products,
        shown_products=product_items,
        n_exact=len(exact_js),
        n_close=len(close_js),
        judge_result=judge_result,
        requested_count=constraints.requested_count,
        brand=merged_state.brand,
        min_price=merged_state.min_price,
        max_price=merged_state.max_price,
    )
    try:
        message = compose_turn(bundle, api_key, config.ORCHESTRATOR_TEMPERATURE)
    except LLMUnavailableError as exc:
        return _llm_error_response(action, session_id, str(exc), req.search_state)

    print(f"[AGENT]   → showing {len(product_items)} products", flush=True)

    return AgentResponse(
        intent=action,
        message=message,
        products=product_items,
        session_id=session_id,
        context_action="replace",
        search_state=new_state,
    )


def _run_image_search(
    req: AgentRequest,
    state,
    session_id: str,
    api_key: str,
) -> AgentResponse:
    if state.image_index is None:
        return AgentResponse(
            intent="image_search",
            message=(
                "Image search is not available yet. "
                "Run `python -m app.data.build_image_embeddings` first."
            ),
            session_id=session_id,
        )

    try:
        from PIL import Image as PILImage
        img_bytes = base64.b64decode(req.image_b64)
        pil_image = PILImage.open(io.BytesIO(img_bytes)).convert("RGB")
        # Re-encode as JPEG to guarantee consistent media type for the multimodal judge.
        # The original upload may be PNG; hardcoding image/jpeg in judge.py would cause
        # the Anthropic API to reject mismatched bytes.
        _jpeg_buf = io.BytesIO()
        pil_image.save(_jpeg_buf, format="JPEG", quality=85)
        query_image_b64_jpeg = base64.b64encode(_jpeg_buf.getvalue()).decode()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_b64 payload: {exc}")

    print(
        f"[AGENT]   → image received: yes  re-encoded-as-jpeg: yes",
        flush=True,
    )

    from app.services.image_search import search_by_image, rerank_image_results

    # Step 1: broad CLIP retrieval
    candidates = search_by_image(
        pil_image,
        state.image_index,
        state.clip_processor,
        state.clip_model,
        top_k=_SHORTLIST_SIZE,
    )

    # Step 2: parse text constraints if any
    text_constraints = query_parser.ParsedQuery(free_text=req.text or "")
    if req.text.strip():
        text_constraints = query_parser.parse_with_keywords(req.text)

    # Step 3: visual reranking for better shortlist ordering
    ranked, dominant_color = rerank_image_results(
        candidates,
        pil_image,
        text_constraints=text_constraints,
        top_k=_SHORTLIST_SIZE,  # keep all for judge
    )

    print(
        f"[AGENT]   → image search executed: yes  candidates={len(ranked)} "
        f"dominant_color={dominant_color!r}",
        flush=True,
    )

    # Step 4: multimodal judge
    record_llm_call(session_id)
    print(f"[AGENT]   → calling image judge  llm_called=yes", flush=True)
    try:
        judge_result = judge_image_candidates(
            user_text=req.text or "",
            query_image_b64=query_image_b64_jpeg,  # guaranteed JPEG
            candidates=ranked,
            api_key=api_key,
            dataset_root=str(config.DATASET_ROOT),
            thumbnail_dir="data/catalog_thumbnails",
            max_visual=12,
        )
    except JudgeUnavailableError as exc:
        return _llm_error_response("image_search", session_id, str(exc), req.search_state)

    exact_js = judge_result.exact_matches()
    close_js = judge_result.close_alternatives()
    n_rejected = len(judge_result.rejected())

    print(
        f"[AGENT]   → image judge: exact={len(exact_js)} close={len(close_js)} rejected={n_rejected}",
        flush=True,
    )

    # Select shown products
    if exact_js:
        shown_js = exact_js[:_MAX_SHOWN]
    elif close_js:
        shown_js = close_js[:_MAX_SHOWN]
    else:
        shown_js = []

    ranked_by_id = {p.id: p for p in ranked}
    product_items: list[ProductItem] = []

    for j in shown_js:
        p = ranked_by_id.get(j.product_id)
        if p:
            product_items.append(_product_item_from_row(
                {
                    "id": p.id, "product_name": p.product_name, "brand": p.brand,
                    "category": p.category, "base_color": p.base_color,
                    "usage": p.usage, "season": p.season, "gender": p.gender,
                    "price": p.price, "image_path": p.image_path,
                },
                j.verdict, j.reason,
            ))

    # Build pool
    all_accepted = exact_js + close_js
    accepted_pool = [
        AcceptedProduct(product_id=j.product_id, verdict=j.verdict, reason=j.reason)
        for j in all_accepted
    ]
    shown_ids = [p.id for p in product_items]

    image_summary = dominant_color or "uploaded image"
    new_state = SearchState(
        last_action="image_search",
        last_intent_action="image_search",
        result_count=len(product_items),
        had_perfect_matches=len(exact_js) > 0,
        rejected_summary=judge_result.overall_summary,
        accepted_pool=accepted_pool,
        shown_product_ids=shown_ids,
        image_active=True,
        image_summary=image_summary,
    )

    record_llm_call(session_id)
    bundle = ToolBundle(
        user_message=req.text or "find visually similar items",
        action="image_search",
        chat_history=req.chat_history,
        search_state_summary=f"image search, dominant color: {dominant_color or 'unknown'}",
        active_products=req.active_products,
        shown_products=product_items,
        n_exact=len(exact_js),
        n_close=len(close_js),
        judge_result=judge_result,
        image_dominant_color=dominant_color or "",
    )
    try:
        message = compose_turn(bundle, api_key, config.ORCHESTRATOR_TEMPERATURE)
    except LLMUnavailableError as exc:
        return _llm_error_response("image_search", session_id, str(exc), req.search_state)

    print(f"[AGENT]   → showing {len(product_items)} image results", flush=True)

    return AgentResponse(
        intent="image_search",
        message=message,
        products=product_items,
        session_id=session_id,
        context_action="replace",
        search_state=new_state,
    )


# ── Bundle search helpers ─────────────────────────────────────────────────────

def _extract_bundle_categories(text: str) -> tuple[str, str] | None:
    """
    Extract exactly 2 product categories from a bundle request text.
    Returns (cat_a, cat_b) or None if fewer than 2 distinct categories found.
    """
    found: list[str] = []
    t = text.lower()
    for kw, cat in _BUNDLE_CAT_MAP.items():
        if re.search(rf"\b{re.escape(kw)}s?\b", t) and cat not in found:
            found.append(cat)
        if len(found) == 2:
            break
    return (found[0], found[1]) if len(found) >= 2 else None


def _run_bundle_search(
    req: AgentRequest,
    state,
    session_id: str,
    api_key: str,
) -> AgentResponse:
    """
    Decompose a 2-item bundle/combo request, retrieve + judge each category separately,
    then build and rank bundle pairs under the shared total budget.
    """
    cats = _extract_bundle_categories(req.text)
    if not cats:
        # Can't parse 2 categories — fall back to plain new_search
        print("[AGENT]   → bundle: could not extract 2 categories, falling back to new_search", flush=True)
        return _run_text_search(req, state, session_id, api_key, "new_search")

    cat_a, cat_b = cats

    # Parse constraints (budget, color, gender, usage)
    record_llm_call(session_id)
    constraints = query_parser.parse_query(req.text, api_key)
    total_budget = constraints.max_price   # shared budget for the pair

    print(
        f"[AGENT]   → bundle: cat_a={cat_a!r} cat_b={cat_b!r} "
        f"total_budget={total_budget} color={constraints.color!r}",
        flush=True,
    )

    retriever = state.retriever
    common_kwargs = dict(
        color=constraints.color,
        gender=constraints.gender,
        usage=constraints.usage,
        top_k=15,
        # Intentionally no max_price per item — budget is applied to the pair total
    )

    shortlist_a = retriever.search(
        query=f"{constraints.color or ''} {cat_a}".strip(),
        category=cat_a,
        **common_kwargs,
    )
    shortlist_b = retriever.search(
        query=f"{constraints.color or ''} {cat_b}".strip(),
        category=cat_b,
        **common_kwargs,
    )

    print(
        f"[AGENT]   → bundle retrieval: {cat_a}={len(shortlist_a)} {cat_b}={len(shortlist_b)}",
        flush=True,
    )

    if not shortlist_a or not shortlist_b:
        missing = cat_a if not shortlist_a else cat_b
        msg = (
            f"I wasn't able to find any {missing} in the catalog that match your request. "
            "Try adjusting the constraints or searching for each item separately."
        )
        return AgentResponse(
            intent="bundle_search",
            message=msg,
            session_id=session_id,
            context_action="keep",
            search_state=req.search_state,
        )

    # Judge each category separately
    budget_note = f" under ${total_budget:.0f} total for both items" if total_budget else ""
    record_llm_call(session_id)
    try:
        judge_a = judge_text_candidates(
            user_ask=f"{req.text} — evaluate only the {cat_a} category",
            search_state_summary=f"{cat_a}{budget_note}",
            candidates=shortlist_a,
            api_key=api_key,
        )
    except JudgeUnavailableError as exc:
        return _llm_error_response("bundle_search", session_id, str(exc), req.search_state)

    record_llm_call(session_id)
    try:
        judge_b = judge_text_candidates(
            user_ask=f"{req.text} — evaluate only the {cat_b} category",
            search_state_summary=f"{cat_b}{budget_note}",
            candidates=shortlist_b,
            api_key=api_key,
        )
    except JudgeUnavailableError as exc:
        return _llm_error_response("bundle_search", session_id, str(exc), req.search_state)

    # Build ProductItem lists from accepted judgments
    def _accepted_items(judge_result, shortlist) -> list[ProductItem]:
        by_id = {p.id: p for p in shortlist}
        accepted_js = judge_result.exact_matches() + judge_result.close_alternatives()
        items: list[ProductItem] = []
        for j in accepted_js:
            p = by_id.get(j.product_id)
            if p:
                items.append(_product_item_from_row(
                    {
                        "id": p.id, "product_name": p.product_name, "brand": p.brand,
                        "category": p.category, "base_color": p.base_color,
                        "usage": p.usage, "season": p.season, "gender": p.gender,
                        "price": p.price, "image_path": p.image_path,
                    },
                    j.verdict, j.reason,
                ))
        return items

    items_a = _accepted_items(judge_a, shortlist_a)
    items_b = _accepted_items(judge_b, shortlist_b)

    print(
        f"[AGENT]   → bundle judge: {cat_a}={len(items_a)} accepted  {cat_b}={len(items_b)} accepted",
        flush=True,
    )

    if not items_a or not items_b:
        missing = cat_a if not items_a else cat_b
        msg = (
            f"I found candidates for both items but the judge couldn't confirm any suitable {missing}. "
            "Try relaxing constraints or searching each item separately."
        )
        return AgentResponse(
            intent="bundle_search",
            message=msg,
            session_id=session_id,
            context_action="keep",
            search_state=req.search_state,
        )

    # Build valid bundle pairs under shared total budget
    bundle_pairs: list[list[ProductItem]] = []
    for ia in items_a[:6]:
        for ib in items_b[:6]:
            total = ia.price + ib.price
            if total_budget and total > total_budget:
                continue
            bundle_pairs.append([ia, ib])

    # Rank by: perfect-match count desc, then total price asc
    bundle_pairs.sort(
        key=lambda bp: (
            -sum(1 for p in bp if p.match_tier == "perfect"),
            bp[0].price + bp[1].price,
        )
    )
    top_bundles = bundle_pairs[:3]

    if not top_bundles:
        msg = (
            f"I couldn't find a complete {cat_a} + {cat_b} combo"
            + (f" within your ${total_budget:.0f} total budget" if total_budget else "")
            + " in this catalog. "
            "Try increasing the budget or searching each item separately."
        )
        return AgentResponse(
            intent="bundle_search",
            message=msg,
            session_id=session_id,
            context_action="keep",
            search_state=req.search_state,
        )

    # Flat product list for context (all unique items across top bundles)
    seen_ids: set[int] = set()
    flat_products: list[ProductItem] = []
    for bp in top_bundles:
        for p in bp:
            if p.id not in seen_ids:
                flat_products.append(p)
                seen_ids.add(p.id)

    # Compose response via orchestrator
    totals_str = ", ".join(
        f"Bundle {i+1}: ${bp[0].price + bp[1].price:.2f}"
        for i, bp in enumerate(top_bundles)
    )
    filter_ctx = (
        f"Bundle search: {cat_a} + {cat_b}. "
        + (f"Total budget: ${total_budget:.0f}. " if total_budget else "")
        + f"Bundle totals: {totals_str}. "
        "Present as complete outfit/combo bundles."
    )

    record_llm_call(session_id)
    tb = ToolBundle(
        user_message=req.text,
        action="bundle_search",
        chat_history=req.chat_history,
        search_state_summary=f"{cat_a} + {cat_b} bundle",
        active_products=req.active_products,
        shown_products=flat_products,
        n_exact=len(top_bundles),
        n_close=0,
        filter_context=filter_ctx,
        max_price=total_budget,
    )
    try:
        message = compose_turn(tb, api_key, config.ORCHESTRATOR_TEMPERATURE)
    except LLMUnavailableError as exc:
        return _llm_error_response("bundle_search", session_id, str(exc), req.search_state)

    print(f"[AGENT]   → bundle: returning {len(top_bundles)} bundle pair(s)", flush=True)

    new_state = SearchState(
        last_action="bundle_search",
        last_intent_action="bundle_search",
        last_query_summary=f"{cat_a} + {cat_b} bundle",
        result_count=len(top_bundles),
        had_perfect_matches=any(
            any(p.match_tier == "perfect" for p in bp) for bp in top_bundles
        ),
    )

    return AgentResponse(
        intent="bundle_search",
        message=message,
        products=flat_products,
        bundle_pairs=top_bundles,
        session_id=session_id,
        context_action="replace",
        search_state=new_state,
    )


# ── Main endpoint ─────────────────────────────────────────────────────────────

@router.post("/agent", response_model=AgentResponse)
def agent(req: AgentRequest, request: Request):
    api_key = config.ANTHROPIC_API_KEY
    session_id = req.session_id or str(uuid.uuid4())

    # ── Rate limit ────────────────────────────────────────────────────────────
    limit_msg = check_rate_limit(session_id)
    if limit_msg:
        return AgentResponse(
            intent="rate_limited",
            message=limit_msg,
            session_id=session_id,
        )

    state = request.app.state

    # ── Planner ───────────────────────────────────────────────────────────────
    decision = make_plan(
        user_message=req.text,
        search_state=req.search_state.model_dump(),
        has_image=bool(req.image_b64),
        has_active_products=bool(req.active_products),
        api_key=api_key,
        chat_history=req.chat_history,
    )
    action = decision.action

    # ── Observability logging ─────────────────────────────────────────────────
    print(
        f"[AGENT] text={req.text[:60]!r} "
        f"has_image={bool(req.image_b64)} "
        f"active={len(req.active_products)} "
        f"planner={action}  ({decision.reasoning})",
        flush=True,
    )
    logger.info(
        "[AGENT] text=%r has_image=%s active=%d planner=%s",
        req.text[:80], bool(req.image_b64), len(req.active_products), action,
    )

    # ── Non-retrieval actions ─────────────────────────────────────────────────

    if action == "reset_context":
        return AgentResponse(
            intent=action,
            message="Starting fresh! What would you like to find today?",
            session_id=session_id,
            context_action="clear",
        )

    if action == "out_of_scope":
        record_llm_call(session_id)
        return AgentResponse(
            intent=action,
            message=generate_scope_refusal(req.text, api_key),
            session_id=session_id,
            context_action="keep",
        )

    if action == "ask_capabilities":
        record_llm_call(session_id)
        return AgentResponse(
            intent=action,
            message=generate_chat(req.text, api_key),
            session_id=session_id,
            context_action="keep",
        )

    if action == "ask_current_state":
        return _handle_ask_current_state(req, session_id)

    if action == "explain_match_quality":
        return _handle_explain_match_quality(req, session_id, api_key)

    if action == "next_result":
        return _handle_next_result(req, state, session_id, api_key)

    if action in ("followup_on_results", "compare_results"):
        return _handle_followup(req, session_id, api_key)

    # ── Retrieval actions ─────────────────────────────────────────────────────

    if action == "bundle_search":
        return _run_bundle_search(req, state, session_id, api_key)

    if action == "image_search":
        return _run_image_search(req, state, session_id, api_key)

    # clarify_request: ask LLM to formulate a clarification question
    if action == "clarify_request":
        record_llm_call(session_id)
        return AgentResponse(
            intent=action,
            message=generate_clarification(req.text, api_key),
            session_id=session_id,
            context_action="keep",
        )

    # Default: new_search or refine_search
    return _run_text_search(req, state, session_id, api_key, action)
