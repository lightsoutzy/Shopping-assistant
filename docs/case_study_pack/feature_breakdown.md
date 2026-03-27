# Feature Breakdown

## 1. General Conversation

**What the user sees:**
A friendly, domain-bounded shopping assistant that answers questions about itself and its capabilities. Example exchanges:
- "What's your name?" → brief identity response
- "What can you help me with?" → lists supported capabilities
- "How does image search work?" → short explanation
- "What kinds of products do you have?" → describes catalog categories
- "What's the weather in Tokyo?" → polite refusal, invitation to ask a fashion question

**How it works under the hood:**
The planner checks for capability questions (`_CAPABILITIES_RE`) and out-of-scope patterns (`_OUT_OF_SCOPE_RE`). Matched queries route to `response_generator.py` which calls Claude Haiku with a short system prompt capping response length at 256 tokens. Out-of-scope queries get a scope refusal response. Neither path touches the retrieval pipeline.

If the Anthropic API is unavailable, `response_generator.py` falls back to pre-written template responses — the only path in the system with a safe fallback, because general chat has no correctness requirement tied to the catalog.

**In scope:** Agent identity, capabilities, catalog scope, how image search works, what categories are available.

**Out of scope:** General knowledge questions, coding help, product recommendations for items outside the catalog, opinions on external brands.

---

## 2. Text-Based Product Recommendation

**What the user sees:**
The user types a natural language shopping query. The assistant returns up to 5 product cards with name, category, color, price, brand, and a one-sentence explanation of why each item was selected. The response text briefly characterizes the result set ("I found 3 exact matches and 2 close alternatives").

**How it works under the hood:**
1. **Planner** classifies as `new_search` or `refine_search`.
2. **Query Parser** calls Claude Haiku with a structured JSON prompt to extract: `{category, color, gender, usage, season, max_price, min_price, brand, plain_only}`. Falls back to regex keyword extraction if LLM is unavailable.
3. **Retriever** hard-filters `catalog.parquet` by extracted constraints, then ranks the filtered set by TF-IDF cosine similarity (unigram+bigram) against the query. Returns 30 candidates. If hard filters eliminate too many results (< 5), filtering is relaxed to the full catalog.
4. **Ranker** applies weighted additive scoring: TF-IDF score, category match, color match, usage match, gender match, season match, price within bounds.
5. **Judge** sends the shortlist to Claude Haiku with a strict system prompt. Each product is labeled `exact_match`, `close_alternative`, or `reject`. Hard constraints (price, brand, plain_only) are enforced — violations produce explicit rejections.
6. **Orchestrator** composes a natural language response with exact/close counts, using the shown product list to prevent hallucination.
7. **SearchState** is updated with the new filters and accepted pool for potential follow-up turns.

**In scope:** Category, color, gender, usage, season, price range, brand, plain/patterned preference. Combinations of multiple constraints in one query.

**Out of scope:** Multi-brand comparison shopping, live price data, availability/stock, size recommendations, style advice beyond basic attribute matching.

---

## 3. Image-Based Product Search

**What the user sees:**
The user uploads an image (JPEG or PNG) via the sidebar uploader. The assistant shows visually similar products from the catalog, noting the detected dominant color. The user can optionally add text to refine ("find something like this but more casual").

**How it works under the hood:**
1. Frontend hashes the image bytes, re-encodes as JPEG (handles RGBA PNG), base64-encodes, and sends in `image_b64`.
2. **Planner** detects image attached → `image_search`.
3. Backend decodes → PIL Image → re-encodes as JPEG (normalizes media type for Anthropic API).
4. **CLIP** encodes the query image to a 512-dim embedding. Cosine similarity is computed against the pre-built catalog embedding index (`image_embeddings.npy`). Returns top 30 results.
5. **Visual reranker** adjusts CLIP scores: `+0.10` for exact color match, `+0.05` for same color family (e.g., white → beige), `−0.06` for color family mismatch, complexity penalty if query is plain but product is patterned.
6. **Image judge** sends query image + up to 12 product thumbnail images to Claude Haiku in a multimodal message. Judge evaluates visual similarity (garment type, silhouette, color, plain vs patterned). Falls back to text-only judge if product images are unavailable.
7. **Orchestrator** composes response noting dominant color if detected.
8. Subsequent text messages are treated as follow-up filtering on these results (not new image queries) unless a new image is uploaded.

**In scope:** Garment type similarity, color similarity, basic silhouette matching, combining with text constraints ("under $80", "more casual").

**Out of scope:** Brand logo recognition, fine-grained pattern matching (stripes vs plaid), style transfer, multi-item outfit detection from a single image.

---

## 4. Follow-Up Questions Over Prior Results

**What the user sees:**
After receiving product recommendations, the user can ask questions about the shown results without starting a new search:
- "Show me the casual ones" → filters to usage=Casual
- "Which is cheapest?" → sorts by price, shows lowest
- "Only the black ones" → filters by color
- "Which one would you recommend?" → picks single highest-scoring item
- "Show me another" → serves next items from the accepted pool

**How it works under the hood:**
The frontend sends `active_products` (currently shown items) and `search_state` (prior filter context) with every request.

**Planner** classifies as `followup_on_results` or `next_result`.

For `followup_on_results`:
- `_handle_followup()` applies keyword-based hard filters to `active_products`: color mentions, usage mentions, brand mentions.
- For "cheapest/most expensive" queries: sorts by price.
- For "best one/which would you recommend" queries: sorts by `match_score`, returns top 1.
- Filtered products are passed to Orchestrator with a `filter_context` description.
- `search_state=ss` is always included in the response to preserve context.

For `next_result`:
- Pulls the next batch (3 items) from `SearchState.accepted_pool`.
- Excludes IDs already in `shown_product_ids`.

**In scope:** Filtering the shown result set by attributes, sorting by price, picking a single recommendation, paginating to more results.

**Out of scope:** Multi-turn negotiation ("make it cheaper" repeatedly cycling through catalog), comparison tables, side-by-side attribute diffs (compare_results is implemented but limited in scope).

---

## 5. Simple Bundle / Combo Requests

**What the user sees:**
The user asks for a 2-item pairing ("suggest a hoodie and sneaker combo" or "what jacket would go with sneakers under $150?"). The assistant returns up to 3 bundle options, each showing both items side by side with individual prices and a combined total.

**How it works under the hood:**
1. **Planner** detects `_BUNDLE_RE` pattern → `bundle_search`.
2. `_extract_bundle_categories()` maps keywords to canonical category names using `_BUNDLE_CAT_MAP`.
3. `_run_bundle_search()` runs the full text retrieval + judge pipeline independently for each of the two categories.
4. If either category has no accepted items, the response says so explicitly and returns no bundles.
5. Valid pairs are built by cross-joining accepted items from each category. Pairs exceeding a total budget constraint (if specified) are filtered out.
6. Top 3 pairs are ranked by combined match score.
7. `AgentResponse.bundle_pairs` carries the result as `list[list[ProductItem]]`.
8. Frontend renders each bundle as a grouped card pair with total price.

**In scope:** Exactly 2-category pairings from the supported catalog categories. Budget constraint applied to total pair price.

**Out of scope:** 3+ item outfits, accessories, accessories pairing, style coherence beyond basic category pairing, color coordination between paired items.
