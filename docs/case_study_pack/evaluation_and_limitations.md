# Evaluation and Limitations

## What the Demo Does Well

**End-to-end pipeline coherence.** The system behaves like a single agent from the user's perspective. A user can upload an image, get similar products, ask "which is cheapest," say "show me the casual ones," and then ask "suggest a bundle with sneakers" — all in one conversation with maintained context.

**Honest failure modes.** When the LLM is unavailable, the system says so. When a query returns zero matching products, the response explains why. There are no silent degradations, no fake fallback recommendations, no hallucinated product details. Product names, prices, colors, and brands in responses come from the catalog records passed to the orchestrator — the LLM cannot invent them.

**Practical multi-modal integration.** The image judge sends actual product thumbnail images to Claude's multimodal API alongside the query image. Results reflect genuine visual comparison, not just text-label similarity. The JPEG re-encoding fix ensures PNG uploads (including RGBA transparency) work correctly without user-visible errors.

**Clean separation of deterministic and LLM logic.** Retrieval, filtering, ranking, rate limiting, and guardrail checks are all deterministic code. LLM is called for three specific tasks: constraint extraction, product judgment, and response composition. This makes the system's behavior predictable and debuggable.

**Deployed and accessible.** The demo runs live at a public URL. A reviewer can interact with it without installing anything.

## Known Limitations

**TF-IDF retrieval is shallow.** TF-IDF matches keywords, not intent. A query for "something cozy for winter evenings" will not reliably retrieve the most semantically relevant items because "cozy" and "evenings" may not appear in product descriptions. Dense embeddings (e.g., a sentence-transformer over product text) would handle this better.

**Mock prices are not real.** Prices are synthetically generated from a deterministic hash of product ID and category. The price ranges are plausible but not actual retail prices. Budget filtering works correctly, but the numbers are not meaningful outside the demo.

**CLIP similarity is approximate.** CLIP embeds visual semantics broadly but is not trained on fashion-specific fine-grained attributes. Two items that look similar to CLIP may differ significantly in material, fit, or style. The visual reranker's color and complexity heuristics partially compensate but are simplistic.

**In-memory rate limiting resets on restart.** The global and per-session LLM call counters are Python dictionaries. A Railway restart (which can happen on deployment or crash) resets all counters. A demo period with frequent restarts could exceed the intended API spend cap.

**No authentication.** Any visitor to the Streamlit URL can start a session. The per-session limit applies per `session_id` UUID (set in browser session state), not per user account. A user could clear their browser state and reset their session counter.

**Catalog size is small.** The demo catalog is ~2,500 products across 8 categories. A real fashion retailer would have tens of thousands of SKUs per category. At this scale the flat `.npy` embedding index and single-process TF-IDF would not perform acceptably.

**Follow-up filtering is keyword-based.** `_handle_followup()` detects color/usage/brand mentions via regex and string matching against the active products list. It does not use the LLM for follow-up constraint extraction. Complex or ambiguous follow-up constraints ("show me the one that would work better for a date night") will not filter correctly.

**Bundle planner does only 2-item pairs.** The bundle feature demonstrates the concept but does not produce complete outfits. Color coordination between paired items is not evaluated — a red hoodie may be paired with a blue sneaker if both score well individually.

## Failure Cases

- Vague queries with no extractable constraints ("show me something nice") will retrieve products that happen to score highest on free-text similarity, which may not reflect the user's intent. The planner may route to `clarify_request` for very short vague queries, but this is not guaranteed.
- Queries for items outside the 8 supported categories ("show me a dress") will return zero results or approximate alternatives from the nearest available category.
- Catalog items with missing or low-quality images produce weak image judge results, since the multimodal judge relies on thumbnail quality.
- Very long follow-up filter chains ("only the ones that are casual, black, under $60, and from Benetton") may produce zero results if the active product set is small (e.g., 3 items shown, none matching all four constraints simultaneously).

## What Would Be Improved in a Production Version

**Dense text embeddings.** Replace TF-IDF with a sentence-transformer model (e.g., `all-MiniLM-L6-v2`) for semantic text retrieval. Pair with FAISS or a vector database for scalable ANN search.

**Persistent rate limiting.** Use Redis or a database-backed counter so LLM call limits survive server restarts and scale across multiple backend replicas.

**Real product data.** Integrate with a live inventory API (Shopify, internal catalog service) instead of a static preprocessed file. The normalized schema is already defined — it's a data pipeline change.

**Fine-grained image model.** A fashion-specific CLIP fine-tune (e.g., trained on FashionIQ or iMaterialist) would improve attribute-level retrieval (fabric, pattern, fit) beyond what the general CLIP model captures.

**User authentication.** Session-based rate limiting requires knowing who the user is across visits. A simple auth system (OAuth or API key per user) would make usage caps reliable.

**Evaluation dataset.** A small labeled test set (query → expected product IDs) would allow quantitative measurement of retrieval precision and LLM judgment quality, enabling regression detection when models or prompts change.

## What Was Deliberately Not Built

- Web search or open-internet product retrieval
- Real-time price or stock data
- Size recommendation
- User purchase history or personalization
- Checkout or cart functionality
- Admin interface for catalog management
- A/B testing or feature flags
- Streaming responses (the API call completes before rendering)
- Any form of model fine-tuning
