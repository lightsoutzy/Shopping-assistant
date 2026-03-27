You are helping me build a take-home prototype for an AI Engineer role.

I want a clean, practical, demo-ready project with good engineering structure, not an overengineered research system.

PROJECT GOAL
Build a single AI agent for a fashion ecommerce website using the Kaggle "Fashion Product Images Dataset" (with styles.csv, product JSON metadata, and product images). The system must support:

1. General conversation with the agent
   - Example: "What’s your name?"
   - Example: "What can you do?"

2. Text-based product recommendation
   - Example: "Recommend me a t-shirt for sports"
   - Example: "Find me black casual sneakers under $80"

3. Image-based product search
   - User uploads an image
   - System returns visually similar products from the predefined catalog

IMPORTANT PRODUCT REQUIREMENT
This should behave like ONE unified agent, not three separate apps. Internally it can route to different modules, but the user experience should feel like one shopping assistant.

TARGET OUTCOME
I want a demo-ready prototype with:
- a lightweight frontend
- a FastAPI backend
- a documented agent API
- a clean README
- modular code
- simple test coverage for core logic
- server-side only LLM API usage
- usage limits suitable for a take-home demo

DO NOT build a giant production system. Keep it clean, practical, and easy to explain in an interview.

==================================================
TECHNICAL DESIGN
==================================================

DOMAIN
Fashion ecommerce shopping assistant.

DATASET
Use the Kaggle Fashion Product Images Dataset as the predefined catalog.
I only need a manageable subset for the demo, not the full dataset.

Please assume we will curate a smaller catalog subset from the full dataset, such as:
- T-shirts
- Shirts
- Shoes
- Sneakers
- Jackets
- Hoodies
- Shorts
- Bags

SYSTEM ARCHITECTURE
Build a unified agent pipeline:

User Input (text and optional image)
  -> Intent Router
  -> Query Parser / Constraint Extractor
  -> Retrieval Layer
       - text retrieval over product metadata
       - image similarity search for uploaded image
       - metadata filtering
  -> Reranker
  -> Response Generator
  -> Lightweight Evaluation / Guardrail Check
  -> Final Response

FRONTEND
Use either:
- Streamlit for speed
or
- a very minimal React frontend if you think it is still simple enough

Default preference: Streamlit unless there is a strong reason otherwise.

BACKEND
Use FastAPI.

The backend should expose one primary endpoint:
POST /agent

Optional internal helper endpoints are fine, but the demo should revolve around one unified /agent endpoint.

LLM USAGE
Use an LLM only where it adds value. Do not use the LLM for everything.

Prefer this pattern:
- general chat: LLM
- text response generation: LLM
- maybe ambiguity handling / clarification: LLM or simple logic
- retrieval: non-LLM
- filtering: non-LLM
- ranking: non-LLM
- image similarity: non-LLM
- core control logic: non-LLM

The API key must remain server-side only and never be exposed to the frontend.

DEMO PROTECTION
This is very important:
- API key must be hidden on backend only
- add per-session rate limits
- add a small total usage cap for demo
- add a friendly "demo limit reached" fallback
- optionally support demo expiration by config
- do not make the app a free unlimited public tool

==================================================
FEATURE REQUIREMENTS
==================================================

1. GENERAL CONVERSATION
The agent should be able to answer:
- what it is
- what it can do
- what kinds of products it can help find
- how image search works
- what catalog it searches

Keep this domain-bounded. It does not need to be a general-purpose assistant.

2. TEXT-BASED PRODUCT RECOMMENDATION
The system should:
- parse user needs from natural language
- extract constraints such as category, color, gender, usage, season, budget if possible
- retrieve candidate products from the predefined catalog
- rerank products
- return the top recommendations with short reasons

Examples:
- "Recommend me a t-shirt for sports"
- "Find me black casual sneakers"
- "Show me a winter jacket for men"
- "I want something comfortable for summer"

3. IMAGE-BASED PRODUCT SEARCH
The system should:
- accept uploaded image input
- compute visual similarity against catalog images
- return visually similar items from the catalog
- optionally combine image search with text refinement, such as:
  - "Find something like this but more casual"
  - "Find something like this under $50"

4. OPTIONAL NICE-TO-HAVE
Only do these if they are simple and do not derail the build:
- ask a concise clarification question when query is too vague
- support product comparison
- log latency per request
- include a lightweight groundedness / constraint check before returning recommendations

==================================================
DATA HANDLING
==================================================

CATALOG SUBSET
Please design the system so I can easily work with a curated subset of the dataset instead of the entire dataset.

Define a preprocessing step that:
- loads styles.csv
- joins useful JSON metadata if needed
- selects a subset of categories
- cleans the data
- produces a normalized catalog file or SQLite database for runtime use

NORMALIZED PRODUCT SCHEMA
Create a practical normalized schema such as:

- id
- product_name
- brand
- category
- subcategory
- gender
- base_color
- usage
- season
- price (if available; if not available, support mock/generated placeholder price)
- description
- image_path
- searchable_text
- image_embedding_path or stored vector reference

If price is missing from the dataset, handle it gracefully for the demo, for example:
- mock prices deterministically
- or make price filtering optional
But structure the system so price can exist.

TEXT SEARCH
Implement a retrieval approach using product metadata and descriptions.
Good options:
- TF-IDF / BM25 style
- embeddings-based semantic search
- or a hybrid approach if simple enough

Default preference:
Use a simple practical approach first. I prefer reliability and fast implementation over fancy complexity.

IMAGE SEARCH
Use a practical image embedding method suitable for similarity search.
Good options:
- CLIP embeddings
- or another lightweight vision embedding approach

Precompute catalog image embeddings offline.

RERANKING
Implement a simple reranker that combines:
- retrieval similarity
- metadata matches
- hard constraints if extracted
- optional heuristic boosts for strong matches

Example factors:
- category match
- color match
- usage match
- gender match
- seasonal fit

==================================================
LLM / AGENT BEHAVIOR
==================================================

INTENT ROUTING
The agent should classify the user request into something like:
- general_chat
- recommend_products
- image_search
- clarify_request

It is okay to do this with simple rules first, and only use an LLM fallback if needed.

QUERY PARSING
The system should extract structured constraints when possible.
Example:
Input: "Recommend me a black t-shirt for sports"
Output:
- category: t-shirt
- color: black
- usage: sports

RESPONSE GENERATION
The final response should be concise and useful.
For recommendations:
- recommend at most 3 items
- include product name
- include key attributes
- explain briefly why each product fits
- do not invent facts not present in the catalog

For general chat:
- keep the agent domain-bounded
- explain capabilities clearly

GUARDRAIL / EVAL CHECK
Add a lightweight final check if feasible:
- make sure recommended products actually came from retrieved results
- make sure hard constraints are not obviously violated
- if a violation is found, retry once or fall back safely

==================================================
CODEBASE REQUIREMENTS
==================================================

I want a clean repo structure like this:

project_root/
  README.md
  requirements.txt
  .env.example
  app/
    main.py
    config.py
    schemas.py
    api/
      routes_agent.py
    services/
      intent_router.py
      query_parser.py
      retriever.py
      image_search.py
      ranker.py
      response_generator.py
      evaluator.py
      rate_limiter.py
    data/
      catalog_loader.py
      preprocess_dataset.py
    utils/
      logging_utils.py
      text_utils.py
  frontend/
    streamlit_app.py
  tests/
    test_intent_router.py
    test_query_parser.py
    test_retriever.py

If you think a slightly different structure is better, keep it simple and explain why.

TESTING
Include basic tests for:
- intent routing
- query parsing
- retrieval returning plausible results

README
Write a strong README that includes:
- project overview
- supported features
- architecture summary
- setup instructions
- dataset preprocessing steps
- how to run backend and frontend
- environment variables needed
- design tradeoffs
- future improvements
- demo protection notes (server-side secrets, usage limits)

==================================================
IMPLEMENTATION PLAN
==================================================

Please help me build this incrementally in phases:

PHASE 1
- define final architecture
- define normalized schema
- define data preprocessing plan
- define repo structure

PHASE 2
- implement dataset preprocessing
- create normalized catalog subset
- create runtime data loading

PHASE 3
- implement text retrieval and metadata filtering
- implement image embedding + similarity search

PHASE 4
- implement unified agent endpoint
- implement routing, parsing, ranking, response generation

PHASE 5
- implement frontend
- implement rate limiting and demo safeguards
- write tests
- polish README

IMPORTANT BUILD STYLE
- Prefer simple, robust choices
- Avoid unnecessary abstractions
- Avoid advanced frameworks unless clearly justified
- Prioritize something I can finish and explain in 1–2 days
- Code should be readable and interview-friendly
- When in doubt, choose the simpler implementation

IMPORTANT RESPONSE STYLE
When helping me:
- be concrete
- give code files one by one
- explain why key decisions are being made
- do not dump giant walls of code without context
- if a decision is uncertain, call it out clearly
- if there are alternatives, recommend one and explain why

FIRST TASK
Start with PHASE 1 only.
Give me:
1. the final architecture
2. the normalized product schema
3. the preprocessing plan
4. the exact repo structure
5. a short explanation of why this design satisfies the take-home requirements and also makes a strong reusable portfolio piece

END OF INSTRUCTION SCRIPT