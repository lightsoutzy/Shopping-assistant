# Project Overview: Fashion Shopping Assistant

## What It Is

A scoped, multimodal AI shopping assistant for a predefined fashion catalog. The system accepts natural language and image queries from a user and returns product recommendations, explanations, and follow-up filtering — all through a single conversational interface.

The project was built as a take-home / job-application demo for an AI Engineer role. The goal was a clean, deployable prototype that demonstrates practical AI system design: retrieval pipelines, LLM orchestration, multimodal input, and multi-turn conversation — without overbuilding.

## What User Problem It Solves

Finding products in a fashion catalog by browsing category trees or keyword filters is tedious. This assistant lets users describe what they want in plain language ("something casual in black under $80 for men") or upload a photo of an item they like and find visually similar products. It supports follow-up questions over the results it already showed, so users do not need to start over to narrow down a search.

## Why the Scope Is Intentionally Limited

The assistant is bounded to a predefined catalog subset (~2,500 products across 8 categories). It does not search the web, does not call external retail APIs, and does not answer questions outside the fashion domain.

This was a deliberate constraint, not a limitation of ambition. A scoped demo:
- Is honest about what it can and cannot do
- Has testable, predictable behavior
- Is explainable end-to-end in an interview
- Does not require safety guardrails against open-ended misuse

Out-of-scope queries (programming questions, weather, general knowledge) receive a polite refusal and an invitation to ask a fashion question instead.

## Deployed Demo

Frontend: https://shopping-assistant-tdpvxkzwtjc5tect6wi6app.streamlit.app/

Backend: FastAPI service hosted on Railway (URL configured via Streamlit secrets).

## Major Capabilities

| Capability | Description |
|---|---|
| General conversation | Agent identity, capabilities, catalog scope, how image search works |
| Text-based recommendation | Natural language query → constraint extraction → TF-IDF retrieval → LLM judge → response |
| Image-based search | Upload image → CLIP similarity → visual reranking → LLM multimodal judge → response |
| Follow-up filtering | "Show me the casual ones" / "which is cheapest" over currently shown results |
| Best-one selection | "Which would you recommend?" → picks single highest-scoring item from shown set |
| Simple bundle planning | "Suggest a hoodie and sneaker combo" → retrieves and pairs items from two categories |
| State-aware pagination | "Show me another" → serves next items from the accepted pool without re-querying |
| Scope refusal | Non-fashion queries refused with explanation and redirect |
