"""api/ — FastAPI backend over the quant-fund engine.

Phase 0 of the UI migration (Streamlit -> Next.js/Tailwind, strangler-fig): a clean HTTP
layer that exposes the existing engine + agent constellation to any frontend. Read-only,
0-LLM-in-DECISION preserved (endpoints serve deterministic engine output / persisted
artifacts; the agent chat layer streams the existing chat_turn loop).

Run:  uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs  (auto OpenAPI)
"""
