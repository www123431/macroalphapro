"""engine/agents/governance — institutional AI governance controls.

data_egress.py — data-residency governance: what data-sensitivity class each LLM provider
may receive. The threat-model #1 overlooked control (DeepSeek = CN residency for a quant
fund's position/PII data). Deterministic classifier + per-provider policy + guard + audit.
"""
