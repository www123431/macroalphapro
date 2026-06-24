"""engine/research/mcp_server.py — Phase 4a.5: MCP stdio server
exporting the 9 Session-3 LLM tools to external Claude clients.

Why MCP: the same 9 tools that Session 3 made callable for our
in-process L4 agent council are equally valuable to senior daily
research via Claude Code desktop. MCP is the wire protocol that lets
an external Claude client invoke them.

Run as stdio server (the form Claude Code / Claude Desktop expect):

  python -m engine.research.mcp_server

Or wire from Claude Code config (.mcp.json) as a stdio server.

Design: thin shim over the existing `engine.research.llm_tools.TOOLS`
registry — same Pydantic schemas, same dispatch — so the inner L4 loop
and external clients share one source of truth. Adding a new tool to
TOOLS auto-exposes it here.
"""
from __future__ import annotations

import inspect
import json
import logging
from typing import Any

from mcp.server.fastmcp import FastMCP

from engine.research.llm_tools import TOOLS, dispatch

logger = logging.getLogger(__name__)

# ── Server bootstrap ──────────────────────────────────────────────────

mcp = FastMCP(
    name="intern-research",
    instructions=(
        "Quant-research toolkit for the intern codebase. Use these tools "
        "to query the intuition-rules base, graveyard, mechanism library, "
        "master paper index, geometry (cosine), Sharpe SE, family n_trials, "
        "and L4 outcome/override ledgers. Always check intuition_rules and "
        "graveyard BEFORE proposing or critiquing a new factor candidate."
    ),
)


def _result_to_json(result: Any) -> str:
    """Coerce tool result (typically dict) to JSON string for MCP wire."""
    if isinstance(result, str):
        return result
    try:
        return json.dumps(result, default=str, indent=2)
    except Exception as exc:
        logger.exception("result serialization failed")
        return json.dumps({"error": f"serialization failed: {exc}"})


def _make_handler(tool_name: str, schema_cls: type) -> Any:
    """Build a handler whose visible signature mirrors the schema fields
    individually — so MCP wire schema is FLAT (one input per field), not
    nested under {"args": {...}}.

    FastMCP introspects via inspect.signature(handler), which respects
    handler.__signature__ + handler.__annotations__ overrides. We
    construct both manually from the Pydantic schema metadata.
    """
    field_names = list(schema_cls.model_fields.keys())

    def handler(**kwargs):
        # Forward only the schema fields; ignore stray kwargs
        clean = {k: kwargs.get(k) for k in field_names if k in kwargs}
        try:
            result = dispatch(tool_name, **clean)
        except Exception as exc:
            logger.exception("MCP tool %s failed", tool_name)
            return _result_to_json({"error": str(exc), "tool": tool_name})
        return _result_to_json(result)

    # Build a Signature whose parameters are the schema fields. FastMCP
    # uses inspect.signature() (PEP 362) which honours __signature__,
    # bypassing PEP-563-stringified __annotations__.
    params = []
    annotations: dict[str, Any] = {}
    for fname, finfo in schema_cls.model_fields.items():
        annotation = finfo.annotation if finfo.annotation is not None else Any
        # Pydantic v2 sentinel for "required" is PydanticUndefined; we
        # treat anything truthy / non-None as a real default to match
        # the schema's Field(default=...) semantics.
        if finfo.is_required():
            default = inspect.Parameter.empty
        else:
            default = finfo.default
        params.append(inspect.Parameter(
            name=fname,
            kind=inspect.Parameter.KEYWORD_ONLY,
            default=default,
            annotation=annotation,
        ))
        annotations[fname] = annotation
    annotations["return"] = str

    handler.__name__ = tool_name
    handler.__signature__ = inspect.Signature(
        parameters=params, return_annotation=str,
    )
    handler.__annotations__ = annotations
    return handler


def _register_all_tools() -> None:
    """Register every entry of TOOLS as an MCP tool.

    The handler takes the schema as a single typed parameter; FastMCP
    flattens this into the MCP wire-level input schema automatically.
    Adding a tool to TOOLS auto-exposes it here — no per-tool MCP code.
    """
    for tool_name, (fn, schema_cls, description) in TOOLS.items():
        handler = _make_handler(tool_name, schema_cls)
        mcp.add_tool(
            handler,
            name=tool_name,
            description=description,
            structured_output=False,
        )


_register_all_tools()


def main() -> None:
    """Entry point for `python -m engine.research.mcp_server`."""
    # stdio transport — matches Claude Desktop / Claude Code expectations
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
