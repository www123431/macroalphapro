"""engine/_streamlit_shim.py — let legacy engine modules import safely
in headless cron contexts (paper-trade daily, NAV rollup, scheduled
jobs) that don't have Streamlit installed.

Usage in a legacy module that uses `@st.cache_data(...)`:

    from engine._streamlit_shim import streamlit as st

When Streamlit IS installed, this just returns the real module.
When it ISN'T, a small stub object provides no-op cache_data /
cache_resource decorators so module import + decorator application
still succeed. Caching is silently disabled in that case, which is
fine for cron processes — short-lived runs don't benefit from
session-scoped caches anyway.

Created 2026-06-02 while wiring roll_daily_nav into the daily cron;
the transitive engine.signal → engine.quant → engine.risk_metrics
chain all depend on Streamlit decorators that needed neutralizing.
"""
from __future__ import annotations


try:
    import streamlit as _streamlit
    streamlit = _streamlit
except ImportError:
    class _StreamlitStub:
        """Mimics enough of the streamlit module surface so that the
        decorators legacy engine modules apply at import time work."""

        @staticmethod
        def cache_data(*args, **kwargs):
            # Supports both forms:
            #   @st.cache_data
            #   @st.cache_data(ttl=...)
            if args and callable(args[0]):
                return args[0]
            def _wrap(fn):
                return fn
            return _wrap

        cache_resource = cache_data

        # Common dummy attrs other modules may touch defensively.
        def __getattr__(self, name):     # type: ignore[no-redef]
            # Anything else accessed becomes a no-op callable returning None
            def _noop(*a, **k):
                return None
            return _noop

    streamlit = _StreamlitStub()    # type: ignore[assignment]
