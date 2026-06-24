"""engine/line_c/text_features.py — Loughran-McDonald dictionary scalars (CPU).

Uses the FULL LM Master Dictionary bundled with pysentiment2 (LM.csv, 86,486
words, all categories) — no download. Per transcript (aggregated full text) we
compute interpretable, economically-motivated scalars:

  lm_net_tone     (pos-neg)/(pos+neg)        Loughran-McDonald (2011) tone
  lm_pos_prop, lm_neg_prop                   raw category proportions
  lm_uncertainty_prop                        LM Uncertainty (the headline LM finding)
  lm_litigious_prop, lm_constraining_prop    risk/constraint language
  lm_modal_prop                              modal (hedging) words
  numeric_density                            numeric-token share (quantitative density)
  log_n_words                                call length control

DOCTRINE: deterministic encoder — immune to LLM narrative-memory contamination
(unlike re-running a generative LLM). Pure feature extraction, no returns seen.
"""
from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"[A-Za-z']+")
_NUMERIC_RE = re.compile(r"[\$%]|\d")

_LM_CATEGORIES = ["Negative", "Positive", "Uncertainty", "Litigious", "Constraining", "Modal"]


def _load_lm_wordmap() -> dict[str, frozenset]:
    """word(UPPER) -> frozenset of categories it belongs to, from pysentiment2 LM.csv."""
    import pysentiment2 as ps
    csv = Path(os.path.dirname(ps.__file__)) / "static" / "LM.csv"
    df = pd.read_csv(csv)
    wmap: dict[str, frozenset] = {}
    cats = {c: (df[c].astype(float).values > 0) for c in _LM_CATEGORIES}
    words = df["Word"].astype(str).str.upper().values
    for i, w in enumerate(words):
        belongs = [c for c in _LM_CATEGORIES if cats[c][i]]
        if belongs:
            wmap[w] = frozenset(belongs)
    logger.info("LM dict loaded: %d categorized words", len(wmap))
    return wmap


_WORDMAP: dict[str, frozenset] | None = None


def _wordmap() -> dict[str, frozenset]:
    global _WORDMAP
    if _WORDMAP is None:
        _WORDMAP = _load_lm_wordmap()
    return _WORDMAP


def lm_features_for_text(text: str, wmap: dict[str, frozenset]) -> dict:
    """Single-document LM scalars."""
    if not text:
        return {}
    toks = _TOKEN_RE.findall(text.upper())
    n = len(toks)
    if n == 0:
        return {}
    counts = {c: 0 for c in _LM_CATEGORIES}
    for t in toks:
        cs = wmap.get(t)
        if cs:
            for c in cs:
                counts[c] += 1
    n_numeric = len(_NUMERIC_RE.findall(text))
    pos, neg = counts["Positive"], counts["Negative"]
    return {
        "n_words": n,
        "log_n_words": float(np.log(n)),
        "lm_pos_prop": pos / n,
        "lm_neg_prop": neg / n,
        "lm_net_tone": (pos - neg) / (pos + neg + 1.0),
        "lm_uncertainty_prop": counts["Uncertainty"] / n,
        "lm_litigious_prop": counts["Litigious"] / n,
        "lm_constraining_prop": counts["Constraining"] / n,
        "lm_modal_prop": counts["Modal"] / n,
        "numeric_density": n_numeric / max(len(text), 1),
    }


def extract_lm_features(text_df: pd.DataFrame, *, text_col="full_text", id_col="transcript_id",
                        log_every=2000) -> pd.DataFrame:
    """LM scalars for a transcript text DataFrame (transcript_id, full_text)."""
    wmap = _wordmap()
    rows = []
    n_total = len(text_df)
    for i, r in enumerate(text_df.itertuples(index=False), 1):
        rec = lm_features_for_text(getattr(r, text_col) or "", wmap)
        if rec:
            rec[id_col] = int(getattr(r, id_col))
            rows.append(rec)
        if i % log_every == 0:
            logger.info("  LM features %d/%d", i, n_total)
    out = pd.DataFrame(rows)
    return out


def extract_lm_from_parquet(path: str, *, text_col="full_text", id_col="transcript_id",
                            batch_size=2000) -> pd.DataFrame:
    """Streaming LM extraction: read the (large) text parquet in row-group batches
    so peak memory holds only one batch of full text at a time (avoids OOM on the
    full corpus)."""
    import pyarrow.parquet as pq
    wmap = _wordmap()
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    parts = []
    done = 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=[id_col, text_col]):
        df = batch.to_pandas()
        rows = []
        for r in df.itertuples(index=False):
            rec = lm_features_for_text(getattr(r, text_col) or "", wmap)
            if rec:
                rec[id_col] = int(getattr(r, id_col))
                rows.append(rec)
        if rows:
            parts.append(pd.DataFrame(rows))
        done += len(df)
        del df
        logger.info("  LM %d/%d", done, total)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


if __name__ == "__main__":
    import sys
    import warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # smoke test on the existing 2024-26 cache by default
    src = sys.argv[1] if len(sys.argv) > 1 else "data/d_pead_plus/_transcripts_text.parquet"
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    txt = pd.read_parquet(src)
    print(f"loaded {len(txt)} transcripts from {src}")
    feats = extract_lm_features(txt.head(1500) if out_path is None else txt)
    print(f"\nLM features: {feats.shape}")
    print(feats.describe().round(4).T.to_string())
    if out_path:
        feats.to_parquet(out_path)
        print(f"saved -> {out_path}")
