"""engine/line_c/finbert_features.py — FinBERT tone + embeddings (GPU, batched).

Model: yiyanghkust/finbert-tone (BERT fine-tuned on financial communications incl.
analyst reports / earnings calls; 3-way tone: positive / negative / neutral).

Per transcript (aggregated full text): split into 512-token chunks (capped), then
FLATTEN chunks across many documents into large GPU batches (fp16) so the RTX
3060 stays saturated, softmax, and MEAN-POOL chunk probabilities back per doc:
    finbert_pos, finbert_neg, finbert_neu  (mean chunk probabilities)
    finbert_tone = finbert_pos - finbert_neg
Optionally a 768-d mean-pooled encoder embedding (Line C ② showcase).

DOCTRINE: FinBERT weights FROZEN, pretrained pre-sample -> deterministic feature
extraction, immune to LLM narrative-memory contamination. No returns seen.
"""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODEL_NAME = "yiyanghkust/finbert-tone"
MAX_LEN = 512
MAX_CHUNKS = 10            # cap ~5k tokens/doc (prepared remarks + substantial Q&A)
MAX_CHARS = 32000         # truncate text before tokenizing (≈ enough for MAX_CHUNKS)
GPU_BATCH = 128           # chunks per GPU forward (fp16, fits 6GB easily)


class FinBertToner:
    def __init__(self, device: str | None = None, return_embedding: bool = False, fp16: bool = True):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.return_embedding = return_embedding
        self.fp16 = fp16 and self.device == "cuda"
        self.tok = AutoTokenizer.from_pretrained(MODEL_NAME)
        m = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME, output_hidden_states=return_embedding)
        m = m.to(self.device).eval()
        if self.fp16:
            m = m.half()
        self.model = m
        id2label = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.idx = {v: k for k, v in id2label.items()}
        logger.info("FinBERT on %s fp16=%s id2label=%s", self.device, self.fp16, id2label)

    def _chunks_for(self, text: str) -> list[list[int]]:
        text = (text or "")[:MAX_CHARS]
        ids = self.tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        body = MAX_LEN - 2
        raw = [ids[i:i + body] for i in range(0, len(ids), body)][:MAX_CHUNKS]
        cls, sep = self.tok.cls_token_id, self.tok.sep_token_id
        return [[cls] + c + [sep] for c in raw] if raw else [[cls, sep]]

    def _run_batch(self, chunk_batch: list[list[int]]):
        torch = self.torch
        maxl = max(len(c) for c in chunk_batch)
        input_ids = torch.full((len(chunk_batch), maxl), self.tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(chunk_batch), maxl), dtype=torch.long)
        for j, c in enumerate(chunk_batch):
            input_ids[j, :len(c)] = torch.tensor(c)
            attn[j, :len(c)] = 1
        input_ids = input_ids.to(self.device); attn = attn.to(self.device)
        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attn)
            probs = torch.softmax(out.logits.float(), dim=-1).cpu().numpy()
            emb = None
            if self.return_embedding:
                hs = out.hidden_states[-1].float()
                mask = attn.unsqueeze(-1).float()
                emb = ((hs * mask).sum(1) / mask.sum(1).clamp(min=1)).cpu().numpy()
        return probs, emb

    def score_texts(self, texts: list[str]):
        # flatten chunks across docs, remember doc ownership
        all_chunks: list[list[int]] = []
        owner: list[int] = []
        for di, t in enumerate(texts):
            cs = self._chunks_for(t)
            all_chunks.extend(cs)
            owner.extend([di] * len(cs))
        owner = np.asarray(owner)
        n_docs = len(texts)
        prob_sum = np.zeros((n_docs, 3), dtype=np.float64)
        cnt = np.zeros(n_docs, dtype=np.int64)
        emb_sum = None
        for b in range(0, len(all_chunks), GPU_BATCH):
            sub = all_chunks[b:b + GPU_BATCH]
            sub_owner = owner[b:b + GPU_BATCH]
            probs, emb = self._run_batch(sub)
            for k, o in enumerate(sub_owner):
                prob_sum[o] += probs[k]; cnt[o] += 1
            if emb is not None:
                if emb_sum is None:
                    emb_sum = np.zeros((n_docs, emb.shape[1]), dtype=np.float64)
                for k, o in enumerate(sub_owner):
                    emb_sum[o] += emb[k]
        p = prob_sum / cnt[:, None].clip(min=1)
        df = pd.DataFrame({
            "finbert_neu": p[:, self.idx["neutral"]],
            "finbert_pos": p[:, self.idx["positive"]],
            "finbert_neg": p[:, self.idx["negative"]],
        })
        df["finbert_tone"] = df["finbert_pos"] - df["finbert_neg"]
        if emb_sum is not None:
            emb = emb_sum / cnt[:, None].clip(min=1)
            return df, emb.astype(np.float32)
        return df, None


def extract_finbert_features(text_df: pd.DataFrame, *, text_col="full_text", id_col="transcript_id",
                             return_embedding=False, batch_docs=64, log_every=2000):
    toner = FinBertToner(return_embedding=return_embedding)
    texts = text_df[text_col].fillna("").tolist()
    ids = text_df[id_col].astype(int).tolist()
    n = len(texts)
    feat_parts, emb_parts = [], []
    t0 = time.time()
    for i in range(0, n, batch_docs):
        df, emb = toner.score_texts(texts[i:i + batch_docs])
        df[id_col] = ids[i:i + batch_docs]
        feat_parts.append(df)
        if emb is not None:
            emb_parts.append(emb)
        done = i + len(df)
        if done % log_every < batch_docs or done == n:
            logger.info("  FinBERT %d/%d (%.1f docs/s)", done, n, done / max(time.time() - t0, 1e-6))
    feats = pd.concat(feat_parts, ignore_index=True)
    if return_embedding:
        emb_all = np.concatenate(emb_parts, axis=0)
        emb_df = pd.DataFrame(emb_all, columns=[f"emb_{k}" for k in range(emb_all.shape[1])])
        emb_df[id_col] = feats[id_col].values
        return feats, emb_df
    return feats


def extract_finbert_from_parquet(path: str, out_tone: str, *, emb_out: str | None = None,
                                 text_col="full_text", id_col="transcript_id",
                                 read_batch=4000, batch_docs=64, flush_every=5):
    """Streaming + RESUMABLE FinBERT over the large text parquet (avoids OOM; the
    3.5h full run survives interruption). Reads parquet in row-group batches,
    skips transcript_ids already in out_tone, appends incrementally."""
    import os
    import pyarrow.parquet as pq

    done: set[int] = set()
    if os.path.exists(out_tone):
        done = set(pd.read_parquet(out_tone, columns=[id_col])[id_col].astype(int).tolist())
    logger.info("FinBERT resumable: %d already done", len(done))

    toner = FinBertToner(return_embedding=emb_out is not None)
    pf = pq.ParquetFile(path)
    total = pf.metadata.num_rows
    seen = 0
    pend_tone, pend_emb = [], []

    def _flush():
        if pend_tone:
            new = pd.concat(pend_tone, ignore_index=True)
            if os.path.exists(out_tone):
                new = pd.concat([pd.read_parquet(out_tone), new], ignore_index=True).drop_duplicates(id_col, keep="last")
            new.to_parquet(out_tone)
        if emb_out is not None and pend_emb:
            ne = pd.concat(pend_emb, ignore_index=True)
            if os.path.exists(emb_out):
                ne = pd.concat([pd.read_parquet(emb_out), ne], ignore_index=True).drop_duplicates(id_col, keep="last")
            ne.to_parquet(emb_out)
        pend_tone.clear(); pend_emb.clear()

    nb = 0
    for batch in pf.iter_batches(batch_size=read_batch, columns=[id_col, text_col]):
        df = batch.to_pandas()
        df = df[~df[id_col].astype(int).isin(done)]
        seen += len(batch)
        if df.empty:
            continue
        if emb_out is not None:
            feats, emb_df = extract_finbert_features(df, text_col=text_col, id_col=id_col,
                                                     return_embedding=True, batch_docs=batch_docs)
            pend_emb.append(emb_df)
        else:
            feats = extract_finbert_features(df, text_col=text_col, id_col=id_col, batch_docs=batch_docs)
        pend_tone.append(feats)
        nb += 1
        logger.info("FinBERT read %d/%d (this-batch %d new)", seen, total, len(df))
        if nb % flush_every == 0:
            _flush()
    _flush()
    return pd.read_parquet(out_tone)


if __name__ == "__main__":
    import sys, warnings
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    src = sys.argv[1] if len(sys.argv) > 1 else "data/d_pead_plus/_transcripts_text.parquet"
    n_sample = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    txt = pd.read_parquet(src).head(n_sample)
    print(f"validating FinBERT on {len(txt)} transcripts (batched/fp16)...")
    t0 = time.time()
    feats = extract_finbert_features(txt)
    dt = time.time() - t0
    print(f"\nFinBERT {feats.shape} in {dt:.1f}s -> est full 91k: {91000/(len(txt)/dt)/3600:.2f} h")
    print(feats[["finbert_pos", "finbert_neg", "finbert_neu", "finbert_tone"]].describe().round(4).T.to_string())
