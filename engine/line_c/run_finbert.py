"""engine/line_c/run_finbert.py — full FinBERT tone + embeddings over 91k corpus.

GPU (RTX 3060, fp16), streaming + RESUMABLE: survives interruption (re-run to
resume from where it stopped). ~3.5h for the full corpus at ~14 docs/s.
"""
import warnings
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

from engine.line_c.finbert_features import extract_finbert_from_parquet

TEXT = "data/line_c/_transcripts_text_2011_2024.parquet"
TONE_OUT = "data/line_c/_finbert_features_full.parquet"
EMB_OUT = "data/line_c/_finbert_emb_full.parquet"

if __name__ == "__main__":
    df = extract_finbert_from_parquet(TEXT, TONE_OUT, emb_out=EMB_OUT)
    print(f"FinBERT full done: {len(df)} transcripts")
