# chatbot/config.py
import os

use_openai = False  # Set to False to use HF models

OPENAI_API_KEY = "your-openai-api-key"
HF_MODEL_NAME = "Qwen/Qwen3-8B"


def _flag(name: str, default: bool) -> bool:
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


# --- LLM ----------------------------------------------------------------------
# gemma4 thinks (writes hidden reasoning) before every reply by default. Off is ~4x
# faster, but in a quick test the bot then stopped replying in Hindi/Hinglish, so it
# stays on until the eval shows otherwise. LLM_THINKING=0 turns it off.
LLM_THINKING = _flag("LLM_THINKING", True)

# --- Policy retrieval -------------------------------------------------------
# Defaults are the winner of the Phase 2 retrieval benchmark (evals/results/v0-summary.md):
# multilingual-e5-base embeddings alone, top 4 sections. Hybrid search and the reranker
# can be switched on per server with environment variables, e.g. RETRIEVAL_HYBRID=1.
RETRIEVAL_EMBED_MODEL = os.getenv("RETRIEVAL_EMBED_MODEL", "intfloat/multilingual-e5-base")
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "4"))
RETRIEVAL_HYBRID = _flag("RETRIEVAL_HYBRID", False)  # fuse BM25 keyword search with embeddings (RRF)
RETRIEVAL_RERANK = _flag("RETRIEVAL_RERANK", False)  # re-score the top candidates with a cross-encoder
RETRIEVAL_RERANK_MODEL = os.getenv("RETRIEVAL_RERANK_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")
