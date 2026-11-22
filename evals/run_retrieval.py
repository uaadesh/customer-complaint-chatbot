"""
Retrieval benchmark: for every FAQ question in the test set, does the retriever find the
policy section that answers it?

Each FAQ conversation in test_set.jsonl lists the section(s) that answer it in
expected.sources, e.g. "refunds.md > Refunds that need approval". Each retriever setup
ranks all policy sections for the question's first turn, and we record where the first
correct section lands. The setups are built with the bot's own PolicyRetriever
(chatbot/rag_retriever.py), so the scores describe the code the bot runs.

Metrics, overall and per language:
- recall@1: the top result is a correct section
- recall@3 / recall@4: a correct section is in the top 3 / 4 (the bot is given the top 4)
- MRR: mean of 1/rank of the first correct section (0 if not in the top 10)
- ms_per_query: search time for one question, on the CPU

Needs no Ollama and no backend. Results are added to the "retrieval" block of the
combined results file.

    python evals/run_retrieval.py --name p2-retrieval-v1
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from chatbot.knowledge_base import load_sections  # noqa: E402
from chatbot.rag_retriever import PolicyRetriever  # noqa: E402

TOP_K = 10  # how far down the ranking we look for MRR
LANGUAGES = ["en", "hi", "hinglish"]

EN_MINILM = "sentence-transformers/all-MiniLM-L6-v2"  # what the bot used before Phase 2
MULTI_MINILM = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ME5_SMALL = "intfloat/multilingual-e5-small"
ME5_BASE = "intfloat/multilingual-e5-base"

# name -> PolicyRetriever settings
SETUPS = {
    "bm25": dict(embed_model=None, hybrid=True),
    "en-minilm": dict(embed_model=EN_MINILM),
    "multi-minilm": dict(embed_model=MULTI_MINILM),
    "me5-small": dict(embed_model=ME5_SMALL),
    "me5-base": dict(embed_model=ME5_BASE),
    "hybrid(bm25+multi-minilm)": dict(embed_model=MULTI_MINILM, hybrid=True),
    "hybrid(bm25+me5-small)": dict(embed_model=ME5_SMALL, hybrid=True),
    "hybrid(bm25+me5-base)": dict(embed_model=ME5_BASE, hybrid=True),
    "me5-base+rerank": dict(embed_model=ME5_BASE, rerank=True),
    "hybrid(bm25+multi-minilm)+rerank": dict(embed_model=MULTI_MINILM, hybrid=True, rerank=True),
    "hybrid(bm25+me5-small)+rerank": dict(embed_model=ME5_SMALL, hybrid=True, rerank=True),
    "hybrid(bm25+me5-base)+rerank": dict(embed_model=ME5_BASE, hybrid=True, rerank=True),
}


def first_hit_rank(ranking, sections, wanted):
    for rank, idx in enumerate(ranking[:TOP_K], start=1):
        if sections[idx].source in wanted:
            return rank
    return None  # not in the top TOP_K


def summarise(ranks: list):
    n = len(ranks)
    if not n:
        return {"n": 0}
    found = [r for r in ranks if r]
    return {
        "n": n,
        "recall@1": round(sum(r == 1 for r in found) / n, 3),
        "recall@3": round(sum(r <= 3 for r in found) / n, 3),
        "recall@4": round(sum(r <= 4 for r in found) / n, 3),
        "mrr": round(sum(1 / r for r in found) / n, 3),
    }


def git_commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
        # Only the code and data that decide the scores; results and docs don't count.
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "chatbot", "knowledge-base",
                                 "evals/run_retrieval.py", "evals/conversations"], cwd=REPO_ROOT) != 0
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def save(out_path: Path, run: dict):
    combined = json.loads(out_path.read_text(encoding="utf-8")) if out_path.exists() else {"index": [], "runs": {}}
    runs = combined.get("retrieval", {}).get("runs", {})
    runs[run["run"]["name"]] = run
    # One row per run and setup, so setups are easy to compare at the top.
    index = [{"run": name, "setup": setup, **m["overall"], "ms_per_query": m["ms_per_query"]}
             for name, r in runs.items() for setup, m in r["setups"].items()]
    combined["retrieval"] = {"index": index, "runs": runs}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="run name, e.g. p2-retrieval-v1")
    parser.add_argument("--tests", default="evals/conversations/test_set.jsonl")
    parser.add_argument("--out", default="evals/results/v0-benchmark.json")
    parser.add_argument("--only", help="comma-separated setup names to run")
    args = parser.parse_args()

    sections = load_sections()
    convs = [json.loads(l) for l in (REPO_ROOT / args.tests).read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = [c for c in convs if c.get("expected", {}).get("sources")]
    print(f"{len(sections)} sections, {len(questions)} questions with source labels")

    setups = {k: v for k, v in SETUPS.items() if not args.only or k in args.only.split(",")}
    results = {}
    for name, settings in setups.items():
        retriever = PolicyRetriever(sections=sections, **settings)
        retriever.rank("warm-up")  # load models before timing
        ranks, start = {}, time.perf_counter()
        for q in questions:
            ranks[q["id"]] = first_hit_rank(retriever.rank(q["turns"][0]), sections, set(q["expected"]["sources"]))
        ms = (time.perf_counter() - start) * 1000 / len(questions)

        overall = summarise(list(ranks.values()))
        by_lang = {lang: summarise([ranks[q["id"]] for q in questions if q["language"] == lang]) for lang in LANGUAGES}
        results[name] = {"retriever": retriever.describe(), "overall": overall, "by_language": by_lang,
                         "ms_per_query": round(ms, 1),
                         "ranks": ranks}  # rank of the first correct section; null = not in the top 10
        print(f"{name:34} R@1 {overall['recall@1']:.2f}  R@3 {overall['recall@3']:.2f}  "
              f"R@4 {overall['recall@4']:.2f}  MRR {overall['mrr']:.2f}   "
              + "  ".join(f"{l} R@3 {by_lang[l]['recall@3']:.2f}" for l in LANGUAGES) + f"   {ms:6.1f} ms/q")

    run_record = {
        "run": {
            "name": args.name,
            "git_commit": git_commit(),
            "test_file": args.tests,
            "sections": len(sections),
            "questions": len(questions),
            "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
        "setups": results,
    }
    out_path = REPO_ROOT / args.out  # an absolute --out replaces REPO_ROOT
    save(out_path, run_record)
    print(f"Saved to {out_path} (retrieval.runs.{args.name})")


if __name__ == "__main__":
    main()
