"""
Baseline eval runner (Phase 1, step 4).

Runs every conversation in a test file through the real agent, records which
tools it called and what it saved, and scores the result.

Usage (from the repo root, with Ollama and the backend running):
    python evals/run_eval.py --name v0-baseline
    python evals/run_eval.py --name debug --only en-file-001,hi-faq-001
    python evals/run_eval.py --name v0.1-qwen3-4b-instruct --model qwen3:4b-instruct

All runs are stored in ONE combined file (--out, default
evals/results/v0-benchmark.json): each run is added under its --name, and a
run with the same name replaces the old one. The file starts with an "index"
of every run's headline numbers, followed by the full details per run.

Fold existing single-run result files into the combined file:
    python evals/run_eval.py --merge evals/results/v0-baseline.json evals/results/v0.1-*.json

Scores per conversation:
    tool_correct  - the expected tool was called with the expected arguments
                    (or no complaint tool was called, when none is expected)
    task_success  - tool_correct AND the saved complaint row matches AND the
                    final reply passes its answer checks
    rubric_grade  - left as null; graded by hand against the "rubric" text
"""

import argparse
import inspect
import json
import sqlite3
import statistics
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# The backend opens "complaints.db" relative to where uvicorn was started,
# so it must be started from the repo root for this path to be the same file.
DB_PATH = REPO_ROOT / "complaints.db"
API_BASE = "http://localhost:8000"
COMPLAINT_FIELDS = ["complaint_id", "name", "phone_number", "email", "complaint_details", "created_at"]


# ---------------------------------------------------------------------------
# Matching rules used in the test file: equals / contains_any / contains_none
# All comparisons are case-insensitive and ignore surrounding whitespace.
# ---------------------------------------------------------------------------
def check_value(value, rule: dict) -> list[str]:
    """Return a list of failure messages (empty list = passed)."""
    text = (value or "").strip().lower()
    failures = []
    if "equals" in rule and text != rule["equals"].strip().lower():
        failures.append(f"expected {rule['equals']!r}, got {value!r}")
    if "contains_any" in rule and not any(s.lower() in text for s in rule["contains_any"]):
        failures.append(f"none of {rule['contains_any']} found in {value!r}")
    if "contains_none" in rule:
        bad = [s for s in rule["contains_none"] if s.lower() in text]
        if bad:
            failures.append(f"forbidden {bad} found in {value!r}")
    return failures


# ---------------------------------------------------------------------------
# Database helpers: reset and seed the complaints table between conversations.
# The backend opens a fresh connection per request, so it sees these changes.
# ---------------------------------------------------------------------------
def reset_db(seed: list[dict]) -> None:
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("DELETE FROM complaints")
        for c in seed:
            conn.execute(
                "INSERT INTO complaints (complaint_id, name, phone_number, email, complaint_details, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (c["complaint_id"], c["name"], c["phone_number"], c["email"], c["complaint_details"],
                 datetime.now().isoformat()),
            )


def new_complaints(seeded_ids: set[str]) -> list[dict]:
    """Complaints created during the conversation, oldest first."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("SELECT * FROM complaints ORDER BY created_at").fetchall()
    return [dict(zip(COMPLAINT_FIELDS, r)) for r in rows if r[0] not in seeded_ids]


# ---------------------------------------------------------------------------
# Tool recording: wrap the agent's tools so every call is logged, then passed
# through unchanged to the real tool. The bot's own code is not modified.
# ---------------------------------------------------------------------------
TOOL_CALLS: list[dict] = []


def recording(name, real_tool):
    def wrapper(tool_input):
        result = real_tool(tool_input)  # same call the agent would make
        TOOL_CALLS.append({"tool": name, "input": tool_input, "output": result})
        return result
    return wrapper


def build_recorded_agent(model: str | None):
    import chatbot.agent as agent_module
    for name in ("create_complaint_tool", "get_complaint_tool"):
        setattr(agent_module, name, recording(name, getattr(agent_module, name)))

    # build_agent() calls get_local_llm() with no arguments, so it always gets
    # the default model. To benchmark another model without editing the bot,
    # replace get_local_llm with a version whose default is the chosen model.
    default_model = inspect.signature(agent_module.get_local_llm).parameters["model_name"].default
    if model and model != default_model:
        original = agent_module.get_local_llm
        agent_module.get_local_llm = lambda model_name=model: original(model_name)
    return agent_module.build_agent(), model or default_model


# ---------------------------------------------------------------------------
# Scoring one conversation
# ---------------------------------------------------------------------------
def score(conv: dict, calls: list[dict], saved: list[dict], final_reply: str) -> dict:
    expected = conv["expected"]
    tool_failures, other_failures = [], []

    called = [c["tool"] for c in calls]
    want = expected.get("tool")
    if want is None:
        if called:
            tool_failures.append(f"expected no tool call, got {called}")
    elif want not in called:
        tool_failures.append(f"expected {want}, got {called or 'no tool call'}")
    else:
        # Check the arguments of the last call to the expected tool.
        last_input = [c["input"] for c in calls if c["tool"] == want][-1]
        for arg, rule in expected.get("tool_args", {}).items():
            tool_failures += [f"tool arg {arg}: {m}" for m in check_value(str(last_input), rule)]

    if "saved_complaint" in expected:
        if not saved:
            other_failures.append("no complaint was saved")
        else:
            row = saved[-1]  # judge the most recent complaint created
            for field, rule in expected["saved_complaint"].items():
                other_failures += [f"saved {field}: {m}" for m in check_value(row[field], rule)]
            if len(saved) > 1:
                other_failures.append(f"{len(saved)} complaints created (expected 1)")

    if "answer" in expected:
        other_failures += [f"answer: {m}" for m in check_value(final_reply, expected["answer"])]

    return {
        "tool_correct": not tool_failures,
        "task_success": not tool_failures and not other_failures,
        "failures": tool_failures + other_failures,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def preflight() -> None:
    for url, what in [(f"{API_BASE}/docs", "backend (uvicorn backend.main:app --port 8000)"),
                      ("http://127.0.0.1:11434/", "Ollama")]:
        try:
            urllib.request.urlopen(url, timeout=5)
        except Exception:
            sys.exit(f"Cannot reach {what} at {url}. Start it first.")
    if not DB_PATH.exists():
        sys.exit(f"{DB_PATH} not found - start the backend from the repo root.")


def rate(items, key):
    return round(sum(i[key] for i in items) / len(items), 3) if items else None


# ---------------------------------------------------------------------------
# Combined results file: {"index": [headline per run], "runs": {name: run}}
# ---------------------------------------------------------------------------
def load_combined(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"index": [], "runs": {}}


def save_combined(path: Path, combined: dict) -> None:
    # Rebuild the index so it always matches the runs below it.
    combined["index"] = [
        {
            "name": name,
            "model": r["run"]["model"],
            "git_commit": r["run"]["git_commit"],
            "test_file": r["run"]["test_file"],
            "task_success": r["summary"]["task_success"],
            "tool_correct": r["summary"]["tool_correct"],
            "p50_seconds": r["summary"]["turn_latency_seconds"]["p50"],
            "p95_seconds": r["summary"]["turn_latency_seconds"]["p95"],
        }
        for name, r in combined["runs"].items()
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    # Index first, easy to read. Keep other blocks (e.g. "retrieval" from run_retrieval.py).
    ordered = {"index": combined["index"], "runs": combined["runs"],
               **{k: v for k, v in combined.items() if k not in ("index", "runs")}}
    path.write_text(json.dumps(ordered, indent=2, ensure_ascii=False), encoding="utf-8")


def merge_files(files: list[str], out_path: Path) -> None:
    combined = load_combined(out_path)
    for f in files:
        run = json.loads(Path(f).read_text(encoding="utf-8"))
        combined["runs"][run["run"]["name"]] = run
        print(f"merged {run['run']['name']}")
    save_combined(out_path, combined)
    print(f"{len(combined['runs'])} runs in {out_path.relative_to(REPO_ROOT)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="run name, e.g. v0-baseline (required unless --merge)")
    parser.add_argument("--tests", default="evals/conversations/test_set.jsonl")
    parser.add_argument("--only", help="comma-separated conversation ids to run")
    parser.add_argument("--model", help="Ollama model to use instead of the bot's default, e.g. qwen3:4b-instruct")
    parser.add_argument("--out", default="evals/results/v0-benchmark.json",
                        help="combined results file that every run is added to")
    parser.add_argument("--merge", nargs="+", metavar="FILE",
                        help="add existing single-run result files to --out, then exit")
    args = parser.parse_args()
    out_path = REPO_ROOT / args.out

    if args.merge:
        merge_files(args.merge, out_path)
        return
    if not args.name:
        parser.error("--name is required")

    tests = [json.loads(line) for line in open(REPO_ROOT / args.tests, encoding="utf-8") if line.strip()]
    if args.only:
        wanted = set(args.only.split(","))
        tests = [t for t in tests if t["id"] in wanted]

    preflight()
    agent, model = build_recorded_agent(args.model)
    print(f"Built agent with model {model}")

    results = []
    for i, conv in enumerate(tests, 1):
        seed = conv.get("setup", {}).get("seed_complaints", [])
        reset_db(seed)
        TOOL_CALLS.clear()
        agent.memory.chat_memory.messages.clear()  # fresh conversation

        transcript = []
        for user_msg in conv["turns"]:
            start = time.time()
            try:
                reply = agent.run(user_msg)
            except Exception as e:  # record crashes as a failed turn, keep going
                reply = f"<agent error: {e!r}>"
            transcript.append({"user": user_msg, "bot": reply, "seconds": round(time.time() - start, 2)})

        saved = new_complaints({c["complaint_id"] for c in seed})
        result = score(conv, list(TOOL_CALLS), saved, transcript[-1]["bot"])
        results.append({
            "id": conv["id"], "language": conv["language"], "category": conv["category"],
            **result,
            "tool_calls": list(TOOL_CALLS), "saved_complaints": saved, "transcript": transcript,
            "rubric": conv.get("rubric"), "rubric_grade": None,
        })
        mark = "PASS" if result["task_success"] else "FAIL"
        print(f"[{i}/{len(tests)}] {mark} {conv['id']}" + (f"  - {result['failures'][0]}" if result["failures"] else ""))

    # Summary, overall and broken down by language and category
    turn_times = [t["seconds"] for r in results for t in r["transcript"]]
    breakdown = lambda key: {
        k: {"n": len(v), "task_success": rate(v, "task_success"), "tool_correct": rate(v, "tool_correct")}
        for k, v in sorted(_group(results, key).items())
    }
    summary = {
        "n": len(results),
        "task_success": rate(results, "task_success"),
        "tool_correct": rate(results, "tool_correct"),
        "by_language": breakdown("language"),
        "by_category": breakdown("category"),
        "turn_latency_seconds": {
            "p50": round(statistics.median(turn_times), 2),
            "p95": round(statistics.quantiles(turn_times, n=20)[-1], 2) if len(turn_times) > 1 else turn_times[0],
        },
    }

    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                            cwd=REPO_ROOT).stdout.strip()
    output = {
        "run": {
            "name": args.name,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": commit,
            "model": model,
            "test_file": args.tests,
        },
        "summary": summary,
        "conversations": results,
    }
    # Re-read the combined file just before writing, so runs added by another
    # process while this one was running are kept.
    combined = load_combined(out_path)
    combined["runs"][args.name] = output
    save_combined(out_path, combined)

    print(f"\nTask success: {summary['task_success']:.0%}   Tool correct: {summary['tool_correct']:.0%}   "
          f"p50/p95 turn: {summary['turn_latency_seconds']['p50']}s / {summary['turn_latency_seconds']['p95']}s")
    print(f"Saved as run '{args.name}' in {out_path.relative_to(REPO_ROOT)}")


def _group(items, key):
    groups = defaultdict(list)
    for item in items:
        groups[item[key]].append(item)
    return groups


if __name__ == "__main__":
    main()
