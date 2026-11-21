# v0 evaluation summary

How well does the original complaint chatbot work, and which free local model should it run on?

## Setup

| | |
|---|---|
| Laptop | NVIDIA GTX 1650 Ti (4 GB VRAM), AMD Ryzen 5 4600H (6 cores), 8 GB RAM, Windows 11 |
| Model server | Ollama (0.17.7 for the v0 baseline, 0.40.17 for the benchmark) |
| Test set | 27 conversations: 11 English, 8 Hindi, 8 Hinglish, covering filing a complaint, looking one up, FAQ questions, missing information and off-topic requests |
| Scoring | **Task success**: the right tool was called, the saved complaint is correct, and the reply contains the right facts. **Tool correct**: the right tool was called with the right arguments. |

Full results for every conversation: [v0-benchmark.json](v0-benchmark.json).

## Two versions of the bot

- **v0** is the original prototype, unchanged (git tag `v0-baseline`).
- **v0.1** is v0 plus basic bug fixes:
  - complaint text may contain commas;
  - no false "registered" message when filing fails;
  - no example values in the prompt for the model to copy;
  - replies in the customer's language;
  - temperature 0;
  - a reply-length cap;
  - more robust JSON reading.

  The FAQ is still not connected in either version.

## Models tested

| Model | Download size | Fits in 4 GB GPU? |
|---|---|---|
| llama3.2:1b | 1.3 GB | Yes |
| qwen2.5:3b | 1.9 GB | Yes |
| llama3.2:3b | 2.0 GB | Yes |
| qwen3:4b-instruct | 2.5 GB | Yes |
| qwen3.5:4b | 3.3 GB | No, runs half on CPU |
| gemma4:e2b | 4.6 GB | No, runs partly on CPU |

All models support Hindi. Models without official Hindi support were excluded.

## Results

| Bot | Model | Task success | Tool correct | File complaint | Lookup | Missing info | Typical reply (p50) | Slow reply (p95) |
|---|---|---|---|---|---|---|---|---|
| v0 | llama3.2:3b | 56% | 89% | 0% | 100% | 67% | 2.5 s | 13.4 s |
| v0.1 | **gemma4:e2b** | **70%** | **100%** | **100%** | 100% | 100% | 5.8 s | 10.8 s |
| v0.1 | qwen3:4b-instruct | 63% | 100% | 71% | 100% | 100% | 5.5 s | 30.2 s |
| v0.1 | qwen2.5:3b | 63% | 89% | 57% | 100% | 67% | **1.0 s** | **3.8 s** |
| v0.1 | llama3.2:3b | 48% | 78% | 57% | 40% | 100% | 1.5 s | 4.5 s |
| v0.1 | llama3.2:1b | 26% | 74% | 29% | 0% | 33% | 0.7 s | 5.0 s |
| v0.1 | qwen3.5:4b | 15% | 56% | 0% | 0% | 0% | 65 s | 97 s |

## What this shows

- **The bug fixes mattered more than the model.** Filing a complaint went from 0% to as high as 100%.
- **Three finalists:** gemma4:e2b (most accurate), qwen3:4b-instruct (equally reliable with tools) and qwen2.5:3b (5× faster).
- **Ruled out:**
  - **llama3.2 (1B and 3B):** weak lookups, and sensitive to small changes in prompt wording.
  - **qwen3.5:4b:** its built-in reasoning used up the whole reply budget, leaving 39 of 41 replies empty.

## Limits

- **27 conversations is a small sample.** A difference of less than about 20 points can be chance, so the finalists are not separated yet.
- **FAQ answers are not compared.** Every model scored 0–25% because the FAQ isn't connected yet, so they guess. This will be fixed in the next phase.
- **The test sets differ slightly.** The v0 baseline used the original test set, and v0.1 used a copy with two stricter FAQ checks.

## Hand grading of the finalists

_To be added._

## Decision

_To be added after hand grading._
