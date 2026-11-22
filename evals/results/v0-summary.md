# v0 evaluation summary

How well does the original complaint chatbot work, and which free local model should it run on?

## Setup

| | |
|---|---|
| Laptop | NVIDIA GTX 1650 Ti (4 GB VRAM), AMD Ryzen 5 4600H (6 cores), 8 GB RAM, Windows 11 |
| Model server | Ollama (0.17.7 for the v0 baseline, 0.40.17 for the benchmark) |
| Test set | 27 conversations: 11 English, 8 Hindi, 8 Hinglish, covering filing a complaint, looking one up, FAQ questions, missing information and off-topic requests. These are now the start of [`test_set.jsonl`](../conversations/test_set.jsonl), which will grow to 150–200 conversations. |
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
- **The test sets differ slightly.** The v0 baseline used the original test set, and v0.1 used a copy with two stricter FAQ checks. Both files are kept in git history at commit `b883481`, under `evals/conversations/`.
- **The exact wording depends on the setup.** Temperature 0 gives the same reply every run on the same setup, but moving layers between CPU and GPU changed the wording of 22 of 27 conversations. The pass/fail outcomes stayed the same.

## Hand grading of the finalists

Each conversation was checked by hand against a one-line description of good behaviour (its rubric). The model's own replies had to be in the customer's language; the fixed ✅/❌ messages written by the code were not counted.

| Model | Automatic | Hand-graded (all 27) | Hand-graded (without FAQ, 19) | Wrong customer data saved | Policy facts made up | Off-topic requests carried out |
|---|---|---|---|---|---|---|
| **gemma4:e2b** | 70% | **59** | **84** | **0** | **1** | 2 of 4 |
| qwen2.5:3b | 63% | 56 | 68 | 2 | 2 | **0 of 4** |
| qwen3:4b-instruct | 63% | 52 | 74 | 1 | 5 | 2 of 4 |

The automatic checks missed off-topic mistakes, such as writing the requested cricket poem, because those tests only checked that no tool was called.

## gemma4:e2b experiments

| Setup | Task success | Typical reply (p50) | Slow reply (p95) | GPU memory |
|---|---|---|---|---|
| Default (Ollama puts 26 of 36 layers on the GPU) | 70% | 5.8 s | 10.8 s | 2.7 GB |
| **All layers on the GPU** (`num_gpu 99`) | 70% | **4.1 s** | **7.3 s** | 3.1 GB |
| QAT version (`gemma4:e2b-it-qat`), all on the GPU | 70% | 5.1 s | 8.8 s | 2.8 GB |

- Ollama holds back GPU memory for gemma's image and audio parts, which this bot never uses. Putting all layers on the GPU anyway makes replies about 30% faster, with no change in accuracy.
- The QAT version (quantization-aware training) gave the same results and was a little slower. Google does not publish accuracy figures comparing the Gemma 4 QAT versions with the full-precision model.

## Decision

**Model: `gemma4:e2b` (standard Q4_K_M version), with all layers on the GPU.**

- **Most accurate, and never saved wrong data.** It filed all 7 complaints correctly in all three languages and made up the fewest facts.
- **The trade-off is speed:** about 4 s per reply, against about 1 s for qwen2.5:3b. qwen2.5's mistakes (wrong saved data, broken Hindi) are harder to fix than gemma's slower replies.
- **Known weaknesses to fix next:**
  - policy answers, by connecting the FAQ;
  - carrying out off-topic requests, by adding guardrails.
