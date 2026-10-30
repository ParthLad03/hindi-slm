# Phase 0 — Setup & Baselines

**Status:** planned · **Target duration:** ~1 week (part-time) · **GPU budget:** ~8–12 RTX 4090 hours

## Why this phase exists

Training without a baseline gives you no way to tell whether a change helped. Phase 0 sets up every measurement the later phases depend on. It also confirms that the chosen base model actually trains on our hardware before Phase 3 depends on it.

## Goals

1. A reproducible repo and environment, plus a RunPod workflow we can repeat without thinking.
2. A **frozen evaluation harness (v0)** that scores any checkpoint on Hindi (Devanagari), romanized Hindi, code-mixed Hindi–English, and English.
3. **Baseline numbers** for the chosen base model, **Qwen3.5-0.8B-Base**, and two reference models.
4. **Verification** that Qwen3.5-0.8B trains on our 4090 stack, with measured throughput and memory.
5. **Hinglish Chat Eval v0**: a native-speaker-written prompt set we'll use from Phase 4 on.

## Non-goals

- No training or fine-tuning. The one exception is a throughput smoke test, whose weights are thrown away.
- No data-cleaning pipeline yet (Phase 2). Phase 0 only pulls small held-out slices for evaluation.
- No deep tokenizer work such as vocabulary extension (Phase 1). Phase 0 only measures fertility.

---

## 0.1 Repo & engineering conventions

| Item | Decision |
|---|---|
| Visibility | Public on GitHub from day one |
| Python env | `uv` + `pyproject.toml`, Python 3.11, versions pinned in `uv.lock` |
| Lint / test | `ruff` + `pytest`. Tests cover anything numeric we could get wrong (BPB, fertility, normalization). |
| Configs | One YAML file per experiment under `configs/`. No hard-coded hyperparameters. |
| Experiment tracking | Weights & Biases (free tier). Every eval also writes a JSON result to `reports/results/`. |
| Artifacts | Checkpoints and datasets on the Hugging Face Hub, never in git |
| Secrets | `.env` (gitignored): `HF_TOKEN`, `WANDB_API_KEY` |
| Commits | At most 2–3 per day, each one a meaningful unit. Messages say *what changed and why*. |
| Private material | `career/` is gitignored and never pushed |

```
LLM/
├── README.md
├── PLAN.md                   # roadmap across phases
├── docs/phases/              # one design doc per phase (this file)
├── configs/                  # eval / train YAMLs
├── scripts/                  # pod_setup.sh, helpers
├── src/hindi_slm/            # importable package
│   ├── eval/                 # bpb.py, benchmarks.py, fertility.py
│   └── utils/
├── eval_sets/                # frozen held-out slices (manifests + hashes) and the Hinglish chat eval
├── tests/
├── reports/                  # write-ups + results/*.json
└── Self-Learn/               # Parth's own rewrite of each component (local only for now)
```

**Working mode:** Claude writes the reference implementation in `src/`. Parth studies it and rewrites each piece from scratch in `Self-Learn/`, then compares the two to find and fix mistakes.

## 0.2 Compute workflow (RunPod, RTX 4090)

- **Local Mac:** code, tokenizer and fertility work, unit tests. These don't need a GPU.
- **Pod:** evals and the throughput test. Start from a PyTorch 2.x + CUDA 12.x template.
- **Network volume:** mounted at `/workspace`, holding `HF_HOME` (model and dataset cache) and the eval slices. Pods stay disposable, and nothing is downloaded twice.
- **`scripts/pod_setup.sh`:** clone the repo, install `uv`, run `uv sync`, log in to HF and W&B from env vars, then run `nvidia-smi` and a bf16 sanity check. One command from a fresh pod to ready.
- **Cost discipline:** stop pods as soon as a job ends. Log GPU-hours per phase in `PLAN.md`.

## 0.3 Base model: Qwen3.5-0.8B-Base (decided 2026-10-30)

**Why:** it's the newest small Qwen. Its 248k vocabulary is 63% larger than Qwen3's, which should help Hindi fertility. And it's natively multimodal, which carries over to the omni phase later. We accept the cost: a hybrid architecture that is harder to learn from and has younger tooling.

Checked against the Hugging Face configs and the `transformers` source on 2026-10-30:

| | **Qwen3.5-0.8B-Base** (chosen) | Qwen3-0.6B-Base (reference) | Qwen2.5-0.5B (reference) |
|---|---|---|---|
| Released | Feb 2026 | 2025 | Sep 2024 |
| Params | 0.87B checkpoint (includes a vision encoder; we load the text model only) | 0.60B | 0.49B |
| Layer layout | 6 × [3 × (Gated DeltaNet → FFN) → 1 × (Gated Attention → FFN)] | 28 × standard attention → FFN | 24 × standard attention → FFN |
| Full attention | 8 Q / 2 KV heads, head_dim 256, output gate, RoPE on 25% of dims | 16 Q / 8 KV, head_dim 128 | 14 Q / 2 KV, head_dim 64 |
| Linear attention | Gated DeltaNet: 16 heads, key/value dim 128, conv kernel 4 | – | – |
| Hidden / FFN | 1024 / 3584 | 1024 / 3072 | 896 / 4864 |
| Vocab | **248,320** (tied embeddings) | 151,936 | 151,936 |
| Extra | 1 multi-token-prediction (MTP) layer | – | – |
| Context | 262k | 32k | 32k |
| License | Apache-2.0 | Apache-2.0 | Apache-2.0 |

**What this means for the stack:**
- Load with `Qwen3_5ForCausalLM`, the text-only class, so the vision tower never touches GPU memory.
- `transformers` has fast-kernel paths for the Gated DeltaNet and short-conv operations, with pure-PyTorch fallbacks. On the pod we install `flash-linear-attention` and `kernels` so the fast path runs, and the throughput test confirms it **is** running (the fallback is much slower).
- **The 248k vocabulary makes the logits expensive.** One 2,048-token sequence produces about 0.5B logit values: ≈1 GB in bf16, ≈2 GB in fp32 for the loss. We'll use a **fused linear + cross-entropy** loss (Liger kernel) or chunked cross-entropy, so the full logits tensor is never materialized.
- Full fine-tuning memory: about 16 bytes/param × ~0.8B ≈ 13 GB before activations. That fits on 24 GB with gradient checkpointing and modest micro-batches. The smoke test confirms it.

**Fallback:** if a hard blocker appears (kernels unusable on the 4090, or broken training), we drop to Qwen3-0.6B-Base and record why. The reference models stay in every baseline table either way. They show what the bigger vocabulary and newer architecture actually buy.

## 0.4 Tokenizer fertility (quick pass)

Phase 1 goes deeper. Here we only measure how Qwen3.5's tokenizer compares, which tells Phase 1 whether vocabulary extension is still worth pursuing.

- **Tokenizers:** Qwen3, Qwen3.5, Sarvam-1 (Indic-optimized reference), Gemma 3 (gated: accept the license on HF first; skip it if that's a hassle).
- **Text:** 5k documents each from FineWeb-2 `hin_Deva`, FineWeb-2 `hin_Latn`, and English FineWeb; plus the code-mixed prompts from §0.6.
- **Metrics:** tokens per whitespace word, characters per token, **bytes per token**. Report per script, with a short table and bar chart.
- **Sanity check:** decoding the encoded text must give back the exact input for every sample.

## 0.5 Evaluation harness v0

Base models can't chat yet, so v0 uses **likelihood-based** and **few-shot** evaluation. Chat evaluation starts in Phase 4.

### a) Intrinsic: bits per byte (BPB) on frozen held-out slices

Perplexity can't be compared across tokenizers, because a tokenizer that splits text more finely has easier per-token predictions. BPB normalizes by UTF-8 bytes:

```
BPB = (sum of token NLLs in nats) / (ln 2 × number of UTF-8 bytes)
```

| Slice | Source | Size |
|---|---|---|
| `hi_deva` | FineWeb-2 `hin_Deva` test split | ~2M bytes |
| `hi_latn` | FineWeb-2 `hin_Latn` test split | ~1M bytes |
| `hi_mixed` | Sangraha `synthetic` (romanized) sample, plus Hinglish eval prompts | ~0.5M bytes |
| `en` | FineWeb (English) sample | ~1M bytes |

The slices are frozen: a manifest of document IDs plus a SHA-256 of the contents, committed under `eval_sets/`. Phase 2 will **decontaminate** its training data against them.

Implementation details that matter (unit-tested):
- Use sliding windows with stride, so no token is scored without context. Count every byte exactly once.
- Run the model in fp32 or bf16, and accumulate the loss in fp32.
- The result must not change with batch size.

### b) Benchmarks (few-shot, log-likelihood scoring)

| Benchmark | Measures | Notes |
|---|---|---|
| MILU (Hindi subset) | Indic knowledge, MCQ | Gated, auto-approved: accept the terms on HF |
| IndicXNLI (hi) | Natural-language inference | |
| FLORES+ hi↔en | Translation (chrF++) | 5-shot generation |
| ARC-Easy, HellaSwag (en) | **English retention** | Tracks forgetting in Phase 3 |

These run through `lm-evaluation-harness` where a task already exists, with custom task YAMLs otherwise. Shot counts, prompts and seeds are fixed in `configs/eval/v0.yaml`.

### c) Outputs

Each run writes `reports/results/<model>__<eval-version>.json` and logs to W&B. One script renders the comparison table.

## 0.6 Hinglish Chat Eval v0 (written by you)

There's no good public eval for how real Indian users type, so we write one. **This is a portfolio artifact in its own right.**

- **~150 prompts**, each written in **3 forms**: Devanagari Hindi, romanized Hindi, and natural code-mixed Hindi–English. That's about 450 items.
- **Categories:** everyday Q&A · explanation and teaching · writing (email, message, story) · reasoning and math word problems · instruction-following with constraints · Indian cultural and local knowledge · safety and refusals.
- **For each prompt:** the category, the expected answer script (should the reply match the user's script?), and reference points a good answer must cover.
- **Rubric** (used from Phase 4 on): helpfulness, correctness, fluency, and *script fidelity*, i.e. whether the reply is in the script the user expects.
- **Format:** `eval_sets/hinglish_chat_v0.jsonl` plus a datasheet in `eval_sets/README.md`.

Claude can draft the category plan and a template. **You write the prompts**, because native phrasing is the point.

## 0.7 Throughput & memory smoke test

For Qwen3.5-0.8B (and Qwen3-0.6B as a comparison point), on the 4090:
- Real tokens, sequence length 2,048, bf16, AdamW, about 50 steps. Try gradient checkpointing on and off, standard vs fused cross-entropy, at the largest micro-batch that fits.
- **Check that the fast kernels are active**, not the PyTorch fallback, and record which path ran.
- Record tokens/sec, peak memory, and **projected hours per 1B tokens**.
- Rough expectation: 10–18k tokens/sec for 0.8B. Phase 0 replaces that guess with a measured number, and Phase 3's token budget is set from it.

## 0.8 Definition of done

- [ ] Repo public, `uv sync` works locally and on a fresh pod, CI runs ruff + pytest
- [ ] `scripts/pod_setup.sh` goes from a fresh pod to ready in one command
- [ ] Fertility table for 4 tokenizers × 4 text types
- [ ] Eval harness v0: BPB on 4 frozen slices + 5 benchmarks, with tests for BPB
- [ ] Baseline results for Qwen3.5-0.8B-Base, plus Qwen3-0.6B-Base and Qwen2.5-0.5B as references
- [ ] Throughput and memory numbers on the 4090, with the fast kernel path confirmed
- [ ] `reports/00-baselines.md`: baselines, fertility, throughput, and what they imply for Phases 1–3
- [ ] `eval_sets/hinglish_chat_v0.jsonl` (~150 × 3) + datasheet

## 0.9 Day-by-day plan (≤ 2–3 commits/day)

| Day | Work | Commits |
|---|---|---|
| 1 | Repo skeleton, README, PLAN, this doc, `pyproject`, CI | 1 |
| 2 | `pod_setup.sh`, pod sanity run; fertility script + tests | 1–2 |
| 3 | Fertility results; freeze the eval slices (manifests + hashes) | 2 |
| 4 | BPB evaluator + tests; first BPB numbers | 1–2 |
| 5 | Benchmark configs; run all 3 models | 1–2 |
| 6 | Throughput test; baselines report | 2 |
| 7 | Hinglish Chat Eval v0 (your writing day); datasheet | 1–2 |

## 0.10 Risks & open questions

| Risk | Mitigation |
|---|---|
| Qwen3.5's Gated DeltaNet kernels fail or silently fall back to slow PyTorch | The throughput test checks which path ran; Qwen3-0.6B is the documented fallback |
| Checkpoint includes a vision tower | Load with `Qwen3_5ForCausalLM`; confirm param count and memory in the smoke test |
| 248k-vocab logits blow up memory | Fused linear + cross-entropy (Liger) or chunked loss |
| Younger tooling (TRL/PEFT, GGUF, vLLM) for a hybrid architecture | Check each tool's support before its phase starts; budget time for workarounds |
| Gated datasets and models (MILU, Gemma, FLORES+) | Accept the terms once on HF; drop Gemma if needed |
| No clean code-mixed held-out text | Use the synthetic romanized text + our own prompts in v0; improve the slice in Phase 2 |
| Benchmark scores near chance for 0.5B base models | Expected for MCQ. That's why BPB is the primary signal and benchmarks secondary |

## What you'll learn in this phase

- Why **BPB** rather than perplexity, and how sliding-window evaluation works → career guide, Part G (cross-entropy / perplexity)
- How **log-likelihood MCQ scoring** works in lm-eval-harness, and why few-shot matters for base models
- **Tokenizer fertility** and why it changes cost, context length and learning difficulty → Part H §1
- Reading a **model config**: GQA, head_dim, tied embeddings, partial RoPE, output gating
- **Linear attention / Gated DeltaNet**: why a recurrent-state layer makes long context cheap, and why hybrids keep some full-attention layers
- **Measuring throughput honestly:** warmup steps, synchronization, peak memory
