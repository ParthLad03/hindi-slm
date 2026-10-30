# Roadmap

**Milestone 1:** a ~0.8B **Hindi + Hinglish chat assistant** on Qwen3.5-0.8B-Base, built end to end in public:
data cleaning → tokenizer study → continued pretraining → SFT → DPO → quantization.

**Target users:** people who type Devanagari Hindi, romanized Hindi, and Hindi–English code-mixed text.
**Hardware:** a single RTX 4090 (24 GB) on RunPod.
**Code style:** hybrid. A hand-written training loop where the learning is (continued pretraining, the from-scratch model); Hugging Face TRL/PEFT for SFT and DPO.

| Phase | What | Design doc | Status | GPU-h |
|---|---|---|---|---|
| 0 | Setup, eval harness, baselines | [phase-0](docs/phases/phase-0-setup-and-baselines.md) | 🟡 planned | – |
| 1 | Tokenizer study: fertility, vocabulary extension | – | ⚪ | – |
| 2 | Data pipeline: FineWeb-2, Sangraha, IndicCorp → clean, dedup, quality-filter | – | ⚪ | – |
| 3 | Continued pretraining (own training loop), data-mix ablations | – | ⚪ | – |
| 4 | SFT (Hindi + Hinglish), LoRA vs full fine-tuning | – | ⚪ | – |
| 5 | Preference tuning (DPO) | – | ⚪ | – |
| 6 | Quantization (GGUF / AWQ, Hindi calibration) + public demo | – | ⚪ | – |

Every phase ends with a public artifact: a report in `reports/`, a dataset or model on the Hugging Face Hub, or both.

**Later:** a small Hindi model pretrained from scratch; then speech and omni models.
