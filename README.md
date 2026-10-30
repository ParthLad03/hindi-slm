# Hindi SLM, built from the data up

Building a small (~0.8B, on Qwen3.5-0.8B-Base) **Hindi + Hinglish** language model end to end, on a single 24 GB GPU, in the open:
data cleaning → tokenizer study → continued pretraining → instruction tuning → preference tuning → quantization.

Most Indian users don't type only Devanagari. They mix scripts and languages ("kal meeting hai, please remind karna"). This project targets that reality, and every step is measured against baselines with a frozen evaluation harness.

- **Roadmap:** [PLAN.md](PLAN.md)
- **Current phase:** [Phase 0: setup & baselines](docs/phases/phase-0-setup-and-baselines.md)
- **Results and write-ups:** `reports/` (coming as each phase lands)

## Status

🟡 Phase 0 in planning. No models or datasets released yet.

## License

Code: Apache-2.0. Datasets and models will carry their own cards and licenses.
