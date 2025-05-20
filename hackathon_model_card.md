---
license: mit
tags:
  - modular-llm
  - registers
  - continual-learning
  - plug-and-play
  - hackathon
  - french-ai
  - llama
  - fine-tuning
  - domain-adaptation
model-index:
  - name: Modular LLMs - Learning with Registers
    results:
      - task:
          type: text-generation
        dataset:
          name: FineWebEDU
          type: text
        metrics:
          - type: mmlu
            value: see table below
---

# 🧠 Modular LLMs: Learning with Registers

## Model Card

### Model Summary

Modular LLMs: Learning with Registers is an open-source project demonstrating a new approach to extending LLMs without full retraining, via **modular learning**. This method allows dynamic injection of domain expertise modules at inference, without modifying the base model weights. 
This methods is an alternative to both RAG and LORA approaches, especially in the case where we need to inject a corpus of knowledge in a condensed form :
- Compared to RAG, it allows to inject a large corpus in very few tokens and inference compute equivalent
- Compared to LoRA and other adapters, it decouples knowledge from capabilities and focuses on modifying the "knowledge" related weights without touching the broader capabilities and common sens of the original model.

It is especially relevant for small language models that cannot retain by heart all the information in their weight at the same time.

---

## Training Details

- **Base**: LLM ~2B parameters, Llama-like architecture
- **Pre-training**: FineWebEDU (10B tokens)
- **Config**:
    - dim: 2048
    - n_layers: 25
    - n_heads: 16
    - sequence: 4096 tokens
    - batch_size: 2 (peak mem: 55G for 4 samples/GPU)
    - dtype: bf16
    - optimizer: AdamW, 
    - steps: 38,000
    - Tokenizer: Llama 3 (SentencePiece)
- **Modules**: trained separately on specialized datasets (e.g.: PubMed, Wikipedia)

---

## Evaluation

### MMLU (0–100%) on different modules

| Modules enabled | Base | Pubmed | TBD |
| --------------- | ----- | ------- | ---------- |
| ✅ MMLU Anatomy        | xx%   | xx%     | xx%        |

*Modules improve performance on their target domain.*

---

## Citation

If you use this work, please cite:

> Modular LLMs: Learning with Registers, Hackathon IDRIS CNRS/NVIDIA 2025, ANITI, IRT Saint Exupery
Thibaut Boissin, Lucas Hervier, Agustin Picard, Jonathan Sprauel

---

## License

MIT License

---

## Acknowledgements

This project received support from the "IA Cluster" program (ANITI), the "France 2030" program (IRT Saint Exupery), and was carried out as part of the 2025 GPU Hackathon organized by IDRIS CNRS and NVIDIA.

---

## Contact

For any questions or contributions, open an issue on the GitHub repo or contact the ANITI/DEEL team.
