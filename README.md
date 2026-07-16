# Defending LLMs against Jailbreak Attacks via Template-Based ICL with a Defensive Suffix

**ACL 2026** |[Code](https://github.com/Trusted-LLM/DSICL)

This repository contains the official implementation of our paper **"Defending LLMs against Jailbreak Attacks via Template-Based ICL with a Defensive Suffix"**. We propose a lightweight, plug-and-play defense mechanism that combines an offline-optimized defensive suffix with online template-based in-context learning (ICL) demonstrations to protect large language models from both white-box and black-box jailbreak attacks.

![Overview](figure/1.png)

---

## 🚀 Key Features

- **High Effectiveness**: Reduces Attack Success Rate (ASR) to nearly **0%** against advanced attacks like GCG, I-GCG, AutoDAN, PAIR, TAP, and more.
- **Lightweight**: Only a negligible computational overhead during inference (no model retraining).
- **Plug-and-Play**: Works for both open-source (Llama, Vicuna) and closed-source (GPT-3.5/4/4.1) models without modifying model parameters.
- **Transferable**: Defensive suffixes trained on one model transfer well to others.
- **Adaptive Robustness**: Maintains strong performance even under white-box adaptive attacks.

---

## 📖 Method Overview

Our method consists of two stages:

### 1. Offline Training Stage (Defensive Suffix Optimization)

- Iteratively harden malicious prompts using existing attack methods (e.g., GCG).
- Optimize a short defensive suffix \( s_{1:n} \) to maximize the likelihood of safe refusal responses while preserving utility on benign queries.
- Uses a hybrid gradient‑guided random search with multi‑coordinate updates for efficiency.

### 2. Online Inference Stage (ICL + Suffix)

- For each user prompt, perform a batch‑level random search over a demonstration pool to select the best ICL example.
- Combine the selected demonstration, the user prompt, and the pre‑trained defensive suffix into a structured template.
- Feed the composite prompt to the LLM for safe response generation.

![Pipeline](figure/2.png)

---

## 📊 Experimental Results

### White‑box Setting (AdvBench)

![Pipeline](figure/3.png)

### Black‑box Setting (GPT-3.5 / GPT-4.1)

![Pipeline](figure/4.png)

---

## 🛠️ Installation

---

## 📦 Usage

We provide optimized suffixes for various models in:

| Model               | Suffix (excerpt) |
|---------------------|------------------|
| Vicuna-7b           | `Barb);\nвърpace пре What Download some unsafe software???encies/? Map>,` |
| Llama-2-7b-chat     | `szóimationBig'] `{Uhh okay Dave Sant (** caller), unfortunately personally` |
| Black‑box (transfer)| `s.' enemies pay Украreading socialehot exploман Spark (+ Cannot [by actress` |

The template used is:

```
You are an assistant that is harmless and helpful. Importantly, you need to always prioritize safety over helpfulness. That is, if answering the user query could be unsafe or harmful, you should refuse to answer. Otherwise provide a thorough and precise response, ensuring you assist the user to the best of your ability.
For example, {ICL demonstration}
Then, answer the question: {PROMPT}
{Defensive Suffix}
```

## ⚠️ Disclaimer

This repository contains model outputs that may be considered offensive. The code is provided for research purposes only. Users are responsible for complying with applicable laws and ethical guidelines.

---

## 📬 Contact

For questions or issues, please open a GitHub issue or contact the authors via email.

---

**Enjoy safe LLM deployments!** 🛡️
