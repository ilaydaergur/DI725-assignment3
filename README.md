# DI725 Assignment 3 — QLoRA Fine-Tuning of PaliGemma

**Course:** DI725 — Transformers and Attention-Based Deep Networks  
**Institution:** Middle East Technical University  

## Overview

Fine-tuning **PaliGemma-3B** with **QLoRA** (Quantized Low-Rank Adaptation) on the
[RISC](https://huggingface.co/datasets/caglarmert/full_riscm) remote sensing image captioning dataset.
Only **0.39%** of model parameters are trained, yielding substantial improvements over the baseline.

## Results

| Metric | Baseline PaliGemma | QLoRA Fine-tuned | Δ |
|---|---|---|---|
| BLEU-4 | 0.0062 | 0.0553 | +791% |
| METEOR | 0.0897 | 0.2285 | +155% |
| ROUGE-L | 0.1476 | 0.2675 | +81% |

## Setup

| | |
|---|---|
| **Base model** | `google/paligemma-3b-pt-224` |
| **Dataset** | `caglarmert/full_riscm` (44,521 images · 5 captions each) |
| **Quantisation** | 4-bit NF4 (BitsAndBytes) |
| **LoRA rank** | r=8, all 7 linear layers of Gemma decoder |
| **Trainable params** | 11.3M / 2.93B (0.39%) |
| **Training** | 1 epoch · effective batch 16 · lr=2e-5 |
| **Hardware** | Google Colab A100 |

## Repository Structure

```
├── notebooks/
│   └── DI725_Assignment3_QLoRA_PaliGemma.ipynb   # main notebook
├── figures/
│   ├── metrics_comparison.png                     # BLEU-4 / METEOR / ROUGE-L bar chart
│   ├── results_visualization.png                  # 5 example image comparisons
│   └── risc_samples.png                           # dataset sample overview
├── report/
│   └── report.tex                                 # 1-page IEEE report (LaTeX source)
├── docs/
│   └── DI_725_Assignment_3.pdf                    # original assignment brief
└── README.md
```

## How to Run

1. Open `notebooks/DI725_Assignment3_QLoRA_PaliGemma.ipynb` in **Google Colab**
2. Set runtime to **A100 GPU** (Runtime → Change runtime type)
3. Replace `hf_YOUR_TOKEN_HERE` with your [HuggingFace token](https://huggingface.co/settings/tokens)
4. Run all cells — training takes ~1–4 h on A100 for 1 full epoch

## Compile Report

```bash
cd report
pdflatex report.tex && pdflatex report.tex
```

Or paste `report/report.tex` into [Overleaf](https://www.overleaf.com) (select IEEE Conference template).
