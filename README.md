# RAG Assignment: Baseline Retrieval vs. Cross-Encoder Re-Ranking

**Vikash Kumar**

## Overview

In this project I build a small RAG system over the *Magic: The Gathering Comprehensive Rules* (19 June 2026, 311 pages) and compare two retrieval pipelines:

| | Retrieval | Chunks sent to the LLM |
|---|---|---|
| **Pipeline 1 (baseline)** | Bi-Encoder `BAAI/bge-small-en-v1.5` + FAISS cosine search, top 10 | top 3 by cosine score |
| **Pipeline 2 (re-ranked)** | the same top 10 | top 3 after re-scoring with Cross-Encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` |

Both pipelines send their top 3 chunks to Gemini (`gemini-3.5-flash-lite`, temperature 0) with the same prompt.

The notebook covers:
1. Rule-aware chunking of the PDF (971 chunks).
2. FAISS index.
3. Both pipelines.
4. 49 test questions (negation, exception, keyword trap, buried rule, control), with retrieval metrics (Hit@3, MRR@10, Recall@10).
5. Answer 1 vs. Answer 2 on 23 selected questions, with rank-shift analysis.
6. Latency profiling.
7. Scaling analysis, including a full-corpus Cross-Encoder check.

**Key results:**
- Hit@3 went from 0.673 to 0.837 and MRR@10 from 0.539 to 0.710.
- Correct answers went from 6 to 14 out of 23.
- Re-ranking adds about 0.7 s per query on CPU.

## Repository layout

```
├── RAG_Baseline_vs_Reranking.ipynb   # main notebook (all code + outputs)
├── data/MagicCompRules_20260619.pdf  # source document
├── results/
│   ├── answers.json                  # Answer 1 / Answer 2 for the 23 final questions
│   └── llm_cache.json                # cached Gemini responses (makes re-runs free and reproducible)
├── report/
│   ├── implementation_analysis.tex   # report source
│   └── implementation_analysis.pdf   # final report
├── PRD/RAG_LMA_Assignment_v1.pdf     # assignment brief
└── requirements.txt
```

## Setup

Requirements:
- Python 3.10 or newer.
- About 2 GB of disk for PyTorch and the models.
- Internet on the first run, to download the models from Hugging Face.
- No GPU is needed; everything runs on CPU.

```bash
git clone https://github.com/viserion999/LMA_Assignment_1.git
cd LMA_Assignment_1

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Create a `.env` file in the project root with a Gemini API key (free key from https://aistudio.google.com/apikey):

```bash
echo "GEMINI_API_KEY=your-key-here" > .env
# optional: use a different Gemini model
# echo "GEMINI_MODEL=gemini-3.5-flash-lite" >> .env
```

`.env` is in `.gitignore`, so the key is never committed.

## Run

Run the whole notebook from the command line (outputs are written back into the notebook):

```bash
jupyter nbconvert --to notebook --execute --inplace RAG_Baseline_vs_Reranking.ipynb
```

Or open it in VS Code / JupyterLab, select the `.venv` kernel and use **Run All**.

Notes:
- With `results/llm_cache.json` present, no Gemini calls are made and the answers are reproduced exactly. Delete the file to regenerate the answers; the free tier is rate-limited, and the notebook retries with back-off.
- The slowest parts on CPU are the latency and scaling sections; the scaling section alone scores all 971 chunks with the Cross-Encoder several times.
- Timings depend on the machine.
