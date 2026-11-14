# Project Sentinel: Factuality & Citation Verification in RAG 

**UT Austin CS388 Natural Language Processing – Final Project (Fall 2026)**  
**Track:** Factuality and Verification  

## Overview
This repository contains an automated fact-checking and citation verification pipeline designed to 
evaluate hallucinations and evidence faithfulness in Retrieval-Augmented Generation (RAG) systems. 

Instead of relying solely on unverified generation, this framework breaks complex LLM responses down 
into **atomic claims**, retrieves grounding evidence using dense embeddings (Microsoft E5), and 
applies an **LLM-as-a-judge** (Qwen 2.5) to deterministically score the relevance and faithfulness 
(Entailed, Contradicted, Neutral) of each claim against the retrieved corpus.

The baseline evaluation utilizes the **WICE** (Real-World Entailment for Claims in Wikipedia) dataset 
to simulate natural, sub-sentence entailment challenges.

## Repository Structure
```text
.
├── data/
│   ├── raw/             # Raw WICE dataset exports
│   ├── processed/       # Parsed and chunked atomic claims (corpus_chunks.csv)
│   └── ground_truth/    # LLM judge outputs and human-labeled validation sets
├── notebooks/           # EDA, metrics calculation, and evaluation visualizations
├── report/              # NeurIPS 2026 LaTeX template for the final 6-8 page report
├── src/
│   ├── download_wice.py     # Fetches and formats the WICE dataset from HuggingFace
│   ├── chunk_documents.py   # Uses NLTK to parse paragraphs into atomic verifiable claims
│   └── evaluate_rag.py      # E5 embedding retrieval & local Qwen judge inference pipeline
└── README.md
