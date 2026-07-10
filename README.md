# Pengembangan Generative AI berbasis LLM — Digdaya Hackathon BI

Submission hackathon Digdaya Batch 7 — modul **Pengembangan Generative AI berbasis LLM**.

## Skenario

Asisten AI internal untuk tim legal perusahaan: chatbot tanya-jawab hukum Indonesia berbasis **Small Language Model (SLM)** yang di-fine-tune + **Retrieval-Augmented Generation (RAG)** dengan dokumen UU/PP resmi.

## Struktur Project

```
PGABL_Rayhan/
├── Fine-tuning_submission_PGABL_Rayhan.ipynb   # Fine-tuning SLM dg QLoRA + Unsloth
├── GRPO_submission_PGABL_Rayhan.ipynb           # GRPO training dg 4 reward functions
├── RAG_submission_PGABL_Rayhan.ipynb            # Full RAG pipeline (Advanced)
├── link_huggingface.txt                         # URL model HuggingFace
├── requirements.txt                             # Dependencies
└── Document RAG/                                # 4 dokumen UU/PP (PDF)
    └── Document RAG/
        ├── UU Nomor 6 Tahun 2023.pdf
        ├── PP Nomor 35 Tahun 2021.pdf
        ├── PP Nomor 5 Tahun 2021.pdf
        └── PP Nomor 51 Tahun 2023.pdf
```

## Tech Stack

| Komponen | Pilihan |
|----------|---------|
| Base Model | `Qwen/Qwen2.5-7B-Instruct` |
| Fine-tuning | Unsloth + QLoRA 4-bit + SFTTrainer |
| GRPO | TRL GRPOTrainer + 4 Reward Functions |
| Embedding | `BAAI/bge-m3` (multilingual) |
| Vector DB | ChromaDB |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| Interface | Gradio ChatInterface + IPython Loop |
| PDF Processing | PyPDF + LangChain |

## Fitur Utama

### Fine-tuning (Advanced — 4 pts)
- ChatML template mapping (format native Qwen2.5)
- QLoRA 4-bit + double quantization + NF4
- LoRA pada **Multi-Head Attention** (q_proj, k_proj, v_proj, o_proj)
- SFTTrainer >= 800 steps, train/val split, logging, 2 hyperparameter experiments
- `push_to_hub_merged(saved_method="merged_16bit")`

### GRPO (Advanced — 4 pts)
- 4 Reward Functions: format, reasoning length, correctness (ROUGE-L), language
- Reward shaping bertahap untuk `<think>...</think>` format
- OOM mitigation: `num_generations=2`, `max_completion_length=256`

### RAG (Advanced — 4 pts)
- **HyDE**: 2 hypothetical answers untuk query transformation
- **Ensemble Retriever**: BM25 (0.3) + Semantic ChromaDB (0.7), Reciprocal Rank Fusion
- **Parent-Child Retriever**: Child 512 token → Parent 1024 token
- **Reranker**: BGE-Reranker-v2-m3, Top-3 chunks
- **DuckDuckGo Fallback**: Jika Top-1 relevance score < 0.5
- **Metadata Enrichment**: Ekstraksi nomor UU, tahun, bab, pasal
- **Citation**: Sumber lengkap per dokumen

## Menjalankan Notebook

1. **Download PDF** dari Google Drive (4 dokumen UU/PP) → taruh di `Document RAG/Document RAG/`
2. **Login HuggingFace**: `huggingface-cli login` atau `notebook_login()`
3. Jalankan di **Google Colab GPU T4** (free tier)
4. Urutan: Fine-tuning → GRPO → RAG

## Requirements

Lihat `requirements.txt`. Rekomendasi: Google Colab (sudah pre-installed sebagian besar dependency).

## Model HuggingFace

https://huggingface.co/RayhanLup1n/qwen2.5-7b-indonesian-legal-finetuned

## Author

**Rayhan Ananda Resky** — Digdaya Hackathon BI Batch 7
