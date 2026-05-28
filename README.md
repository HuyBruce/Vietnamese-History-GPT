# Vietnamese-History-GPT

Mini GPT-style Vietnamese language model built from scratch using PyTorch, featuring a custom tokenizer, transformer architecture, and Retrieval-Augmented Generation (RAG) pipeline for Vietnamese historical question answering.

---

# Features

* Custom BPE tokenizer training
* GPT-style transformer architecture
* Multi-Head Self Attention
* Autoregressive text generation
* Mixed Precision Training (AMP)
* Checkpoint resume training
* Cosine Learning Rate Scheduler
* Token caching with NumPy memmap
* Vietnamese corpus preprocessing pipeline
* Experimental RAG integration

---

# Architecture

```text
Vietnamese Corpus
        ↓
Tokenizer Training (BPE)
        ↓
Tokenization
        ↓
Transformer Model
        ↓
Language Modeling
        ↓
Text Generation
```

---

# Model Configuration

| Parameter          | Value  |
| ------------------ | ------ |
| Vocabulary Size    | 30,000 |
| Embedding Size     | 384    |
| Attention Heads    | 6      |
| Transformer Layers | 6      |
| Context Length     | 256    |
| Dropout            | 0.2    |

---

# Tech Stack

* Python
* PyTorch
* NumPy
* HuggingFace Tokenizers
* HuggingFace Datasets

---

# Project Structure

```text
Vietnamese-History-GPT/
│
├── Dataset/
│   ├── vi_tokenizer.json
│   └── vietnamese_tokens.meta.json
│
├── finetune/
├── models/
├── rag/
├── scripts/
│
├── dataset_loader.py
├── download_dataset.py
├── generate.py
├── inference_test.py
├── model.py
├── train.py
├── train_tokenizer.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# Training Pipeline

## 1. Download Dataset

```bash
python download_dataset.py
```

## 2. Train Tokenizer

```bash
python train_tokenizer.py
```

## 3. Train Model

```bash
python train.py
```

## 4. Generate Text

```bash
python generate.py
```

---

# Example Output

Prompt:

```text
Việt Nam là
```

Generated Output:

```text
Việt Nam là một quốc gia nằm ở khu vực Đông Nam Á với lịch sử lâu đời...
```

---

# Current Features

* Transformer-based language model
* Custom Vietnamese tokenizer
* Checkpoint resume support
* Token cache optimization
* Text generation pipeline

---

# Future Improvements

* RAG-based document retrieval
* PDF question answering
* Streamlit web interface
* Chat memory
* LangChain integration
* Multi-agent workflows

---

# Installation

```bash
git clone https://github.com/HuyBruce/Vietnamese-History-GPT.git
cd Vietnamese-History-GPT
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Notes

Large datasets and model weights are excluded from the repository using `.gitignore`.

This project is intended for educational and research purposes focused on Vietnamese NLP and LLM engineering.
