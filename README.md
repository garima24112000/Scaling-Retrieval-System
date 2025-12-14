# Scaling Retrieval Systems  
**A Big Data Approach to Efficient Retrieval-Augmented Generation**

This repository contains the complete implementation and benchmarking framework for **scaling retrieval systems** across multiple backends (FAISS, Chroma, Spark) under constrained cloud hardware. The project was developed as part of **AMS 560 – Big Data Systems, Algorithms & Networks** at **Stony Brook University**.

The work systematically studies how **datastore size, backend architecture, and domain diversity** affect retrieval latency, throughput, and end-to-end RAG performance, with a strong emphasis on **reproducibility and systems engineering**.

---

## 📌 Project Motivation

Retrieval-Augmented Generation (RAG) allows language models to fetch knowledge from external datastores instead of relying purely on parametric memory. Recent research shows that **scaling retrieval is an orthogonal alternative to scaling model size**.

This project investigates:
- How retrieval latency and throughput scale with datastore size
- Trade-offs between single-node, metadata-aware, and distributed retrieval backends
- Whether **small models with RAG** can approach the performance of **large non-RAG models**
- Practical constraints of running large retrieval systems on **commodity cloud hardware (Google Colab)**

---

## 🏗️ System Architecture

The system is designed as a **modular, backend-agnostic retrieval infrastructure**.

**Pipeline Overview**
1. Text ingestion and chunking  
2. Dense embedding generation  
3. Index construction (FAISS / Chroma / Spark)  
4. Backend-specific retrieval execution  
5. Optional RAG-based LLM inference  
6. Automated benchmarking and metric logging  

A unified benchmarking harness (`bench.py`) orchestrates all experiments using **configuration-driven execution**, enabling controlled and reproducible comparisons.

---

## 📊 Experimental Setup

- **Datastore sizes:** 10k, 30k, 50k embedded chunks  
- **Domains:** Single-domain and multi-domain  
- **Hardware:** Google Colab (CPU-only)  
- **Embeddings:** Dense sentence-level embeddings (float32)  
- **Models:**  
  - TinyLlama-1.1B (with RAG)  
  - Llama-3.1-8B (without RAG)

**Metrics Collected**
- Retrieval latency (p50, p95)
- Throughput (queries/sec)
- Index build time
- End-to-end inference latency
- Stability across repeated runs

---

## 🧪 Key Findings

- FAISS Single provides the best overall latency–throughput balance on constrained hardware.
- FAISS Sharded introduces overhead at small scales but improves tail latency at larger datastores.
- Chroma incurs additional cost due to metadata filtering and disk I/O.
- Spark is unsuitable for low-latency retrieval due to distributed coordination overhead.
- Small LLMs with RAG can achieve end-to-end latency comparable to much larger non-RAG models.

---

## 📁 Repository Structure
Scaling-Retrieval-System/
```
├── bench.py # Unified benchmarking entry point
├── configs/ # Experiment and backend configs
├── data/
│ ├── embeddings/ # Precomputed embedding slices
│ ├── metadata/ # Parquet metadata tables
├── indices/ # FAISS / Chroma index artifacts
├── backends/
│ ├── faiss_backend.py
│ ├── chroma_backend.py
│ ├── spark_backend.py
├── scripts/ # Preprocessing and slicing utilities
├── results/
│ ├── json/ # Raw benchmark logs
│ ├── csv/ # Aggregated metrics
├── plots/ # Result visualizations
└── README.md
```
## ▶️ How to Run

### Run a Benchmark
```bash
python bench.py \
  --backend faiss_single \
  --datastore_size 30000 \
  --domain multidomain
```

## Colab files to run and view entire code:
Preprocessing - https://drive.google.com/file/d/1IxK2Ql38tuuGvmDA4EhNc6L1_cXN8Gpi
Retierval - https://drive.google.com/file/d/1gcCB3XfqrcLgF7wx6jgvMmKzWZYT33wg
Scaling - https://drive.google.com/file/d/1ChXLOH4z25YuRObxoK9VcUiig6D12bcv
Visualization - https://drive.google.com/file/d/1aj9gXsHppXlSIdpJ55W6mtnAZjc3WODR

## Team & Contributions

Shruti Jagdale – Data preprocessing, embedding generation, FAISS single-node indexing, visualizations
Kashish Deepak Lalwani – Backend-agnostic retrieval runner and execution layer
Garima Prachi – Unified backend infrastructure, FAISS (single & sharded), Spark-based distributed retrieval, global Top-K merging
Quynh Ho – Experiment orchestration, RAG vs non-RAG pipelines, reproducibility
Elita Hazel Gorimanikonda – Benchmark execution, metric aggregation, performance analysis

## 🚀 Future Work

GPU-accelerated FAISS
Retrieval quality metrics (MRR, nDCG)
Vector compression and learned indexes
Multi-node and cluster-scale evaluation
