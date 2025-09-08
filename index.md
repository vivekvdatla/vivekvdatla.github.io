---
layout: default
title: "About me"
---

About Me
---

I’m a senior AI leader and applied researcher with 16+ years building production-grade NLP systems at scale. My focus is information retrieval (IR) and retrieval-augmented generation (RAG) for high-stakes domains—clinical decision support, post-market surveillance, and large-scale customer support. At Capital One, I lead NLP/IR strategy and architecture for enterprise agent-assist, shipping hybrid retrieval (BM25 + dense + SPLADE), cross-encoder re-ranking (monoT5/monoBERT/CE), ranked fusion, and answerability gating (semantic-entropy/nucleus thresholds) that assist thousands of agents across multiple lines of business.

Previously at Philips Research, I directed cross-functional teams delivering healthcare AI from lab to product, transferring 10+ technologies into production. My research portfolio includes 15+ granted patents, 35+ filings, and 30+ peer-reviewed publications spanning robust QA, multimodal learning (VQA-Med), and knowledge-graph–driven inference.

I operate end-to-end: problem framing → data/ETL → modeling → evaluation → rollout. Tooling includes PyTorch, Hugging Face (Transformers/PEFT: LoRA/QLoRA), Sentence-Transformers, Faiss/Milvus/pgvector, OpenSearch/Elasticsearch, PyKEEN/Neo4j, and modern serving stacks (vLLM, TGI, Triton, TensorRT-LLM) with mixed precision, quantization (8/4-bit), FlashAttention, and KV-cache optimizations. On the platform side: MLflow/W&B, Airflow, Docker/Kubernetes, Ray/Dask, and production observability (Prometheus/Grafana/ELK) with quality SLOs, drift monitors, and A/B harnesses.

My current interests center on trustworthy LLMs—meaning-aware uncertainty (e.g., semantic entropy/MARS-style scoring), confidence calibration, hallucination mitigation, and provenance tracing that distinguishes pretraining vs. context-driven knowledge in RAG. The throughline in my work is turning cutting-edge research into reliable, interpretable, cost-efficient systems that create measurable business and user impact.


## Education

**Ph.D., Computer Science**, University of Memphis, TN — *2009–2014*
**M.S., Computer Science**, University of Memphis, TN — *2006–2008*
**B.Tech., Electronics & Communication Engg.**, JNTU, India — *2001–2005*

---

## Core Strengths

* **IR/RAG Architecture:** Hybrid retrieval (BM25 + dense + SPLADE), **cross‑encoder re‑ranking**, late‑fusion, user‑preference boosting, **entropy/nucleus thresholding**, dynamic index routing, **document chunking & windowing**, query rewriting & intent routing.
* **Trust & Safety for LLMs:** Abstention, refusal sensitivity vs. utility trade‑offs, **semantic entropy**, **MARS‑style meaning‑aware scoring**, DetectGPT‑style detectors, confidence calibration, provenance tracing for **pretrain vs. context** attribution.
* **Scalable ML Delivery:** Model/product roadmaps, experiment platforms, evaluation harnesses, **guardrails**, CI/CD for models, **observability (drift, latency, quality)**, on‑call runbooks.
* **Leadership:** Hiring/mentoring, stakeholder alignment, value stream ownership (\~\$2M), program management across research → product.

---

## Technical Stack

**Languages:** Python, SQL, Java, JavaScript, Bash, C, MATLAB
**DL/ML:** PyTorch, TensorFlow, JAX (intro), Hugging Face Transformers/PEFT, Sentence‑Transformers, scikit‑learn, XGBoost
**LLMs/Serving:** Llama‑2/3, Mistral, GPT‑4/4o/4.1, T5/Flan, **vLLM**, **Text Generation Inference (TGI)**, **TensorRT‑LLM**, **Triton Inference Server**
**Fine‑Tuning:** LoRA/QLoRA, Adapters/IA3, P‑tuning, Prefix/Prompt/Instruction tuning; **mixed‑precision**, gradient/ZeRO sharding
**Optimization:** Quantization (8‑/4‑bit), **FlashAttention**, speculative decoding, KV‑cache management
**RAG & IR:** Faiss, Milvus, pgvector, Elasticsearch/OpenSearch, **ColBERT**, SPLADE; monoT5/monoBERT/CE‑rerankers; query rewriting, hyde/self‑ask, **long‑context chunking**
**Graphs:** PyKEEN (RotatE/QuatE/PairRE/HousE), NetworkX, Neo4j
**MLOps/Platforms:** MLflow, Weights & Biases, Kubeflow, Airflow, **Docker**, **Kubernetes**, **Ray/Dask**, GitHub Actions/CI
**Data/ETL:** Pandas, NumPy, Apache Arrow, Spark, Kafka, Hadoop, RabbitMQ
**Monitoring:** Prometheus, Grafana, ELK; **human‑in‑the‑loop** annotation (Label Studio, Prodigy)
**Cloud/Infra:** AWS (EC2, EKS, S3, SageMaker), Azure, GCP; serverless patterns

---

## Selected Impact & Achievements

* **Capital One Agent‑Assist (IR/RAG):** Led the IR team to a **state‑of‑the‑art RAG** platform for call‑center agents; deployed **cross‑encoder re‑rankers**, **hybrid retrieval**, and **ranked‑fusion** with **entropy‑gated answerability**, improving top‑k precision and reducing handle‑time (HT) and AHT variance.
* **Trustworthy Responses:** Shipped abstention & uncertainty pipelines (semantic entropy + nucleus thresholds) that reduced unsafe generations while maintaining task utility; introduced **user‑preference re‑ranking** and **dynamic profile routing**.
* **Tech Transfer (Philips):** Drove **10+** research‑to‑product transfers (clinical de‑identification, knowledge‑graph‑assisted diagnosis, DSP assets, ICON semantic search).
* **IP & Publications:** **15 granted** patents, **35+ filed**, **64+ invention disclosures**; **30+ publications** (NAACL, COLING, AAAI, WWW, BHI, MLHC, TREC).
* **Awards:** **Circle of Excellence (2025)**—Capital One’s highest honor; **CIO Elite (2023)**; **TechX (2023)**.

---