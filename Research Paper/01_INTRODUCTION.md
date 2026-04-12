# I. INTRODUCTION
## Status: [✅] Done

---

Modern industrial facilities are monitored through Supervisory Control and Data Acquisition (SCADA) and Human-Machine Interface (HMI) systems that collect high-frequency telemetry from hundreds of sensors across rotating machinery, fluid systems, and electrical equipment. Despite this dense instrumentation, industrial maintenance operations remain fundamentally reactive: an asset fails, an alarm fires, and a technician is dispatched. The cost of this paradigm is severe. Unplanned downtime in manufacturing costs an estimated USD 260,000 per hour on average [1], and bearing failures alone account for over 40% of all rotating machinery outages [2].

The core limitation of conventional monitoring infrastructure is architectural. Rule-based threshold systems — the backbone of industrial SCADA — flag anomalies only when a single sensor crosses a pre-configured static limit. This approach is inherently brittle: it fails to detect compound failure signatures that emerge across multiple sensors simultaneously, cannot distinguish between real mechanical faults and transient sensor glitches, and provides no diagnostic context when an alarm fires. Operators are left to manually search through hundreds of pages of technical manuals under time pressure, a process that introduces 30–60 minutes of diagnostic latency and is critically dependent on individual operator expertise [3].

The emergence of machine learning-based anomaly detection has demonstrated significant promise in addressing the detection gap. Unsupervised autoencoder architectures, trained exclusively on healthy operational data, learn to reconstruct normal sensor distributions, with reconstruction error serving as a principled anomaly score [4], [5]. However, these sub-symbolic models present a fundamental limitation: they can detect *that* something is wrong, but they cannot explain *why*, nor can they translate a high reconstruction error into actionable repair guidance. This explainability gap is especially consequential in safety-critical industrial settings, where unjustified maintenance dispatches carry both operational and financial costs.

Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems have emerged as a compelling pathway toward bridging this gap [6]. By grounding LLM reasoning in domain-specific document corpora, RAG systems can produce contextually accurate, citation-backed responses. However, existing RAG deployments in industrial contexts face two critical deficiencies. First, they treat technical manuals as purely textual documents, discarding the rich semantic content embedded in engineering diagrams, assembly drawings, and wiring schematics. Second, they operate as stateless, single-turn systems, offering no mechanism to accumulate organizational knowledge from previously resolved incidents.

Recent advances in agentic AI frameworks, particularly graph-based orchestration systems such as LangGraph [7], have demonstrated the viability of decomposing complex reasoning workflows into specialized, cooperative agent roles. Yet the application of multi-agent generative AI to the full diagnostic loop — from raw telemetry to step-by-step guided repair — with integrated physics-based validation and multimodal knowledge retrieval, remains an open research challenge.

This paper presents the **Zynaptrix Industrial Copilot**, a neuro-symbolic agentic generative AI framework designed to close these gaps. The system unifies sub-symbolic autoencoder-based anomaly detection with symbolic LLM reasoning chains and multimodal knowledge retrieval into a single end-to-end diagnostic pipeline — addressing, simultaneously, the detection, explainability, and remediation deficits that no existing single-paradigm approach resolves. The principal contributions of this work are:

1. **Physics-Aware Hybrid Confidence Scoring**: A novel three-factor confidence formulation that fuses a dual-architecture anomaly detector — a Dense Autoencoder for point deviations and an LSTM Autoencoder for temporal drift — with manufacturer-specification physics constraint validation and multi-window temporal pattern analysis, reducing false-positive maintenance dispatches from 12–15% to below 1%.

2. **Six-Node Agentic LangGraph Pipeline**: A directed acyclic graph (DAG) orchestration of six specialized AI agents — Sensor Analyst, Validation Engineer, Diagnostic Classifier, Knowledge Retriever, Execution Strategist, and Safety Critic — enabling transparent, auditable, and modular diagnostic reasoning.

3. **Multimodal RAG Engine with Caption-Based Vision-Language Alignment**: A four-stage ingestion and retrieval pipeline combining YOLOv8-DocLayNet layout detection, MobileSAM agentic figure decomposition, GPT-4o Vision semantic captioning (producing text-embedded captions placed in the same 1536-dimensional embedding space as prose — enabling direct semantic image retrieval without separate visual indices such as CLIP or ImageBind), and pgvector cosine-similarity search. This approach produces domain-specific, actionable explanations that address the operator-actionability deficit of abstract XAI feature-attribution methods identified in the literature.

4. **Institutional Intelligence**: A continuous learning subsystem that uses GPT-4o to summarize resolved diagnostic incidents into structured narratives, which are then vectorized and stored in a pgvector-backed organisational memory. Future retrievals rank historically successful repairs alongside static manual content, transforming individual technician experience into institution-wide knowledge — without retraining any model.

Evaluation of the fully deployed Zynaptrix Industrial Copilot system — comprising per-machine trained TensorFlow/Keras dual-autoencoder models, a live six-node LangGraph agent pipeline, a multimodal RAG engine backed by PostgreSQL/pgvector spanning 15 ingested technical manuals, InfluxDB real-time telemetry streaming, and a Next.js operator dashboard — demonstrates: a **98% reduction in diagnostic latency** (Mean Time to Diagnosis from 35 minutes to under 2 seconds); end-to-end fault-to-procedure latency under 8 seconds; Safety Critic final approval rate exceeding 94% within a bounded two-attempt iterative refinement loop; false-positive dispatch rate reduced from 12–15% to below 1%; and a 34% improvement in retrieval precision through vision-language captioning over text-only and direct-image-embedding baselines.

The remainder of this paper is organized as follows. Section II reviews related work across anomaly detection, RAG systems, multi-agent architectures, and industrial AI. Section III details the system methodology and architecture. Section IV presents the experimental evaluation of the deployed system. Section V provides a critical discussion of findings, limitations, and future directions, followed by conclusions.

---
*Word count: ~680 words (IEEE compliant for Introduction)*

---

## References Used in This Section

| Tag | Full IEEE Citation |
|-----|--------------------|
| [1] | Y. Peng, M. Dong, and M. J. Zuo, "Current status of machine prognostics in condition-based maintenance: A review," *Int. J. Adv. Manuf. Technol.*, vol. 50, pp. 297–313, 2010. |
| [2] | N. Tandon and A. Choudhury, "A review of vibration and acoustic measurement methods for the detection of defects in rolling element bearings," *Tribol. Int.*, vol. 32, no. 8, pp. 469–480, 1999. |
| [3] | Z. M. Çınar, A. A. Nuhu, Q. Zeeshan, O. Korhan, M. Asmael, and B. Sahraoui, "Machine Learning in Predictive Maintenance towards Sustainable Smart Manufacturing in Industry 4.0," *Sustainability*, vol. 12, no. 19, p. 8211, 2020. |
| [4] | M. Sakurada and T. Yairi, "Anomaly Detection Using Autoencoders with Nonlinear Dimensionality Reduction," in *Proc. MLSDA Workshop*, ACM, Dec. 2014. |
| [5] | P. Malhotra, L. Vig, G. Shroff, and P. Agarwal, "LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection," in *ICML Time Series Workshop*, 2016. |
| [6] | P. Lewis, E. Perez, A. Piktus *et al.*, "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks," in *Adv. Neural Inf. Process. Syst. (NeurIPS)*, vol. 33, pp. 9459–9474, 2020. |
| [7] | LangChain, Inc., "LangGraph: Build Stateful, Multi-Actor Applications with LLMs," GitHub repository, 2024. [Online]. Available: https://github.com/langchain-ai/langgraph. [Accessed: Apr. 2026]. |
