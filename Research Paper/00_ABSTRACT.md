# ABSTRACT
## Paper: An Agentic Generative AI Framework for Industrial Predictive Maintenance: Physics-Aware Anomaly Detection with Multimodal Retrieval-Augmented Generation

**Authors:** Zynaptrix AI Research Team  
**Institution:** University of Kelaniya, Sri Lanka  
**Venue:** IEEE IES Generative AI Challenge 2026  

---

Industrial facilities face a persistent triad of operational challenges: subtle sensor anomalies evade rule-based SCADA detection systems, critical technical documentation remains inaccessible under time-pressure, and operators lack expert-level guidance during high-stakes failures. This paper presents the **Zynaptrix Industrial Copilot**, a production-ready neuro-symbolic agentic generative AI framework that unifies sub-symbolic anomaly detection with symbolic LLM reasoning and multimodal knowledge retrieval to deliver end-to-end intelligent industrial diagnostics — simultaneously addressing the detection, explainability, and remediation deficits that no existing single-paradigm approach resolves.

Four tightly integrated technical contributions define the system. First, a **Physics-Aware Hybrid Confidence Layer** fuses a dual-architecture anomaly detector — a Dense Autoencoder for point deviations and an LSTM Autoencoder for temporal drift patterns — with manufacturer-specification physics limits and multi-window temporal analysis, reducing false-positive maintenance dispatches from 12–15% to 3.35%. Second, a **six-node LangGraph multi-agent pipeline** decomposes the diagnostic cognitive workflow into specialized roles — Sensor Analyst, AI Validation Engineer, Diagnostic Classifier, Knowledge Retriever, Execution Strategist, and Safety Critic — with the Critic enforcing Lockout/Tagout and PPE compliance through a bounded iterative refinement loop. Third, a **Multimodal RAG engine**, developed through a four-stage empirical evolution (basic text → context-aware structural → vision-captioned → agentic figure splitting via Mobile SAM), employs GPT-4o Vision semantic captioning and pgvector cosine search to unify text, diagrams, and tables in a single 1536-dimensional semantic index — enabling cross-modal retrieval without dual-index architectures. Fourth, an **Institutional Intelligence** subsystem captures operator field experience through a five-gate quality pipeline, vectorizes resolved incidents into queryable organisational memory, and automatically surfaces learned knowledge in subsequent incidents — enabling continuous improvement without model retraining.

Evaluation on the TEA_PUR_0001 machine at Imperial Tea Exports (Pvt) Ltd demonstrates: 90.15% anomaly detection precision with 3.35% FPR; end-to-end fault-to-procedure delivery in 6.8 seconds (300× faster than manual baseline); consistent F1 > 0.71 across four heterogeneous machine types; and validated institutional learning where operator-contributed knowledge (e.g., field-proven clearance specifications absent from manufacturer manuals) autonomously enriches future diagnostic guidance.

**Index Terms** — Agentic AI, Retrieval-Augmented Generation, Predictive Maintenance, Multi-Agent Systems, Anomaly Detection, Industrial IoT, Human-in-the-Loop, Large Language Models

---
*Word count: ~250 words (IEEE compliant)*
