# 🏭 Gen AI Industrial Copilot: Research Baseline & Strategic Roadmap

This document establishes a comprehensive technical baseline for the **Zynaptrix Industrial Copilot** research project. It synthesizes the current architecture, multi-asset capabilities, and sets the stage for future enhancements in the domain of autonomous industrial fleet management.

## 🎯 Research Objective
To develop a production-ready, multi-agent AI framework capable of **autonomous telemetry monitoring, real-time anomaly detection, and expert-level multimodal diagnostic resolution** for complex factory fleets using Generative AI (LangGraph + RAG).

---

## 🏗️ System Architecture Snapshot

### 1. Backend Core (Python/FastAPI)
- **Anomaly Detection Layer**: 
  - Uses Dense Autoencoders (TensorFlow/Keras) for reconstruction MSE scoring.
  - **Scaling**: Isolated model registry for Pump, Lathe, and Turbine assets.
- **Orchestration Layer (LangGraph)**:
  - Multi-agent state machine that coordinates between Anomaly Alerts, Manual Inquiries, and Diagnostic Validation.
  - **Criticism Node**: Ensures diagnostic outputs are verified against technical documentation before final delivery.
- **Unified RAG Engine**:
  - **Multimodal**: Uses CLIP embeddings and GPT-4o Vision to index technical manuals (text + diagrams).
  - **Dynamic Routing**: Maps machine telemetry context to specific vector database identifiers (`manual_id`).

### 2. Frontend Layer (Next.js/Redux)
- **State Management**: Centralized **Redux Toolkit** architecture.
- **Side-Effect Control**: Async Thunks handle all API interactions (Simulator, Machine Registry, RAG Inquiries).
- **Visualization**: Recharts for high-frequency telemetry visualization and Carbon-fibre inspired industrial UI.

---

## ✅ Current Multi-Asset Capabilities
The system has graduated from a single-machine PoC to a **Multi-Asset Fleet**:
1.  **Concurrent Simulators**: Independent sub-processes for Pump, Lathe, and Turbine.
2.  **Context-Aware Diagnostics**: AI personas dynamically resolve based on the asset ID (e.g., Gas Turbine Expert vs. Pump Specialist).
3.  **Cross-Manual Isolation**: Retrieval is strictly bounded by asset IDs to prevent cross-asset diagnostic pollution.

---

## 🚀 Strategic Roadmap: Next Enhancements

### Pillar 1: Predictive Health Scoring (PHS)
- **Target**: Move from "Anomaly Detection" (Reactive) to "Remaining Useful Life" (Proactive).
- **Research Point**: Implement a health-decay algorithm based on historical anomaly frequency and reconstruction error trends.

### Pillar 2: Stateful Multi-Agent Memory
- **Target**: Enable the Copilot to "remember" previous repairs and part replacements for a specific machine ID.
- **Research Point**: Integrate a persistent "Maintenance Log" tool within the LangGraph state to inform future diagnostics with past repair history.

### Pillar 3: Edge-Cloud Hybrid Telemetry
- **Target**: Optimize the WebSocket pipeline for 100+ concurrent assets.
- **Research Point**: Research data-compression techniques for telemetry streams and "Silent Anomaly" filtering at the edge (simulator side) before transmission to the Copilot.

### Pillar 4: Interactive Multimodal Guidance
- **Target**: Use the Copilot to provide step-by-step visual maintenance procedures.
- **Research Point**: Implement a "Procedure Agent" that can generate sequential maintenance checklists by interleaving technical diagrams retrieved from the RAG engine.

---

## 🛠️ Developer Continuity: How to Proceed
1.  **Model Training**: Use `backend/models/train_model.py` to refine thresholds for new assets.
2.  **RAG Expansion**: Ingest new PDF manuals via the dashboard UI; the system will automatically handle vectorization.
3.  **Agent Logic**: Modify `backend/agents/copilot_graph.py` to add new "Expert" nodes or validation criteria.
4.  **UI Feedback**: Enhance `frontend/src/app/page.tsx` to display more granular "Sub-component" health (e.g., Bearing Temp vs. Rotor Vibration).

---
*Document Version: 1.0.0 (2026.03.26)*  
*Prepared for: Zynaptrix Advanced Engineering Division*
