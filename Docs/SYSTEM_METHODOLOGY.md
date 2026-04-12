# 🏭 Generative AI Framework for Industrial Operational Intelligence
## Complete System Methodology Documentation

**Document Version:** 2.0.0  
**Last Updated:** 2026-04-08  
**Project:** Zynaptrix Industrial AI Copilot

---

## 📋 Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [System Overview](#2-system-overview)
3. [Core Methodology Components](#3-core-methodology-components)
4. [Digital Twin & Telemetry Pipeline](#4-digital-twin--telemetry-pipeline)
5. [Anomaly Detection Engine](#5-anomaly-detection-engine)
6. [Agentic Orchestration (LangGraph)](#6-agentic-orchestration-langgraph)
7. [Multimodal RAG Engine](#7-multimodal-rag-engine)
8. [Human-in-the-Loop (HITL) Workflow](#8-human-in-the-loop-hitl-workflow)
9. [Continuous Learning Loop](#9-continuous-learning-loop)
10. [Technology Stack](#10-technology-stack)
11. [System Architecture Diagrams](#11-system-architecture-diagrams)
12. [Data Flow Specifications](#12-data-flow-specifications)
13. [Industrial Sensor Data Acquisition](#13-industrial-sensor-data-acquisition)
14. [Validation & Quality Assurance](#14-validation--quality-assurance)
15. [Security & Authentication](#15-security--authentication)
16. [Future Roadmap](#16-future-roadmap)

---

## 1. Executive Summary

The **Zynaptrix Industrial AI Copilot** is a production-ready, agentic AI framework designed for multi-asset industrial environments. The system combines **sub-symbolic AI** (Deep Learning Autoencoders) with **symbolic reasoning** (Multi-Agent LangGraph orchestration) and **Vision-Augmented Retrieval-Augmented Generation (RAG)** to deliver comprehensive operational intelligence.

### 🎯 Core Objectives

| Objective | Description |
|-----------|-------------|
| **Continuous Monitoring** | High-frequency (10Hz) sensor telemetry ingestion from industrial assets |
| **Intelligent Detection** | Unsupervised anomaly detection using trained Autoencoder models |
| **Autonomous Reasoning** | Multi-agent diagnostic pipeline for root cause analysis |
| **Knowledge Synthesis** | Multimodal RAG retrieval from technical manuals and historical fixes |
| **Guided Resolution** | Step-by-step repair procedures with safety validation |
| **Organizational Learning** | Vectorized archival of resolved incidents for continuous improvement |

### 🔑 Key Innovation Points

1. **Neuro-Symbolic Fusion**: Combines neural network pattern recognition with symbolic agent reasoning
2. **Multimodal Knowledge**: Unified embedding space for text, tables, and technical diagrams
3. **HITL Safety Loop**: Critic agent validation with mandatory human sign-off
4. **Self-Improving System**: Every resolved incident enriches the knowledge base

---

## 2. System Overview

### 2.1 System State Machine

The framework models machine health through a finite state machine:

```
┌─────────────────────────┐
│    NORMAL OPERATION     │
└───────────┬─────────────┘
            │ MSE > threshold (3 consecutive ticks)
            ▼
┌─────────────────────────┐
│    ANOMALY DETECTED     │
└───────────┬─────────────┘
            │ AnomalyService escalation
            ▼
┌─────────────────────────┐
│    ESCALATED FAULT      │
└───────────┬─────────────┘
            │ OrchestratorAgent.handle_anomaly()
            ▼
┌─────────────────────────┐
│  AGENTIC INVESTIGATION  │◄──┐
│    (LangGraph DAG)      │   │ Critic retry loop
└───────────┬─────────────┘───┘
            │ Returns execution_plan
            ▼
┌─────────────────────────┐
│  PROCEDURE STREAMING    │
│       TO UI             │
└───────────┬─────────────┘
            │ Operator completes steps + sign-off
            ▼
┌─────────────────────────┐
│   INCIDENT RESOLVED     │
└───────────┬─────────────┘
            │ GPT-4o summarizes → vectorizes → archives
            ▼
┌─────────────────────────┐
│ ARCHIVED IN INTERACTION │
│        MEMORY           │
└───────────┬─────────────┘
            │ Future RAG queries retrieve this fix
            ▼
┌─────────────────────────┐
│    NORMAL OPERATION     │ ← System is now smarter
└─────────────────────────┘
```

### 2.2 Architectural Layers

```
┌────────────────────────────────────────────────────────────────┐
│                     PRESENTATION LAYER                         │
│  Next.js 14 Dashboard │ Redux Toolkit │ WebSocket Client       │
└────────────────────────────────────────────────────────────────┘
                               ▲
                               │ REST API + WebSocket
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                          │
│  FastAPI Server │ Pydantic Validation │ CORS Middleware        │
└────────────────────────────────────────────────────────────────┘
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  DETECTION   │    │  ORCHESTRATION   │    │   KNOWLEDGE      │
│    LAYER     │    │     LAYER        │    │     LAYER        │
│              │    │                  │    │                  │
│ TensorFlow   │    │ LangGraph DAG    │    │ Unified RAG      │
│ Autoencoders │    │ 5-Node Pipeline  │    │ pgvector Search  │
└──────────────┘    └──────────────────┘    └──────────────────┘
       │                       │                       │
       └───────────────────────┼───────────────────────┘
                               ▼
┌────────────────────────────────────────────────────────────────┐
│                     PERSISTENCE LAYER                          │
│  PostgreSQL + pgvector │ InfluxDB Time-Series │ File Storage   │
└────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Methodology Components

### 3.1 Methodology Framework

The system implements a **5-pillar methodology** for industrial operational intelligence:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    OPERATIONAL INTELLIGENCE METHODOLOGY             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│  │  SENSE   │ → │ DETECT   │ → │ REASON   │ → │ ADVISE   │ → │  LEARN   │
│  │          │   │          │   │          │   │          │   │          │
│  │ Digital  │   │ Anomaly  │   │ Multi-   │   │ RAG +    │   │ Feedback │
│  │  Twin    │   │ Engine   │   │ Agent    │   │ HITL     │   │  Loop    │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Pillar 1: SENSE (Digital Twin)
- High-frequency sensor telemetry capture (10Hz)
- Multi-asset fleet simulation with machine-specific distributions
- Real-time streaming via WebSocket to operator dashboards

#### Pillar 2: DETECT (Anomaly Detection)
- Unsupervised learning via Dense/LSTM Autoencoders
- Reconstruction error (MSE) scoring against calibrated thresholds
- Consecutive-count escalation to filter transient noise

#### Pillar 3: REASON (Agentic Orchestration)
- LangGraph state machine with 5 specialized nodes
- Root cause analysis through sensor correlation
- Critic-validated safety verification

#### Pillar 4: ADVISE (Knowledge Synthesis)
- Multimodal RAG across technical manuals
- Historical fix retrieval from organizational memory
- Structured JSON procedures with inline diagrams

#### Pillar 5: LEARN (Continuous Improvement)
- GPT-4o summarization of resolved incidents
- Embedding and archival of operator fixes
- Automatic enrichment of future RAG queries

---

## 4. Digital Twin & Telemetry Pipeline

### 4.1 Sensor Simulation Model

The system generates synthetic telemetry using Gaussian distributions calibrated to real industrial equipment:

#### Machine-Specific Base Configurations

| Machine Type | Temperature (°C) | Motor Current (A) | Vibration (mm/s) | Speed (RPM) | Pressure (bar) |
|--------------|------------------|-------------------|------------------|-------------|----------------|
| **PUMP-001** | μ=180, σ=2 | μ=4.5, σ=0.5 | μ=0.8, σ=0.1 | μ=160, σ=5 | μ=4.5, σ=0.2 |
| **LATHE-002** | μ=45, σ=2 | μ=12.5, σ=1.5 | μ=0.15, σ=0.05 | μ=3200, σ=20 | μ=8.5, σ=0.5 |
| **TURBINE-003** | μ=850, σ=15 | μ=450, σ=5 | μ=1.2, σ=0.2 | μ=15000, σ=50 | μ=32, σ=1 |

### 4.2 State Generation Functions

The `anomaly_injector` module provides deterministic state generators:

```python
STATE_GENERATORS = {
    "normal":        normal_reading,        # Stable Gaussian noise
    "machine_fault": machine_fault_reading, # Bearing wear simulation
    "sensor_freeze": sensor_freeze_reading, # Stuck sensor values
    "sensor_drift":  sensor_drift_reading,  # Gradual temperature rise
    "idle":          idle_reading,          # Powered-off state
}
```

#### Fault Signature: `machine_fault_reading`
| Sensor | Deviation | Physical Explanation |
|--------|-----------|---------------------|
| `motor_current` | **1.6× mean**, 2× std | High current draw from friction |
| `vibration` | **3.5× mean**, 5× std | Severe vibration from imbalance |
| `speed` | **0.8× mean**, 2× std | Rotational slowing from drag |
| `pressure` | **0.9× mean** | Loss of hydraulic efficiency |
| `temperature` | Slightly decreased | Counter-intuitive cooling pre-heat buildup |

### 4.3 Telemetry Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SENSOR SIMULATOR PROCESS                     │
│                                                                 │
│  [1 Hz Tick Loop] ──────────────────────────────────────────►  │
│         │                                                       │
│         ├─► pick_state() → state ∈ {normal, fault, freeze, ...} │
│         │                                                       │
│         ├─► STATE_GENERATORS[state](machine_id) → reading      │
│         │                                                       │
│         ├─► InfluxDB Write (Line Protocol)                     │
│         │     sensor_readings,machine_id=PUMP-001,state=normal │
│         │     temperature=180.1,motor_current=4.52,...         │
│         │                                                       │
│         └─► HTTP POST → /api/telemetry/push                    │
│                   │                                             │
└──────────────────│──────────────────────────────────────────────┘
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BROADCAST                            │
│                                                                 │
│  TelemetryClientManager.broadcast({type: "telemetry", data})   │
│                   │                                             │
│                   ▼                                             │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  WebSocket Connection Pool                              │   │
│  │  [Client 1] [Client 2] [Client 3] ... [Client N]       │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND REDUX                               │
│                                                                 │
│  ws.onmessage → dispatch(addTelemetry({                        │
│      machineId, time, temperature, current, vibration          │
│  }))                                                            │
│         │                                                       │
│         ▼                                                       │
│  state.telemetry ← rolling buffer (max 20 points)              │
│         │                                                       │
│         ▼                                                       │
│  Recharts <LineChart> re-render                                │
└─────────────────────────────────────────────────────────────────┘
```

---

## 5. Anomaly Detection Engine

### 5.1 Three-Tier Detection Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│ TIER 1: AnomalyDetector (models/detect_anomaly.py)             │
│ ──────────────────────────────────────────────────────────────  │
│ • Per-machine model/scaler registry                            │
│ • Cached model loading (lazy initialization)                   │
│ • MSE reconstruction error calculation                         │
│ • Health score computation (0-100 scale)                       │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 2: AnomalyService (services/anomaly_service.py)           │
│ ──────────────────────────────────────────────────────────────  │
│ • Consecutive anomaly tracking                                 │
│ • Escalation threshold management (default: 3)                 │
│ • Alert formatting and logging                                 │
│ • Callback invocation on confirmed faults                      │
└─────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│ TIER 3: MonitoringService (services/monitoring_service.py)     │
│ ──────────────────────────────────────────────────────────────  │
│ • Integration seam between detection and orchestration         │
│ • OrchestratorAgent invocation                                 │
│ • Context construction for LangGraph pipeline                  │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Autoencoder Architecture

#### Dense Autoencoder (Primary Model)

```
INPUT LAYER: 5 features
    │   [temperature, motor_current, vibration, speed, pressure]
    ▼
┌──────────────────────────────┐
│  Dense(32, activation='relu')│  ← Compression begins
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Dense(16, activation='relu')│  ← Bottleneck (learned "normal" representation)
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│  Dense(32, activation='relu')│  ← Expansion begins
└──────────────┬───────────────┘
               ▼
┌──────────────────────────────┐
│ Dense(5, activation='linear')│  ← Reconstructed output
└──────────────────────────────┘

LOSS = Mean Squared Error (MSE)
TRAINING DATA = Only "state == normal" rows
```

**Key Design Decision:** The model is trained *exclusively* on normal operating data. This ensures fault patterns produce high reconstruction errors since the model has never learned to reconstruct them.

#### LSTM Autoencoder (Sequential Alternative)

```
INPUT: (seq_len=10, features=5)
    │
    ▼
┌──────────────────────────────────────┐
│  LSTM(64, return_sequences=False)   │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  RepeatVector(seq_len)              │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  LSTM(64, return_sequences=True)    │
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│  TimeDistributed(Dense(5))          │
└──────────────────────────────────────┘
```

**Use Case:** Detects temporal anomalies where individual readings appear normal but the *trend* is anomalous (e.g., sensor drift).

### 5.3 Detection Inference Pipeline

```python
def detect(reading: dict) -> dict:
    # Step 1: Extract machine context
    machine_id = reading.get("machine_id", "PUMP-001")
    
    # Step 2: Load cached model + scaler
    scaler = self._scaler_cache.get(machine_id)
    model = self._model_cache.get(machine_id)
    
    # Step 3: Prepare input vector
    sensor_values = [reading[col] for col in SENSOR_COLUMNS]
    X_raw = np.array(sensor_values).reshape(1, -1)  # shape: (1, 5)
    
    # Step 4: Normalize using machine-specific scaler
    X_norm = scaler.transform(X_raw)  # Z-score transformation
    
    # Step 5: Reconstruct through autoencoder
    X_reconstructed = model.predict(X_norm, verbose=0)
    
    # Step 6: Compute reconstruction error
    mse = np.mean((X_norm - X_reconstructed) ** 2)
    
    # Step 7: Derive health score
    ANOMALY_THRESHOLD = 0.7187  # Calibrated at 99th percentile
    health_score = max(0, 100 - (mse / ANOMALY_THRESHOLD) * 100)
    
    return {
        "is_anomaly":   mse > ANOMALY_THRESHOLD,
        "score":        float(mse),
        "threshold":    ANOMALY_THRESHOLD,
        "health_score": round(health_score, 1),
        "sensors":      reading,
        "machine_id":   machine_id,
    }
```

### 5.4 Consecutive Count Escalation

Single-tick anomalies are filtered as noise. Only sustained faults trigger orchestration:

```
Tick 1: MSE = 0.82 > 0.7187 → is_anomaly=True, consecutive_count=1, escalated=False
Tick 2: MSE = 0.91 > 0.7187 → is_anomaly=True, consecutive_count=2, escalated=False
Tick 3: MSE = 0.88 > 0.7187 → is_anomaly=True, consecutive_count=3, escalated=TRUE ✓
         │
         └──► _on_anomaly_confirmed(alert) → OrchestratorAgent.handle_anomaly()
```

---

## 6. Agentic Orchestration (LangGraph)

### 6.1 LangGraph Pipeline Overview

The cognitive core uses a 5-node Directed Acyclic Graph (DAG):

```
┌─────────────────────────────────────────────────────────────────────┐
│                      LANGGRAPH DAG PIPELINE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ENTRY ───► [SENSOR_STATUS] ───► [DIAGNOSTIC] ───► [KNOWLEDGE]     │
│                                                           │         │
│                                                           ▼         │
│                        ┌──────────────────────────── [STRATEGY]     │
│                        │                                  │         │
│                        │ retry (critic_approved=False)    ▼         │
│                        │                              [CRITIC]      │
│                        │                                  │         │
│                        └──────────◄───────────────────────┤         │
│                                                           │         │
│                                        approved ──────────┘         │
│                                           │                         │
│                                           ▼                         │
│                                         END                         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 6.2 Shared State Object (CopilotState)

All nodes communicate through an immutable-accumulation TypedDict:

```python
class CopilotState(TypedDict):
    # ═══════ INPUTS ═══════
    machine_id:       str           # e.g., "PUMP-001"
    machine_state:    str           # e.g., "machine_fault"
    anomaly_score:    float         # MSE reconstruction error
    suspect_sensor:   str           # e.g., "vibration"
    recent_readings:  Dict          # Raw sensor values at fault time
    user_query:       str           # Operator's free-text question
    anomaly_id:       Optional[int] # FK into anomaly_records table

    # ═══════ NODE OUTPUTS ═══════
    sensor_status:        str       # "FAULT" | "WARNING" | "NORMAL"
    sensor_analysis:      str       # LLM-generated paragraph
    diagnostic_category:  str       # "MECHANICAL" | "ELECTRICAL" | "SENSOR"
    diagnostic_summary:   str       # Structured diagnosis paragraph
    retrieved_knowledge:  str       # Concatenated RAG text chunks
    retrieved_images:     List[str] # Paths to technical diagrams
    execution_strategy:   str       # Proposed maintenance approach
    final_execution_plan: str       # Full LLM response (Mode 1 or 2)
    critic_approved:      bool      # True = proceed, False = retry
    critic_feedback:      str       # Critique text for retry
```

### 6.3 Node-by-Node Analysis

#### Node 1: SENSOR_STATUS

**Purpose:** Translate raw telemetry into human-readable severity assessment.

```
INPUT:  machine_id, anomaly_score, suspect_sensor, recent_readings
OUTPUT: sensor_status ("FAULT"|"WARNING"|"NORMAL"), sensor_analysis

LOGIC:
  - If anomaly_score > 1.5 × threshold → "FAULT" (critical)
  - If anomaly_score > threshold → "WARNING"
  - Else → "NORMAL"
  
LLM PROMPT: "Describe which sensors are outside normal ranges and 
             what physical phenomena might explain the deviation"
```

#### Node 2: DIAGNOSTIC

**Purpose:** Categorize root cause and produce structured diagnosis.

```
INPUT:  sensor_status, sensor_analysis, suspect_sensor
OUTPUT: diagnostic_category, diagnostic_summary

CATEGORIES:
  ┌──────────────┬────────────────────────────────────────────┐
  │  MECHANICAL  │ vibration/speed/current → bearing, shaft  │
  │  ELECTRICAL  │ motor_current only → winding, VFD         │
  │  SENSOR      │ freeze/drift patterns → calibration       │
  │  THERMAL     │ temperature without mechanical → cooling  │
  └──────────────┴────────────────────────────────────────────┘
```

#### Node 3: KNOWLEDGE

**Purpose:** Retrieve machine-specific technical knowledge from RAG.

```
INPUT:  machine_id, diagnostic_category, diagnostic_summary, user_query
OUTPUT: retrieved_knowledge, retrieved_images

FLOW:
  1. Lookup machine_id → machines table → manual_id
  2. Construct contextual query:
     "Diagnose {category} fault. Symptoms: {summary}. Query: {user_query}"
  3. Embed query → 1536-dim vector
  4. Search ManualChunk table (filtered by manual_id)
  5. Search InteractionMemory (filtered by machine_id)
  6. Assemble retrieved context + image paths
```

**Critical Design: manual_id Isolation**
The retriever always filters `ManualChunk.manual_id == manual_id`. A PUMP fault only searches pump manuals, never lathe or turbine chunks.

#### Node 4: STRATEGY

**Purpose:** Synthesize all prior analysis into actionable response.

```
INPUT:  diagnostic_summary, retrieved_knowledge, retrieved_images, user_query
OUTPUT: final_execution_plan

MODE DETECTION:
  - If query contains "Generate full step-by-step repair procedure" → Mode 2
  - Else → Mode 1 (Summary)

MODE 1 OUTPUT (Summary):
  "Based on sensor readings, elevated vibration (2.8 mm/s) combined with
   increased motor current (7.2A) indicates bearing wear...
   [SUGGESTION: Generate full step-by-step repair procedure]"

MODE 2 OUTPUT (Procedure):
  [PROCEDURE_START]
  {
    "phases": [
      {"id": "safety_01", "type": "safety", "subphases": [...]},
      {"id": "maint_01", "type": "maintenance", "subphases": [...]}
    ]
  }
  [PROCEDURE_END]
```

#### Node 5: CRITIC

**Purpose:** Validate strategy for safety compliance and coherence.

```
INPUT:  final_execution_plan, diagnostic_category
OUTPUT: critic_approved (bool), critic_feedback (str)

VALIDATION RULES:
  ┌────────────────────────────────────────────────────────────┐
  │ For MECHANICAL category:                                   │
  │   ✓ Lockout/Tagout (LOTO) steps present?                  │
  │   ✓ PPE requirements listed?                              │
  │   ✓ Post-repair verification step?                        │
  │                                                            │
  │ For Mode 2 procedures:                                     │
  │   ✓ First phase type == "safety"?                         │
  │   ✓ Critical tasks marked "critical": true?               │
  │                                                            │
  │ General:                                                   │
  │   ✓ Response coherent and related to symptoms?            │
  │   ✓ No contradictions?                                    │
  └────────────────────────────────────────────────────────────┘

ROUTING:
  critic_approved=True  → END
  critic_approved=False → STRATEGY (with critic_feedback injected)
```

---

## 7. Multimodal RAG Engine

### 7.1 Architecture Overview

The RAG engine is an independent subsystem with three responsibilities:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     UNIFIED MULTIMODAL RAG ENGINE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐         │
│  │   INGESTION   │   │   RETRIEVAL   │   │  GENERATION   │         │
│  │               │   │               │   │               │         │
│  │ PDF → Parse   │   │ Query →       │   │ Context →     │         │
│  │ → YOLO →      │   │ Embed →       │   │ LLM →         │         │
│  │ GPT-4o →      │   │ pgvector →    │   │ Response      │         │
│  │ Chunk →       │   │ Retrieve      │   │               │         │
│  │ Embed →       │   │               │   │               │         │
│  │ Store         │   │               │   │               │         │
│  └───────────────┘   └───────────────┘   └───────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.2 Ingestion Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     PDF MANUAL INGESTION FLOW                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  PDF Upload (multipart/form-data)                                  │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STAGE 1: PDF PARSING (parser.py)                            │   │
│  │                                                              │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │ PyMuPDF: Render page to 150 DPI image                │  │   │
│  │   └─────────────────────┬────────────────────────────────┘  │   │
│  │                         ▼                                    │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │ YOLOv8-DocLayNet: Detect layout regions              │  │   │
│  │   │   Classes: picture, figure, text, title, list, table │  │   │
│  │   └─────────────────────┬────────────────────────────────┘  │   │
│  │                         │                                    │   │
│  │         ┌───────────────┴───────────────┐                   │   │
│  │         ▼                               ▼                    │   │
│  │   ┌──────────────┐             ┌──────────────┐             │   │
│  │   │ IMAGE region │             │ TEXT region  │             │   │
│  │   │ Crop → Save  │             │ Clip → Extract│             │   │
│  │   │ PNG file     │             │ text content │             │   │
│  │   └──────────────┘             └──────────────┘             │   │
│  │                                                              │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │ Camelot: Extract tables using lattice detection      │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STAGE 2: VISION CAPTIONING (captioner.py)                   │   │
│  │                                                              │   │
│  │ For each type=="image" item:                                │   │
│  │   GPT-4o Vision API → Generate technical description        │   │
│  │                                                              │   │
│  │ Example Output:                                             │   │
│  │   "[VISUAL DESCRIPTION]: Cross-section of centrifugal pump  │   │
│  │    showing impeller clearance gap of 0.5mm, wear ring       │   │
│  │    position, and discharge volute geometry..."              │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STAGE 3: CONTEXTUAL CHUNKING (chunker.py)                   │   │
│  │                                                              │   │
│  │ Sliding window: chunk_size=500 words, overlap=100 words     │   │
│  │                                                              │   │
│  │   [────────── Chunk 1 ──────────]                           │   │
│  │                    [────────── Chunk 2 ──────────]          │   │
│  │                                 [────────── Chunk 3 ──────] │   │
│  │   ◄─────────── 500 words ───────────►                       │   │
│  │                    ◄─── 100 word overlap ──►                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ STAGE 4: EMBEDDING & STORAGE (embedder.py)                  │   │
│  │                                                              │   │
│  │ For each chunk (batch_size=20):                             │   │
│  │   1. OpenAI text-embedding-3-small → 1536-dim vector        │   │
│  │   2. INSERT INTO manual_chunks (manual_id, type, content,   │   │
│  │                                  embedding, page, path)     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.3 Retrieval Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                     RETRIEVAL FLOW                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User Query: "Why is motor current spiking?"                       │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ DYNAMIC ROUTING                                              │   │
│  │                                                              │   │
│  │   machine_id="PUMP-001"                                     │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │   SELECT manual_id FROM machines                             │   │
│  │   WHERE machine_id = "PUMP-001"                              │   │
│  │         │                                                    │   │
│  │         ▼                                                    │   │
│  │   manual_id = "Zynaptrix_9000"                               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ QUERY ENCODING                                               │   │
│  │                                                              │   │
│  │   query_embedding = text-embedding-3-small(query)           │   │
│  │                   = [1536-dimensional vector]               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ VECTOR SIMILARITY SEARCH (pgvector)                         │   │
│  │                                                              │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │ Search 1: Manual Text + Table Chunks (top-3)         │  │   │
│  │   │   WHERE manual_id = "Zynaptrix_9000"                  │  │   │
│  │   │     AND type IN ("text", "table")                     │  │   │
│  │   │   ORDER BY cosine_distance(embedding, query_embedding)│  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │ Search 2: Manual Image Chunks (top-1)                │  │   │
│  │   │   WHERE manual_id = "Zynaptrix_9000"                  │  │   │
│  │   │     AND type = "image"                                │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  │                                                              │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │ Search 3: Historical Fixes (top-2)                   │  │   │
│  │   │   FROM interaction_memory                            │  │   │
│  │   │   WHERE machine_id = "PUMP-001"                       │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ CONTEXT ASSEMBLY                                             │   │
│  │                                                              │   │
│  │   --- Manual Context 1 (Page 12) ---                        │   │
│  │   Motor current spikes typically indicate excessive load... │   │
│  │                                                              │   │
│  │   --- Image Description 1 (Page 8) ---                      │   │
│  │   [IMAGE REFERENCE: pump_p8_img0.png]                       │   │
│  │   [VISUAL DESCRIPTION]: Cross-section showing...            │   │
│  │                                                              │   │
│  │   --- PREVIOUS FIX 1 (2026-01-15) ---                       │   │
│  │   Summary: Replaced worn bearing set...                     │   │
│  │   Operator Actions: Removed coupling guard...               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│       │                                                             │
│       ▼                                                             │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ LLM GENERATION (GPT-4o, temperature=0.1)                    │   │
│  │                                                              │   │
│  │   System: "You are a [manual_type] specialist..."           │   │
│  │   User: "Technician query: 'Why is motor current spiking?'" │   │
│  │   Context: [assembled context above]                        │   │
│  │                                                              │   │
│  │   Response → final answer + image references                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 7.4 Database Schema

```sql
-- Manual knowledge chunks
CREATE TABLE manual_chunks (
    id          SERIAL PRIMARY KEY,
    manual_id   VARCHAR(255) NOT NULL,     -- "Zynaptrix_9000"
    type        VARCHAR(50)  NOT NULL,     -- "text" | "image" | "table"
    content     TEXT         NOT NULL,     -- Raw text or vision description
    embedding   VECTOR(1536),              -- OpenAI embedding
    page        INTEGER,                   -- Source page number
    path        VARCHAR(500)               -- Image file path (nullable)
);
CREATE INDEX idx_manual_chunks_manual_id ON manual_chunks(manual_id);

-- Machine registry
CREATE TABLE machines (
    id          SERIAL PRIMARY KEY,
    machine_id  VARCHAR(100) UNIQUE NOT NULL,  -- "PUMP-001"
    name        VARCHAR(255),
    location    VARCHAR(255),
    manual_id   VARCHAR(255)                    -- Links to manual_chunks
);

-- Anomaly incident log
CREATE TABLE anomaly_records (
    id          SERIAL PRIMARY KEY,
    machine_id  VARCHAR(100) NOT NULL,
    timestamp   VARCHAR(50)  NOT NULL,
    type        VARCHAR(100),
    score       INTEGER,
    sensor_data TEXT,                          -- JSON
    resolved    BOOLEAN DEFAULT FALSE
);

-- Chat conversation history
CREATE TABLE chat_messages (
    id          SERIAL PRIMARY KEY,
    anomaly_id  INTEGER REFERENCES anomaly_records(id),
    role        VARCHAR(50)  NOT NULL,         -- "agent" | "user"
    content     TEXT         NOT NULL,
    timestamp   VARCHAR(50),
    images      TEXT,                          -- JSON array
    metadata    TEXT                           -- JSON (task completion state)
);

-- Organizational learning memory
CREATE TABLE interaction_memory (
    id          SERIAL PRIMARY KEY,
    machine_id  VARCHAR(100) NOT NULL,
    manual_id   VARCHAR(255),                  -- Always "Historical_Knowledge"
    summary     TEXT         NOT NULL,         -- GPT-4o summary (≤150 words)
    operator_fix TEXT,                         -- Verbatim operator description
    embedding   VECTOR(1536),                  -- Embedding of summary
    timestamp   VARCHAR(50)
);
CREATE INDEX idx_interaction_memory_machine ON interaction_memory(machine_id);
```

---

## 8. Human-in-the-Loop (HITL) Workflow

### 8.1 Procedure Interaction Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HITL PROCEDURE WORKFLOW                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 1. ANOMALY DETECTION                                         │   │
│  │    System detects sustained fault → creates AnomalyRecord    │   │
│  │    → broadcasts anomaly_alert via WebSocket                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 2. INITIAL DIAGNOSIS (Mode 1)                                │   │
│  │    Operator clicks incident → auto-triggers quick summary    │   │
│  │    Agent returns: "Bearing wear suspected..."                │   │
│  │    + [SUGGESTION: Generate full step-by-step procedure]      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 3. PROCEDURE REQUEST (Mode 2)                                │   │
│  │    Operator clicks "🔧 Start Guided Repair Procedure"        │   │
│  │    Agent returns structured JSON procedure                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 4. STEP-BY-STEP EXECUTION                                    │   │
│  │                                                              │   │
│  │    ┌────────────────────────────────────────────────────┐   │   │
│  │    │ PHASE: SAFETY                                       │   │   │
│  │    │                                                     │   │   │
│  │    │  ┌──────────────────────────────────────────────┐  │   │   │
│  │    │  │ [Step 1] 🔴 CRITICAL                          │  │   │   │
│  │    │  │ Power off and perform LOTO                   │  │   │   │
│  │    │  │                                               │  │   │   │
│  │    │  │ [ I have done ] [ Haven't ] [ Can't do ]     │  │   │   │
│  │    │  └──────────────────────────────────────────────┘  │   │   │
│  │    │                                                     │   │   │
│  │    │  ┌──────────────────────────────────────────────┐  │   │   │
│  │    │  │ [Step 2] 🔴 CRITICAL                          │  │   │   │
│  │    │  │ Don PPE: safety glasses, gloves              │  │   │   │
│  │    │  └──────────────────────────────────────────────┘  │   │   │
│  │    └────────────────────────────────────────────────────┘   │   │
│  │                                                              │   │
│  │    ┌────────────────────────────────────────────────────┐   │   │
│  │    │ PHASE: MAINTENANCE                                  │   │   │
│  │    │                                                     │   │   │
│  │    │  ┌──────────────────────────────────────────────┐  │   │   │
│  │    │  │ [Step 3]                                      │  │   │   │
│  │    │  │ Remove coupling guard [IMAGE_0]              │  │   │   │
│  │    │  │                                               │  │   │   │
│  │    │  │ ┌─────────────────────────────────────────┐  │  │   │   │
│  │    │  │ │ [Technical diagram shown inline]       │  │  │   │   │
│  │    │  │ └─────────────────────────────────────────┘  │  │   │   │
│  │    │  │                                               │  │   │   │
│  │    │  │ Comment: ___________________________          │  │   │   │
│  │    │  │ [ Done ] [ Undone ]                          │  │   │   │
│  │    │  └──────────────────────────────────────────────┘  │   │   │
│  │    └────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 5. COMPLETION & ARCHIVAL                                     │   │
│  │                                                              │   │
│  │    ┌────────────────────────────────────────────────────┐   │   │
│  │    │ 🎉 All Steps Completed!                             │   │   │
│  │    │                                                     │   │   │
│  │    │ Document your fix:                                  │   │   │
│  │    │ ┌─────────────────────────────────────────────┐    │   │   │
│  │    │ │ Replaced worn ball bearings at drive end.  │    │   │   │
│  │    │ │ Used bearing puller #BP-3 to extract old   │    │   │   │
│  │    │ │ bearings. Installed SKF 6205-2RS sealed... │    │   │   │
│  │    │ └─────────────────────────────────────────────┘    │   │   │
│  │    │                                                     │   │   │
│  │    │ [ Archive Incident ]                                │   │   │
│  │    └────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ 6. LEARNING LOOP                                             │   │
│  │                                                              │   │
│  │    1. AnomalyRecord.resolved = True                         │   │
│  │    2. GPT-4o summarizes chat + operator_fix → ≤150 words    │   │
│  │    3. Embed summary → 1536-dim vector                       │   │
│  │    4. INSERT INTO interaction_memory                        │   │
│  │    5. Future PUMP-001 queries will retrieve this fix        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 Procedure JSON Schema

```json
{
  "phases": [
    {
      "id": "safety_01",
      "type": "safety",
      "title": "Pre-Work Safety Preparation",
      "subphases": [
        {
          "title": "Lockout/Tagout",
          "tasks": [
            {
              "id": "s1",
              "text": "Power off main breaker and apply LOTO lock",
              "critical": true
            },
            {
              "id": "s2",
              "text": "Verify zero energy state with multimeter",
              "critical": true
            }
          ]
        }
      ]
    },
    {
      "id": "maint_01",
      "type": "maintenance",
      "title": "Bearing Replacement Procedure",
      "subphases": [
        {
          "title": "Component Removal",
          "tasks": [
            {
              "id": "t1",
              "text": "Remove coupling guard [IMAGE_0]"
            },
            {
              "id": "t2",
              "text": "Extract shaft bearing using puller tool"
            }
          ]
        }
      ]
    }
  ]
}
```

---

## 9. Continuous Learning Loop

### 9.1 Knowledge Enrichment Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     ORGANIZATIONAL LEARNING LOOP                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ TIME T: Initial Anomaly                                      │   │
│  │                                                              │   │
│  │   PUMP-001 fault detected                                   │   │
│  │   RAG retrieves: Manual chunks only (no historical fixes)   │   │
│  │   Operator performs repair + documents fix                  │   │
│  │   → "Replaced worn ball bearings at drive end housing"      │   │
│  │                                                              │   │
│  │   GPT-4o Summary:                                           │   │
│  │   "Bearing degradation in PUMP-001 drive end housing        │   │
│  │    caused elevated vibration (2.8mm/s) and motor current    │   │
│  │    spike (7.2A). Root cause: mechanical wear after          │   │
│  │    8,500 operating hours. Resolution: Replaced SKF 6205     │   │
│  │    bearings using puller BP-3."                             │   │
│  │                                                              │   │
│  │   → Embedded and stored in interaction_memory               │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                               │                                     │
│                               ▼                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ TIME T+30 days: Similar Anomaly                              │   │
│  │                                                              │   │
│  │   PUMP-001 fault detected (similar symptoms)                │   │
│  │   RAG retrieves:                                            │   │
│  │     1. Manual chunks (technical procedures)                 │   │
│  │     2. PREVIOUS FIX from interaction_memory ◄─── NEW!       │   │
│  │                                                              │   │
│  │   Agent response now includes:                              │   │
│  │   "Historical records show a similar incident resolved      │   │
│  │    30 days ago by replacing the drive end bearings.         │   │
│  │    Recommend inspecting bearing condition first."           │   │
│  │                                                              │   │
│  │   → Faster resolution, informed by organizational memory    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 9.2 Memory Table Growth

| Incident # | Machine | Summary | Impact |
|------------|---------|---------|--------|
| 1 | PUMP-001 | Bearing replacement | Baseline |
| 2 | LATHE-002 | Spindle alignment | Cross-machine learning |
| 3 | PUMP-001 | Seal replacement | Enriches PUMP-001 history |
| 4 | TURBINE-003 | Blade inspection | New asset type learning |
| N | ... | ... | Continuously improving |

---

## 10. Technology Stack

### 10.1 Backend Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **API Framework** | FastAPI | 0.109+ | REST endpoints + WebSocket |
| **ASGI Server** | Uvicorn | 0.27+ | High-performance async server |
| **Validation** | Pydantic v2 | 2.5+ | Request/response contracts |
| **ORM** | SQLAlchemy | 2.0+ | Database sessions and schema |
| **Vector Extension** | pgvector | 0.5+ | Cosine similarity search |
| **ML Framework** | TensorFlow/Keras | 2.15+ | Autoencoder training/inference |
| **PDF Parsing** | PyMuPDF (fitz) | 1.23+ | PDF rendering + text extraction |
| **Layout Detection** | YOLOv8 (Ultralytics) | 8.0+ | Document segmentation |
| **Vision AI** | OpenAI GPT-4o Vision | latest | Image captioning |
| **Text Embeddings** | text-embedding-3-small | latest | 1536-dim vector encoding |
| **LLM** | OpenAI GPT-4o | latest | Diagnosis + procedure generation |
| **Agent Framework** | LangGraph | 0.0.30+ | Multi-node DAG orchestration |
| **Time-Series DB** | InfluxDB Cloud | 2.0+ | 10Hz telemetry storage |
| **Vector/Relational DB** | PostgreSQL + pgvector | 15+ | Chunks, machines, anomalies |

### 10.2 Frontend Technologies

| Category | Technology | Version | Purpose |
|----------|------------|---------|---------|
| **Framework** | Next.js | 14+ | App Router + SSR/CSR |
| **Language** | TypeScript | 5+ | Type-safe development |
| **State Management** | Redux Toolkit | 2+ | Centralized state |
| **Styling** | Tailwind CSS | 4+ | Utility-first CSS |
| **Charts** | Recharts | 3+ | Real-time line charts |
| **Markdown** | react-markdown + remark-gfm | 10+ | Agent response rendering |
| **Icons** | Lucide React | 0.3+ | Consistent iconography |
| **Animation** | Framer Motion | 12+ | UI micro-animations |

---

## 11. System Architecture Diagrams

### 11.1 High-Level System Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                          FACTORY FLOOR                              │
│                                                                     │
│    [PUMP-001]          [LATHE-002]          [TURBINE-003]          │
│        │                    │                    │                  │
│        └────────────────────┼────────────────────┘                  │
│                             │ Sensor Telemetry (10Hz)               │
└─────────────────────────────┼───────────────────────────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                BACKEND - INDUSTRIAL COPILOT CORE                    │
│                        (FastAPI/Python)                             │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ SIMULATOR LAYER                                             │    │
│  │ • Per-machine telemetry generation                         │    │
│  │ • Independent START/STOP controls                          │    │
│  │ • Synthetic anomaly injection                              │    │
│  └─────────────────────────┬──────────────────────────────────┘    │
│            ▼               ▼                                        │
│   InfluxDB Write      WebSocket Broadcast                           │
│            ▼               ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ ANOMALY DETECTION                                           │    │
│  │ • Per-machine Autoencoder (TensorFlow)                     │    │
│  │ • Per-machine StandardScaler registry                      │    │
│  │ • MSE-based anomaly scoring                                │    │
│  │ • Consecutive-count escalation                             │    │
│  └─────────────────────────┬──────────────────────────────────┘    │
│                            ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ MULTI-AGENT ORCHESTRATION (LangGraph)                       │    │
│  │                                                             │    │
│  │  Alert → [Sensor Status] → [Diagnostic] → [RAG] →          │    │
│  │           [Strategy] → [Critic] → Execution Plan            │    │
│  └─────────────────────────┬──────────────────────────────────┘    │
│                            ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ UNIFIED MULTIMODAL RAG ENGINE                               │    │
│  │                                                             │    │
│  │  PDF → YOLO Detection → GPT-4o Captioning → Embedding      │    │
│  │  Query → Vector Search → Context Assembly → LLM Generation │    │
│  └─────────────────────────┬──────────────────────────────────┘    │
│                            ▼                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ PERSISTENCE LAYER (PostgreSQL + pgvector)                   │    │
│  │                                                             │    │
│  │  • Machine registry + manual_id mapping                    │    │
│  │  • AnomalyRecord incident tracking                         │    │
│  │  • ChatMessage conversation history                        │    │
│  │  • InteractionMemory organizational learning               │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────┬───────────────────────────────────────┘
                              │ REST API + WebSocket
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                FRONTEND - INDUSTRIAL DASHBOARD                      │
│                     (Next.js/React/Redux)                           │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ Redux Store (Centralized State)                             │    │
│  │ • copilotSlice: telemetry, chatHistory, anomalies          │    │
│  │ • machineSlice: machines, currentMachineId                 │    │
│  │ • simulatorSlice: activeSimulators                         │    │
│  │ • ingestionSlice: uploadStatus                             │    │
│  └────────────────────────────────────────────────────────────┘    │
│                                                                     │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │ UI Components                                               │    │
│  │ • Real-time Telemetry Charts (Recharts)                    │    │
│  │ • Anomaly History / Incident Registry                      │    │
│  │ • Chat Window (Markdown rendering)                         │    │
│  │ • Machine Selector (Dropdown)                              │    │
│  │ • Simulator Controls (Play/Stop)                           │    │
│  │ • Procedure Step Cards (HITL workflow)                     │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 12. Data Flow Specifications

### 12.1 Normal Telemetry Flow

```
sensor_simulator.py
  ├─ [1Hz tick]
  ├─ anomaly_injector.normal_reading("PUMP-001")
  │    └─ returns {temperature: 180.1, motor_current: 4.52, ...}
  │
  ├─ [InfluxDB Write]:
  │    InfluxWriter.write_sensor_reading(reading, state="normal")
  │
  └─ [UI Push]:
       requests.post("/api/telemetry/push", json=reading)
         └─ manager.broadcast({"type": "telemetry", "data": reading})
              └─ [WebSocket to all clients]
                   └─ dispatch(addTelemetry({...}))
                        └─ Recharts re-renders
```

### 12.2 Anomaly Detection → Orchestration Flow

```
stream_listener.py (polling InfluxDB)
  └─ requests.post("/anomaly/detect", json=reading)
       └─ anomaly_service.process(reading)
            └─ AnomalyDetector.detect(reading)
                 └─ mse = 1.42 > 0.7187 → is_anomaly=True
                      └─ consecutive_count++
                           └─ if count >= 3: ESCALATED
                                └─ OrchestratorAgent.handle_anomaly()
                                     └─ workflow.invoke(initial_state)
                                          └─ [5-Node LangGraph Pipeline]
                                               └─ Returns final_execution_plan
```

---

## 13. Industrial Sensor Data Acquisition

### 13.1 Overview: How Real Factory Data is Collected

In production industrial environments, sensor data acquisition follows established industrial protocols and architectures. While this research framework uses a **digital twin simulator** for development and testing, the architecture is designed to integrate with real industrial data sources.

```
┌─────────────────────────────────────────────────────────────────────┐
│                 INDUSTRIAL DATA ACQUISITION LANDSCAPE               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐            │
│   │  SENSORS    │    │    PLC      │    │   SCADA     │            │
│   │             │───►│             │───►│             │            │
│   │ Temperature │    │ Siemens S7  │    │ WinCC       │            │
│   │ Pressure    │    │ Allen-Brad. │    │ Ignition    │            │
│   │ Vibration   │    │ Mitsubishi  │    │ Wonderware  │            │
│   │ Current     │    │             │    │             │            │
│   └─────────────┘    └──────┬──────┘    └──────┬──────┘            │
│                             │                   │                   │
│         ┌───────────────────┴───────────────────┘                   │
│         ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    INDUSTRIAL PROTOCOLS                      │  │
│   │                                                              │  │
│   │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐    │  │
│   │  │ Modbus   │  │ OPC UA   │  │EtherNet/ │  │PROFINET  │    │  │
│   │  │ TCP/RTU  │  │          │  │   IP     │  │          │    │  │
│   │  └──────────┘  └──────────┘  └──────────┘  └──────────┘    │  │
│   └─────────────────────────────────────────────────────────────┘  │
│         │                                                           │
│         ▼                                                           │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │                    IOT GATEWAY / EDGE                        │  │
│   │                                                              │  │
│   │  Siemens IoT2040 │ Advantech │ Raspberry Pi │ Custom Edge   │  │
│   └────────────────────────────┬────────────────────────────────┘  │
│                                │                                    │
│                                ▼ MQTT / HTTP / WebSocket           │
│   ┌─────────────────────────────────────────────────────────────┐  │
│   │              ZYNAPTRIX INDUSTRIAL COPILOT                    │  │
│   │                   (This Framework)                           │  │
│   └─────────────────────────────────────────────────────────────┘  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 13.2 Method 1: Direct PLC Communication (Most Common)

**Overview:** PLCs (Programmable Logic Controllers) store real-time sensor values in memory registers. This is the **primary method** for acquiring raw, real-time industrial data.

#### Supported Industrial Protocols

| Protocol | Description | Common Use Cases | Port |
|----------|-------------|------------------|------|
| **Modbus TCP/RTU** | Simple, widely supported | Legacy systems, simple sensors | 502 |
| **OPC UA** | Modern, secure, unified | Industry 4.0, complex systems | 4840 |
| **EtherNet/IP** | Allen-Bradley standard | Rockwell automation | 44818 |
| **PROFINET** | Siemens standard | Siemens PLC ecosystem | Dynamic |
| **S7 Protocol** | Siemens proprietary | Direct S7-300/400/1200/1500 | 102 |

#### PLC Register Mapping Example

```
┌────────────────────────────────────────────────────────────────┐
│                    PLC MEMORY REGISTER MAP                      │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│  HOLDING REGISTERS (Read/Write):                               │
│  ┌──────────┬─────────────────────────────────────────┐        │
│  │ 40001    │ Temperature Sensor 1 (scaled: ×0.1°C)   │        │
│  │ 40002    │ Motor Current (scaled: ×0.01A)          │        │
│  │ 40003    │ Vibration Sensor (scaled: ×0.001mm/s)   │        │
│  │ 40004    │ Speed Encoder (RPM, direct)             │        │
│  │ 40005    │ Pressure Transducer (scaled: ×0.1 bar)  │        │
│  └──────────┴─────────────────────────────────────────┘        │
│                                                                 │
│  INPUT REGISTERS (Read-Only):                                  │
│  ┌──────────┬─────────────────────────────────────────┐        │
│  │ 30001    │ Ambient Temperature                     │        │
│  │ 30002    │ Coolant Level                           │        │
│  └──────────┴─────────────────────────────────────────┘        │
│                                                                 │
│  COILS (Digital I/O):                                          │
│  ┌──────────┬─────────────────────────────────────────┐        │
│  │ 00001    │ Motor Running Status (1=ON, 0=OFF)      │        │
│  │ 00002    │ Emergency Stop Active                   │        │
│  │ 00003    │ Safety Guard Closed                     │        │
│  └──────────┴─────────────────────────────────────────┘        │
│                                                                 │
└────────────────────────────────────────────────────────────────┘
```

#### Python Integration Example (pymodbus)

```python
from pymodbus.client import ModbusTcpClient
import time

# Connect to PLC
client = ModbusTcpClient('192.168.1.100', port=502)
client.connect()

def read_sensor_data():
    """Read sensor values from PLC holding registers."""
    # Read 5 registers starting at address 40001 (Modbus uses 0-based: 0)
    result = client.read_holding_registers(address=0, count=5, slave=1)
    
    if not result.isError():
        return {
            "temperature":    result.registers[0] * 0.1,    # Scale factor
            "motor_current":  result.registers[1] * 0.01,
            "vibration":      result.registers[2] * 0.001,
            "speed":          result.registers[3],           # Direct value
            "pressure":       result.registers[4] * 0.1,
            "timestamp":      time.time()
        }
    return None

# Polling loop (integrate with Zynaptrix telemetry endpoint)
while True:
    data = read_sensor_data()
    if data:
        # POST to Zynaptrix: /api/telemetry/push
        requests.post("http://localhost:8000/api/telemetry/push", json=data)
    time.sleep(0.1)  # 10Hz polling
```

#### OPC UA Integration Example

```python
from opcua import Client

# Connect to OPC UA server
client = Client("opc.tcp://192.168.1.100:4840")
client.connect()

# Browse and read nodes
temperature_node = client.get_node("ns=2;s=PUMP001.Temperature")
current_node = client.get_node("ns=2;s=PUMP001.MotorCurrent")
vibration_node = client.get_node("ns=2;s=PUMP001.Vibration")

def read_opc_data():
    return {
        "temperature":    temperature_node.get_value(),
        "motor_current":  current_node.get_value(),
        "vibration":      vibration_node.get_value(),
        "machine_id":     "PUMP-001"
    }
```

#### Recommended Tools

| Language | Library | Protocol Support |
|----------|---------|------------------|
| **Python** | `pymodbus` | Modbus TCP/RTU |
| **Python** | `opcua` / `asyncua` | OPC UA |
| **Python** | `snap7` | Siemens S7 |
| **Node.js** | `modbus-serial` | Modbus TCP/RTU |
| **Node.js** | `node-opcua` | OPC UA |
| **Go** | `gomodbus` | Modbus TCP |

**👉 This is the BEST method for raw, real-time data with minimal latency.**

---

### 13.3 Method 2: SCADA System Integration

**Overview:** SCADA (Supervisory Control and Data Acquisition) systems already collect and historize sensor data. Integration provides access to processed, validated data.

#### Common SCADA Platforms

| Platform | Vendor | Data Access Methods |
|----------|--------|---------------------|
| **WinCC** | Siemens | SQL Server, OPC, REST API |
| **Ignition** | Inductive Automation | SQL, MQTT, REST API |
| **Wonderware InTouch** | AVEVA | Historian, SQL, OPC |
| **FactoryTalk** | Rockwell | SQL Server, OPC |
| **AVEVA PI** | AVEVA | PI Web API, JDBC |

#### SCADA Database Integration Example

```python
import pyodbc
import pandas as pd

# Connect to SCADA historian database
conn = pyodbc.connect(
    'DRIVER={SQL Server};'
    'SERVER=scada-server.factory.local;'
    'DATABASE=WinCC_History;'
    'UID=readonly_user;'
    'PWD=secure_password'
)

def query_recent_telemetry(machine_id: str, minutes: int = 5):
    """Query last N minutes of sensor data from SCADA historian."""
    query = f"""
        SELECT 
            Timestamp,
            Temperature,
            MotorCurrent,
            Vibration,
            Speed,
            Pressure
        FROM SensorHistory
        WHERE MachineID = '{machine_id}'
          AND Timestamp >= DATEADD(minute, -{minutes}, GETDATE())
        ORDER BY Timestamp DESC
    """
    return pd.read_sql(query, conn)

# Ignition SCADA REST API example
import requests

def get_ignition_tags(tag_paths: list):
    """Read tags from Ignition SCADA via REST API."""
    response = requests.post(
        "http://ignition-server:8088/system/webdev/readTags",
        json={"paths": tag_paths},
        auth=("api_user", "api_key")
    )
    return response.json()
```

**👉 Easier than PLC integration, but data may be pre-processed or aggregated.**

---

### 13.4 Method 3: IoT Gateway / Edge Device

**Overview:** Modern factories use IoT gateways as intermediaries between PLCs and cloud systems, providing protocol translation and edge processing.

#### Common IoT Gateways

| Device | Vendor | Protocols Supported |
|--------|--------|---------------------|
| **IoT2040** | Siemens | Modbus, S7, MQTT, HTTP |
| **EdgeLink** | Advantech | Modbus, OPC UA, MQTT |
| **ThingsPro** | Moxa | Modbus, MQTT, RESTful |
| **Raspberry Pi** | Foundation | Custom (w/ HATs) |
| **NVIDIA Jetson** | NVIDIA | Edge AI + protocols |

#### Gateway Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IOT GATEWAY ARCHITECTURE                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌───────────────────────────────────────────────────────────────┐ │
│  │                     SOUTHBOUND (Field Side)                    │ │
│  │                                                                │ │
│  │   [PLC 1] ──Modbus──►┐                                        │ │
│  │   [PLC 2] ──S7─────►├──► [IoT Gateway]                        │ │
│  │   [Sensor]──4-20mA─►┘     │                                   │ │
│  │                           │ Edge Processing:                   │ │
│  │                           │ • Data normalization               │ │
│  │                           │ • Local anomaly detection          │ │
│  │                           │ • Data buffering                   │ │
│  │                           │ • Protocol translation             │ │
│  └───────────────────────────┼───────────────────────────────────┘ │
│                              │                                      │
│  ┌───────────────────────────┼───────────────────────────────────┐ │
│  │                     NORTHBOUND (Cloud Side)                    │ │
│  │                              │                                 │ │
│  │                              ▼                                 │ │
│  │   ┌────────────────────────────────────────────────────────┐  │ │
│  │   │                    MQTT BROKER                         │  │ │
│  │   │                                                        │  │ │
│  │   │  Topics:                                               │  │ │
│  │   │    factory/line1/pump001/temperature                   │  │ │
│  │   │    factory/line1/pump001/motor_current                 │  │ │
│  │   │    factory/line1/pump001/vibration                     │  │ │
│  │   │    factory/line1/pump001/anomaly_alert                 │  │ │
│  │   └────────────────────────────────────────────────────────┘  │ │
│  │                              │                                 │ │
│  │                              ▼                                 │ │
│  │   ┌────────────────────────────────────────────────────────┐  │ │
│  │   │           ZYNAPTRIX MQTT SUBSCRIBER                    │  │ │
│  │   │                                                        │  │ │
│  │   │  Subscribe: factory/+/+/# (wildcard)                   │  │ │
│  │   │  → Parse message → POST /api/telemetry/push            │  │ │
│  │   └────────────────────────────────────────────────────────┘  │ │
│  └───────────────────────────────────────────────────────────────┘ │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### MQTT Subscriber Integration

```python
import paho.mqtt.client as mqtt
import json
import requests

ZYNAPTRIX_API = "http://localhost:8000/api/telemetry/push"

def on_message(client, userdata, msg):
    """Handle incoming MQTT messages from IoT gateway."""
    topic_parts = msg.topic.split('/')
    # topic: factory/line1/pump001/temperature
    
    machine_id = topic_parts[2].upper()  # "PUMP001" → "PUMP-001"
    sensor_type = topic_parts[3]
    value = float(msg.payload.decode())
    
    # Aggregate readings and push to Zynaptrix
    # (Implementation depends on message format)

def on_connect(client, userdata, flags, rc):
    # Subscribe to all factory telemetry
    client.subscribe("factory/+/+/#")

mqtt_client = mqtt.Client()
mqtt_client.on_connect = on_connect
mqtt_client.on_message = on_message
mqtt_client.connect("mqtt-broker.factory.local", 1883)
mqtt_client.loop_forever()
```

**👉 Subscribe to topics like `factory/line1/temperature` for real-time streaming.**

---

### 13.5 Method 4: Direct Sensor Interface (Advanced)

**Overview:** In rare cases without PLC infrastructure, sensors can be read directly using analog-to-digital converters (ADCs) and microcontrollers.

#### Analog Signal Types

| Signal Type | Range | Common Use | Resolution |
|-------------|-------|------------|------------|
| **4-20mA** | Current loop | Process instruments | High noise immunity |
| **0-10V** | Voltage | Industrial sensors | Medium noise immunity |
| **0-5V** | Voltage | Embedded sensors | Common microcontroller |
| **PT100/PT1000** | Resistance | Temperature RTDs | Requires signal conditioning |
| **Thermocouple** | mV | High-temp measurement | Requires cold junction compensation |

#### Hardware Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  DIRECT SENSOR INTERFACE STACK                      │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ PHYSICAL LAYER                                               │   │
│  │                                                              │   │
│  │   [Temperature]   [Pressure]   [Vibration]   [Current]      │   │
│  │   PT100 RTD       4-20mA       Accelerometer  CT Sensor     │   │
│  │       │              │              │             │          │   │
│  │       ▼              ▼              ▼             ▼          │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │         SIGNAL CONDITIONING                          │  │   │
│  │   │                                                       │  │   │
│  │   │  • RTD-to-voltage converter (e.g., MAX31865)         │  │   │
│  │   │  • 4-20mA to 0-3.3V resistor (250Ω)                  │  │   │
│  │   │  • Charge amplifier for piezo accelerometer          │  │   │
│  │   │  • Burden resistor for current transformer           │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  │       │              │              │             │          │   │
│  │       ▼              ▼              ▼             ▼          │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │         ADC (Analog-to-Digital Converter)            │  │   │
│  │   │                                                       │  │   │
│  │   │  Options:                                            │  │   │
│  │   │  • ADS1115 (16-bit, 4-channel, I²C)                 │  │   │
│  │   │  • MCP3008 (10-bit, 8-channel, SPI)                 │  │   │
│  │   │  • Built-in ADC (ESP32, STM32)                      │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ PROCESSING LAYER                                             │   │
│  │                                                              │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │  Microcontroller / SBC                                │  │   │
│  │   │                                                       │  │   │
│  │   │  • Raspberry Pi 4 (Python, high-level)               │  │   │
│  │   │  • ESP32 (MicroPython/C++, low power)                │  │   │
│  │   │  • Arduino + Ethernet Shield (simple)                │  │   │
│  │   │  • STM32 (industrial-grade)                          │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  │                              │                               │   │
│  │                              ▼ HTTP / MQTT / WebSocket      │   │
│  │   ┌──────────────────────────────────────────────────────┐  │   │
│  │   │           ZYNAPTRIX TELEMETRY ENDPOINT               │  │   │
│  │   └──────────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

#### Raspberry Pi + ADS1115 Example

```python
import board
import busio
import adafruit_ads1x15.ads1115 as ADS
from adafruit_ads1x15.analog_in import AnalogIn
import requests
import time

# Initialize I2C and ADC
i2c = busio.I2C(board.SCL, board.SDA)
ads = ADS.ADS1115(i2c)

# Create analog input channels
chan_temp = AnalogIn(ads, ADS.P0)      # Temperature (4-20mA → voltage)
chan_current = AnalogIn(ads, ADS.P1)   # Motor current
chan_vibration = AnalogIn(ads, ADS.P2) # Vibration sensor
chan_pressure = AnalogIn(ads, ADS.P3)  # Pressure transducer

def convert_4_20ma_to_value(voltage, min_val, max_val):
    """Convert 4-20mA signal (via 250Ω resistor) to engineering units."""
    # 4mA = 1V, 20mA = 5V (with 250Ω resistor)
    ma = voltage / 0.25  # Convert voltage to mA
    normalized = (ma - 4) / 16  # Normalize 4-20mA to 0-1
    return min_val + normalized * (max_val - min_val)

def read_sensors():
    return {
        "temperature":   convert_4_20ma_to_value(chan_temp.voltage, 0, 200),
        "motor_current": convert_4_20ma_to_value(chan_current.voltage, 0, 20),
        "vibration":     convert_4_20ma_to_value(chan_vibration.voltage, 0, 10),
        "pressure":      convert_4_20ma_to_value(chan_pressure.voltage, 0, 50),
        "speed":         0,  # Requires separate encoder interface
        "machine_id":    "PUMP-001",
        "timestamp":     time.time()
    }

# Main loop
while True:
    data = read_sensors()
    requests.post("http://zynaptrix-server:8000/api/telemetry/push", json=data)
    time.sleep(0.1)  # 10Hz
```

**👉 This gives true raw signal access but requires hardware design and signal conditioning.**

---

### 13.6 Integration Architecture Summary

```
┌─────────────────────────────────────────────────────────────────────┐
│              SENSOR DATA ACQUISITION METHOD COMPARISON              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  METHOD           │ COMPLEXITY │ LATENCY │ DATA QUALITY │ USE CASE │
│  ─────────────────┼────────────┼─────────┼──────────────┼──────────│
│  Direct PLC       │ Medium     │ <10ms   │ Raw          │ Best     │
│  (Modbus/OPC UA)  │            │         │              │ default  │
│  ─────────────────┼────────────┼─────────┼──────────────┼──────────│
│  SCADA System     │ Low        │ 1-5s    │ Processed    │ Existing │
│  (SQL/REST)       │            │         │              │ systems  │
│  ─────────────────┼────────────┼─────────┼──────────────┼──────────│
│  IoT Gateway      │ Medium     │ 50-500ms│ Normalized   │ Modern   │
│  (MQTT/HTTP)      │            │         │              │ IIoT     │
│  ─────────────────┼────────────┼─────────┼──────────────┼──────────│
│  Direct Sensor    │ High       │ <1ms    │ Raw analog   │ No PLC   │
│  (ADC/MCU)        │            │         │              │ infra    │
│  ─────────────────┼────────────┼─────────┼──────────────┼──────────│
│  Simulator        │ None       │ N/A     │ Synthetic    │ Dev/Test │
│  (This Framework) │            │         │              │          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 14. Validation & Quality Assurance

### 14.1 Multi-Layer Validation Architecture

The system implements validation at every layer to ensure data integrity, safety compliance, and reliable operation:

```
┌─────────────────────────────────────────────────────────────────────┐
│                  VALIDATION ARCHITECTURE LAYERS                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ LAYER 1: INPUT VALIDATION (Pydantic Models)                  │   │
│  │                                                              │   │
│  │ • API request schema validation                             │   │
│  │ • Type coercion and constraints                             │   │
│  │ • Required field enforcement                                │   │
│  │ • Range validation for sensor values                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ LAYER 2: DATA QUALITY VALIDATION (Preprocessing)            │   │
│  │                                                              │   │
│  │ • Missing value detection and handling                      │   │
│  │ • Outlier clipping (4-sigma bounds)                         │   │
│  │ • Sensor freeze detection (rolling std < 1e-6)              │   │
│  │ • Temporal continuity checks                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ LAYER 3: MODEL VALIDATION (Anomaly Detection)                │   │
│  │                                                              │   │
│  │ • Threshold calibration (99th percentile on normal data)    │   │
│  │ • Consecutive count filtering (noise rejection)             │   │
│  │ • Machine-specific model isolation                          │   │
│  │ • Health score normalization (0-100 scale)                  │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ LAYER 4: AGENT VALIDATION (LangGraph Critic Node)           │   │
│  │                                                              │   │
│  │ • Safety procedure verification (LOTO, PPE)                 │   │
│  │ • Response coherence validation                             │   │
│  │ • Manual compliance checking                                │   │
│  │ • Retry loop for failed validations                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                              │                                      │
│                              ▼                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ LAYER 5: HUMAN VALIDATION (HITL Workflow)                    │   │
│  │                                                              │   │
│  │ • Mandatory step-by-step sign-off                           │   │
│  │ • Critical task acknowledgment                              │   │
│  │ • Operator fix documentation                                │   │
│  │ • Incident resolution confirmation                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 API Input Validation (Pydantic)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any

class SensorReading(BaseModel):
    """Validated sensor reading input model."""
    
    machine_id: str = Field(..., min_length=1, max_length=50)
    temperature: float = Field(..., ge=-50, le=1500)      # °C
    motor_current: float = Field(..., ge=0, le=1000)      # Amps
    vibration: float = Field(..., ge=0, le=100)           # mm/s
    speed: float = Field(..., ge=0, le=50000)             # RPM
    pressure: float = Field(..., ge=0, le=500)            # bar
    timestamp: Optional[str] = None
    
    @validator('machine_id')
    def validate_machine_id_format(cls, v):
        """Ensure machine_id follows naming convention."""
        import re
        if not re.match(r'^[A-Z]+-\d{3}$', v):
            raise ValueError('machine_id must match pattern: TYPE-NNN')
        return v

class AnomalyEvent(BaseModel):
    """Validated anomaly event for copilot invocation."""
    
    machine_id: str
    machine_state: str = Field(..., regex='^(machine_fault|machine_warning|manual_inquiry)$')
    anomaly_id: Optional[int] = Field(None, ge=1)
    anomaly_score: Optional[float] = Field(None, ge=0, le=10)
    user_query: Optional[str] = Field(None, max_length=2000)
    suspect_sensor: Optional[str] = None
    recent_readings: Optional[Dict[str, Any]] = None

class ProcedureTaskUpdate(BaseModel):
    """Validated task completion update."""
    
    task_id: str = Field(..., min_length=1, max_length=20)
    completed: bool
    comment: Optional[str] = Field(None, max_length=500)
```

### 14.3 Data Quality Validation Rules

| Validation Type | Implementation | Threshold | Action on Failure |
|-----------------|----------------|-----------|-------------------|
| **Missing Values** | `df.isna().sum()` | >50% row | Drop row |
| **Forward Fill** | `df.ffill()` | Remaining NaN | Fill from last valid |
| **Outlier Clipping** | `clip(μ ± 4σ)` | 4 std dev | Clip to bounds |
| **Sensor Freeze** | `rolling_std < 1e-6` | 10-tick window | Flag as anomaly |
| **Value Range** | Per-sensor bounds | SENSOR_SCHEMA | Reject reading |
| **Temporal Gap** | `diff(timestamp)` | >5 seconds | Log warning |

### 14.4 Anomaly Detection Validation

```python
class AnomalyValidator:
    """Validates anomaly detection results before escalation."""
    
    def __init__(self, threshold: float = 0.7187, consecutive_required: int = 3):
        self.threshold = threshold
        self.consecutive_required = consecutive_required
        self._consecutive_count = 0
    
    def validate(self, mse_score: float, reading: dict) -> dict:
        """
        Validate anomaly detection result.
        
        Returns:
            dict with validation status and metadata
        """
        validation_result = {
            "is_valid_reading": True,
            "validation_errors": [],
            "is_anomaly": False,
            "should_escalate": False,
        }
        
        # Check sensor value ranges
        for sensor, value in reading.items():
            if sensor in SENSOR_SCHEMA:
                bounds = SENSOR_SCHEMA[sensor]["normal_range"]
                if not (bounds[0] <= value <= bounds[1]):
                    validation_result["validation_errors"].append(
                        f"{sensor} out of range: {value}"
                    )
        
        # Validate MSE score
        if mse_score < 0:
            validation_result["is_valid_reading"] = False
            validation_result["validation_errors"].append("Negative MSE score")
            return validation_result
        
        # Check anomaly threshold
        validation_result["is_anomaly"] = mse_score > self.threshold
        
        # Apply consecutive count logic
        if validation_result["is_anomaly"]:
            self._consecutive_count += 1
        else:
            self._consecutive_count = 0
        
        validation_result["consecutive_count"] = self._consecutive_count
        validation_result["should_escalate"] = (
            self._consecutive_count >= self.consecutive_required
        )
        
        return validation_result
```

### 14.5 LangGraph Critic Validation Rules

The Critic node validates agent outputs before delivery to operators:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    CRITIC VALIDATION RULESET                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  RULE CATEGORY        │ VALIDATION CHECK                │ REQUIRED │
│  ─────────────────────┼─────────────────────────────────┼──────────│
│                                                                     │
│  SAFETY COMPLIANCE:                                                 │
│  ─────────────────────┼─────────────────────────────────┼──────────│
│  LOTO Procedure       │ Contains lockout/tagout steps   │ CRITICAL │
│  PPE Requirements     │ Lists required safety equipment │ HIGH     │
│  Energy Isolation     │ Specifies isolation points      │ CRITICAL │
│  Post-Work Verify     │ Includes verification steps     │ MEDIUM   │
│                                                                     │
│  PROCEDURE STRUCTURE:                                               │
│  ─────────────────────┼─────────────────────────────────┼──────────│
│  Safety Phase First   │ type="safety" is first phase    │ CRITICAL │
│  Critical Marking     │ Safety tasks have critical=true │ HIGH     │
│  JSON Well-Formed     │ Valid JSON between tags         │ CRITICAL │
│  Step Ordering        │ Logical sequence maintained     │ MEDIUM   │
│                                                                     │
│  CONTENT QUALITY:                                                   │
│  ─────────────────────┼─────────────────────────────────┼──────────│
│  Symptom Relevance    │ Response addresses fault type   │ HIGH     │
│  No Contradictions    │ Steps don't conflict            │ HIGH     │
│  Image References     │ [IMAGE_N] tags are valid        │ MEDIUM   │
│  Completeness         │ All required sections present   │ MEDIUM   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.6 Procedure JSON Validation

```python
import json
from typing import Optional
from dataclasses import dataclass

@dataclass
class ValidationResult:
    is_valid: bool
    errors: list
    warnings: list

def validate_procedure_json(content: str) -> ValidationResult:
    """
    Validate procedure JSON structure and content.
    
    Checks:
    1. JSON is well-formed
    2. Required fields present
    3. Safety phase is first
    4. Critical tasks marked correctly
    5. Image references are valid
    """
    errors = []
    warnings = []
    
    # Extract JSON from tags
    start_tag = "[PROCEDURE_START]"
    end_tag = "[PROCEDURE_END]"
    
    if start_tag not in content or end_tag not in content:
        errors.append("Missing procedure tags")
        return ValidationResult(False, errors, warnings)
    
    json_str = content.split(start_tag)[1].split(end_tag)[0].strip()
    
    try:
        procedure = json.loads(json_str)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {str(e)}")
        return ValidationResult(False, errors, warnings)
    
    # Validate structure
    if "phases" not in procedure:
        errors.append("Missing 'phases' key")
        return ValidationResult(False, errors, warnings)
    
    phases = procedure["phases"]
    
    if len(phases) == 0:
        errors.append("No phases defined")
        return ValidationResult(False, errors, warnings)
    
    # Check safety phase is first
    if phases[0].get("type") != "safety":
        errors.append("First phase must be type='safety'")
    
    # Validate each phase
    for i, phase in enumerate(phases):
        phase_id = phase.get("id", f"phase_{i}")
        
        if "subphases" not in phase:
            warnings.append(f"Phase {phase_id} has no subphases")
            continue
        
        for j, subphase in enumerate(phase["subphases"]):
            if "tasks" not in subphase:
                warnings.append(f"Subphase {j} in {phase_id} has no tasks")
                continue
            
            for task in subphase["tasks"]:
                # Check critical marking for safety tasks
                if phase.get("type") == "safety":
                    if not task.get("critical", False):
                        warnings.append(
                            f"Safety task '{task.get('id')}' should be marked critical"
                        )
                
                # Validate image references
                text = task.get("text", "")
                import re
                image_refs = re.findall(r'\[IMAGE_(\d+)\]', text)
                for ref in image_refs:
                    # Image index validation would happen against retrieved_images
                    pass
    
    is_valid = len(errors) == 0
    return ValidationResult(is_valid, errors, warnings)
```

### 14.7 End-to-End Validation Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    END-TO-END VALIDATION FLOW                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. SENSOR DATA ARRIVES                                            │
│     │                                                               │
│     ├─► Pydantic validates schema ─────────────► 400 Bad Request   │
│     │   (type, range, required fields)                             │
│     │                                                               │
│     ▼                                                               │
│  2. PREPROCESSING                                                   │
│     │                                                               │
│     ├─► Missing values filled ──────────────────► Forward fill     │
│     ├─► Outliers clipped ───────────────────────► 4-sigma bounds   │
│     ├─► Freeze detection ───────────────────────► Flag if frozen   │
│     │                                                               │
│     ▼                                                               │
│  3. ANOMALY DETECTION                                              │
│     │                                                               │
│     ├─► Model inference ────────────────────────► MSE score        │
│     ├─► Threshold check ────────────────────────► is_anomaly       │
│     ├─► Consecutive count ──────────────────────► should_escalate  │
│     │                                                               │
│     ▼                                                               │
│  4. LANGGRAPH PIPELINE                                             │
│     │                                                               │
│     ├─► Sensor Status validation ───────────────► severity check   │
│     ├─► Diagnostic validation ──────────────────► category valid   │
│     ├─► RAG retrieval validation ───────────────► chunks found     │
│     ├─► Strategy generation ────────────────────► response formed  │
│     │                                                               │
│     ▼                                                               │
│  5. CRITIC VALIDATION                                              │
│     │                                                               │
│     ├─► Safety compliance ──────────────────────► LOTO present?    │
│     ├─► Structure validation ───────────────────► JSON valid?      │
│     ├─► Content quality ────────────────────────► coherent?        │
│     │                                                               │
│     ├─► IF FAILS: retry with feedback ──────────► back to Strategy │
│     │                                                               │
│     ▼                                                               │
│  6. HUMAN VALIDATION (HITL)                                        │
│     │                                                               │
│     ├─► Operator reviews procedure ─────────────► displayed in UI  │
│     ├─► Step-by-step confirmation ──────────────► manual sign-off  │
│     ├─► Critical task acknowledgment ───────────► safety verified  │
│     │                                                               │
│     ▼                                                               │
│  7. RESOLUTION VALIDATION                                          │
│     │                                                               │
│     ├─► Operator documents fix ─────────────────► free-text input  │
│     ├─► Summary generation ─────────────────────► GPT-4o summary   │
│     ├─► Memory archival ────────────────────────► vectorized       │
│     │                                                               │
│     ▼                                                               │
│  ✅ VALIDATED INCIDENT RESOLUTION                                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 15. Security & Authentication

### 15.1 Current Security Posture

> ⚠️ **Important:** The current implementation is designed for research/development environments. Production deployment requires additional security measures.

| Component | Current State | Production Requirement |
|-----------|---------------|------------------------|
| **CORS** | Wildcard (`*`) | Restrict to frontend URL |
| **Authentication** | None | OAuth2 + JWT |
| **Authorization** | None | RBAC (Operator, Engineer, Admin) |
| **API Rate Limiting** | None | Request throttling |
| **Data Encryption** | TLS (if configured) | TLS 1.3 mandatory |
| **Secret Management** | `.env` file | Vault / AWS Secrets Manager |
| **Audit Logging** | Basic console logs | Structured audit trail |

### 15.2 Recommended Security Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  PRODUCTION SECURITY ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ NETWORK LAYER                                                │   │
│  │                                                              │   │
│  │   [Internet] ──► [WAF] ──► [Load Balancer] ──► [API]        │   │
│  │                    │                                         │   │
│  │                    ├─► DDoS protection                       │   │
│  │                    ├─► SQL injection filtering               │   │
│  │                    └─► XSS protection                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ APPLICATION LAYER                                            │   │
│  │                                                              │   │
│  │   ┌────────────────────────────────────────────────────┐    │   │
│  │   │ FastAPI Security Middleware                         │    │   │
│  │   │                                                     │    │   │
│  │   │ • OAuth2 + JWT authentication                       │    │   │
│  │   │ • Role-based access control (RBAC)                  │    │   │
│  │   │ • API key validation for machine-to-machine         │    │   │
│  │   │ • Request rate limiting (100 req/min per user)      │    │   │
│  │   │ • Input sanitization                                │    │   │
│  │   └────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ DATA LAYER                                                   │   │
│  │                                                              │   │
│  │   • Encryption at rest (PostgreSQL TDE)                     │   │
│  │   • Encryption in transit (TLS 1.3)                         │   │
│  │   • Database access via service accounts only               │   │
│  │   • Secrets in HashiCorp Vault / AWS Secrets Manager        │   │
│  │   • PII data masking in logs                                │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 15.3 Role-Based Access Control (RBAC) Design

| Role | Permissions | Use Case |
|------|-------------|----------|
| **Viewer** | View telemetry, view anomaly history | Read-only monitoring |
| **Operator** | Viewer + execute procedures, resolve incidents | Floor technicians |
| **Engineer** | Operator + upload manuals, register machines | Maintenance engineers |
| **Admin** | Engineer + user management, system config | System administrators |
| **API Service** | Machine-to-machine telemetry push | PLC/Gateway integration |

### 15.4 JWT Authentication Implementation

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except JWTError:
        raise credentials_exception

def require_role(required_role: str):
    """Dependency for role-based access control."""
    async def role_checker(current_user: dict = Depends(get_current_user)):
        role_hierarchy = ["viewer", "operator", "engineer", "admin"]
        user_level = role_hierarchy.index(current_user["role"])
        required_level = role_hierarchy.index(required_role)
        if user_level < required_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{required_role}' or higher required"
            )
        return current_user
    return role_checker

# Usage in endpoints
@router.post("/api/copilot/invoke")
async def invoke_copilot(
    event: AnomalyEvent,
    current_user: dict = Depends(require_role("operator"))
):
    # Only operators and above can invoke copilot
    pass

@router.post("/ingest-manual")
async def ingest_manual(
    file: UploadFile,
    current_user: dict = Depends(require_role("engineer"))
):
    # Only engineers and above can upload manuals
    pass
```

---

## 16. Future Roadmap

### 16.1 Planned Enhancements

| Pillar | Target | Research Direction |
|--------|--------|-------------------|
| **Predictive Health** | Move from reactive to proactive | LSTM-Autoencoders for time-series forecasting (4-hour failure prediction) |
| **Stateful Memory** | Per-machine repair history | LangGraph tool integration for maintenance log queries |
| **Edge-Cloud Hybrid** | 100+ concurrent assets | Telemetry compression, silent anomaly filtering at edge |
| **Interactive Guidance** | AR/VR integration | Stream procedures to HoloLens for hands-free repair |
| **Multi-Language** | Global deployment | Multilingual RAG with translation layer |

### 16.2 Technical Debt Items

| Issue | Location | Priority | Solution |
|-------|----------|----------|----------|
| Wildcard CORS | `main_api.py` | HIGH | Restrict to frontend URL |
| No Authentication | All endpoints | HIGH | Add OAuth2 + JWT middleware |
| Synchronous LangGraph | `/api/copilot/invoke` | MEDIUM | Run in thread pool executor |
| No Unit Tests | Backend | MEDIUM | Add pytest coverage |
| Global WebSocket | `TelemetryClientManager` | MEDIUM | Per-machine subscription rooms |

---

## 📚 Related Documentation

- `CODEBASE_ANALYSIS.md` - Deep dive into code structure
- `GEN_AI_COPILOT_BASELINE.md` - Research baseline and objectives
- `PROJECT_RESEARCH_REPORT.md` - Supervisor research summary
- `DETAILED/01-08_*.md` - Component-specific technical documentation

---

**Document Prepared By:** Industrial AI Research Team  
**Framework Version:** 2.0.0  
**Last Updated:** 2026-04-08
