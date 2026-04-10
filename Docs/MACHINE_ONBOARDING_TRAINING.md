# Machine Onboarding and Training Process

## Industrial Copilot - Equipment Registration and Knowledge Integration

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [System Architecture for Machine Onboarding](#2-system-architecture-for-machine-onboarding)
3. [Machine Registration Process](#3-machine-registration-process)
4. [Sensor Configuration and Validation](#4-sensor-configuration-and-validation)
5. [ML Pipeline Training](#5-ml-pipeline-training)
6. [Knowledge Base Integration](#6-knowledge-base-integration)
7. [Machine-to-Documentation Linkage](#7-machine-to-documentation-linkage)
8. [Frontend User Experience](#8-frontend-user-experience)
9. [End-to-End Workflow Example](#9-end-to-end-workflow-example)
10. [Data Architecture](#10-data-architecture)
11. [Configuration Reference](#11-configuration-reference)

---

## 1. Executive Overview

### 1.1 Purpose

The Machine Onboarding and Training Process is the foundational prerequisite that enables the Industrial Copilot's multi-agent system to function. Before any intelligent agent can diagnose problems, retrieve relevant documentation, or generate repair strategies, the system must:

1. **Know what machines exist** in the facility
2. **Understand each machine's sensor configuration** (what readings are normal vs. abnormal)
3. **Have trained ML models** capable of detecting anomalies specific to each machine type
4. **Link machines to their technical documentation** for context-aware assistance

This document describes the complete onboarding workflow that transforms a raw machine into an intelligent, monitored asset within the Industrial Copilot ecosystem.

### 1.2 Onboarding Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MACHINE ONBOARDING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   MACHINE   │    │   SENSOR    │    │    ML       │    │  KNOWLEDGE  │  │
│  │ REGISTRATION│───▶│ VALIDATION  │───▶│  TRAINING   │───▶│  LINKING    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│        │                  │                  │                  │           │
│        ▼                  ▼                  ▼                  ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │  Database   │    │   Config    │    │ LSTM Model  │    │   Manual    │  │
│  │   Record    │    │   Files     │    │   Weights   │    │   Chunks    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.3 Key Outcomes

Upon successful onboarding, each machine gains:

| Capability | Description |
|------------|-------------|
| **Identity** | Unique identifier and location tracking in the fleet registry |
| **Sensor Intelligence** | Calibrated thresholds for normal operation and fault detection |
| **Anomaly Detection** | Machine-specific LSTM Autoencoder trained on realistic patterns |
| **Documentation Access** | Linked technical manual for context-aware RAG retrieval |
| **Historical Memory** | Empty interaction memory ready to accumulate resolved incidents |

---

## 2. System Architecture for Machine Onboarding

### 2.1 Component Overview

The onboarding system consists of four primary components working in concert:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        ONBOARDING ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌──────────────────────┐                             │
│                        │     FRONTEND         │                             │
│                        │  (React + Redux)     │                             │
│                        └──────────┬───────────┘                             │
│                                   │                                         │
│                                   │ HTTP POST                               │
│                                   ▼                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                        BACKEND API LAYER                              │  │
│  │                                                                        │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │  │
│  │   │   Machine    │    │  Datasheet   │    │     AI       │           │  │
│  │   │     API      │    │   Parser     │    │  Validation  │           │  │
│  │   └──────┬───────┘    └──────┬───────┘    └──────┬───────┘           │  │
│  │          │                   │                   │                    │  │
│  └──────────┼───────────────────┼───────────────────┼────────────────────┘  │
│             │                   │                   │                       │
│             ▼                   ▼                   ▼                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      ML TRAINING PIPELINE                             │  │
│  │                                                                        │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │  │
│  │   │   Dataset    │    │    Data      │    │    Model     │           │  │
│  │   │  Generator   │───▶│ Normalizer   │───▶│   Trainer    │           │  │
│  │   └──────────────┘    └──────────────┘    └──────────────┘           │  │
│  │                                                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│             ▼                   ▼                   ▼                       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                     PERSISTENT STORAGE                                │  │
│  │                                                                        │  │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐           │  │
│  │   │  PostgreSQL  │    │    JSON      │    │   PyTorch    │           │  │
│  │   │   (Neon)     │    │  Configs     │    │   Models     │           │  │
│  │   └──────────────┘    └──────────────┘    └──────────────┘           │  │
│  │                                                                        │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow Summary

The onboarding process follows a strict sequential flow where each stage depends on the successful completion of the previous stage:

1. **User Input** → Machine metadata and sensor configurations arrive via the frontend form
2. **Validation** → AI validates sensor parameters against physics and domain knowledge
3. **Pattern Generation** → Synthetic anomaly patterns are created for training data
4. **Dataset Creation** → Realistic time-series data is generated using the patterns
5. **Normalization** → Data is standardized for optimal model training
6. **Model Training** → LSTM Autoencoder learns normal behavior patterns
7. **Registration** → Machine record is persisted to the database
8. **Linking** → Machine is connected to its technical documentation

---

## 3. Machine Registration Process

### 3.1 Machine Data Model

Each machine in the Industrial Copilot system is represented by a database record containing four essential fields:

```
┌─────────────────────────────────────────────────────────────────┐
│                      MACHINE ENTITY                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │  machine_id   │  Primary identifier (e.g., "PUMP-001") │   │
│   ├───────────────┼─────────────────────────────────────────┤   │
│   │  name         │  Descriptive name for operators         │   │
│   ├───────────────┼─────────────────────────────────────────┤   │
│   │  location     │  Physical deployment location           │   │
│   ├───────────────┼─────────────────────────────────────────┤   │
│   │  manual_id    │  Link to technical documentation        │   │
│   └───────────────┴─────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

**Field Descriptions:**

- **machine_id**: A unique string identifier that follows a naming convention such as "PUMP-001", "MOTOR-002", or "CONVEYOR-003". This identifier is used throughout the system to reference the specific machine.

- **name**: A human-readable descriptive name that operators see in the user interface, such as "Zynaptrix-9000 Turbo Pump" or "Precision CNC Lathe Model X2".

- **location**: The physical location within the facility where the machine is deployed. This helps operators quickly locate the machine when maintenance is required. Examples include "Hall A - Section 4" or "Building B - Production Line 3".

- **manual_id**: A string that links this machine to its technical documentation in the knowledge base. Multiple machines of the same type can share a single manual_id, enabling efficient documentation management.

### 3.2 Registration API Endpoint

The machine registration process is initiated through a single HTTP endpoint that accepts multipart form data. This design allows simultaneous upload of machine metadata and sensor datasheets in a single request.

**Endpoint Specification:**

| Property | Value |
|----------|-------|
| Route | POST /api/machines |
| Content-Type | multipart/form-data |
| Authentication | Required (API key or session token) |
| Response | JSON object containing the registered machine |

### 3.3 Eight-Step Registration Workflow

The backend processes each registration request through eight sequential stages:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REGISTRATION WORKFLOW (8 STEPS)                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   STEP 1: PARSE FORM DATA                                                   │
│   ─────────────────────────                                                 │
│   │ Extract machine_id, name, location, manual_id                           │
│   │ Extract sensors array (JSON)                                            │
│   │ Collect datasheet PDF files (if any)                                    │
│   ▼                                                                         │
│                                                                             │
│   STEP 2: SENSOR CONFIGURATION EXTRACTION                                   │
│   ───────────────────────────────────────                                   │
│   │ For each sensor with uploaded PDF:                                      │
│   │   → Parse PDF using DatasheetParser                                     │
│   │   → Extract operating ranges via GPT-4o                                 │
│   │ For sensors without PDF:                                                │
│   │   → Estimate parameters from sensor name                                │
│   ▼                                                                         │
│                                                                             │
│   STEP 3: AI CROSS-VALIDATION                                               │
│   ───────────────────────────                                               │
│   │ Validate sensor relationships for physics consistency                   │
│   │ Detect machine type from name (pump/motor/conveyor/etc.)                │
│   │ Generate cross-validation metadata                                      │
│   ▼                                                                         │
│                                                                             │
│   STEP 4: ANOMALY PATTERN GENERATION                                        │
│   ──────────────────────────────────                                        │
│   │ Create synthetic anomaly patterns for ML training                       │
│   │ Save to: data/processed/anomaly_patterns_{machine_id}.json              │
│   ▼                                                                         │
│                                                                             │
│   STEP 5: PERSIST SENSOR CONFIGURATIONS                                     │
│   ─────────────────────────────────────                                     │
│   │ Save to: data/processed/sensor_configs.json                             │
│   │ Save validation metadata: data/processed/validation_{machine_id}.json   │
│   ▼                                                                         │
│                                                                             │
│   STEP 6: TRIGGER ML PIPELINE (ASYNC)                                       │
│   ───────────────────────────────────                                       │
│   │ Spawn subprocess for dataset generation                                 │
│   │ Spawn subprocess for data normalization                                 │
│   │ Spawn subprocess for model training                                     │
│   ▼                                                                         │
│                                                                             │
│   STEP 7: UPDATE DATABASE                                                   │
│   ────────────────────────                                                  │
│   │ Insert or update Machine record in PostgreSQL                           │
│   │ Enforce unique constraint on machine_id                                 │
│   ▼                                                                         │
│                                                                             │
│   STEP 8: RETURN RESPONSE                                                   │
│   ────────────────────────                                                  │
│   │ Return JSON object with machine details                                 │
│   │ Include success status and any warnings                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Additional Machine Management Endpoints

Beyond registration, the API provides endpoints for comprehensive machine lifecycle management:

| Operation | HTTP Method | Route | Description |
|-----------|-------------|-------|-------------|
| List All | GET | /api/machines | Retrieves all registered machines |
| Get Single | GET | /api/machines/{machine_id} | Retrieves a specific machine by ID |
| Decommission | POST | /api/machines/delete/{machine_id} | Removes a machine from active monitoring |

---

## 4. Sensor Configuration and Validation

### 4.1 Sensor Data Model

Each sensor attached to a machine is characterized by a comprehensive configuration that defines its operational parameters:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SENSOR CONFIGURATION SCHEMA                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   IDENTIFICATION                                                            │
│   ──────────────                                                            │
│   sensor_id        │  Unique identifier (e.g., "pressure_01")               │
│   sensor_name      │  Display name (e.g., "Discharge Pressure")             │
│   unit             │  Measurement unit (e.g., "bar", "°C", "mm/s")          │
│                                                                             │
│   OPERATING PARAMETERS                                                      │
│   ────────────────────                                                      │
│   mu (μ)           │  Mean value during normal operation                    │
│   sigma (σ)        │  Standard deviation during normal operation            │
│   min_normal       │  Lower bound of normal range (μ - 2σ typically)        │
│   max_normal       │  Upper bound of normal range (μ + 2σ typically)        │
│                                                                             │
│   FAULT THRESHOLDS                                                          │
│   ────────────────                                                          │
│   fault_high       │  High threshold triggering fault alert                 │
│   fault_low        │  Low threshold triggering fault alert                  │
│                                                                             │
│   UI METADATA                                                               │
│   ───────────                                                               │
│   icon_type        │  Dashboard icon (pressure/temperature/vibration/etc.)  │
│                                                                             │
│   VALIDATION STATUS                                                         │
│   ─────────────────                                                         │
│   ai_validation    │  Object containing:                                    │
│                    │    - is_valid: Boolean                                 │
│                    │    - confidence: Float (0.0 - 1.0)                     │
│                    │    - issues: Array of detected problems                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Datasheet Parsing Process

When operators upload sensor datasheets (PDF documents from manufacturers), the system automatically extracts operating parameters using AI-powered document understanding:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATASHEET PARSING PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   INPUT: Sensor datasheet PDF                                               │
│   ─────                                                                     │
│                                                                             │
│      ┌──────────────────┐                                                   │
│      │  sensor_001.pdf  │                                                   │
│      │  (Manufacturer   │                                                   │
│      │   Datasheet)     │                                                   │
│      └────────┬─────────┘                                                   │
│               │                                                             │
│               ▼                                                             │
│      ┌──────────────────┐                                                   │
│      │  PDF Text        │                                                   │
│      │  Extraction      │                                                   │
│      └────────┬─────────┘                                                   │
│               │                                                             │
│               ▼                                                             │
│      ┌──────────────────────────────────────────────────────────────┐       │
│      │                    GPT-4o ANALYSIS                            │       │
│      │                                                               │       │
│      │  Prompt: "Extract the following parameters from this         │       │
│      │           sensor datasheet:                                   │       │
│      │           - Normal operating range                            │       │
│      │           - Maximum safe operating value                      │       │
│      │           - Minimum safe operating value                      │       │
│      │           - Unit of measurement                               │       │
│      │           - Accuracy specifications"                          │       │
│      │                                                               │       │
│      └────────┬─────────────────────────────────────────────────────┘       │
│               │                                                             │
│               ▼                                                             │
│      ┌──────────────────┐                                                   │
│      │  EXTRACTED       │                                                   │
│      │  PARAMETERS      │                                                   │
│      │  ─────────────   │                                                   │
│      │  mu: 25.0        │                                                   │
│      │  sigma: 2.5      │                                                   │
│      │  fault_high: 45  │                                                   │
│      │  fault_low: 10   │                                                   │
│      │  unit: "bar"     │                                                   │
│      └──────────────────┘                                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3 AI Cross-Validation

After individual sensor parameters are extracted, the AI Automation Engineer validates the entire sensor suite for physical consistency. This step catches configuration errors that would otherwise produce meaningless anomaly detection:

**Validation Checks Performed:**

1. **Physical Plausibility**: Are the values within realistic ranges for the sensor type?
2. **Inter-Sensor Consistency**: Do related sensors have compatible ranges (e.g., inlet pressure < outlet pressure for a pump)?
3. **Machine Type Inference**: Based on the machine name and sensor types, what kind of equipment is this?
4. **Threshold Sanity**: Are fault thresholds appropriately distant from normal ranges?

**Example Validation Issues Detected:**

| Issue Type | Example |
|------------|---------|
| Range Violation | Temperature sensor with max_normal = 500°C for a standard bearing |
| Relationship Error | Discharge pressure lower than suction pressure for a pump |
| Missing Critical | Motor without current or vibration monitoring |
| Threshold Overlap | fault_low = 25 when min_normal = 20 (too close) |

### 4.4 Fallback Estimation

When no datasheet is provided, the system estimates reasonable parameters based on the sensor name and common industrial standards:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FALLBACK ESTIMATION LOGIC                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SENSOR NAME ANALYSIS                                                      │
│   ────────────────────                                                      │
│                                                                             │
│   "Discharge Pressure"                                                      │
│        │                                                                    │
│        ├─ Keywords: pressure, discharge                                     │
│        ├─ Inferred type: Pressure sensor                                    │
│        ├─ Estimated unit: bar or psi                                        │
│        └─ Default ranges:                                                   │
│             mu: 25.0, sigma: 3.0                                            │
│             fault_high: 45, fault_low: 5                                    │
│                                                                             │
│   "Bearing Temperature"                                                     │
│        │                                                                    │
│        ├─ Keywords: temperature, bearing                                    │
│        ├─ Inferred type: Temperature sensor                                 │
│        ├─ Estimated unit: °C                                                │
│        └─ Default ranges:                                                   │
│             mu: 55.0, sigma: 8.0                                            │
│             fault_high: 85, fault_low: 20                                   │
│                                                                             │
│   "Motor Vibration"                                                         │
│        │                                                                    │
│        ├─ Keywords: vibration, motor                                        │
│        ├─ Inferred type: Vibration sensor                                   │
│        ├─ Estimated unit: mm/s                                              │
│        └─ Default ranges:                                                   │
│             mu: 2.5, sigma: 0.5                                             │
│             fault_high: 7.0, fault_low: 0.0                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. ML Pipeline Training

### 5.1 Training Pipeline Overview

Once sensor configurations are validated, the system automatically trains a machine-specific anomaly detection model. This process runs asynchronously to avoid blocking the registration response:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ML TRAINING PIPELINE                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────┐                                                      │
│   │  SENSOR CONFIGS  │                                                      │
│   │  (JSON)          │                                                      │
│   └────────┬─────────┘                                                      │
│            │                                                                │
│            ▼                                                                │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                     STEP 1: DATASET GENERATION                      │    │
│   │                     ─────────────────────────                       │    │
│   │                                                                     │    │
│   │   • Reads sensor configs (mu, sigma, fault thresholds)              │    │
│   │   • Reads AI-generated anomaly patterns                             │    │
│   │   • Generates 10,000+ realistic time-series samples                 │    │
│   │   • Injects anomalies at realistic frequencies                      │    │
│   │   • Output: data/processed/{machine_id}_training.csv                │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│            │                                                                │
│            ▼                                                                │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                     STEP 2: DATA NORMALIZATION                      │    │
│   │                     ─────────────────────────                       │    │
│   │                                                                     │    │
│   │   • Z-score normalization: (x - μ) / σ                              │    │
│   │   • Per-sensor standardization                                      │    │
│   │   • Saves scaler parameters for inference                           │    │
│   │   • Output: data/processed/{machine_id}_normalized.csv              │    │
│   │   • Scaler: models/{machine_id}_scaler.pkl                          │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│            │                                                                │
│            ▼                                                                │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                     STEP 3: MODEL TRAINING                          │    │
│   │                     ─────────────────────                           │    │
│   │                                                                     │    │
│   │   Architecture: LSTM Autoencoder                                    │    │
│   │   ─────────────────────────────────                                 │    │
│   │                                                                     │    │
│   │   Input Layer (N sensors × T timesteps)                             │    │
│   │        │                                                            │    │
│   │        ▼                                                            │    │
│   │   LSTM Encoder (128 units)                                          │    │
│   │        │                                                            │    │
│   │        ▼                                                            │    │
│   │   Latent Space (32 dimensions)                                      │    │
│   │        │                                                            │    │
│   │        ▼                                                            │    │
│   │   LSTM Decoder (128 units)                                          │    │
│   │        │                                                            │    │
│   │        ▼                                                            │    │
│   │   Output Layer (reconstruction)                                     │    │
│   │                                                                     │    │
│   │   Loss: MSE (reconstruction error)                                  │    │
│   │   Threshold: 95th percentile of training MSE                        │    │
│   │                                                                     │    │
│   │   Output: models/{machine_id}_autoencoder.pth                       │    │
│   │                                                                     │    │
│   └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5.2 AI-Generated Anomaly Patterns

A unique feature of our training pipeline is the use of AI to generate realistic anomaly patterns. Instead of relying solely on random noise injection, GPT-4o analyzes the machine type and sensor configuration to create domain-appropriate failure scenarios:

**Example Anomaly Patterns for a Centrifugal Pump:**

| Pattern Name | Affected Sensors | Behavior |
|--------------|------------------|----------|
| Cavitation | Suction pressure, vibration | Pressure drops with high-frequency vibration spikes |
| Bearing Wear | Temperature, vibration | Gradual temperature rise with increasing vibration amplitude |
| Seal Leak | Discharge pressure, flow | Pressure decay with flow reduction |
| Impeller Damage | Current, vibration, pressure | Current spikes, erratic vibration, pressure fluctuation |
| Motor Overload | Current, temperature | Current exceeds normal, temperature rises steadily |

These patterns are saved to JSON files and used by the dataset generator to inject realistic anomalies into the synthetic training data.

### 5.3 Training Artifacts

Upon completion, the ML pipeline produces the following artifacts:

| Artifact | Path | Purpose |
|----------|------|---------|
| Training Data | data/processed/{machine_id}_training.csv | Raw generated data |
| Normalized Data | data/processed/{machine_id}_normalized.csv | Standardized data |
| Scaler | models/{machine_id}_scaler.pkl | Parameters for normalizing new data |
| Model Weights | models/{machine_id}_autoencoder.pth | Trained LSTM Autoencoder |
| Threshold | models/{machine_id}_threshold.json | Anomaly detection threshold |
| Anomaly Patterns | data/processed/anomaly_patterns_{machine_id}.json | AI-generated patterns |

---

## 6. Knowledge Base Integration

### 6.1 Manual Ingestion Process

Technical documentation for each machine type must be ingested into the knowledge base before the RAG system can provide contextual assistance. This is a separate process from machine registration but is essential for full functionality.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE BASE INTEGRATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        MANUAL UPLOAD                                        │
│                        ─────────────                                        │
│                                                                             │
│   ┌──────────────────┐     ┌──────────────────┐                             │
│   │  Technical PDF   │────▶│  /ingest-manual  │                             │
│   │  (100+ pages)    │     │  API Endpoint    │                             │
│   └──────────────────┘     └────────┬─────────┘                             │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    RAG INGESTION PIPELINE                            │   │
│   │                                                                      │   │
│   │   ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐             │   │
│   │   │  Parse  │──▶│  Chunk  │──▶│ Enrich  │──▶│  Embed  │             │   │
│   │   └─────────┘   └─────────┘   └─────────┘   └─────────┘             │   │
│   │       │             │             │             │                    │   │
│   │       ▼             ▼             ▼             ▼                    │   │
│   │   Layout        Semantic      Caption       Vector                   │   │
│   │   Detection     Splitting     Generation    Storage                  │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │                    POSTGRESQL + PGVECTOR                             │   │
│   │                                                                      │   │
│   │   manual_chunks table:                                               │   │
│   │   ┌────────────┬────────┬─────────────┬───────────┬──────────┐      │   │
│   │   │ manual_id  │  type  │   content   │ embedding │   page   │      │   │
│   │   ├────────────┼────────┼─────────────┼───────────┼──────────┤      │   │
│   │   │ Pump_9000  │  text  │ "..."       │ [1536-d]  │    5     │      │   │
│   │   │ Pump_9000  │ image  │ "caption"   │ [1536-d]  │    12    │      │   │
│   │   │ Pump_9000  │ table  │ "summary"   │ [1536-d]  │    23    │      │   │
│   │   └────────────┴────────┴─────────────┴───────────┴──────────┘      │   │
│   │                                                                      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Ingestion Endpoint

| Property | Value |
|----------|-------|
| Route | POST /ingest-manual |
| Content-Type | multipart/form-data |
| Parameters | manual_id (string), file (PDF) |
| Response | Success message with chunk count |

### 6.3 Chunk Types Created

The ingestion pipeline creates three types of searchable chunks from each PDF:

| Type | Content | Embedding Source |
|------|---------|------------------|
| **Text** | Paragraphs and sections from the document | Direct text content |
| **Image** | AI-generated captions describing diagrams | Caption text |
| **Table** | AI-generated summaries of data tables | Summary text |

All chunks share the same 1536-dimensional embedding space, enabling unified semantic search across text, images, and tables.

---

## 7. Machine-to-Documentation Linkage

### 7.1 Linkage Mechanism

The connection between machines and their documentation is established through a string-based foreign key relationship:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MACHINE-DOCUMENTATION LINKAGE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                         MACHINES TABLE                              │    │
│   ├────────────┬────────────────────┬─────────────┬──────────────────┤    │
│   │ machine_id │       name         │  location   │    manual_id     │    │
│   ├────────────┼────────────────────┼─────────────┼──────────────────┤    │
│   │ PUMP-001   │ Turbo Pump 9000    │ Hall A      │ → Zynaptrix_9000 │    │
│   │ PUMP-002   │ Turbo Pump 9000    │ Hall B      │ → Zynaptrix_9000 │    │
│   │ MOTOR-001  │ Drive Motor X2     │ Section 4   │ → Motor_X2_v3    │    │
│   └────────────┴────────────────────┴─────────────┴─────────┬────────┘    │
│                                                              │             │
│                                                              │             │
│                           LINKAGE                            │             │
│                              │                               │             │
│                              ▼                               │             │
│   ┌────────────────────────────────────────────────────────────────────┐    │
│   │                      MANUAL_CHUNKS TABLE                            │    │
│   ├──────────────────┬────────┬─────────────┬───────────┬──────────┐    │
│   │    manual_id     │  type  │   content   │ embedding │   page   │    │
│   ├──────────────────┼────────┼─────────────┼───────────┼──────────┤    │
│   │ ← Zynaptrix_9000 │  text  │ "..."       │ [1536-d]  │    5     │    │
│   │   Zynaptrix_9000 │ image  │ "caption"   │ [1536-d]  │    12    │    │
│   │   Motor_X2_v3    │  text  │ "..."       │ [1536-d]  │    8     │    │
│   └──────────────────┴────────┴─────────────┴───────────┴──────────┘    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Flexible Association Patterns

This design supports multiple common industrial scenarios:

**Pattern 1: Multiple Machines, Single Manual**

Multiple machines of the same model can share documentation:

- PUMP-001 (Hall A) → Zynaptrix_9000
- PUMP-002 (Hall B) → Zynaptrix_9000
- PUMP-003 (Hall C) → Zynaptrix_9000

All three pumps retrieve from the same knowledge base.

**Pattern 2: One Machine, One Manual**

Critical or unique equipment may have dedicated documentation:

- TURBINE-001 → Turbine_Unit_1_Custom

**Pattern 3: Documentation-First**

Manuals can be ingested before machines are registered. When machines are later added, they simply reference the existing manual_id.

### 7.3 Provenance Checking

Before retrieving documentation, the system verifies that the referenced manual exists in the knowledge base:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROVENANCE CHECK FLOW                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   User Query                                                                │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 1. Get machine_id from query context                                │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 2. Lookup: SELECT manual_id FROM machines WHERE machine_id = ?      │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────────────────────────────────────────────────────────────┐   │
│   │ 3. Check: SELECT COUNT(*) FROM manual_chunks WHERE manual_id = ?    │   │
│   └─────────────────────────────────────────────────────────────────────┘   │
│        │                                                                    │
│        ├────────────────────────────────────────┐                           │
│        │                                        │                           │
│        ▼ (count > 0)                            ▼ (count = 0)               │
│   ┌─────────────────┐                    ┌─────────────────────────────┐   │
│   │   PROCEED WITH  │                    │   ADD DISCLAIMER:            │   │
│   │   RAG RETRIEVAL │                    │   "Manual not yet ingested.  │   │
│   └─────────────────┘                    │    Response based on general │   │
│                                          │    knowledge only."          │   │
│                                          └─────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

This ensures operators always know when responses are grounded in machine-specific documentation versus general AI knowledge.

---

## 8. Frontend User Experience

### 8.1 Machine Registry Page

The frontend provides a dedicated interface for machine management at the `/machines` route:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MACHINE REGISTRY PAGE LAYOUT                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────┐  ┌────────────────────────────────┐   │
│  │     REGISTRATION FORM           │  │      FLEET REGISTRY TABLE       │   │
│  │     (Left Panel)                │  │      (Right Panel)              │   │
│  │                                 │  │                                 │   │
│  │  ┌───────────────────────────┐  │  │  ┌───────────────────────────┐  │   │
│  │  │ Machine ID                │  │  │  │ Asset ID │ Name  │ Status │  │   │
│  │  │ [PUMP-001              ]  │  │  │  ├──────────┼───────┼────────┤  │   │
│  │  └───────────────────────────┘  │  │  │ PUMP-001 │ Turbo │   🟢   │  │   │
│  │                                 │  │  │ PUMP-002 │ Turbo │   🟢   │  │   │
│  │  ┌───────────────────────────┐  │  │  │ MOTOR-01 │ Drive │   🟡   │  │   │
│  │  │ Machine Name              │  │  │  │ CONV-001 │ Belt  │   🔴   │  │   │
│  │  │ [Turbo Pump 9000       ]  │  │  │  └───────────────────────────┘  │   │
│  │  └───────────────────────────┘  │  │                                 │   │
│  │                                 │  │  Actions: [Edit] [Delete]       │   │
│  │  ┌───────────────────────────┐  │  │                                 │   │
│  │  │ 📍 Location               │  │  └────────────────────────────────┘   │
│  │  │ [Hall A - Section 4    ]  │  │                                       │
│  │  └───────────────────────────┘  │                                       │
│  │                                 │                                       │
│  │  ┌───────────────────────────┐  │                                       │
│  │  │ 📚 Target Manual ID       │  │                                       │
│  │  │ [Zynaptrix_9000        ]  │  │                                       │
│  │  └───────────────────────────┘  │                                       │
│  │                                 │                                       │
│  │  ─────── SENSORS ───────────   │                                       │
│  │                                 │                                       │
│  │  ┌───────────────────────────┐  │                                       │
│  │  │ Sensor 1: [pressure_01]   │  │                                       │
│  │  │ Name: [Discharge Press ]  │  │                                       │
│  │  │ 📄 [Upload Datasheet]     │  │                                       │
│  │  └───────────────────────────┘  │                                       │
│  │                                 │                                       │
│  │  [+ Add Sensor]                 │                                       │
│  │                                 │                                       │
│  │  [🚀 ENROLL MACHINE]            │                                       │
│  │                                 │                                       │
│  └─────────────────────────────────┘                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.2 Knowledge Management Page

A separate page at `/ingestion` handles documentation uploads:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE MANAGEMENT PAGE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     UPLOAD TECHNICAL MANUAL                           │   │
│  │                                                                       │   │
│  │   ┌────────────────────────────────────────────────────────────┐     │   │
│  │   │  Manual ID                                                  │     │   │
│  │   │  [Zynaptrix_9000                                         ]  │     │   │
│  │   │                                                             │     │   │
│  │   │  (This ID will be used to link machines to this manual)    │     │   │
│  │   └────────────────────────────────────────────────────────────┘     │   │
│  │                                                                       │   │
│  │   ┌────────────────────────────────────────────────────────────┐     │   │
│  │   │                                                             │     │   │
│  │   │    ┌─────────────────────────────────────────────────┐     │     │   │
│  │   │    │                                                  │     │     │   │
│  │   │    │    📄 Drop PDF here or click to browse          │     │     │   │
│  │   │    │                                                  │     │     │   │
│  │   │    │    Supported: PDF files up to 50MB              │     │     │   │
│  │   │    │                                                  │     │     │   │
│  │   │    └─────────────────────────────────────────────────┘     │     │   │
│  │   │                                                             │     │   │
│  │   └────────────────────────────────────────────────────────────┘     │   │
│  │                                                                       │   │
│  │   [🔄 INITIALIZE INGESTION]                                           │   │
│  │                                                                       │   │
│  │   Status: Ready                                                       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                     INGESTION PROTOCOL                                │   │
│  │                                                                       │   │
│  │   Step 1: Upload PDF → Layout analysis begins                         │   │
│  │   Step 2: Processing → Images captioned, tables summarized            │   │
│  │   Step 3: Complete → Knowledge base updated                           │   │
│  │                                                                       │   │
│  │   ⚠️  Processing time depends on document size (1-15 minutes)        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 8.3 Redux State Management

The frontend uses Redux Toolkit for state management with the following async operations:

| Action | Purpose | API Call |
|--------|---------|----------|
| fetchMachines | Load fleet registry on page load | GET /api/machines |
| registerMachine | Submit new machine registration | POST /api/machines |
| deleteMachine | Decommission a machine | POST /api/machines/delete/{id} |
| fetchMachineConfig | Get detailed sensor configurations | GET /api/machines/{id}/config |

---

## 9. End-to-End Workflow Example

### 9.1 Scenario: Adding a New CNC Lathe

This section walks through the complete process of onboarding a new CNC lathe machine from initial registration to first diagnostic query.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE ONBOARDING WORKFLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PHASE 1: MACHINE REGISTRATION                                             │
│   ─────────────────────────────                                             │
│                                                                             │
│   Operator navigates to /machines and fills the registration form:          │
│                                                                             │
│   • Machine ID: LATHE-002                                                   │
│   • Name: Precision Lathe X2                                                │
│   • Location: Hall B - Machine Shop                                         │
│   • Manual ID: lathe_manual_v3                                              │
│   • Sensors:                                                                │
│       - Spindle Temperature (spindle_temp) + datasheet.pdf                  │
│       - Vibration Z-Axis (vibration_z) [no datasheet]                       │
│                                                                             │
│   Clicks "Enroll Machine"                                                   │
│                                                                             │
│   ──────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   PHASE 2: BACKEND PROCESSING                                               │
│   ───────────────────────────                                               │
│                                                                             │
│   [1] Form data received by POST /api/machines                              │
│   [2] DatasheetParser extracts temperature limits from PDF                  │
│   [3] Vibration sensor gets estimated defaults (no PDF provided)            │
│   [4] AI Automation Engineer validates sensor relationships:                │
│       ✓ Temperature monitoring appropriate for spindle                      │
│       ✓ Vibration monitoring appropriate for rotating machinery             │
│       ✓ Machine type detected: CNC Lathe                                    │
│   [5] Anomaly patterns generated (overheating, tool chatter, bearing wear)  │
│   [6] ML pipeline spawned asynchronously:                                   │
│       → Dataset: 10,000 synthetic samples with realistic patterns           │
│       → Normalization: Z-score per sensor                                   │
│       → Training: LSTM Autoencoder (5 epochs, ~2 minutes)                   │
│   [7] Machine record inserted into database                                 │
│   [8] Response returned to frontend                                         │
│                                                                             │
│   ──────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   PHASE 3: MANUAL UPLOAD                                                    │
│   ──────────────────────                                                    │
│                                                                             │
│   Operator navigates to /ingestion:                                         │
│                                                                             │
│   • Manual ID: lathe_manual_v3                                              │
│   • File: precision_lathe_manual_2024.pdf (150 pages)                       │
│                                                                             │
│   Clicks "Initialize Ingestion"                                             │
│                                                                             │
│   ──────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   PHASE 4: INGESTION PIPELINE                                               │
│   ───────────────────────────                                               │
│                                                                             │
│   [1] PDF parsed with YOLOv8 DocLayNet (layout detection)                   │
│   [2] Content extracted:                                                    │
│       - 423 text passages                                                   │
│       - 67 diagrams and figures                                             │
│       - 28 specification tables                                             │
│   [3] Semantic chunking with section context preservation                   │
│   [4] Enrichment:                                                           │
│       - 67 images → GPT-4o Vision captions                                  │
│       - 28 tables → GPT-4o-mini summaries                                   │
│   [5] All chunks embedded (text-embedding-3-small, 1536 dimensions)         │
│   [6] 518 total chunks inserted into manual_chunks table                    │
│                                                                             │
│   ──────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   PHASE 5: SYSTEM READY                                                     │
│   ─────────────────────                                                     │
│                                                                             │
│   The lathe is now fully integrated:                                        │
│                                                                             │
│   ✓ Real-time anomaly detection active (LSTM monitoring sensor streams)     │
│   ✓ RAG retrieval linked to 518 documentation chunks                        │
│   ✓ Agents can diagnose problems using machine-specific knowledge           │
│   ✓ Historical memory ready to accumulate resolved incidents                │
│                                                                             │
│   ──────────────────────────────────────────────────────────────────────    │
│                                                                             │
│   PHASE 6: FIRST DIAGNOSTIC QUERY                                           │
│   ─────────────────────────────                                             │
│                                                                             │
│   Day 3: Anomaly detected - spindle temperature elevated                    │
│                                                                             │
│   Operator clicks on alert, asks:                                           │
│   "My lathe spindle is running hot. What should I check?"                   │
│                                                                             │
│   System process:                                                           │
│   [1] machine_id = LATHE-002 (from anomaly context)                         │
│   [2] Lookup: manual_id = lathe_manual_v3                                   │
│   [3] Provenance check: 518 chunks found (manual is ingested)               │
│   [4] Embed query, search manual_chunks                                     │
│   [5] Retrieve: 3 text chunks + 2 diagram captions + 1 troubleshooting      │
│       table about spindle cooling                                           │
│   [6] RAG generates response with:                                          │
│       - Diagnostic checklist (coolant level, chip buildup, bearing)         │
│       - Reference to page 67 spindle cooling diagram                        │
│       - Specification table for temperature limits                          │
│                                                                             │
│   Operator resolves issue, enters notes:                                    │
│   "Cleared chip buildup from coolant intake. Temperature normalized."       │
│                                                                             │
│   [7] Feedback vectorized and stored in interaction_memory                  │
│   [8] Next similar query will retrieve this historical fix                  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 10. Data Architecture

### 10.1 Database Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATABASE ENTITY RELATIONSHIPS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                                                                             │
│       ┌────────────────────┐                                                │
│       │      MACHINES      │                                                │
│       │ ────────────────── │                                                │
│       │ machine_id (PK)    │                                                │
│       │ name               │                                                │
│       │ location           │                                                │
│       │ manual_id (FK)  ───┼──────────────────────┐                         │
│       └─────────┬──────────┘                      │                         │
│                 │                                 │                         │
│       ┌─────────┴───────────────────┐             │                         │
│       │                             │             │                         │
│       ▼                             ▼             ▼                         │
│   ┌────────────────────┐    ┌────────────────────────────────────────┐     │
│   │  ANOMALY_RECORDS   │    │            MANUAL_CHUNKS                │     │
│   │ ────────────────── │    │ ────────────────────────────────────── │     │
│   │ id (PK)            │    │ id (PK)                                 │     │
│   │ machine_id (FK)    │    │ manual_id (indexed)                     │     │
│   │ timestamp          │    │ type (text/image/table)                 │     │
│   │ type               │    │ content                                 │     │
│   │ score              │    │ embedding (VECTOR 1536)                 │     │
│   │ sensor_data (JSON) │    │ page                                    │     │
│   │ resolved           │    │ path (image file path)                  │     │
│   │ ai_validation_*    │    └────────────────────────────────────────┘     │
│   └─────────┬──────────┘                                                    │
│             │                                                               │
│             │ (1:N)                                                         │
│             ▼                                                               │
│   ┌────────────────────┐                                                    │
│   │   CHAT_MESSAGES    │                                                    │
│   │ ────────────────── │                                                    │
│   │ id (PK)            │                                                    │
│   │ anomaly_id (FK)    │                                                    │
│   │ role (user/agent)  │                                                    │
│   │ content            │                                                    │
│   │ timestamp          │                                                    │
│   └────────────────────┘                                                    │
│                                                                             │
│                                                                             │
│   ┌────────────────────────────────────────┐                                │
│   │        INTERACTION_MEMORY              │                                │
│   │ ────────────────────────────────────── │                                │
│   │ id (PK)                                │                                │
│   │ machine_id (indexed)                   │                                │
│   │ manual_id                              │                                │
│   │ summary                                │                                │
│   │ operator_fix                           │                                │
│   │ embedding (VECTOR 1536)                │                                │
│   │ timestamp                              │                                │
│   └────────────────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 10.2 File-Based Storage

In addition to database storage, certain artifacts are stored as files for performance and flexibility:

| File Path | Content | Format |
|-----------|---------|--------|
| data/processed/sensor_configs.json | All sensor configurations keyed by machine_id | JSON |
| data/processed/anomaly_patterns_{machine_id}.json | AI-generated anomaly patterns | JSON |
| data/processed/validation_{machine_id}.json | AI validation results | JSON |
| models/{machine_id}_autoencoder.pth | Trained LSTM Autoencoder weights | PyTorch |
| models/{machine_id}_scaler.pkl | Data normalization parameters | Pickle |
| models/{machine_id}_threshold.json | Anomaly detection threshold | JSON |
| data/extracted_images/{manual_id}/ | Extracted images from PDFs | PNG/JPG |

### 10.3 Embedding Strategy

All embeddings in the system use the same model and dimensionality to enable unified vector search:

| Content | Embedding Model | Dimensions | Storage |
|---------|-----------------|------------|---------|
| Text chunks | text-embedding-3-small | 1536 | pgvector |
| Image captions | text-embedding-3-small | 1536 | pgvector |
| Table summaries | text-embedding-3-small | 1536 | pgvector |
| Historical fixes | text-embedding-3-small | 1536 | pgvector |
| User queries | text-embedding-3-small | 1536 | Runtime only |

This unified embedding space allows a single query to simultaneously search across text documentation, image descriptions, table summaries, and historical fix records.

---

## 11. Configuration Reference

### 11.1 Environment Variables

The following environment variables control the machine onboarding and training system:

**Required Variables:**

| Variable | Description | Example |
|----------|-------------|---------|
| OPENAI_API_KEY | API key for GPT-4o and embeddings | sk-... |
| DATABASE_URL | PostgreSQL connection string | postgresql://user:pass@host:port/db |

**Optional Tuning Variables:**

| Variable | Description | Default |
|----------|-------------|---------|
| ML_TRAINING_EPOCHS | LSTM Autoencoder training epochs | 5 |
| ML_BATCH_SIZE | Training batch size | 32 |
| ML_SEQUENCE_LENGTH | Time-series window size | 20 |
| SYNTHETIC_DATA_SIZE | Number of training samples | 10000 |
| ANOMALY_THRESHOLD_PERCENTILE | MSE percentile for threshold | 95 |

### 11.2 Directory Structure

```
industrial_copilot/
├── backend/
│   ├── api/
│   │   └── machine_api.py          # Registration endpoints
│   ├── services/
│   │   └── datasheet_parser.py     # PDF datasheet extraction
│   ├── unified_rag/
│   │   └── db/
│   │       └── models.py           # SQLAlchemy models
│   ├── generate_dataset.py         # Synthetic data generation
│   └── preprocessing/
│       └── normalization.py        # Data standardization
│
├── models/
│   ├── {machine_id}_autoencoder.pth
│   ├── {machine_id}_scaler.pkl
│   └── train_model.py              # LSTM training script
│
├── data/
│   ├── processed/
│   │   ├── sensor_configs.json
│   │   ├── anomaly_patterns_{machine_id}.json
│   │   └── {machine_id}_training.csv
│   └── extracted_images/
│       └── {manual_id}/
│
└── frontend/
    └── src/
        └── app/
            ├── machines/
            │   └── page.tsx        # Machine registry UI
            └── ingestion/
                └── page.tsx        # Manual upload UI
```

### 11.3 API Rate Limits and Performance

| Operation | Typical Duration | Notes |
|-----------|------------------|-------|
| Machine Registration (without ML) | 5-10 seconds | Datasheet parsing with GPT-4o |
| ML Training Pipeline | 2-5 minutes | Runs asynchronously |
| Manual Ingestion (100 pages) | 10-15 minutes | Depends on image count |
| Single Sensor Datasheet Parse | 3-5 seconds | GPT-4o API call |

---

## Summary

The Machine Onboarding and Training Process is the critical prerequisite that enables the Industrial Copilot to function effectively. By following this comprehensive workflow:

1. **Machines gain identity** through unique IDs and location tracking
2. **Sensors are calibrated** using AI-powered datasheet extraction or intelligent defaults
3. **Anomaly detection becomes possible** through machine-specific LSTM Autoencoders
4. **Documentation is searchable** via the multimodal RAG knowledge base
5. **Historical learning accumulates** as incidents are resolved and vectorized

This foundation enables the multi-agent system to provide context-aware, machine-specific diagnostic assistance that bridges the gap between raw sensor alerts and actionable maintenance procedures.

---

*Document Version: 1.0*
*Last Updated: April 2026*
*Team Zynaptrix - University of Kelaniya*
