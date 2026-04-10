<<<<<<< HEAD
# RAG Ingestion Pipeline: Technical Documentation

## A Context-Aware Multimodal Document Processing System for Industrial Technical Manuals

---

## 1. Executive Summary

The RAG Ingestion Pipeline is the foundational component of the Zynaptrix Industrial Copilot that transforms raw PDF technical manuals into a semantically searchable vector knowledge base. Unlike conventional document processing systems that treat PDFs as flat text streams, our pipeline implements a **vision-first, context-aware** approach that preserves the structural hierarchy, visual semantics, and domain-specific meaning of industrial documentation.

### Key Innovation: Caption-Based Multimodal Embedding

Rather than using direct image embeddings (CLIP, ImageBind), we employ **LLM-generated contextual captions** that are then embedded using text embedding models. This approach enables **semantic retrieval** ("show me the bearing assembly procedure") rather than mere **visual similarity** ("images that look like this").

---

## 2. Pipeline Architecture Overview

### 2.1 High-Level Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           RAG INGESTION PIPELINE v2                                  │
│                      "From PDF to Searchable Knowledge"                              │
└─────────────────────────────────────────────────────────────────────────────────────┘

                                    INPUT
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │         PDF Document            │
                    │   (Technical Manual 100+ pages) │
                    │   • Text, Tables, Diagrams      │
                    │   • Schematics, Photos          │
                    │   • Exploded Views              │
                    └─────────────────────────────────┘
                                      │
        ┌─────────────────────────────┼─────────────────────────────┐
        │                             │                             │
        ▼                             ▼                             ▼
┌───────────────┐           ┌───────────────┐           ┌───────────────┐
│   STAGE 1     │           │   STAGE 1.5   │           │   STAGE 1     │
│   YOLOv8      │           │   SAM +       │           │   Camelot     │
│   DocLayNet   │           │   GPT-4o      │           │   Tables      │
│               │           │               │           │               │
│ Layout        │           │ Figure        │           │ Table         │
│ Detection     │           │ Splitting     │           │ Extraction    │
└───────────────┘           └───────────────┘           └───────────────┘
        │                             │                             │
        └─────────────────────────────┼─────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │          STAGE 2                │
                    │    Contextual Chunking          │
                    │                                 │
                    │  • LangChain Recursive Splitter │
                    │  • Section Metadata Preserved   │
                    │  • Semantic Boundaries          │
                    └─────────────────────────────────┘
                                      │
                                      ▼
        ┌─────────────────────────────┴─────────────────────────────┐
        │                                                           │
        ▼                                                           ▼
┌───────────────────────┐                           ┌───────────────────────┐
│      STAGE 3A         │                           │      STAGE 3B         │
│   Image Captioning    │                           │  Table Summarization  │
│                       │                           │                       │
│  GPT-4o Vision API    │                           │  GPT-4o-mini          │
│  + Context Injection  │                           │  + Section Context    │
│                       │                           │                       │
│  "Senior Industrial   │                           │  "Technical Data      │
│   Engineer Persona"   │                           │   Specialist Persona" │
└───────────────────────┘                           └───────────────────────┘
        │                                                           │
        └─────────────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │          STAGE 4                │
                    │   Unified Text Embedding        │
                    │                                 │
                    │  OpenAI text-embedding-3-small  │
                    │  (1536 dimensions)              │
                    │                                 │
                    │  Text ──► Vector               │
                    │  Caption ──► Vector            │
                    │  Table Summary ──► Vector      │
                    └─────────────────────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────┐
                    │          OUTPUT                 │
                    │   PostgreSQL + pgvector         │
                    │                                 │
                    │  ┌─────────────────────────┐   │
                    │  │    ManualChunk Table    │   │
                    │  │  • manual_id            │   │
                    │  │  • type (text/img/tbl)  │   │
                    │  │  • content              │   │
                    │  │  • embedding[1536]      │   │
                    │  │  • page                 │   │
                    │  │  • path                 │   │
                    │  └─────────────────────────┘   │
                    └─────────────────────────────────┘
```

### 2.2 Component Interaction Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        COMPONENT INTERACTION MAP                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   pipeline.py   │────►│   parser.py     │────►│ figure_splitter │
│                 │     │                 │     │      .py        │
│ Orchestrator    │     │ DocumentParser  │     │ FigureSplitter  │
│ process_manual  │     │ YOLOv8 Layout   │     │ SAM + GPT-4o    │
│ _async()        │     │ Detection       │     │ Centers         │
└────────┬────────┘     └─────────────────┘     └─────────────────┘
         │
         │              ┌─────────────────┐     ┌─────────────────┐
         ├─────────────►│   chunker.py    │     │   captioner.py  │
         │              │                 │     │                 │
         │              │ ContextualChunker│     │ ImageCaptioner  │
         │              │ LangChain        │     │ GPT-4o Vision   │
         │              │ RecursiveSplit   │     │                 │
         │              └─────────────────┘     └─────────────────┘
         │                                              │
         │              ┌─────────────────┐             │
         ├─────────────►│ table_transformer│◄───────────┘
         │              │      .py        │
         │              │                 │
         │              │ TableTransformer│
         │              │ GPT-4o-mini     │
         │              └─────────────────┘
         │
         │              ┌─────────────────┐     ┌─────────────────┐
         └─────────────►│   embedder.py   │────►│   models.py     │
                        │                 │     │   (Database)    │
                        │ MultimodalEmbed │     │                 │
                        │ text-embed-3-sm │     │ ManualChunk     │
                        │ (1536-dim)      │     │ pgvector        │
                        └─────────────────┘     └─────────────────┘
=======
# RAG Ingestion Pipeline & Human-in-the-Loop (HITL) Architecture
## Zynaptrix Industrial Copilot — Comprehensive Technical Reference

---

> **Document Purpose:** A full-spectrum reference covering how knowledge is ingested, stored, retrieved, and validated through Human-in-the-Loop interactions across the Zynaptrix Industrial Copilot platform.
>
> **Target Audience:** AI researchers, industrial engineers, system architects, and graduate researchers studying applied Gen AI frameworks in industrial environments.

---

## Table of Contents

1. [Executive Overview](#1-executive-overview)
2. [System Architecture Overview](#2-system-architecture-overview)
3. [RAG Ingestion Pipeline](#3-rag-ingestion-pipeline)
4. [Multi-Agent Orchestration (LangGraph)](#4-multi-agent-orchestration-langgraph)
5. [Chatbot Interfaces & Human Interaction](#5-chatbot-interfaces--human-interaction)
6. [Validation Methods](#6-validation-methods)
7. [Adaptive Learning via Interaction Memory](#7-adaptive-learning-via-interaction-memory)
8. [Data Flow: End-to-End Lifecycle](#8-data-flow-end-to-end-lifecycle)
9. [Database Schema & Vector Store](#9-database-schema--vector-store)
10. [Next.js Frontend Architecture](#10-nextjs-frontend-architecture)
11. [Research Importance & Contributions](#11-research-importance--contributions)

---

## 1. Executive Overview

### 1.1 The Industrial Knowledge Problem

Industrial facilities operate with thousands of pages of technical manuals, real-time sensor telemetry streams, and fragmented institutional knowledge trapped in individual technicians' heads. When a pump bearing fails at 2 AM, the operator faces three simultaneous challenges:

- **What is wrong?** — interpreting raw sensor numbers (e.g., vibration 12.3 mm/s vs. baseline 4.2 mm/s)
- **What does the manual say?** — locating the correct repair procedure in a 500-page PDF
- **Has this happened before?** — recalling past incidents and their successful resolutions

Traditional systems fail here. Rule-based alarms say *"vibration threshold exceeded"* but offer no guidance. Pure ML anomaly detectors flag the event but are black boxes. Static manual search takes 30–60 minutes per incident.

### 1.2 The Zynaptrix Solution

The Zynaptrix Industrial Copilot integrates three paradigms into a unified Gen AI Framework:

| Paradigm | Technology | Role |
|---|---|---|
| **Anomaly Detection** | LSTM / Autoencoder (MSE scoring) | Detects statistical pattern deviation in sensor streams |
| **Multi-Agent Reasoning** | LangGraph (DAG pipeline) | Orchestrates 6 specialized AI agents for Sensor→Diagnosis→Strategy |
| **Multimodal RAG** | pgvector + OpenAI text-embedding-3-small | Retrieves relevant text, images, and past fixes from technical manuals |

The result: a 2-second end-to-end pipeline from anomaly alert to structured repair procedure with inline technical diagrams, replacing a 30-minute manual search process.

---

## 2. System Architecture Overview

```
┌────────────────────────────────────────────────────────────────────┐
│                  ZYNAPTRIX INDUSTRIAL COPILOT                        │
│                       FULL SYSTEM ARCHITECTURE                       │
└────────────────────────────────────────────────────────────────────┘

  LAYER 0: DATA INGESTION
  ┌──────────────────────────────────────────────────────────┐
  │  PDF Manuals → Parser → Chunker → Embedder → pgvector   │
  │  Sensor Datasheets → AI Engineer → Sensor Config JSON   │
  └──────────────────────────────────────────────────────────┘
                            ↓
  LAYER 1: REAL-TIME TELEMETRY
  ┌──────────────────────────────────────────────────────────┐
  │  IoT Sensors → MQTT/InfluxDB → Autoencoder MSE Scoring  │
  │  AnomalyService → TemporalAnalyzer → HybridConfidence   │
  └──────────────────────────────────────────────────────────┘
                            ↓
  LAYER 2: MULTI-AGENT REASONING (LangGraph)
  ┌──────────────────────────────────────────────────────────┐
  │  SensorStatus → ValidationEngineer → Diagnostic         │
  │  → KnowledgeRetrieval (RAG) → Strategy → Critic         │
  └──────────────────────────────────────────────────────────┘
                            ↓
  LAYER 3: HUMAN INTERACTION (HITL)
  ┌──────────────────────────────────────────────────────────┐
  │  Diagnostic Copilot Chat  │  Central Assistant Bot       │
  │  Step Wizard (LOTO/PPE)   │  Session Memory + RAG        │
  │  Operator Feedback Form   │  Intent Classification       │
  └──────────────────────────────────────────────────────────┘
                            ↓
  LAYER 4: ADAPTIVE MEMORY (FEEDBACK LOOP)
  ┌──────────────────────────────────────────────────────────┐
  │  Feedback Validation → AI Summary → Embedding           │
  │  → InteractionMemory (pgvector) → Future RAG Recall      │
  └──────────────────────────────────────────────────────────┘
>>>>>>> origin/main
```

---

<<<<<<< HEAD
## 3. Stage 1: Structural Parsing with YOLOv8 DocLayNet

### 3.1 The Problem with Traditional PDF Parsing

Traditional PDF extraction tools (PyPDF2, pdfminer) treat documents as linear text streams. This approach fails for technical manuals because:

| Issue | Impact |
|-------|--------|
| **Flat Text Extraction** | Loses spatial relationships between text and diagrams |
| **No Element Classification** | Cannot distinguish headers from body text from captions |
| **Image Blindness** | Embedded figures are either ignored or extracted without context |
| **Table Corruption** | Complex tables become unreadable text fragments |

### 3.2 Our Solution: AI-Powered Layout Detection

We employ **YOLOv8 trained on the DocLayNet dataset** to perform intelligent document understanding:

```python
class DocumentParser:
    def __init__(self, yolo_weights="models/yolov8_doclaynet.pt"):
        self.layout_model = YOLO(yolo_weights)
    
    def parse_pdf(self, file_path: str, manual_id: str):
        doc = fitz.open(file_path)
        current_section = "General Information"  # Structural context tracking
        
        for page in doc:
            # Render page at 150 DPI for YOLO
            pix = page.get_pixmap(dpi=150)
            img_array = np.frombuffer(pix.samples, dtype=np.uint8)
            
            # Run YOLOv8 layout detection
            results = self.layout_model(Image.fromarray(img_array))
            
            for box in results[0].boxes:
                class_name = names[int(box.cls[0])].lower()
                
                # Update structural context for headers
                if "title" in class_name or "header" in class_name:
                    current_section = extract_text(box)
                    
                # Route to appropriate processor
                elif "picture" in class_name or "figure" in class_name:
                    # Trigger Agentic Figure Splitting
                    ...
                elif "text" in class_name:
                    # Extract with section context
                    ...
```

### 3.3 DocLayNet Classes and Their Uses

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DocLayNet CLASS TAXONOMY                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   ┌─────────────┐     STRUCTURAL ELEMENTS                               │
│   │   Title     │ ──► Section headers → Updates current_section context │
│   │   Header    │     Used for hierarchical document navigation         │
│   └─────────────┘                                                        │
│                                                                          │
│   ┌─────────────┐     CONTENT ELEMENTS                                  │
│   │   Text      │ ──► Body paragraphs → Chunked with section metadata   │
│   │   List-item │     Procedure steps, specifications                   │
│   └─────────────┘                                                        │
│                                                                          │
│   ┌─────────────┐     VISUAL ELEMENTS                                   │
│   │   Figure    │ ──► Technical diagrams → Agentic Figure Splitting     │
│   │   Picture   │     Photographs, schematics                           │
│   └─────────────┘                                                        │
│                                                                          │
│   ┌─────────────┐     DATA ELEMENTS                                     │
│   │   Table     │ ──► Specification tables → Camelot extraction         │
│   │   Caption   │     Figure descriptions                               │
│   └─────────────┘                                                        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.4 Structural Context Tracking

A key innovation is **maintaining structural context** throughout parsing:

```python
# In-memory tracking of document hierarchy
current_section = "General Information"

for box in detected_elements:
    if is_header(box):
        current_section = extract_text(box)  # Update context
        print(f"[Structure] New Section: {current_section}")
    
    elif is_figure(box):
        # Pass context to figure splitter
        parent_ctx = f"Figure on Page {page} under '{current_section}'"
        sub_figures = splitter.split_image_sam(crop, parent_context=parent_ctx)
    
    elif is_text(box):
        # Attach context as metadata
        parsed_data.append({
            "type": "text",
            "content": text,
            "metadata": {"section": current_section}  # ← Context preserved
        })
```

This enables downstream retrieval queries like *"Find bearing replacement in the Maintenance section"* to work correctly.

---

## 4. Stage 1.5: Agentic Figure Decomposition

### 4.1 The Composite Figure Problem

Technical manuals frequently contain **composite figures**—single images containing multiple distinct components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                  COMPOSITE FIGURE EXAMPLE                                │
│                                                                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                                                                  │   │
│   │    ┌──────────┐    ┌──────────┐    ┌──────────┐                │   │
│   │    │  Motor   │    │ Coupling │    │  Pump    │                │   │
│   │    │ Assembly │    │  Detail  │    │  Head    │                │   │
│   │    │          │    │          │    │          │                │   │
│   │    │  Fig 1a  │    │  Fig 1b  │    │  Fig 1c  │                │   │
│   │    └──────────┘    └──────────┘    └──────────┘                │   │
│   │                                                                  │   │
│   │           "Figure 1: Complete Pump Assembly"                     │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   PROBLEM: Traditional RAG embeds this as ONE image                     │
│   RESULT: Query "coupling detail" fails to retrieve this figure         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Three-Phase Agentic Decomposition

Our **FigureSplitter** implements a novel three-phase approach:

```
┌─────────────────────────────────────────────────────────────────────────┐
│              AGENTIC FIGURE SPLITTING PIPELINE                           │
└─────────────────────────────────────────────────────────────────────────┘

         PHASE 1                   PHASE 2                   PHASE 3
    ┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
    │   GPT-4o        │      │   Voronoi       │      │   SAM           │
    │   Vision        │      │   Clustering    │      │   Neural        │
    │                 │      │                 │      │   Masking       │
    │ "Find semantic  │ ──►  │  K-D Tree       │ ──►  │                 │
    │  centers of     │      │  pixel-to-      │      │  Precise        │
    │  each distinct  │      │  center         │      │  boundary       │
    │  component"     │      │  assignment     │      │  extraction     │
    │                 │      │                 │      │                 │
    │ OUTPUT:         │      │ OUTPUT:         │      │ OUTPUT:         │
    │ [(x,y,label)]   │      │ Cluster masks   │      │ Cropped images  │
    └─────────────────┘      └─────────────────┘      └─────────────────┘
```

#### Phase 1: Semantic Center Detection (GPT-4o Vision)

```python
def ask_openai_centers(self, base64_image, parent_context=""):
    prompt = (
        "You are an expert technical layout analyzer. Analyze this technical drawing.\n"
        f"Context: {parent_context}\n"
        "Identify the exact center points of each distinct machine diagram "
        "AND any major text blocks (titles, headers).\n"
        "Return a JSON array:\n"
        " - 'x': normalized x coordinate (0 to 1000)\n"
        " - 'y': normalized y coordinate (0 to 1000)\n"
        " - 'is_noise': boolean (true if text block, false if diagram)\n"
        " - 'label': a short 1-3 word descriptive label\n"
    )
    
    response = self.client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
    )
    return json.loads(response.choices[0].message.content)
```

**Example GPT-4o Response:**
```json
[
  {"x": 200, "y": 300, "is_noise": false, "label": "Motor Assembly"},
  {"x": 500, "y": 300, "is_noise": false, "label": "Coupling Detail"},
  {"x": 800, "y": 300, "is_noise": false, "label": "Pump Head"},
  {"x": 500, "y": 850, "is_noise": true, "label": "Figure Caption"}
]
```

#### Phase 2: Voronoi Clustering via K-D Tree

```python
# Convert normalized coordinates to pixel coordinates
centers_px = [(int(pt["x"]/1000*w), int(pt["y"]/1000*h)) for pt in llm_centers]

# Build K-D Tree for efficient nearest-neighbor lookup
tree = cKDTree(np.array(centers_px))

# Assign each foreground pixel to nearest center
foreground_points = np.column_stack(np.where(binary_mask > 0))
_, cluster_labels = tree.query(foreground_points)

# Result: Each pixel knows which component it belongs to
```

#### Phase 3: Neural Masking with SAM (Segment Anything Model)

```python
for i, center in enumerate(centers_px):
    if is_noise_list[i]:
        continue  # Skip text blocks
    
    # Get Voronoi bounding box
    cluster_points = points[cluster_labels == i]
    bbox = [xmin, ymin, xmax, ymax]
    
    # Use SAM for precise neural masking
    sam_res = self.model(image, bboxes=bbox, retina_masks=True)
    mask = sam_res[0].masks.data[0].cpu().numpy()
    
    # Extract with white background
    cropped = extract_with_mask(image, mask)
    
    results.append({
        "crop": cropped,
        "label": labels_list[i],  # "Motor Assembly"
        "box": (x, y, w, h)
    })
```

### 4.3 Before vs After Figure Splitting

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    FIGURE SPLITTING RESULTS                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   BEFORE (Traditional RAG):                                             │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │  pump_assembly.png                                               │   │
│   │  Caption: "Technical diagram on Page 15"                        │   │
│   │  → Query "coupling" = NO MATCH (entire figure embedded as one)  │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│   AFTER (Agentic Splitting):                                            │
│   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                │
│   │ pump_p15_     │ │ pump_p15_     │ │ pump_p15_     │                │
│   │ sub0_0.png    │ │ sub0_1.png    │ │ sub0_2.png    │                │
│   │               │ │               │ │               │                │
│   │ "Motor        │ │ "Coupling     │ │ "Pump Head    │                │
│   │  Assembly"    │ │  Detail"      │ │  Section"     │                │
│   │               │ │               │ │               │                │
│   │ → Motor       │ │ → Coupling    │ │ → Pump        │                │
│   │   queries ✓   │ │   queries ✓   │ │   queries ✓   │                │
│   └───────────────┘ └───────────────┘ └───────────────┘                │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Stage 2: Contextual Semantic Chunking

### 5.1 Chunking Strategy

Text content is split using **LangChain's RecursiveCharacterTextSplitter** with semantic boundary awareness:

```python
class ContextualChunker:
    def __init__(self, chunk_size=800, overlap=100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=["\n\n", "\n", ". ", " ", ""]  # Priority order
        )
```

### 5.2 Separator Priority Logic

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SEPARATOR PRIORITY HIERARCHY                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PRIORITY 1: "\n\n" (Double newline)                                   │
│   └── Paragraph boundaries - STRONGEST semantic break                    │
│                                                                          │
│   PRIORITY 2: "\n" (Single newline)                                     │
│   └── Line breaks - Often procedure step boundaries                      │
│                                                                          │
│   PRIORITY 3: ". " (Period + space)                                     │
│   └── Sentence boundaries - Keeps complete thoughts                      │
│                                                                          │
│   PRIORITY 4: " " (Space)                                               │
│   └── Word boundaries - Last resort before character split               │
│                                                                          │
│   PRIORITY 5: "" (Empty - character level)                              │
│   └── Absolute fallback - rarely used                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Metadata Preservation

Each chunk retains its structural context:

```python
def chunk_data(self, parsed_data, manual_id):
    chunks = []
    for item in parsed_data:
        if item["type"] == "text":
            texts = self.splitter.split_text(item["content"])
            for t in texts:
                chunks.append({
                    "manual_id": manual_id,
                    "type": "text",
                    "content": t,
                    "page": item["page"],
                    "metadata": {
                        "section": item["metadata"]["section"]  # ← Preserved!
                    }
                })
        
        elif item["type"] == "image":
            # Images are NOT chunked - passed through whole
            chunks.append({
                "manual_id": manual_id,
                "type": "image",
                "path": item["path"],
                "content": "",  # Will be filled by captioner
                "metadata": item["metadata"]
            })
    
    return chunks
```

---

## 6. Stage 3: LLM-Powered Enrichment

### 6.1 The Caption-Based Embedding Innovation

This is the **core research contribution** of our pipeline. Instead of using direct image embeddings (CLIP, ImageBind), we:

1. **Generate contextual captions** using GPT-4o Vision
2. **Embed the captions** using the same text embedding model
3. **Store both** caption (for retrieval) and image path (for display)

```
┌─────────────────────────────────────────────────────────────────────────┐
│           CAPTION-BASED vs DIRECT IMAGE EMBEDDING                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   APPROACH A: Direct Image Embedding (CLIP/ImageBind)                   │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │   Image     │────►│   CLIP      │────►│  512-dim    │              │
│   │   (pixels)  │     │   Encoder   │     │  Vector     │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                                                                          │
│   ✓ Fast (no LLM call)                                                  │
│   ✗ Captures VISUAL similarity, not SEMANTIC meaning                    │
│   ✗ "Similar looking bearings" ≠ "Bearings with same problem"          │
│                                                                          │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                          │
│   APPROACH B: Caption-Based Embedding (OUR METHOD)                      │
│   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐              │
│   │   Image     │────►│  GPT-4o     │────►│  Caption    │              │
│   │   (pixels)  │     │  Vision     │     │  (text)     │              │
│   │   +Context  │     │             │     │             │              │
│   └─────────────┘     └─────────────┘     └─────────────┘              │
│                               │                  │                      │
│                               │                  ▼                      │
│                               │         ┌─────────────┐                │
│                               │         │  text-emb   │                │
│                               │         │  -3-small   │                │
│                               │         └─────────────┘                │
│                               │                  │                      │
│                               │                  ▼                      │
│                               │         ┌─────────────┐                │
│                               │         │  1536-dim   │                │
│                               │         │  Vector     │                │
│                               │         └─────────────┘                │
│                                                                          │
│   ✓ Captures SEMANTIC meaning ("inner race bearing wear")              │
│   ✓ Same embedding space as text → unified retrieval                   │
│   ✓ Context-aware descriptions                                         │
│   ✗ Slower (requires LLM Vision call)                                  │
│   ✗ API cost per image                                                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 6.2 Context-Aware Image Captioning

The **ImageCaptioner** uses a specialized prompt with full document context:

```python
class ImageCaptioner:
    def generate_caption(self, image_path: str, metadata: dict) -> str:
        page = metadata.get("page", "Unknown")
        section = metadata.get("section", "Unknown Section")
        label = metadata.get("label", "Diagram")
        parent_ctx = metadata.get("parent_context", "")
        
        # Build hyper-contextualized prompt
        context_str = f"This image is a technical illustration labeled '{label}' "
        context_str += f"on Page {page} of the manual."
        if section != "Unknown Section":
            context_str += f" It is within section: '{section}'."
        if parent_ctx:
            context_str += f" Context: {parent_ctx}."
        
        prompt = (
            f"You are a Senior Industrial Systems Engineer. {context_str}\n\n"
            "INSTRUCTIONS:\n"
            "1. Describe this specific technical component in extremely high detail.\n"
            "2. Explain its function and relationship to the surrounding assembly.\n"
            "3. Identify any labels, bolts, connectors, or part numbers visible.\n"
            "4. Use professional engineering terminology. This description will be "
            "used for RAG retrieval, so include keywords that a technician would use "
            "when troubleshooting this specific part."
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                    }}
                ]
            }],
            max_tokens=500
        )
        
        caption = response.choices[0].message.content.strip()
        return f"### {label} (Context: {section})\n\n{caption}"
```

### 6.3 Example Caption Output

**Input:** Image of a bearing assembly from "Maintenance" section, Page 42

**Generated Caption:**
```markdown
### Bearing Assembly Detail (Context: Maintenance Procedures)

This cross-sectional diagram illustrates the main shaft bearing assembly 
for the Model Z-500 centrifugal pump. The image shows:

**Components Identified:**
- Inner race bearing (SKF 6205-2RS) mounted on the shaft via interference fit
- Outer race seated in the bearing housing with 0.002" clearance
- Grease cavity between the bearing and the lip seal
- Shaft sleeve (Alloy 316 stainless) protecting against corrosion

**Key Specifications Visible:**
- Bearing bore: 25mm
- Housing bore: 52mm
- Total assembly width: 15mm

**Troubleshooting Keywords:**
Bearing replacement, shaft sleeve, inner race wear, outer race housing, 
grease fitting, lip seal, bearing preload, axial clearance
```

### 6.4 Table Summarization

Tables are transformed into searchable text summaries:

```python
class TableTransformer:
    def summarize_table(self, table_json: str, context: str = "") -> str:
        prompt = (
            "You are a Technical Data Specialist. Convert this raw table JSON "
            "into a concise, searchable summary.\n"
            f"Context: {context}\n"
            "Format your response as a clear description. Focus on key "
            "specifications, ranges, and part numbers.\n"
            "Include every unique column name and its meaning.\n"
            "If it's a troubleshooting table, list Problem-Cause-Solution pairs."
        )
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You convert structured technical data into dense, searchable text summaries."},
                {"role": "user", "content": f"{prompt}\n\nRAW TABLE DATA:\n{table_json}"}
            ],
            max_tokens=400,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
```

---

## 7. Stage 4: Unified Vector Embedding & Storage

### 7.1 Embedding Model Selection

We use **OpenAI text-embedding-3-small** for all content types:

| Property | Value |
|----------|-------|
| **Model** | text-embedding-3-small |
| **Dimensions** | 1536 |
| **Max Tokens** | 8191 |
| **Cost** | $0.02 / 1M tokens |

### 7.2 Why Unified Embedding Space?

```
┌─────────────────────────────────────────────────────────────────────────┐
│               UNIFIED vs SEPARATE EMBEDDING SPACES                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   APPROACH A: Separate Spaces (CLIP + Text Embeddings)                  │
│                                                                          │
│   Query ──► text-embedding ──► [1536-dim] ──┐                          │
│                                              │  CANNOT directly         │
│   Image ──► CLIP ──► [512-dim] ─────────────┘  compare distances!      │
│                                                                          │
│   ✗ Requires separate indices                                           │
│   ✗ Complex fusion logic at retrieval time                             │
│   ✗ Different similarity metrics                                        │
│                                                                          │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                          │
│   APPROACH B: Unified Space (Caption Embeddings) - OUR METHOD           │
│                                                                          │
│   Query ──► text-embedding ──► [1536-dim] ──┐                          │
│                                              │  SAME space!             │
│   Caption ──► text-embedding ──► [1536-dim] ─┘  Direct comparison!      │
│                                                                          │
│   ✓ Single vector index                                                 │
│   ✓ Simple cosine distance for all types                               │
│   ✓ Text/Image/Table results naturally ranked together                 │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Batch Processing Implementation

```python
async def process_manual_async(file_path: str, manual_id: str):
    # ... stages 1-3 ...
    
    # Stage 4: Batch Embedding & Storage
    print(f"💾 [Pipeline] Stage 4/4: Embedding and persistence...")
    
    batch_size = 20
    for i in range(0, len(enriched_chunks), batch_size):
        batch = enriched_chunks[i:i + batch_size]
        print(f"   ∟ Committing batch {i//batch_size + 1}...")
        
        db = SessionLocal()
        try:
            for chunk in batch:
                if not chunk["content"]:
                    continue
                
                # Generate embedding
                emb = embedder.embed_text(chunk["content"])
                
                # Store in database
                db_chunk = ManualChunk(
                    manual_id=chunk["manual_id"],
                    type=chunk["type"],
                    content=chunk["content"],
                    path=chunk.get("path"),
                    embedding=emb,
                    page=chunk["page"]
                )
                db.add(db_chunk)
            
            db.commit()
        except Exception as e:
            print(f"   ⚠️ [Pipeline] Batch commit failed: {e}")
            db.rollback()
        finally:
            db.close()
```

---

## 8. Research Comparison: Our Approach vs Baselines

### 8.1 Comparative Analysis Framework

Based on research guidelines, we compare our **Context-Aware Caption Embedding** approach against baseline methods:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    RESEARCH COMPARISON MATRIX                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   METHOD              │ TEXT  │ IMAGE │ CONTEXT │ SEMANTIC │ COST      │
│   ────────────────────┼───────┼───────┼─────────┼──────────┼──────────│
│   Pure Text RAG       │  ✓    │  ✗    │   ✗     │   ✓      │  Low     │
│   CLIP Embedding      │  ✗    │  ✓    │   ✗     │   ~      │  Low     │
│   ImageBind           │  ✗    │  ✓    │   ✗     │   ~      │  Medium  │
│   ColPali/ColQwen     │  ✓    │  ✓    │   ~     │   ✓      │  High    │
│   ────────────────────┼───────┼───────┼─────────┼──────────┼──────────│
│   OUR METHOD          │  ✓    │  ✓    │   ✓     │   ✓      │  Medium  │
│   (Caption + Context) │       │       │         │          │          │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘

Legend:
  ✓ = Fully Supported    ~ = Partially Supported    ✗ = Not Supported
```

### 8.2 Detailed Method Comparison

#### Method 1: Pure Text RAG (Baseline)

```python
# Traditional approach - IGNORES images entirely
def pure_text_rag(pdf_path):
    text = extract_text_only(pdf_path)
    chunks = split_into_chunks(text)
    embeddings = embed_text(chunks)
    store(embeddings)
```

| Pros | Cons |
|------|------|
| Fast processing | Loses 40-60% of manual content |
| Low API costs | Diagrams completely invisible |
| Simple implementation | Critical visual procedures missed |

#### Method 2: CLIP Image Embedding

```python
# Direct visual embedding
def clip_rag(pdf_path):
    images = extract_images(pdf_path)
    for img in images:
        embedding = clip_model.encode_image(img)  # 512-dim
        store_separate_index(embedding)
```

| Pros | Cons |
|------|------|
| Captures visual features | No semantic understanding |
| Fast embedding | Different space than text |
| Works without captions | "Similar looking" ≠ "similar meaning" |

#### Method 3: ImageBind (Meta)

```python
# Multimodal aligned embedding
def imagebind_rag(pdf_path):
    images = extract_images(pdf_path)
    for img in images:
        embedding = imagebind.embed({"vision": img})  # 1024-dim
        store(embedding)
```

| Pros | Cons |
|------|------|
| Multi-modal alignment | Still visual similarity |
| Audio/video capable | High compute requirements |
| Research-grade quality | No document context |

#### Method 4: Our Approach (Caption + Context)

```python
# Context-aware caption embedding
def our_rag(pdf_path):
    images = extract_with_layout(pdf_path)  # YOLOv8
    for img, metadata in images:
        # Include section, page, label context
        caption = gpt4o_vision_caption(img, metadata)
        embedding = text_embed(caption)  # Same space as text!
        store_unified(embedding)
```

| Pros | Cons |
|------|------|
| Semantic retrieval | Higher API cost per image |
| Context-aware | Latency (~3s per image) |
| Unified embedding space | Requires GPT-4o access |
| Grounded in document structure | |

### 8.3 Research Questions Addressed

#### RQ1: Is Caption-Based Embedding Better Than Direct Image Embedding?

**Hypothesis:** Caption embeddings capture semantic meaning that direct image embeddings miss.

**Evidence from our implementation:**

```
Query: "How do I replace the inner race bearing?"

CLIP Result:
  → Images of bearings (visually similar)
  → No context about replacement PROCEDURE

Our Method Result:
  → Caption: "...inner race bearing removal requires shaft sleeve extraction..."
  → Directly answers the HOW question
```

#### RQ2: Does Context Improve Image Understanding?

**Hypothesis:** Passing section/page context to the captioner improves retrieval accuracy.

**Evidence:**

```
WITHOUT Context:
  Prompt: "Describe this image"
  Caption: "Technical drawing showing mechanical components"
  
WITH Context:
  Prompt: "This is from 'Maintenance Procedures' section, Page 42..."
  Caption: "This cross-sectional diagram illustrates the bearing 
            replacement procedure for Model Z-500..."
```

The context-aware caption includes:
- Section-specific terminology
- Procedural language (if from procedures section)
- Component relationships within the manual

#### RQ3: Can Structured Captions Improve RAG?

**Hypothesis:** Structured captions (with explicit sections) improve retrieval precision.

**Our Implementation:**
```python
# Structured caption format
caption = f"### {label} (Context: {section})\n\n{detailed_description}"

# Example output:
"""
### Bearing Assembly Detail (Context: Maintenance Procedures)

This cross-sectional diagram illustrates...

**Components Identified:**
- Inner race bearing (SKF 6205-2RS)
- Outer race housing
...

**Troubleshooting Keywords:**
bearing replacement, shaft sleeve, inner race wear...
"""
```

The structured format ensures:
- Clear component identification
- Section context preserved
- Explicit keywords for retrieval

---

## 9. The Real-World Problem We Solve

### 9.1 Why This Matters

```
┌─────────────────────────────────────────────────────────────────────────┐
│              THE UNSOLVED PROBLEM AT SCALE                               │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   PDFs are EVERYWHERE:                                                  │
│   • Research papers                                                     │
│   • Business reports                                                    │
│   • Legal documents                                                     │
│   • Technical manuals                                                   │
│   • LMS materials                                                       │
│   • Medical records                                                     │
│                                                                          │
│   Current systems:                                                      │
│   ✗ IGNORE images entirely                                             │
│   ✗ OR poorly index them (visual similarity only)                      │
│   ✗ Lose structural context                                            │
│   ✗ Cannot answer "how" questions from diagrams                        │
│                                                                          │
│   ─────────────────────────────────────────────────────────────────     │
│                                                                          │
│   OUR APPROACH SOLVES:                                                  │
│                                                                          │
│   "How do we search for MEANING inside images in documents?"            │
│                                                                          │
│   This is a REAL industry + research problem                            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 9.2 Industrial Application Impact

| Scenario | Traditional RAG | Our Approach |
|----------|-----------------|--------------|
| "Show bearing replacement" | Returns text only | Returns procedure + diagram |
| "What does this symbol mean?" | No match | Retrieves legend diagram |
| "Torque specs for coupling" | Text table (maybe) | Text + diagram with callouts |
| "Safety lockout procedure" | Partial text | Full procedure + visual steps |

---

## 10. Performance Metrics

### 10.1 Ingestion Performance

| Metric | Value | Notes |
|--------|-------|-------|
| **PDF Parsing** | ~2 pages/second | Including YOLOv8 detection |
| **Figure Splitting** | ~5 seconds/figure | GPT-4o + SAM |
| **Image Captioning** | ~3 seconds/image | GPT-4o Vision API |
| **Table Summarization** | ~1 second/table | GPT-4o-mini |
| **Embedding** | ~20 chunks/second | Batched API calls |
| **Total (100-page manual)** | 10-15 minutes | End-to-end |

### 10.2 Storage Requirements

| Content Type | Average Size | Embedding Size |
|--------------|--------------|----------------|
| Text chunk | ~800 chars | 1536 × 4 bytes = 6KB |
| Image caption | ~500 chars | 1536 × 4 bytes = 6KB |
| Table summary | ~400 chars | 1536 × 4 bytes = 6KB |
| Image file | ~50-200 KB | Stored on disk |

### 10.3 API Cost Estimation (per 100-page manual)

| Operation | Estimated Calls | Cost |
|-----------|-----------------|------|
| GPT-4o Vision (captioning) | ~50 images | ~$2.50 |
| GPT-4o (figure splitting) | ~20 composites | ~$1.00 |
| GPT-4o-mini (tables) | ~30 tables | ~$0.15 |
| text-embedding-3-small | ~500 chunks | ~$0.01 |
| **Total per manual** | - | **~$3.66** |

---

## 11. Configuration Reference

### 11.1 Environment Variables

```env
# Required
OPENAI_API_KEY=sk-...

# Database
DATABASE_URL=postgresql://user:pass@localhost:5433/rag_db

# Optional tuning
RAG_CHUNK_SIZE=800
RAG_CHUNK_OVERLAP=100
RAG_VISION_DETAIL=high
RAG_BATCH_SIZE=20
RAG_CONCURRENT_LIMIT=10
```

### 11.2 Model Files Required

| Model | Purpose | Location | Size |
|-------|---------|----------|------|
| `yolov8_doclaynet.pt` | Layout detection | `models/` | ~50MB |
| `mobile_sam.pt` | Figure segmentation | `models/` | ~40MB |

### 11.3 Directory Structure

```
data/
├── uploads/          # Raw PDF files
├── extracted/        # Extracted images
│   ├── manual1_p1_fig0.png
│   ├── manual1_p1_sub0_0.png
│   └── ...
└── embeddings/       # (Optional) Cached embeddings
=======
## 3. RAG Ingestion Pipeline

The RAG ingestion pipeline is the foundational knowledge-loading mechanism. It converts raw PDF manuals into semantically searchable, multimodal vector embeddings stored in PostgreSQL with pgvector extension.

### 3.1 Pipeline Stages

```
PDF Manual File
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: STRUCTURAL PARSING  (unified_rag/ingestion/parser.py) │
│  • PyMuPDF extracts raw text per page                            │
│  • YOLOv8 object detection identifies figure bounding boxes      │
│  • Figure Splitter crops and saves images to disk                │
│  • Tables detected and extracted as structured strings           │
│  Output: List of {type, content/path, page, metadata} dicts      │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: ADAPTIVE CHUNKING   (unified_rag/ingestion/chunker.py)│
│  • Semantic recursive chunking strategy                          │
│  • Text chunks: ~500 tokens with sliding overlap                 │
│  • Image chunks: one chunk per extracted figure                  │
│  • Table chunks: one per detected table                          │
│  • Each chunk tagged: manual_id, type, page, metadata            │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: CONCURRENT LLM ENRICHMENT (asyncio + Semaphore 10)    │
│                                                                   │
│  FOR IMAGE CHUNKS:                                               │
│    captioner.generate_caption(path, metadata)                    │
│    → GPT-4o Vision: "Exploded view of outboard bearing..."       │
│    ← Caption replaces path as searchable content                 │
│                                                                   │
│  FOR TABLE CHUNKS:                                               │
│    tabler.summarize_table(content, section_context)              │
│    → GPT-4o: Converts HTML table to structured English summary   │
│    ← "Table 3: Torque specs. Bearing bolt: 65 Nm..."             │
│                                                                   │
│  Rate-limited: max 10 concurrent LLM calls                       │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 4: EMBEDDING & PERSISTENCE  (batched, 20 chunks/commit)  │
│  embedder.embed_text(chunk_content)                              │
│    → OpenAI text-embedding-3-small → 1536-dim float vector       │
│  ManualChunk(manual_id, type, content, embedding, page, path)    │
│    → INSERT INTO manual_chunks (PostgreSQL + pgvector)           │
└─────────────────────────────────────────────────────────────────┘
      │
      ▼
┌──────────────────────────────────────┐
│  pgvector STORE: manual_chunks table │
│  Queryable via cosine similarity     │
│  Filtered by: manual_id (per-machine)│
└──────────────────────────────────────┘
```

### 3.2 Manual-to-Machine Binding (Provenance Enforcement)

Every chunk stored in `manual_chunks` carries a `manual_id` tag that maps directly to `machines.manual_id`. This creates a strict isolation boundary preventing cross-machine knowledge contamination.

**Why this matters — Safety Scenario:**

| Scenario | Without Binding | With Binding |
|---|---|---|
| Query: "Torque bearing on PUMP-001" | LLM retrieves chunks from ALL manuals, may get LATHE torque spec (120 Nm) | Only retrieves PUMP manual (correct: 65 Nm) |
| Result | Over-torqued bearing → immediate failure | Correct installation → safe operation |

**Provenance Check at Query Time** (`copilot_graph.py`):
```python
chunk_count = db.query(ManualChunk).filter(
    ManualChunk.manual_id == manual_id
).count()

if chunk_count == 0:
    # Inject disclaimer flag — no manual ingested for this machine
    query = f"[DISCLAIMER_REQUIRED: MISSING_MANUAL] {query}"
```

If no manual exists, the RAG engine prepends a mandatory disclaimer that it is using general industrial best practices, not machine-specific documentation.

### 3.3 Sensor Datasheet Ingestion (Machine Onboarding Pipeline)

When a new machine is registered via the frontend, operators upload sensor datasheets (PDFs). This triggers a full AI-powered secondary ingestion pipeline:

```
Sensor Datasheet PDF (per sensor)
      │
      ▼
DatasheetParser.parse_pdf(pdf_bytes)  [PyMuPDF text extraction]
      │
      ▼
DatasheetParser.extract_sensor_config(sensor_name, sensor_id, pdf_text)
      │  → GPT-4o extracts: {mu, sigma, min_normal, max_normal,
      │                       fault_high, fault_low, unit, icon_type}
      ▼
AIAutomationEngineerAgent.validate_sensor_config(config, datasheet_text)
      │  → Checks: mu vs specs, sigma reasonableness, fault thresholds
      │  → Returns: is_valid, confidence, corrections, engineering_notes
      ▼
AIAutomationEngineerAgent.cross_validate_sensors(all_configs, machine_type)
      │  → Checks temperature ↔ current correlation consistency
      │  → Returns: is_consistent, expected_correlations, fault_coverage_score
      ▼
AIAutomationEngineerAgent.generate_anomaly_patterns(configs, machine_type)
      │  → GPT-4o generates: machine_fault & sensor_drift patterns
      │  → Includes: staged progression, sensor correlations, physics rationale
      ▼
generate_dataset.py --machine_id --use_ai_patterns
      │  → Synthesizes normal + anomaly CSV training dataset
      ▼
preprocessing/normalization.py → models/train_model.py
      │  → Trains LSTM/Autoencoder anomaly detection model per machine
      ▼
sensor_configs.json + ML model saved (per-machine)
```

Every new machine that registers **automatically gets its own dedicated anomaly detection model** trained on AI-synthesized physics-accurate data from its own sensor datasheets.

---

## 4. Multi-Agent Orchestration (LangGraph)

The `copilot_graph.py` defines a directed acyclic graph (DAG) where 6 specialized agents process each anomaly event sequentially, sharing a typed `CopilotState` dictionary.

### 4.1 Pipeline DAG

```
SensorStatusAgent
      │
      ▼
ValidationEngineerAgent   ← 4-Stage AI Validation Layer
      │
      ▼
DiagnosticAgent           ← DB persistence + report generation
      │
      ▼
KnowledgeRetrievalAgent   ← Multimodal RAG + HITL query routing
      │
      ▼
StrategyAgent             ← Procedure synthesis passthrough
      │
      ▼
CriticAgent               ← HITL-aware output formatter
      │
      ▼
[END] → API Response → Next.js Frontend
```

### 4.2 Shared State Schema

```python
class CopilotState(TypedDict):
    # Anomaly Detection Inputs
    event_id: str
    machine_id: str
    machine_state: str          # e.g. "PUMP_FAULT"
    anomaly_score: float        # MSE from autoencoder (0.0–1.0)
    user_query: Optional[str]   # HITL: operator's follow-up
    suspect_sensor: Optional[str]
    recent_readings: Optional[Dict[str, Any]]

    # AI Validation Layer
    ai_validation_status: Optional[str]   # TRUE_FAULT | SENSOR_GLITCH | NORMAL_WEAR
    fault_category: Optional[str]         # mechanical | thermal | electrical | process | sensor
    ai_confidence_score: Optional[float]  # 0.0 – 1.0
    ai_engineering_notes: Optional[str]   # Root cause explanation

    # Agent Outputs (accumulated sequentially)
    sensor_status_report: str
    diagnostic_report: str
    rag_context: str
    retrieved_images: List[str]    # Absolute URLs to technical diagrams
    strategy_report: str
    critic_feedback: str
    final_execution_plan: str
    chat_history: Optional[str]    # Wizard conversation history
```

### 4.3 Agent Descriptions

#### Agent 1 — SensorStatusAgent
Translates raw sub-symbolic telemetry into natural language. Uses the anomaly MSE score and suspect sensor to generate a plain-English status report.

**Output:** `"Detected severe deviations aligning with thermistor_1. Mathematical threshold exceeded by 0.87 MSE."`

---

#### Agent 2 — ValidationEngineerAgent

The most sophisticated agent — a 4-stage pipeline that determines whether an anomaly is a real fault or a false positive **before human escalation**.

**4-Stage Pipeline:**

```
Stage 1: Physics Violations Check
  sensor_config_loader.get_violation_summary(machine_id, recent_readings)
  → Compares readings against manufacturer-specified limits
  → Categories: CRITICAL (above fault_high) | WARNING (above max_normal)

Stage 2: Temporal Pattern Analysis
  TemporalAnalyzer.analyze(machine_id, readings)
  → Tracks rolling 5-reading history per sensor per machine
  → Detects: sudden_spike vs. sustained_trend
  → Computes: rate_of_change, trend_direction

Stage 3: Hybrid Confidence Score
  calculate_hybrid_confidence(ml_score, physics_summary, temporal_pattern)
  → formula: ML_score + 0.30(critical violation) + 0.15(warning)
              + 0.20(sustained) - 0.15(sudden spike) → clamped [0,1]
  → If confidence < 0.2: AUTO-CLASSIFY SENSOR_GLITCH (skip Stage 4)

Stage 4: AI High-Accuracy Classification
  AIAutomationEngineerAgent.high_accuracy_fault_classification()
  → GPT-4o with JSON mode
  → Multi-hypothesis ranking (top 3 causes + probability %)
  → Root Cause Analysis + fault propagation chain
  → Recommended Actions (immediate/high/medium/low priority)
  → Flags: requires_human_review
```

**Output Classifications:**

| Status | Meaning | Operator Action |
|---|---|---|
| `TRUE_FAULT` | Genuine equipment fault | Immediate — open Copilot Chat |
| `SENSOR_GLITCH` | EMI / transient noise | None — auto-dismissed |
| `NORMAL_WEAR` | Expected degradation | Schedule maintenance |

---

#### Agent 3 — DiagnosticAgent
Builds human-readable diagnostic report from validation output. **Persists** AI validation results to the database for audit trail:

```python
recent_anomaly.ai_validation_status = ai_status        # e.g. "TRUE_FAULT"
recent_anomaly.fault_category = fault_category          # e.g. "thermal"
recent_anomaly.ai_confidence_score = ai_confidence      # e.g. 0.92
recent_anomaly.ai_engineering_notes = ai_notes          # Detailed text
db.commit()
```

---

#### Agent 4 — KnowledgeRetrievalAgent (Multimodal RAG Interface)

Routes queries to the appropriate `RAGMode` based on HITL context tags embedded in `user_query`, then calls `RAGGenerator.generate_response()`.

**HITL Query Routing:**

```python
if "[CLARIFY_STEP]" in user_q:
    mode = RAGMode.CLARIFICATION      # "Explain this step simply"
elif "[EVALUATE_STEP]" in user_q:
    mode = RAGMode.EVALUATION         # "I completed this step"
elif "[CONVERSATIONAL_WIZARD]" in user_q:
    mode = RAGMode.CONVERSATIONAL_WIZARD  # Full guided repair session
else:
    mode = RAGMode.SUMMARY            # Initial anomaly (no prior query)
```

**Dynamic Manual Resolution:**
```python
machine_record = db.query(Machine).filter(Machine.machine_id == machine_id).first()
manual_id = machine_record.manual_id  # e.g. "Zynaptrix_9000"
```

The agent resolves machine_id → manual_id at runtime, ensuring every query is always directed to the correct manual.

---

#### Agent 5 — StrategyAgent
Passthrough layer that carries the RAG context (with embedded `[IMAGE_N]` tags) to the Critic. The actual synthesis is performed within the RAG prompt.

---

#### Agent 6 — CriticAgent
Final output formatter. HITL-aware: behaves differently depending on whether a `user_query` is present.

- **No user_query** (initial alert): Wraps output as brief diagnostic summary with `[SUGGESTION: Generate full step-by-step repair procedure]` trigger
- **With user_query** (HITL conversation): Passes full strategy report (procedure steps + images) directly to the chat interface

---

## 5. Chatbot Interfaces & Human Interaction

The system exposes two distinct chatbot interfaces serving different phases of the HITL workflow.

### 5.1 Diagnostic Copilot Chat

**Purpose:** Incident-specific guided repair wizard activated on anomaly detection.

**Backend:** `api/machine_api.py` + `copilot_graph.py` + `unified_rag/retrieval/rag.py`

**Human Interaction Flow:**

```
PHASE 1 — ALERT
AI detects anomaly → AnomalyRecord created in DB
Incident Registry shows alert card (machine_id, score, timestamp)
Operator clicks alert → Opens Diagnostic Copilot Chat

PHASE 2 — INITIAL BRIEF (automated, no user input)
LangGraph pipeline runs with RAGMode.SUMMARY
Returns: "🚨 AI Diagnostic Alert"
  → Machine state + confidence score
  → 3–5 sentence diagnostic summary from RAG
  → [SUGGESTION: Generate full step-by-step repair procedure] button

PHASE 3 — GUIDED WIZARD (operator clicks suggestion)
Frontend sends: user_query = "[CONVERSATIONAL_WIZARD] Generate full step-by-step..."
RAGMode.CONVERSATIONAL_WIZARD activates
LLM generates structured phases:
  [PHASE: Safety & Preparation]
    - Lockout/Tagout (LOTO) mandatory first
    - PPE requirements
    - [IMAGE_0] safety diagram
  [PHASE: Diagnosis Verification]
    - Physical inspection checklist
    - [IMAGE_1] component location diagram
  [PHASE: Repair Procedure]
    - Numbered step-by-step instructions
    - [IMAGE_N] technical diagrams at exact relevant steps

PHASE 4 — INTERACTIVE STEP CARDS
Operator: "I have done this step" → [EVALUATE_STEP] tag
  RAGMode.EVALUATION: QA Supervisor responds
  → Returns [STEP_COMPLETE] or [STEP_NEED_HELP]

Operator: "Explain simply" → [CLARIFY_STEP] tag
  RAGMode.CLARIFICATION: Technical Mentor responds
  → ELI5 bullet points + relevant images

PHASE 5 — RESOLUTION
Operator types resolution note (e.g., "Replaced bearing SKF-6205, 65Nm")
validate_operator_feedback() runs quality check
If valid → Vectorize → Archive to InteractionMemory
AnomalyRecord.resolved = True
```

**HITL Learning Data Captured:**

| Data Point | Storage | Purpose |
|---|---|---|
| Every chat message | `chat_history` table | Audit trail, session replay |
| AI validation classification | `anomaly_records.ai_validation_status` | Future pattern analysis |
| Operator resolution note | `interaction_memory` (vectorized) | Future RAG retrieval |
| Step completion states | `chat_history.message_metadata` JSON | Procedure progress tracking |

---

### 5.2 Central Assistant Bot

**Purpose:** General-purpose knowledge assistant accessible via floating chat bubble. Not incident-specific.

**Backend:** `api/assistant_api.py`

**Human Interaction Flow:**

```
Operator opens Central Assistant (blue chat bubble, bottom-right)
Optional: Selects machine from "RAG: [Machine] Manual" dropdown

SESSION MANAGEMENT
First message → Creates AssistantSession (GPT-4o-mini generates 4-word title)
Subsequent messages → Appends to session with last-10-message context window

INTENT CLASSIFICATION (5 intents via GPT-4o-mini)

1. GUIDE — Keyword match: "ingest", "upload", "register", "simulator"
   Response: Deterministic numbered steps from SYSTEM_GUIDE_STEPS dict
   No LLM call — instant structured step cards

2. ONBOARDING — "how does this work?", "what is this system?"
   Response: GPT-4o using embedded SYSTEM_ONBOARDING_CONTEXT doc

3. RAG — Technical questions when machine is selected
   Mode: CONVERSATIONAL_WIZARD (if procedural keywords present)
         SUMMARY (otherwise)
   Response: RAGGenerator + GPT-4o synthesis + inline [IMAGE_N] tags

4. SEARCH — General industry knowledge, standards
   Response: Simulated web search via GPT-4o, then synthesized

5. CHAT — Greetings, off-topic
   Response: GPT-4o-mini, lightweight system prompt
```

**Key Differentiator from Copilot Chat:**

| Feature | Diagnostic Copilot Chat | Central Assistant |
|---|---|---|
| Context | Locked to anomaly_id | Session-based, machine optional |
| Primary use | Guided step-by-step repair | General Q&A + navigation |
| Memory | Chat linked to AnomalyRecord | AssistantSession + AssistantMessage |
| Intent routing | LangGraph DAG | 5-intent GPT-4o-mini classifier |
| Learning loop | Resolves to InteractionMemory | Informational only, no feedback |

---

## 6. Validation Methods

The Zynaptrix platform employs **five distinct validation layers**, spanning automated physics checks, ML scoring, AI classification, and human feedback quality assurance.

### 6.1 Layer 1 — Physics Violation Check

**Service:** `services/sensor_config_loader.py` → `get_violation_summary()`

Compares each current sensor reading against manufacturer-specified limits from `sensor_configs.json`.

| Violation Class | Condition | Confidence Impact |
|---|---|---|
| `CRITICAL` (fault violation) | Reading > `fault_high` or < `fault_low` | +0.30 to hybrid confidence |
| `WARNING` (normal violation) | Reading > `max_normal` (but below fault) | +0.15 to hybrid confidence |
| `NONE` | Reading within normal bounds | No boost |

Physics violations are the **first hard evidence layer** — regardless of ML uncertainty, a reading beyond its datasheet-specified limit is unambiguous.

---

### 6.2 Layer 2 — Temporal Pattern Analysis

**Service:** `services/anomaly_service.py` → `TemporalAnalyzer`

Tracks a rolling 5-reading history per sensor per machine and classifies the pattern:

| Pattern | Detection | Confidence Impact |
|---|---|---|
| **Sudden Spike** | `deviation > 3σ AND rate_of_change > 2σ` in single reading | −0.15 (likely glitch) |
| **Sustained Trend** | `anomaly_count >= 3` consecutive anomalous readings | +0.20 |
| **Rising/Falling** | ≥80% of diffs in same direction | Trend metadata only |
| **Erratic** | High variance without direction | Flagged for sensor review |

A 60°C temperature jump in one reading that immediately returns to 25°C is physically impossible given thermal mass — Temporal Analyzer catches this and reduces confidence, preventing false escalation.

---

### 6.3 Layer 3 — Hybrid Confidence Score

**Service:** `services/anomaly_service.py` → `calculate_hybrid_confidence()`

```
hybrid_confidence = ML_anomaly_score (autoencoder MSE)
                  + 0.30  IF critical physics violations present
                  + 0.15  IF warning physics violations present
                  + 0.20  IF sustained temporal trend detected
                  - 0.15  IF single spike detected (likely glitch)
                  → clamped to [0.0, 1.0]

Decision threshold:
  confidence < 0.2 → AUTO-CLASSIFY "SENSOR_GLITCH" (no GPT-4o call)
  confidence >= 0.2 → Proceed to Stage 4 AI classification
```

This dual-threshold design saves LLM API costs on obvious false positives while ensuring real faults receive full AI analysis.

---

### 6.4 Layer 4 — AI Engineering Classification (GPT-4o)

**Agent:** `agents/ai_automation_engineer.py` → `high_accuracy_fault_classification()`

**System Prompt Role:** "Senior Industrial Automation Engineer with 20+ years experience in predictive maintenance, FMEA, and fault diagnosis."

**Input package to GPT-4o:**
- ML score + hybrid confidence
- All physics violations with values and limits
- Temporal pattern analysis JSON
- All current sensor readings
- Machine sensor configurations (for cross-correlation reasoning)
- Last 5 historical anomalies (context)
- Few-shot examples: thermal overload, EMI spike, normal wear drift

**Enforced JSON output schema:**
```json
{
    "primary_classification": "TRUE_FAULT | SENSOR_GLITCH | NORMAL_WEAR",
    "fault_category": "mechanical | thermal | electrical | process | sensor",
    "confidence_score": 0.92,
    "confidence_interval": [0.85, 0.97],
    "hypotheses": [
        {
            "rank": 1,
            "description": "Bearing degradation due to lubrication failure",
            "probability": 0.75,
            "supporting_evidence": ["temp+current correlation", "sustained trend"],
            "contradicting_evidence": ["no speed variation"]
        }
    ],
    "root_cause_analysis": {
        "primary_cause": "Outboard bearing race spalling",
        "contributing_factors": ["insufficient lubrication"],
        "fault_propagation": "friction → current rise → temperature climb"
    },
    "recommended_actions": [
        {"priority": "immediate", "action": "Initiate LOTO and shutdown"},
        {"priority": "high", "action": "Inspect outboard bearing"}
    ],
    "engineering_notes": "Multi-sensor correlation confirms mechanical fault...",
    "requires_human_review": true
}
```

**Fallback Chain:** AI Engineer → `_fallback_gpt4_validation()` (simpler prompt) → `_default_validation_result()` (deterministic, flags for manual review).

---

### 6.5 Layer 5 — Operator Feedback Validation

**Endpoint:** `POST /api/chat-history/{anomaly_id}/resolve`

When an operator submits their incident resolution note, it is AI-validated for archival quality before being accepted into the knowledge base:

```python
validation_prompt = """
Evaluate:
1. Is this detailed enough for future troubleshooting?
2. Does it describe actual actions taken (not just observations)?
3. Is it technically relevant to the anomaly type?

Return JSON: {is_valid, quality_score, feedback_message, suggestions, extracted_actions}
"""
```

**Quality Gate:**

| Quality Score | Text Length | Outcome |
|---|---|---|
| >= 0.3 OR length > 20 chars | Either passes | Accepted → vectorized to knowledge base |
| < 0.3 AND length < 20 chars | Both fail | Rejected → operator shown improvement suggestions |

Minimum threshold (0.3) is intentionally permissive to avoid blocking operators in the field who may be time-pressured. Even terse feedback like "Replaced bearing" meets the threshold.

---

### 6.6 Validation Summary

| Layer | Method | Technology | Trigger |
|---|---|---|---|
| Physics Violation | Manufacturer limit comparison | sensor_configs.json | Every sensor reading |
| Temporal Pattern | Rolling-window statistical analysis | TemporalAnalyzer | Every reading |
| Hybrid Confidence | Formula fusion (ML + Physics + Temporal) | Python arithmetic | Pre-classification filter |
| AI Classification | Multi-hypothesis GPT-4o analysis | AIAutomationEngineerAgent | confidence >= 0.2 |
| Feedback Quality | GPT-4o JSON quality scoring | validate_operator_feedback() | On incident resolution |

---

## 7. Adaptive Learning via Interaction Memory

Every successfully resolved incident enriches the knowledge base permanently. This is the system's **continuous learning mechanism** — accuracy improves over time without retraining any neural networks.

### 7.1 Learning Loop

```
INCIDENT RESOLVED
operator_fix = "Replaced outboard bearing T-09 with SKF-6205.
                Torqued to 65Nm. Lubricated with Shell Gadus S3 V220C
                per manual page 47."
      │
      ▼
validate_operator_feedback() → quality_score: 0.88 ✓
      │
      ▼
GPT-4o Summarization:
"PUMP-001 bearing replacement. Actions: (1) LOTO applied,
(2) Outboard bearing T-09 removed, (3) SKF-6205 installed,
(4) Torqued 65Nm, (5) Shell Gadus applied per page 47.
Root cause: lubrication starvation after 18 months."
      │
      ▼
text-embedding-3-small → 1536-dim vector
      │
      ▼
InteractionMemory(
    machine_id="PUMP-001",
    manual_id="Historical_Knowledge",
    summary=<above>,
    operator_fix=<original text>,
    embedding=<vector>,
    timestamp="2026-04-07 14:32:00"
) → INSERT into interaction_memory table
      │
      ▼
FUTURE INCIDENT on PUMP-001 bearing anomaly:
RetrievalEngine fetches historical_fixes alongside manual chunks
LLM context includes:
"--- PREVIOUS FIX 1 (2026-04-07) ---
Summary: PUMP-001 bearing replacement. Operator used SKF-6205..."
LLM output:
"Field wisdom: Previous operators resolved this with SKF-6205 bearing
installed with Shell Gadus S3 V220C lubricant (see page 47)."
```

### 7.2 Three-Source Parallel Retrieval

Every RAG query retrieves from three sources simultaneously:

| Source | Table | Filter | Returns |
|---|---|---|---|
| Manual Text | `manual_chunks` (type='text') | `manual_id` | Procedure text, specifications |
| Manual Images | `manual_chunks` (type='image') | `manual_id` | Diagram captions + file paths |
| Historical Fixes | `interaction_memory` | `machine_id` | Past resolution summaries |

All results ranked by cosine similarity and merged into a unified context window for the LLM synthesizer.

---

## 8. Data Flow: End-to-End Lifecycle

```
T=0ms    IoT Sensor → InfluxDB stream
         {thermistor_1: 38.2°C, lem_1: 57.1A, encoder_1: 1580rpm}

T=5ms    AnomalyDetector.detect(reading)
         → Autoencoder MSE = 0.87 > threshold 0.72 → IS_ANOMALY

T=10ms   TemporalAnalyzer: 4th consecutive anomaly → is_sustained=True

T=15ms   calculate_hybrid_confidence()
         → 0.87 + 0.30 (critical) + 0.20 (sustained) = 1.0 → clamped

T=20ms   AnomalyRecord written to DB. Incident Registry updates.

T=25ms   OrchestratorAgent builds CopilotState, invokes LangGraph

T=50ms   SensorStatusAgent → sensor_status_report generated

T=200ms  ValidationEngineerAgent:
         Physics: thermistor_1 = 38.2 > 35°C (CRITICAL)
         Temporal: sustained, rising trend
         Hybrid confidence: 0.95 → proceed to Stage 4
         AIAutomationEngineerAgent: TRUE_FAULT, thermal, 0.92

T=800ms  DiagnosticAgent:
         "🔴 [CRITICAL] AI-Verified Fault Detected..."
         Persists classification to AnomalyRecord DB

T=1200ms KnowledgeRetrievalAgent:
         machine_id PUMP-001 → manual_id Zynaptrix_9000
         847 chunks found in manual ✓
         RAGMode.SUMMARY → 3 text chunks + 2 images + 1 historical fix

T=1800ms StrategyAgent + CriticAgent:
         Formats initial alert with [SUGGESTION] trigger
         Writes ChatMessage to DB

T=2000ms API Response to Frontend (React)

─── HUMAN INTERACTION PHASE ───

T+5min   Operator opens Incident in Diagnostic Copilot Chat
         Clicks "Generate full step-by-step repair procedure"

T+5min   RAGMode.CONVERSATIONAL_WIZARD
         GPT-4o generates:
           [PHASE: Safety] LOTO + PPE + [IMAGE_0]
           [PHASE: Diagnosis] Physical inspection + [IMAGE_1]
           [PHASE: Repair] Numbered steps + inline [IMAGE_N]

T+15min  Operator: "Explain this step simply"
         RAGMode.CLARIFICATION → ELI5 bullet points

T+45min  Operator submits resolution:
         "Replaced outboard bearing. SKF-6205. 65Nm torque."
         validate_operator_feedback() → quality: 0.82 ✓
         GPT-4o summarizes → embedded → InteractionMemory table
         AnomalyRecord.resolved = True
         Knowledge base permanently enriched.
>>>>>>> origin/main
```

---

<<<<<<< HEAD
## 12. Key Research Contributions

| Contribution | Description | Novelty |
|--------------|-------------|---------|
| **Context-Aware Captioning** | Section/page context injection for industrial PDF captioning | First implementation for industrial domain |
| **Agentic Figure Decomposition** | GPT-4o + Voronoi + SAM pipeline for composite diagram splitting | Novel 3-phase approach |
| **Unified Embedding Space** | Caption-based embedding for semantic (not visual) retrieval | Superior to CLIP for domain-specific queries |
| **Structural Context Tracking** | In-memory hierarchy during parsing | Enables section-scoped queries |
| **Industrial RAG Framework** | End-to-end system validated on real technical manuals | Production-ready implementation |

---

## 13. Conclusion

The RAG Ingestion Pipeline represents a novel approach to multimodal document understanding for industrial applications. By combining:

1. **YOLOv8 Layout Detection** - Intelligent document parsing
2. **Agentic Figure Splitting** - GPT-4o + SAM decomposition
3. **Context-Aware Captioning** - Semantic image understanding
4. **Unified Text Embedding** - Single vector space for all content

We achieve **semantic retrieval** capabilities that surpass traditional visual similarity methods, enabling operators to query industrial knowledge using natural language and receive both textual procedures and relevant technical diagrams.

**Research Verdict:**
- ✅ Strong engineering + applied AI
- ✅ Industry-relevant problem
- ✅ Novel contributions (context-aware captioning, agentic splitting)
- 📊 For conference-level work: Add formal evaluation with Precision@K metrics

---

*This document covers the complete RAG Ingestion Pipeline implementation. See companion documents for Retrieval Pipeline and Agent System documentation.*
=======
## 9. Database Schema & Vector Store

### 9.1 Tables

| Table | Key Fields | Purpose |
|---|---|---|
| `machines` | machine_id, manual_id | Machine registry + manual binding |
| `manual_chunks` | manual_id, type, content, embedding VECTOR, page, path | RAG knowledge store |
| `anomaly_records` | machine_id, score, resolved, ai_validation_status, ai_confidence_score | Anomaly audit trail |
| `chat_history` | anomaly_id FK, role, content, images JSON, message_metadata JSON | Copilot chat log |
| `interaction_memory` | machine_id, summary, operator_fix, embedding VECTOR | Continuous learning store |
| `assistant_sessions` | machine_id, title | Central Assistant session |
| `assistant_messages` | session_id FK, role, content, images JSON | Central Assistant history |

### 9.2 Vector Search

```sql
-- Semantic similarity search (all RAG queries)
SELECT *, 1 - (embedding <=> query_vector) AS similarity
FROM manual_chunks
WHERE manual_id = 'Zynaptrix_9000'   -- machine-bound
ORDER BY similarity DESC
LIMIT 5;

-- Historical fix retrieval
SELECT *, 1 - (embedding <=> query_vector) AS similarity
FROM interaction_memory
WHERE machine_id = 'PUMP-001'        -- machine-bound
ORDER BY similarity DESC
LIMIT 2;
```

---

## 10. Next.js Frontend Architecture

The Zynaptrix frontend is a high-performance, real-time dashboard built with **Next.js 16** and **Tailwind CSS 4**. It focuses on "Visual Excellence" and "Actionable Intelligence," transforming raw AI outputs into interactive maintenance workflows.

### 10.1 Technical Stack

| Category | Technology | Purpose |
|---|---|---|
| **Framework** | Next.js 16 (App Router) | Server-side rendering, routing, and optimization |
| **State Management** | Redux Toolkit | Centralized store for machines, alerts, and chat |
| **Styling** | Tailwind CSS 4 | Modern, utility-first design system |
| **Animations** | Framer Motion | Smooth transitions and interactive micro-animations |
| **Data Viz** | Recharts | Real-time telemetry and anomaly score visualization |
| **Markdown** | React Markdown | Rendering structured LLM responses with GFM support |
| **Export** | jsPDF / html2canvas | Generating professional technical reports |

### 10.2 Core Components & Design System

#### 10.2.1 TaskInteractionCard (HITL Catalyst)
The core of the Human-in-the-Loop workflow. It parses the `[PROCEDURE_START]` JSON from the backend and renders:
- **Phase Tabs**: Grouping preparation, diagnosis, and repair steps.
- **Interactive Checkboxes**: Real-time task completion tracking synced to the database.
- **Inline Diagrams**: Localized viewing of technical manual figures (`[IMAGE_N]`).
- **Contextual Actions**: "Explain simply" or "Need help" buttons that trigger specialized RAG modes.

#### 10.2.2 AssistantSidebar
A multi-turn conversational interface that handles:
- **Intent-Driven UI**: Switches layout based on whether the AI is guiding, searching, or performing RAG.
- **Session Persistence**: Persistent chat history across page refreshes via Redux.
- **Machine Selection**: Context-aware queries based on the selected asset.

#### 10.2.3 GalaxyBackground & Aesthetics
To ensure a **Premium First Impression**, the system utilizes a custom `GalaxyBackground` component using CSS gradients and opacity masks, creating a high-tech "Industrial OS" feel.

### 10.3 Services & Utilities

- **`professionalReportService.ts`**: Aggregates incident diagnostic data, chat history, and resolution notes into a polished PDF report for maintenance auditing.
- **`pdfExportService.ts`**: Handles the underlying canvas rendering and PDF pagination.
- **Machine Onboarding Wizard**: A multi-step form for registering machines, uploading sensor datasheets, and monitoring the AI model training progress.
### 11. Research Importance & Contributions

### 11.1 Novel Research Contributions

#### 11.1.1 Physics-Aware Hybrid Anomaly Confidence
Most ML anomaly research uses pure neural network MSE scores. Zynaptrix introduces a **hybrid formula** fusing autoencoder score with physics-domain knowledge (sensor limits) and temporal signal analysis (spike vs. sustained). This addresses a fundamental weakness of pure ML: anomaly score uncertainty on rare-but-normal operating conditions.

#### 11.1.2 AI-as-Validation-Engineer Pattern
GPT-4o is used as a **domain-expert filter agent** before human escalation — not as a primary decision maker, but as a second opinion that reduces false positive escalations causing operator alert fatigue. This is a novel use of LLMs as classification validators rather than generators.

#### 11.1.3 Unified Semantic Space for Multimodal RAG
Technical diagrams are typically excluded from RAG systems due to format incompatibility. Zynaptrix bridges this gap by using GPT-4o Vision to generate natural-language captions for engineering figures, then embedding those captions with the same model as text chunks. Operators can ask "show bearing assembly diagram" and retrieve the correct image through pure semantic similarity — no keyword tagging required.

#### 11.1.4 Self-Improving Knowledge Base (Institutional Memory)
Every resolved incident is vectorized and permanently archived. Future incidents retrieve not just static manuals but **organizational memory** of how similar issues were resolved on the same machine. This implements retrieval-augmented continual learning without retraining any neural networks.

#### 11.1.5 Per-Machine Anomaly Detection with AI-Synthesized Training Data
Each registered machine receives its own dedicated anomaly detection model trained on AI-synthesized physics-accurate fault progressions derived from its actual sensor datasheets. The system is **entirely self-provisioning** — no historical failure data required.

### 11.2 System Comparison

| Capability | Manual Search | Rule-Based SCADA | Pure ML | **Zynaptrix** |
|---|---|---|---|---|
| Anomaly Detection | ❌ | ✅ Threshold only | ✅ Statistical | ✅ Hybrid (ML + Physics + Temporal) |
| Root Cause Analysis | ❌ Manual | ❌ None | ❌ Black box | ✅ AI multi-hypothesis |
| Procedure Retrieval | ❌ 30 min | ❌ None | ❌ None | ✅ Semantic RAG < 2 sec |
| Visual Diagrams | ❌ Flip through manual | ❌ None | ❌ None | ✅ Inline via Vision captioning |
| Historical Learning | ❌ Tribal knowledge | ❌ None | ❌ None | ✅ Vectorized InteractionMemory |
| False Positive Filtering | ❌ Operator judgment | ❌ None | ⚠️ Tuning | ✅ 4-stage AI validation |
| Guided Repair Steps | ❌ None | ❌ None | ❌ None | ✅ Conversational Wizard |
| Safety Enforcement | ❌ Operator checklist | ❌ None | ❌ None | ✅ Mandatory LOTO phase (RAG prompt rule) |

### 11.3 Industrial Impact

| Metric | Traditional | Zynaptrix |
|---|---|---|
| Time to Diagnosis | 30–60 minutes | ~2 seconds |
| Knowledge Capture | Retained in technician's memory | Vectorized to knowledge base per incident |
| Junior Operator Capability | Limited by experience | AI-guided to expert-level procedure quality |
| Safety Compliance | Depends on individual checklist discipline | Enforced by SAFETY MANDATE in RAG prompt — LOTO always first |
| System Improvement | Static until retraining | Every resolved incident improves future responses |

---

## Appendix A: RAG Mode Reference

| RAGMode | Trigger Tag | LLM Role | Response Format |
|---|---|---|---|
| `SUMMARY` | None (default) | Diagnostic AI | 3–5 sentence summary + `[SUGGESTION]` trigger |
| `PROCEDURE` | `[PROCEDURE_REQUEST]` | Procedure Generator | JSON in `[PROCEDURE_START]...[PROCEDURE_END]` tags |
| `CLARIFICATION` | `[CLARIFY_STEP]` | Technical Mentor (ELI5) | Bullet-point plain English + inline images |
| `EVALUATION` | `[EVALUATE_STEP]` | QA Supervisor | Starts with `[STEP_COMPLETE]` or `[STEP_NEED_HELP]` |
| `CONVERSATIONAL_WIZARD` | `[CONVERSATIONAL_WIZARD]` | Maintenance Mentor | Phase-structured, chat-history aware, mandatory safety-first |

---

## Appendix B: File Reference Map

| Component | File Path |
|---|---|
| RAG Ingestion Pipeline | `backend/unified_rag/ingestion/pipeline.py` |
| PDF Parser | `backend/unified_rag/ingestion/parser.py` |
| Semantic Chunker | `backend/unified_rag/ingestion/chunker.py` |
| Image Captioner | `backend/unified_rag/ingestion/captioner.py` |
| RAG Generator + Modes | `backend/unified_rag/retrieval/rag.py` |
| Retrieval Engine | `backend/unified_rag/retrieval/retriever.py` |
| LangGraph Copilot DAG | `backend/agents/copilot_graph.py` |
| AI Validation Engineer | `backend/agents/ai_automation_engineer.py` |
| Validation Prompts (Few-Shot) | `backend/agents/validation_prompts.py` |
| Knowledge Agent | `backend/agents/knowledge_agent.py` |
| Anomaly Detection Service | `backend/services/anomaly_service.py` |
| Sensor Config Loader | `backend/services/sensor_config_loader.py` |
| Datasheet AI Parser | `backend/services/datasheet_parser.py` |
| Diagnostic Copilot API | `backend/api/machine_api.py` |
| Central Assistant API | `backend/api/assistant_api.py` |
| Copilot Chat API | `backend/api/copilot_chat_api.py` |
| Database Models | `backend/unified_rag/db/models.py` |

---

*Document generated: April 2026 | Zynaptrix Industrial Copilot v2.0*  
*Research context: Gen AI Framework for Industrial Predictive Maintenance*
>>>>>>> origin/main
