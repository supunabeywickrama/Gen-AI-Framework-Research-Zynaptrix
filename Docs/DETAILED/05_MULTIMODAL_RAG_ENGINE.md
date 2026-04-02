# Part 5: Unified Multimodal RAG Engine

## 5.1 Overview

The `unified_rag/` package is an independent subsystem responsible for:
1. **Ingestion**: Parsing PDFs, extracting text/images/tables, captioning images, chunking, embedding, and storing in PostgreSQL/pgvector.
2. **Retrieval**: Semantic vector search across two independent sources (manuals and field history).
3. **Generation**: Synthesizing retrieved context into LLM-generated diagnostic responses.

This subsystem is architecturally isolated — it has its own `config.py`, its own `database.py` session factory, and its own API router. It can theoretically be deployed as a microservice.

---

## 5.2 Configuration (`unified_rag/config.py`)

Uses `pydantic-settings` with a `BaseSettings` class:

```python
class Settings(BaseSettings):
    database_url_env: Optional[str] = Field(None, alias="DATABASE_URL")
    postgres_user:     str = "myuser"
    postgres_password: str = "mypassword"
    postgres_db:       str = "rag_db"
    postgres_host:     str = "localhost"
    postgres_port:     int = 5433
    openai_api_key:    str = os.getenv("OPENAI_API_KEY", "")

    @property
    def database_url(self) -> str:
        if self.database_url_env:
            return self.database_url_env
        return f"postgresql://{user}:{password}@{host}:{port}/{db}"
```

**Priority logic:** If `DATABASE_URL` env var is set (e.g., Neon cloud connection string), it takes precedence. This allows seamless switching between local dev PostgreSQL and cloud-hosted Neon.

---

## 5.3 Database Models (`unified_rag/db/models.py`)

All five tables use `pgvector`'s `Vector` column type for storing embedding arrays. The `extend_existing=True` in `__table_args__` allows re-declaration across multiple imports without SQLAlchemy errors.

### 5.3.1 `ManualChunk`

Primary vector store table. Every searchable unit of knowledge (text block, image description, table) is stored as one row.

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `manual_id` | String (indexed) | Links chunk to a specific machine manual (e.g., "Zynaptrix_9000") |
| `type` | String | `"text"`, `"image"`, or `"table"` — used for retrieval filtering |
| `content` | Text | Raw text or GPT-4o-generated vision description |
| `embedding` | Vector | 1536-dimensional OpenAI embedding array |
| `page` | Integer | Source page number in the PDF |
| `path` | String (nullable) | Filesystem path to extracted image file |

**Why `manual_id` and not `machine_id`?** Multiple machines can share the same manual. The `Machine` table creates the N:1 mapping from `machine_id` → `manual_id`. This allows the Zynaptrix-9000 manual to serve both PUMP-001 and a future PUMP-002 without re-ingesting the document.

### 5.3.2 `Machine`

Registry of deployed industrial assets.

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `machine_id` | String (unique, indexed) | Business key: "PUMP-001", "LATHE-002", etc. |
| `name` | String | Human-readable machine name |
| `location` | String | Physical location (e.g., "Hall A - Section 4") |
| `manual_id` | String | FK-like reference to `ManualChunk.manual_id` |

### 5.3.3 `AnomalyRecord`

Immutable log of every confirmed anomaly event.

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Referenced by `ChatMessage.anomaly_id` |
| `machine_id` | String (indexed) | Which machine triggered the alert |
| `timestamp` | String | ISO-8601 UTC timestamp of detection |
| `type` | String | Anomaly category from `machine_state` |
| `score` | Integer | Normalized health score (0–100) |
| `sensor_data` | Text | JSON-serialized dict of sensor readings at fault time |
| `resolved` | Boolean | Default False; set True by `resolve_incident()` |

### 5.3.4 `ChatMessage`

Stores every message in every diagnostic conversation.

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Sequential message order |
| `anomaly_id` | Integer FK | Links to `anomaly_records.id`. NULL for general queries |
| `role` | String | `"agent"` or `"user"` |
| `content` | Text | Raw message content (may contain JSON, Markdown, procedure tags) |
| `timestamp` | String | ISO-8601 |
| `images` | Text | JSON-encoded list of image URLs served by `/static/` |
| `message_metadata` | Text | JSON dict. Used for `completed_tasks` tracking (HITL sign-off state) |

**Note on `message_metadata`:** The SQLAlchemy column is mapped as `Column("metadata", Text)` but accessed as `message_metadata` in Python to avoid conflict with SQLAlchemy's own `metadata` attribute. This is a common gotcha.

### 5.3.5 `InteractionMemory`

Vectorized institutional knowledge from resolved incidents. This is the "organizational memory" that makes the system smarter with each use.

| Column | Type | Description |
|---|---|---|
| `id` | Integer PK | Auto-increment |
| `machine_id` | String (indexed) | Which machine's incident produced this memory |
| `manual_id` | String | Always `"Historical_Knowledge"` to distinguish from manual chunks |
| `summary` | Text | GPT-4o summarized action steps (≤150 words) |
| `operator_fix` | Text | Verbatim operator description of the physical repair performed |
| `embedding` | Vector | 1536-dimensional embedding of the summary |
| `timestamp` | String | When the incident was resolved |

---

## 5.4 Ingestion Pipeline (`unified_rag/ingestion/`)

### 5.4.1 `pipeline.py` — `process_manual(file_path, manual_id)`

This is the top-level orchestrator for ingesting a new PDF manual. It runs in 4 sequential stages:

**Stage 1: PDF Parsing**
```python
parsed_data = parser.parse_pdf(file_path, manual_id)
```
Returns a list of raw dicts: `[{"type": "text"|"image"|"table", "content": ..., "page": ..., "path": ...}]`

**Stage 2: Vision Captioning**
Iterates over the parsed items. For every `type == "image"` item:
```python
desc = captioner.generate_caption(item["path"])
item["content"] = f"[IMAGE REFERENCE: {path}]\n[VISUAL DESCRIPTION]: {desc}"
```
The vision description is inlined as the `content` field so it can be embedded as text later. This allows the same 1536-dimensional embedding space to search images by meaning.

**Stage 3: Contextual Chunking**
```python
chunks = chunker.chunk_data(parsed_data, manual_id)
```
Adds `manual_id` to each chunk dict and applies sliding-window splitting for long texts.

**Stage 4: Batch Embedding and DB Storage**

Uses a mini-batch loop (batch_size=20) with exponential-backoff retry logic (max 3 retries):

```python
for i in range(0, len(chunks), 20):
    batch = chunks[i:i+20]
    # A. Embed all chunks in batch (no DB connection open)
    for chunk in batch:
        emb = embedder.embed_text(chunk["content"])

    # B. Open DB session only for write
    with SessionLocal() as db:
        for chunk, emb in batch_to_save:
            db_chunk = ManualChunk(...)
            db.add(db_chunk)
        db.commit()
```

**Critical architecture note:** The DB session is opened **only** for the write phase, not during the embedding API calls. This prevents connection timeouts from occurring while waiting for OpenAI API responses (which can take several seconds per chunk).

### 5.4.2 `parser.py` — `DocumentParser`

The most complex module in the RAG system. Uses a multi-pass strategy:

**Pass 1: YOLOv8 DocLayNet Layout Detection**

For each PDF page:
1. Renders the page to a PIL Image at 150 DPI: `page.get_pixmap(dpi=150)`
2. Runs YOLOv8 inference on the PIL image: `self.layout_model(img, verbose=False)`
3. Iterates detected bounding boxes:
   - Classes `"picture"` or `"figure"` → **Image extraction path**:
     - Crops the PIL image to the bounding box
     - Saves it as `{manual_id}_p{page}_img{idx}.png` in `data/extracted/`
     - Adds `{"type": "image", "path": ..., "page": ...}` dict (no content yet — filled by captioner)
   - Classes `"text"`, `"title"`, `"list"` → **Text extraction path**:
     - Scales YOLO pixel coordinates back to PyMuPDF's point coordinate system
     - Calls `page.get_text("text", clip=rect)` to extract only the text within that bounding box
     - Adds `{"type": "text", "content": ..., "page": ...}` dict

**Why 150 DPI?** At 150 DPI, images are large enough for accurate YOLOv8 inference while small enough to not exhaust memory on long documents.

**Scale Conversion Formula:**
```python
x_scale = page.rect.width / pix.width    # pts_per_pixel (x)
y_scale = page.rect.height / pix.height  # pts_per_pixel (y)
rect = fitz.Rect(x1*x_scale, y1*y_scale, x2*x_scale, y2*y_scale)
```
This converts from PIL pixel coordinates (top-left origin) to PyMuPDF's point coordinate system.

**Fallback Pass: PyMuPDF Heuristics** (if YOLO weights unavailable)
Uses `page.get_text("blocks")` which returns raw text blocks as a list of tuples. Block type `6 == 0` filters for text blocks (not image blocks).

**Pass 2: Camelot Table Extraction**
After YOLO processing, runs Camelot on each page using `"lattice"` flavor (for tables with visible grid lines). Output is a pandas DataFrame converted to JSON string for embedding.

**Constructor initialization with graceful degradation:**
All three libraries (ultralytics, easyocr, camelot) are imported inside `try/except ImportError` blocks. If a library is not installed, the corresponding capability is disabled but the rest of the pipeline still functions.

### 5.4.3 `captioner.py` — `ImageCaptioner`

Uses `openai.OpenAI` client with the GPT-4o Vision API.

**`generate_caption(image_path) -> str`**

1. Base64-encodes the image file: `base64.b64encode(open(path, "rb").read()).decode("utf-8")`
2. Sends a multimodal chat completion:
   ```python
   messages=[{
     "role": "user",
     "content": [
       {"type": "text",      "text": "Describe this industrial/technical manual image in extremely high detail..."},
       {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
     ]
   }]
   ```
3. `max_tokens=300` — enough for a detailed technical description without excessive cost per image
4. Temperature is not set (defaults to 1.0 for creative description)
5. Returns the caption string; on API failure returns `"Image description unavailable."`

**Why Vision Captioning Instead of CLIP?**
CLIP embeds images directly into a 512-dimensional space. The problem is CLIP's embedding space is misaligned with OpenAI's 1536-dimensional text embedding space — semantic similarity search would be comparing apples to oranges. By converting images to *text descriptions* first, all chunks live in the same 1536-dimensional OpenAI embedding space, enabling unified similarity search across modalities.

### 5.4.4 `chunker.py` — `ContextualChunker`

**`chunk_data(parsed_data, manual_id) -> list[dict]`**

Applies word-count-based sliding window chunking:

- For text/table items ≤ `chunk_size` (500) words: kept as single chunk
- For longer text: slices into overlapping windows of `chunk_size=500` words with `overlap=100` words of context shared between consecutive chunks:
  ```python
  for i in range(0, len(words), chunk_size - overlap):
      chunk_words = words[i:i + chunk_size]
  ```
  The 100-word overlap ensures that sentences spanning a chunk boundary appear in both chunks, preventing loss of context.

- For image items: passed through unchanged (content is now the vision caption string)

Each output chunk dict contains: `manual_id`, `type`, `content`, `page`, and optionally `path`.

### 5.4.5 `embedder.py` — `MultimodalEmbedder`

A thin singleton wrapper around the OpenAI embeddings API.

```python
class MultimodalEmbedder:
    def __init__(self):
        self.openai_client = OpenAI(api_key=settings.openai_api_key)
        self.text_model = "text-embedding-3-small"  # 1536 dimensions

    def embed_text(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            input=text, model=self.text_model
        )
        return response.data[0].embedding

embedder = MultimodalEmbedder()  # module-level singleton
```

The module-level `embedder` instance is imported by both `pipeline.py` and `retriever.py`. This means OpenAI client initialization happens once at module import time, not on every request.

---

## 5.5 Retrieval System (`unified_rag/retrieval/`)

### 5.5.1 `retriever.py` — `RetrievalEngine`

**`retrieve(db, query, manual_id, machine_id) -> dict`**

Performs three independent vector searches in one call:

**Search 1: Manual Text + Table Chunks**
```python
text_results = db.query(ManualChunk).filter(
    ManualChunk.manual_id == manual_id,
    ManualChunk.type.in_(["text", "table"])
).order_by(
    ManualChunk.embedding.cosine_distance(query_emb)
).limit(self.top_k_text).all()
```
`cosine_distance` is provided by the `pgvector.sqlalchemy` extension. Lower distance = higher semantic similarity. Returns top 3 chunks.

**Search 2: Manual Image Chunks**
Same query but filtered to `type == "image"`. Returns top 1 image chunk. The "content" field of this chunk is the GPT-4o vision caption, which was searched against the query embedding.

**Search 3: Interaction Memory (Field History)**
```python
if machine_id:
    historical_fixes = db.query(InteractionMemory).filter(
        InteractionMemory.machine_id == machine_id
    ).order_by(
        InteractionMemory.embedding.cosine_distance(query_emb)
    ).limit(self.top_k_memory).all()
```
**Critically**, this search filters by `machine_id` (not `manual_id`). This ensures that historical fixes for PUMP-001 are never used to advise on LATHE-002 faults — even if the fault symptoms are textually similar.

**Error Handling:** Each of the three searches is wrapped in `try/except`. If any one fails (e.g., the `pgvector` extension isn't installed, or the `interaction_memory` table doesn't exist yet), it gracefully returns an empty list for that source without failing the entire retrieval.

### 5.5.2 `rag.py` — `RAGGenerator`

**`generate_response(query, manual_id, machine_id) -> dict`**

**Stage 1: Retrieval**
```python
retrieved_data = self.retriever.retrieve(db, query, manual_id, machine_id)
```

**Stage 2: Context Assembly**
Iterates retrieved chunks and builds a structured context string:
```
--- Manual Context 1 (Page 12) ---
[content of chunk 1]

--- Manual Context 2 (Page 15) ---
[content of chunk 2]

--- Image Description 1 (Page 8) ---
[IMAGE REFERENCE: data/extracted/pump_p8_img0.png]
[VISUAL DESCRIPTION]: The diagram shows a cross-section of a centrifugal pump ...

--- PREVIOUS FIX 1 (2026-01-15 09:30:00) ---
Summary: Replaced worn bearing set in drive end housing ...
Operator Actions: Removed coupling guard, extracted shaft bearing ...
```

**Stage 3: Mode Detection**
```python
is_procedure_request = (
    "Generate full step-by-step repair procedure" in query or
    "FULL structured JSON repair procedure" in query
)
```

**Stage 4: LLM Call**
```python
response = openai.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": f"Technician query: '{query}'"}
    ],
    max_tokens=3000,
    temperature=0.1   # very low temperature for deterministic, factual output
)
```

`temperature=0.1` is intentional — diagnostic procedures need to be consistent and factual, not creative.

**Stage 5: Return**
```python
return {
    "answer":  answer_text,
    "images":  image_reference_paths,
    "pages":   sorted(list(pages_set))
}
```

### 5.5.3 Prompt Templates

**Mode 1 Summary Prompt:**
Contains 7 absolute rules preventing the LLM from generating procedure steps. The key rule is the mandatory ending tag: the LLM **must** end its response with `[SUGGESTION: Generate full step-by-step repair procedure]`. This tag is parsed by the frontend copilotSlice to decide whether to show the "Start Guided Repair" button.

**Mode 2 Procedure Prompt:**
Instructs the LLM to output **only** the JSON object wrapped between `[PROCEDURE_START]` and `[PROCEDURE_END]`. It provides the exact JSON schema as a literal string in the prompt, including:
- Nested `phases → subphases → tasks` hierarchy
- Required `"type": "safety"` as the first phase
- Required `"critical": true` on safety-critical tasks
- `[IMAGE_N]` tag syntax for inline diagram references

GPT-4o reliably follows this format at temperature=0.1, but the Critic node validates it to catch any formatting drift.

---

## 5.6 RAG API Endpoints (`unified_rag/api/endpoints.py`)

### `POST /ingest-manual`

**Request:** `multipart/form-data` with `manual_id: str` (Form field) and `file: UploadFile` (Must be `.pdf`)

**Processing:**
1. Validates file extension
2. Saves uploaded PDF to `data/uploads/{manual_id}_{filename}.pdf`
3. Calls `process_manual(file_path, manual_id)` synchronously
4. Returns `{"message": "Manual ingested successfully", "details": {"status": "success", "chunks_processed": N}}`

**Note:** Currently runs synchronously on the main thread. For large manuals (100+ pages), this can block the server for several minutes. A future improvement would use `BackgroundTasks` (the `background_tasks` parameter is already imported but not used).

### `POST /machines`, `GET /machines`, `GET /machines/{machine_id}`, `POST /machines/delete/{machine_id}`

Full CRUD for the `Machine` registry. The `POST /machines` endpoint implements upsert logic — if the machine already exists, it updates all fields rather than throwing a 409 conflict error.

### `POST /chat`

**Request:** `{"manual_id": str, "query": str}`

Direct RAG endpoint bypassing the LangGraph pipeline. Used for manual knowledge base queries without an active anomaly context. Returns `{"answer": str, "images": list[str], "pages": list[int]}`.
