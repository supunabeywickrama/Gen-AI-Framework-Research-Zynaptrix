# Industrial Manual AI Copilot 🏭

This project is a complete, enterprise-grade **Document Intelligence Platform** engineered to parse complex industrial manuals into an interactive Multimodal Retrieval-Augmented Generation (RAG) system. 

It allows engineers and technicians to ask natural language questions about machine maintenance and immediately receive high-quality AI-reasoned answers alongside the exact, context-relevant schematic diagrams, figures, and tables from the manual.

## Key Features
* **PDF Processing**: Converts raw manual pages into high-resolution images.
* **Layout Detection (YOLOv8)**: Uses computer vision to identify headers, text blocks, images, and tables.
* **Optical Character Recognition (EasyOCR)**: Extracts pristine textual data from detected text zones.
* **Contextual Chunking**: Uses a sliding window algorithm to overlap chunks and prevent breaking contextual meaning.
* **OpenAI Embeddings & pgvector**: Vectorizes chunks and stores them in a robust PostgreSQL database.
* **Context-Aware Diagram Retrieval**: Features a novel PyMuPDF Trigram algorithm that dynamically maps semantic text chunks back to their origin physical pages, retrieving the precise diagrams associated with an answer.
* **Streamlit Web Dashboard**: An intuitive graphical interface to query the Copilot.

---

## 🚀 How to Install and Run Locally

If you are cloning this repository for the very first time, follow these precise steps to get the full pipeline running.

### 1. Prerequisites
- **Python 3.10+**
- **Docker Desktop** (Required to spin up the `pgvector` database).
- **Poppler (For Windows Users Only)**: `pdf2image` requires Poppler to extract pages from PDFs. 
  1. Download the latest Poppler Windows release from [GitHub](https://github.com/oschwartz10612/poppler-windows/releases/).
  2. Extract it (e.g., `C:\poppler`).
  3. Add `C:\poppler\Library\bin` to your System `PATH` environment variables.

### 2. Clone the Repository & Environment Setup
Clone the repository and set up a virtual Python environment so dependencies don't conflict:

```bash
git clone <your-repo-link>
cd Gen-AI-Framework-Research-Zynaptrix

# Create and activate a Virtual Environment
python -m venv .venv
# For Windows:
.venv\Scripts\activate
# For Mac/Linux:
source .venv/bin/activate

# Install all requisite dependencies
pip install -r requirements.txt
```

### 3. Setup Open AI Credentials
Create a `.env` file in the root directory and securely add your OpenAI API Key.

```env
OPENAI_API_KEY=sk-proj-your-actual-api-key-here
```

### 4. Provide a Manual
Place any industrial PDF manual you wish to digest inside `industrial_manual_parser/data/manuals/`.
*(By default, the Espressif ESP32 Datasheet is used as an example)*.

---

## ⚙️ Running the Parsing Pipeline

Run the following commands sequentially from the root project directory to digest the manual through the entire AI pipeline:

### Step A: Start the Vector Database
Initialize the `pgvector` PostgreSQL container:
```bash
docker-compose up -d
```

### Step B: Run the parsers
Navigate to the module directory or run the scripts as modules:
```bash
cd industrial_manual_parser

# 1. Convert PDF Manual to Images
python parser/pdf_loader.py

# 2. Detect Document Layouts (YOLOv8)
python parser/layout_detector.py

# 3. Extract Text (EasyOCR) & Chunking
python processing/ocr_and_chunk.py

# 4. Generate OpenAI Embeddings & Store in Vector DB
python embeddings/openai_embed_chunks.py
```

---

## 📊 Running the Dashboard

Once the database is populated, launch the Streamlit graphical interface:

```bash
# Assuming you are still inside the `industrial_manual_parser` directory
streamlit run dashboard/app.py
```

The app will launch in your browser at `http://localhost:8501`. 

Type questions exactly as you would to a knowledgeable maintenance engineer (e.g., *"How do I power up the chip?"*) and observe the multimodal retrieval correctly provide the text guidelines along with the associated technical schematics!
