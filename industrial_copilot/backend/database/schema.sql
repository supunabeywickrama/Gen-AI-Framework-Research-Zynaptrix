-- ─────────────────────────────────────────────────────────────────────────────
-- schema.sql — Neon PostgreSQL + pgvector schema
-- Industrial AI Copilot — Tea Bag Packing Machine
--
-- Run this once on your Neon database to initialise the schema.
-- Usage:  psql $NEON_DB_URL -f database/schema.sql
-- ─────────────────────────────────────────────────────────────────────────────

-- 1. Enable pgvector extension (must be done first)
CREATE EXTENSION IF NOT EXISTS vector;

-- ─────────────────────────────────────────────────────────────────────────────
-- 2. Machine Documents table
--    Stores embedded text chunks from maintenance manuals, SOPs, fault guides.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS machine_documents (
    id              SERIAL PRIMARY KEY,
    doc_id          TEXT        NOT NULL,               -- source document name/path
    chunk_index     INTEGER     NOT NULL,               -- chunk number within document
    title           TEXT        NOT NULL,               -- document title or section heading
    content         TEXT        NOT NULL,               -- raw text chunk
    fault_type      TEXT,                               -- tag: machine_fault | sensor_drift | etc.
    sensor          TEXT,                               -- related sensor tag (optional)
    embedding       VECTOR(768) NOT NULL,               -- OpenAI text-embedding-3-small (768-dim)
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast cosine similarity search
CREATE INDEX IF NOT EXISTS machine_documents_embedding_idx
    ON machine_documents
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- ─────────────────────────────────────────────────────────────────────────────
-- 3. Anomaly Events log table
--    Stores detected anomaly events + agent-generated maintenance advice.
-- ─────────────────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS anomaly_events (
    id              SERIAL PRIMARY KEY,
    event_time      TIMESTAMPTZ DEFAULT NOW(),
    machine_state   TEXT        NOT NULL,               -- e.g. machine_fault, sensor_drift
    suspect_sensor  TEXT,                               -- sensor that triggered the alert
    anomaly_score   FLOAT       NOT NULL,               -- reconstruction error score
    triggered_by    TEXT,                               -- consecutive anomaly count threshold
    agent_advice    TEXT,                               -- knowledge agent retrieval result
    resolved        BOOLEAN     DEFAULT FALSE,
    resolved_at     TIMESTAMPTZ
);

-- ─────────────────────────────────────────────────────────────────────────────
-- 4. Helper view — recent unresolved alerts
-- ─────────────────────────────────────────────────────────────────────────────
CREATE OR REPLACE VIEW active_alerts AS
    SELECT *
    FROM anomaly_events
    WHERE resolved = FALSE
    ORDER BY event_time DESC;
