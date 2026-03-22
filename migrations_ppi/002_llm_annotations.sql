-- Migration 002: Add protein annotations and publication abstracts for LLM benchmark
-- Adds function_description, go_terms, domain_annotations to proteins table
-- Creates ppi_publication_abstracts table for PubMed abstract storage

INSERT OR IGNORE INTO schema_migrations (version) VALUES ('002');

ALTER TABLE proteins ADD COLUMN function_description TEXT;
ALTER TABLE proteins ADD COLUMN go_terms TEXT;
ALTER TABLE proteins ADD COLUMN domain_annotations TEXT;

CREATE TABLE IF NOT EXISTS ppi_publication_abstracts (
    pmid INTEGER PRIMARY KEY,
    title TEXT,
    abstract TEXT NOT NULL,
    publication_year INTEGER,
    journal TEXT,
    fetched_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
