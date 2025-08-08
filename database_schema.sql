-- Production Database Schema for Enhanced Document Intelligence System
-- Supports Azure OpenAI + Pinecone RAG with o3-mini conflict detection

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search optimization
CREATE EXTENSION IF NOT EXISTS "pgcrypto"; -- For secure hashing

-- 1. DOCUMENTS TABLE
-- Store document metadata, processing status, and blob URLs
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_url TEXT NOT NULL UNIQUE,
    document_hash VARCHAR(64) NOT NULL UNIQUE, -- MD5 hash for deduplication
    original_filename TEXT,
    file_size_bytes BIGINT,
    mime_type VARCHAR(100),
    processing_status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (processing_status IN ('pending', 'processing', 'completed', 'failed', 'archived')),
    processed_at TIMESTAMPTZ,
    total_clauses INTEGER DEFAULT 0,
    total_characters INTEGER DEFAULT 0,
    document_health_score DECIMAL(3,3), -- 0.000 to 1.000
    pinecone_namespace VARCHAR(100),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_error TEXT,
    metadata JSONB DEFAULT '{}' -- Store additional document metadata
);

-- 2. CLAUSES TABLE  
-- Individual clauses with Pinecone vector references
CREATE TABLE clauses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    pinecone_vector_id VARCHAR(255) NOT NULL UNIQUE, -- Reference to Pinecone vector
    chunk_index INTEGER NOT NULL,
    clause_text TEXT NOT NULL,
    clause_length INTEGER NOT NULL,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'text-embedding-ada-002',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB DEFAULT '{}',
    -- Ensure unique combination of document and chunk
    UNIQUE(document_id, chunk_index)
);

-- 3. CONFLICTS TABLE
-- Store detected conflicts between clause pairs with o3-mini reasoning
CREATE TABLE conflicts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conflict_hash VARCHAR(64) NOT NULL UNIQUE, -- Hash of clause pair for deduplication
    clause1_id UUID NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    clause2_id UUID NOT NULL REFERENCES clauses(id) ON DELETE CASCADE,
    conflict_type VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL CHECK (severity IN ('critical', 'high', 'medium', 'low', 'informational')),
    reasoning_model VARCHAR(50) NOT NULL DEFAULT 'o3-mini',
    reasoning_text TEXT NOT NULL,
    resolution_recommendation TEXT NOT NULL,
    confidence_score DECIMAL(3,3) NOT NULL, -- 0.000 to 1.000
    requires_human_review BOOLEAN NOT NULL DEFAULT false,
    similarity_score DECIMAL(4,3), -- Semantic similarity that triggered analysis
    detected_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analysis_duration_ms INTEGER, -- Time taken for o3-mini analysis
    metadata JSONB DEFAULT '{}',
    -- Ensure we don't analyze the same pair twice
    CHECK (clause1_id != clause2_id)
);

-- 4. QUERY_CACHE TABLE
-- Cache responses for repeated queries to optimize performance
CREATE TABLE query_cache (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_hash VARCHAR(64) NOT NULL, -- Hash of normalized query + document context
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    original_query TEXT NOT NULL,
    normalized_query TEXT NOT NULL, -- Cleaned/normalized version
    standard_answer TEXT NOT NULL,
    enhanced_answer TEXT, -- Answer after conflict detection
    source_clauses JSONB NOT NULL, -- Array of clause references and scores
    conflicts_detected INTEGER NOT NULL DEFAULT 0,
    conflict_analysis JSONB DEFAULT '[]', -- Detailed conflict information
    confidence_level VARCHAR(20) NOT NULL,
    requires_human_review BOOLEAN NOT NULL DEFAULT false,
    processing_time_ms INTEGER NOT NULL,
    model_versions JSONB NOT NULL, -- Track which models/versions were used
    cache_hits INTEGER NOT NULL DEFAULT 0,
    last_accessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    expires_at TIMESTAMPTZ NOT NULL DEFAULT (NOW() + INTERVAL '7 days'), -- Cache expiration
    -- Composite index for fast lookups
    UNIQUE(query_hash, document_id)
);

-- 5. ANALYTICS TABLE
-- Track query patterns, system performance, and usage metrics
CREATE TABLE analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(50) NOT NULL, -- 'query', 'document_processed', 'conflict_detected', etc.
    document_id UUID REFERENCES documents(id) ON DELETE SET NULL,
    session_id UUID, -- Track user sessions if applicable
    user_id VARCHAR(100), -- For authentication tracking
    query_text TEXT,
    response_time_ms INTEGER,
    api_calls_made INTEGER DEFAULT 0, -- Azure OpenAI API calls
    tokens_consumed INTEGER DEFAULT 0, -- Total tokens used
    cache_hit BOOLEAN DEFAULT false,
    error_occurred BOOLEAN DEFAULT false,
    error_message TEXT,
    metadata JSONB DEFAULT '{}',
    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 6. API_REQUESTS TABLE
-- Track hackathon API requests for monitoring and rate limiting
CREATE TABLE api_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    request_id VARCHAR(100) NOT NULL UNIQUE,
    bearer_token_hash VARCHAR(64), -- Hashed bearer token for security
    client_ip INET,
    user_agent TEXT,
    document_url TEXT NOT NULL,
    questions JSONB NOT NULL, -- Array of questions
    questions_count INTEGER NOT NULL,
    processing_status VARCHAR(20) NOT NULL DEFAULT 'processing' CHECK (processing_status IN ('processing', 'completed', 'failed', 'timeout')),
    total_processing_time_ms INTEGER,
    cache_hits INTEGER DEFAULT 0,
    api_calls_made INTEGER DEFAULT 0,
    tokens_consumed INTEGER DEFAULT 0,
    errors_encountered INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    response_data JSONB, -- Store the final response
    error_details JSONB DEFAULT '{}' -- Store error information
);

-- PERFORMANCE INDEXES
-- Critical indexes for query performance

-- Documents table indexes
CREATE INDEX idx_documents_url_hash ON documents(document_hash);
CREATE INDEX idx_documents_processing_status ON documents(processing_status);
CREATE INDEX idx_documents_created_at ON documents(created_at);
CREATE INDEX idx_documents_health_score ON documents(document_health_score) WHERE document_health_score IS NOT NULL;

-- Clauses table indexes
CREATE INDEX idx_clauses_document_id ON clauses(document_id);
CREATE INDEX idx_clauses_pinecone_id ON clauses(pinecone_vector_id);
CREATE INDEX idx_clauses_length ON clauses(clause_length);
-- GIN index for full-text search on clause text
CREATE INDEX idx_clauses_text_gin ON clauses USING GIN(to_tsvector('english', clause_text));

-- Conflicts table indexes
CREATE INDEX idx_conflicts_hash ON conflicts(conflict_hash);
CREATE INDEX idx_conflicts_clause1 ON conflicts(clause1_id);
CREATE INDEX idx_conflicts_clause2 ON conflicts(clause2_id);
CREATE INDEX idx_conflicts_severity ON conflicts(severity);
CREATE INDEX idx_conflicts_type ON conflicts(conflict_type);
CREATE INDEX idx_conflicts_detected_at ON conflicts(detected_at);
CREATE INDEX idx_conflicts_requires_review ON conflicts(requires_human_review) WHERE requires_human_review = true;

-- Query cache indexes
CREATE INDEX idx_query_cache_hash_doc ON query_cache(query_hash, document_id);
CREATE INDEX idx_query_cache_expires_at ON query_cache(expires_at);
CREATE INDEX idx_query_cache_last_accessed ON query_cache(last_accessed_at);
CREATE INDEX idx_query_cache_conflicts ON query_cache(conflicts_detected) WHERE conflicts_detected > 0;

-- Analytics table indexes
CREATE INDEX idx_analytics_event_type ON analytics(event_type);
CREATE INDEX idx_analytics_timestamp ON analytics(timestamp);
CREATE INDEX idx_analytics_document_id ON analytics(document_id) WHERE document_id IS NOT NULL;
CREATE INDEX idx_analytics_response_time ON analytics(response_time_ms) WHERE response_time_ms IS NOT NULL;
CREATE INDEX idx_analytics_errors ON analytics(error_occurred) WHERE error_occurred = true;

-- API requests indexes
CREATE INDEX idx_api_requests_token_hash ON api_requests(bearer_token_hash) WHERE bearer_token_hash IS NOT NULL;
CREATE INDEX idx_api_requests_created_at ON api_requests(created_at);
CREATE INDEX idx_api_requests_status ON api_requests(processing_status);
CREATE INDEX idx_api_requests_document_url ON api_requests(document_url);

-- TRIGGERS FOR AUTOMATIC TIMESTAMP UPDATES
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply the trigger to documents table
CREATE TRIGGER update_documents_updated_at 
    BEFORE UPDATE ON documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- UTILITY FUNCTIONS

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION clean_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM query_cache WHERE expires_at < NOW();
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get document processing statistics
CREATE OR REPLACE FUNCTION get_processing_stats()
RETURNS TABLE (
    total_documents BIGINT,
    completed_documents BIGINT,
    failed_documents BIGINT,
    pending_documents BIGINT,
    total_clauses BIGINT,
    total_conflicts BIGINT,
    avg_health_score NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*) as total_documents,
        COUNT(*) FILTER (WHERE processing_status = 'completed') as completed_documents,
        COUNT(*) FILTER (WHERE processing_status = 'failed') as failed_documents,
        COUNT(*) FILTER (WHERE processing_status = 'pending') as pending_documents,
        (SELECT COUNT(*) FROM clauses) as total_clauses,
        (SELECT COUNT(*) FROM conflicts) as total_conflicts,
        AVG(document_health_score) as avg_health_score
    FROM documents;
END;
$$ LANGUAGE plpgsql;

-- EXAMPLE QUERIES FOR VERIFICATION

-- Example: Find documents with critical conflicts
-- SELECT d.document_url, COUNT(c.*) as critical_conflicts
-- FROM documents d
-- JOIN clauses cl ON d.id = cl.document_id
-- JOIN conflicts c ON (cl.id = c.clause1_id OR cl.id = c.clause2_id)
-- WHERE c.severity = 'critical'
-- GROUP BY d.id, d.document_url
-- ORDER BY critical_conflicts DESC;

-- Example: Cache hit rate analysis
-- SELECT 
--     DATE_TRUNC('hour', created_at) as hour,
--     COUNT(*) as total_queries,
--     COUNT(*) FILTER (WHERE cache_hits > 0) as cache_hits,
--     ROUND(COUNT(*) FILTER (WHERE cache_hits > 0) * 100.0 / COUNT(*), 2) as cache_hit_rate
-- FROM query_cache
-- WHERE created_at > NOW() - INTERVAL '24 hours'
-- GROUP BY hour
-- ORDER BY hour DESC;
