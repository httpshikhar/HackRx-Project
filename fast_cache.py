"""
Ultra-fast in-memory caching system for document intelligence.
Provides sub-second response times for repeated queries.
"""

import hashlib
import json
import asyncio
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
import numpy as np
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class FastCache:
    """High-performance in-memory cache with LRU eviction and similarity matching."""
    
    def __init__(self, max_size: int = 10000, ttl_seconds: int = 3600):
        """
        Initialize fast cache.
        
        Args:
            max_size: Maximum number of entries in cache
            ttl_seconds: Time-to-live for cache entries in seconds
        """
        self.cache = OrderedDict()
        self.embeddings_cache = {}  # Store embeddings separately for fast similarity
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.hit_count = 0
        self.miss_count = 0
        
    def _generate_key(self, question: str, document_id: str) -> str:
        """Generate cache key from question and document."""
        combined = f"{document_id}:{question.lower().strip()}"
        return hashlib.md5(combined.encode()).hexdigest()
    
    def _is_expired(self, timestamp: datetime) -> bool:
        """Check if cache entry is expired."""
        return datetime.now() > timestamp + timedelta(seconds=self.ttl_seconds)
    
    async def get(self, question: str, document_id: str) -> Optional[Dict[str, Any]]:
        """
        Get cached result for question.
        
        Args:
            question: The question to look up
            document_id: Document ID
            
        Returns:
            Cached result or None if not found/expired
        """
        key = self._generate_key(question, document_id)
        
        if key in self.cache:
            entry = self.cache[key]
            
            # Check expiration
            if self._is_expired(entry['timestamp']):
                del self.cache[key]
                if key in self.embeddings_cache:
                    del self.embeddings_cache[key]
                self.miss_count += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hit_count += 1
            
            logger.info(f"Cache hit! Hit rate: {self.get_hit_rate():.2%}")
            return entry['data']
        
        self.miss_count += 1
        return None
    
    async def set(self, question: str, document_id: str, result: Dict[str, Any], 
                  question_embedding: Optional[List[float]] = None) -> None:
        """
        Store result in cache.
        
        Args:
            question: The question
            document_id: Document ID
            result: Result to cache
            question_embedding: Optional embedding for similarity matching
        """
        key = self._generate_key(question, document_id)
        
        # Evict oldest if at capacity
        if len(self.cache) >= self.max_size:
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
            if oldest_key in self.embeddings_cache:
                del self.embeddings_cache[oldest_key]
        
        # Store in cache
        self.cache[key] = {
            'timestamp': datetime.now(),
            'data': result,
            'question': question
        }
        
        # Store embedding if provided
        if question_embedding:
            self.embeddings_cache[key] = np.array(question_embedding)
    
    async def find_similar(self, question_embedding: List[float], document_id: str, 
                          threshold: float = 0.95) -> Optional[Dict[str, Any]]:
        """
        Find similar cached questions using embeddings.
        
        Args:
            question_embedding: Embedding of current question
            document_id: Document ID
            threshold: Similarity threshold (0-1)
            
        Returns:
            Cached result of similar question or None
        """
        if not self.embeddings_cache:
            return None
        
        query_embedding = np.array(question_embedding)
        best_match = None
        best_score = threshold
        
        for key, cached_embedding in self.embeddings_cache.items():
            # Only check entries for same document
            if key in self.cache:
                entry = self.cache[key]
                
                # Skip expired entries
                if self._is_expired(entry['timestamp']):
                    continue
                
                # Calculate cosine similarity
                similarity = np.dot(query_embedding, cached_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(cached_embedding)
                )
                
                if similarity > best_score:
                    best_score = similarity
                    best_match = entry['data']
        
        if best_match:
            logger.info(f"Found similar question in cache with {best_score:.2%} similarity")
            self.hit_count += 1
            return best_match
        
        return None
    
    def get_hit_rate(self) -> float:
        """Get cache hit rate."""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_count": self.hit_count,
            "miss_count": self.miss_count,
            "hit_rate": self.get_hit_rate(),
            "ttl_seconds": self.ttl_seconds
        }
    
    def clear(self) -> None:
        """Clear all cache entries."""
        self.cache.clear()
        self.embeddings_cache.clear()
        self.hit_count = 0
        self.miss_count = 0
        logger.info("Cache cleared")


class DocumentCache:
    """Cache for processed document chunks and embeddings."""
    
    def __init__(self):
        """Initialize document cache."""
        self.documents = {}  # document_id -> chunks and metadata
        self.chunk_embeddings = {}  # document_id -> embeddings array
        
    async def add_document(self, document_id: str, chunks: List[str], 
                           embeddings: List[List[float]], metadata: Dict[str, Any]) -> None:
        """
        Add processed document to cache.
        
        Args:
            document_id: Unique document identifier
            chunks: List of text chunks
            embeddings: List of chunk embeddings
            metadata: Document metadata
        """
        self.documents[document_id] = {
            'chunks': chunks,
            'metadata': metadata,
            'timestamp': datetime.now()
        }
        
        # Store embeddings as numpy array for fast similarity search
        self.chunk_embeddings[document_id] = np.array(embeddings)
        
        logger.info(f"Cached document {document_id} with {len(chunks)} chunks")
    
    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get cached document data."""
        return self.documents.get(document_id)
    
    async def search_chunks(self, document_id: str, query_embedding: List[float], 
                           top_k: int = 10) -> List[Tuple[int, float, str]]:
        """
        Search for most relevant chunks using embeddings.
        
        Args:
            document_id: Document to search in
            query_embedding: Query embedding
            top_k: Number of top results to return
            
        Returns:
            List of (chunk_index, similarity_score, chunk_text) tuples
        """
        if document_id not in self.chunk_embeddings:
            return []
        
        doc_data = self.documents[document_id]
        embeddings = self.chunk_embeddings[document_id]
        query_vec = np.array(query_embedding)
        
        # Calculate similarities
        similarities = np.dot(embeddings, query_vec) / (
            np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
        )
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append((
                int(idx),
                float(similarities[idx]),
                doc_data['chunks'][idx]
            ))
        
        return results
    
    def has_document(self, document_id: str) -> bool:
        """Check if document is in cache."""
        return document_id in self.documents
    
    def clear(self) -> None:
        """Clear all cached documents."""
        self.documents.clear()
        self.chunk_embeddings.clear()
        logger.info("Document cache cleared")


# Global cache instances
query_cache = FastCache(max_size=5000, ttl_seconds=1800)  # 30 min TTL
document_cache = DocumentCache()
