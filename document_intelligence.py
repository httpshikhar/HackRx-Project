"""
Document Intelligence System using Azure OpenAI and Pinecone

This module provides functionality to:
1. Process PDF documents and extract meaningful chunks
2. Generate embeddings using Azure OpenAI
3. Store embeddings in Pinecone vector database
4. Perform semantic search and answer questions
"""

import os
import re
import hashlib
import asyncio
import logging
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse
import requests
from io import BytesIO

# Third-party imports
import PyPDF2
from openai import AzureOpenAI
from pinecone import Pinecone
from conflict_detector import ConflictDetector, enhance_rag_with_conflict_detection

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentIntelligenceSystem:
    """
    Document Intelligence System that processes PDFs, generates embeddings,
    stores them in Pinecone, and answers questions using Azure OpenAI.
    """
    
    def __init__(self):
        """Initialize the Document Intelligence System with required clients."""
        self._setup_clients()
        self._setup_pinecone_index()
    
    def _setup_clients(self):
        """Setup Azure OpenAI and Pinecone clients."""
        # Azure OpenAI setup for chat completion (o3-mini)
        self.azure_client_chat = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_O3_API_VERSION", "2025-01-31"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Azure OpenAI setup for embeddings (text-embedding-ada-002)
        self.azure_client_embeddings = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_EMBED_API_VERSION", "2024-02-01"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Pinecone setup
        self.pinecone_client = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        
        # Model configurations
        self.embedding_model = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-ada-002")
        self.chat_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "gpt-4")
        
        logger.info("Clients initialized successfully")
    
    def _setup_pinecone_index(self):
        """Setup or connect to Pinecone index."""
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "document-intelligence")
        
        try:
            # Try to connect to existing index
            self.pinecone_index = self.pinecone_client.Index(self.index_name)
            logger.info(f"Connected to existing Pinecone index: {self.index_name}")
        except Exception as e:
            logger.error(f"Could not connect to Pinecone index: {e}")
            logger.info("Please ensure the Pinecone index exists and is properly configured")
            raise
    
    def _download_pdf(self, pdf_url: str) -> bytes:
        """Download PDF from URL or read from local file and return bytes."""
        try:
            # Check if it's a local file path
            if os.path.exists(pdf_url):
                logger.info(f"Reading local PDF file: {pdf_url}")
                with open(pdf_url, 'rb') as f:
                    return f.read()
            else:
                # Download from URL
                logger.info(f"Downloading PDF from URL: {pdf_url}")
                response = requests.get(pdf_url, stream=True, timeout=30)
                response.raise_for_status()
                return response.content
        except Exception as e:
            logger.error(f"Failed to get PDF from {pdf_url}: {e}")
            raise
    
    def _extract_text_from_pdf(self, pdf_content: bytes) -> str:
        """Extract text from PDF content."""
        try:
            pdf_file = BytesIO(pdf_content)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text
        except Exception as e:
            logger.error(f"Failed to extract text from PDF: {e}")
            raise
    
    def _chunk_text_into_clauses(self, text: str, max_chunk_size: int = 1000) -> List[str]:
        """
        Intelligently chunk text into meaningful clauses/paragraphs.
        
        Args:
            text: The text to chunk
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks
        """
        # Clean the text
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Split by common clause/paragraph separators
        clause_separators = [
            r'\n\n+',  # Double newlines (paragraphs)
            r'\.\s+(?=[A-Z])',  # Sentence endings followed by capital letters
            r';\s+',  # Semicolons
            r':\s+(?=[A-Z])',  # Colons followed by capital letters
        ]
        
        chunks = [text]
        
        # Apply separators in order
        for separator in clause_separators:
            new_chunks = []
            for chunk in chunks:
                split_chunks = re.split(separator, chunk)
                new_chunks.extend([c.strip() for c in split_chunks if c.strip()])
            chunks = new_chunks
        
        # Merge small chunks and split large ones
        final_chunks = []
        current_chunk = ""
        
        for chunk in chunks:
            # If adding this chunk exceeds max size and current chunk is not empty
            if len(current_chunk + " " + chunk) > max_chunk_size and current_chunk:
                final_chunks.append(current_chunk.strip())
                current_chunk = chunk
            else:
                current_chunk = (current_chunk + " " + chunk).strip()
        
        # Add the last chunk if it exists
        if current_chunk:
            final_chunks.append(current_chunk)
        
        # Filter out very short chunks (less than 50 characters)
        final_chunks = [chunk for chunk in final_chunks if len(chunk) >= 50]
        
        return final_chunks
    
    async def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text using Azure OpenAI."""
        try:
            response = self.azure_client_embeddings.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            raise
    
    def _generate_document_id(self, pdf_url: str) -> str:
        """Generate a unique document ID from PDF URL."""
        return hashlib.md5(pdf_url.encode()).hexdigest()
    
    async def process_document(self, pdf_url: str) -> Dict[str, Any]:
        """
        Process a PDF document: extract text, chunk it, generate embeddings, and store in Pinecone.
        
        Args:
            pdf_url: URL of the PDF document to process
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Processing document: {pdf_url}")
        
        try:
            # Generate document ID
            doc_id = self._generate_document_id(pdf_url)
            
            # Download and extract text from PDF
            pdf_content = self._download_pdf(pdf_url)
            text = self._extract_text_from_pdf(pdf_content)
            
            if not text.strip():
                raise ValueError("No text extracted from PDF")
            
            # Chunk text into clauses
            chunks = self._chunk_text_into_clauses(text)
            logger.info(f"Created {len(chunks)} chunks from document")
            
            # Generate embeddings for all chunks
            vectors_to_upsert = []
            clauses_data = []  # For database storage
            
            for i, chunk in enumerate(chunks):
                embedding = await self._generate_embedding(chunk)
                vector_id = f"{doc_id}_chunk_{i}"
                
                vector_data = {
                    "id": vector_id,
                    "values": embedding,
                    "metadata": {
                        "document_id": doc_id,
                        "document_url": pdf_url,
                        "chunk_index": i,
                        "text": chunk,
                        "chunk_length": len(chunk)
                    }
                }
                vectors_to_upsert.append(vector_data)
                
                # Prepare clause data for database
                clause_data = {
                    "document_id": doc_id,
                    "pinecone_vector_id": vector_id,
                    "chunk_index": i,
                    "clause_text": chunk,
                    "clause_length": len(chunk),
                    "metadata": {
                        "document_url": pdf_url
                    }
                }
                clauses_data.append(clause_data)
            
            # Upsert vectors to Pinecone in batches
            batch_size = 100
            for i in range(0, len(vectors_to_upsert), batch_size):
                batch = vectors_to_upsert[i:i + batch_size]
                self.pinecone_index.upsert(vectors=batch)
            
            logger.info(f"Successfully processed and stored document {doc_id}")
            
            return {
                "success": True,
                "document_id": doc_id,
                "chunks_processed": len(chunks),
                "total_characters": len(text),
                "pdf_url": pdf_url,
                "clauses_data": clauses_data  # For database integration
            }
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            return {
                "success": False,
                "error": str(e),
                "pdf_url": pdf_url
            }
    
    async def query_document(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Query the document database and generate an answer using retrieved clauses.
        
        Args:
            question: The question to ask
            top_k: Number of top relevant clauses to retrieve
            
        Returns:
            Dictionary with answer and source clauses
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # Generate embedding for the question
            question_embedding = await self._generate_embedding(question)
            
            # Search in Pinecone
            search_results = self.pinecone_index.query(
                vector=question_embedding,
                top_k=top_k,
                include_metadata=True
            )
            
            if not search_results.matches:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "source_clauses": [],
                    "confidence": 0.0
                }
            
            # Extract relevant clauses
            source_clauses = []
            context_text = ""
            
            for match in search_results.matches:
                clause_info = {
                    "text": match.metadata["text"],
                    "document_url": match.metadata["document_url"],
                    "chunk_index": match.metadata["chunk_index"],
                    "similarity_score": float(match.score)
                }
                source_clauses.append(clause_info)
                context_text += f"\n\n{match.metadata['text']}"
            
            # Generate answer using Azure OpenAI Chat Completion
            answer = await self._generate_answer(question, context_text)
            
            # Calculate average confidence score
            avg_confidence = sum(clause["similarity_score"] for clause in source_clauses) / len(source_clauses)
            
            return {
                "success": True,
                "answer": answer,
                "source_clauses": source_clauses,
                "confidence": avg_confidence,
                "total_clauses_found": len(source_clauses)
            }
            
        except Exception as e:
            logger.error(f"Failed to process query: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
    
    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using Azure OpenAI chat completion."""
        try:
            system_prompt = """You are a helpful document analysis assistant. 
            Answer questions based on the provided context from document clauses. 
            If the context doesn't contain enough information to answer the question, 
            say so clearly. Be accurate and cite specific information from the context when possible."""
            
            user_prompt = f"""Context from document clauses:
{context}

Question: {question}

Please provide a comprehensive answer based on the context provided above."""
            
            response = self.azure_client_chat.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Failed to generate answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    async def query_document_with_conflict_detection(self, question: str, top_k: int = 10) -> Dict[str, Any]:
        """
        Enhanced query that includes conflict detection and resolution using o3-mini reasoning.
        
        Args:
            question: The question to ask
            top_k: Number of top relevant clauses to retrieve (increased for better conflict detection)
            
        Returns:
            Dictionary with enhanced response including conflict analysis
        """
        logger.info(f"Processing enhanced query with conflict detection: {question}")
        
        try:
            # First, get the standard RAG response
            standard_response = await self.query_document(question, top_k)
            
            if not standard_response["success"]:
                return standard_response
            
            # Initialize conflict detector
            conflict_detector = ConflictDetector()
            
            # Enhance the response with conflict detection
            enhanced_response = await enhance_rag_with_conflict_detection(
                standard_answer=standard_response["answer"],
                relevant_clauses=standard_response["source_clauses"],
                user_query=question,
                conflict_detector=conflict_detector
            )
            
            # Merge with original response data
            enhanced_response.update({
                "success": True,
                "original_confidence": standard_response.get("confidence", 0.0),
                "total_clauses_found": standard_response.get("total_clauses_found", 0)
            })
            
            return enhanced_response
            
        except Exception as e:
            logger.error(f"Failed to process enhanced query: {e}")
            return {
                "success": False,
                "error": str(e),
                "question": question
            }
    
    async def delete_document(self, pdf_url: str) -> Dict[str, Any]:
        """Delete all vectors for a specific document from Pinecone."""
        try:
            doc_id = self._generate_document_id(pdf_url)
            
            # Get all vector IDs for this document
            # First, let's try to query for vectors with this document_id in metadata
            try:
                # Query to find vectors for this document
                query_response = self.pinecone_index.query(
                    vector=[0] * 1536,  # Dummy vector for metadata filtering
                    top_k=10000,  # Large number to get all vectors
                    filter={"document_id": doc_id},
                    include_metadata=False
                )
                
                # Extract vector IDs
                vector_ids = [match.id for match in query_response.matches]
                
                if vector_ids:
                    # Delete by specific IDs
                    self.pinecone_index.delete(ids=vector_ids)
                    logger.info(f"Deleted {len(vector_ids)} vectors for document {doc_id}")
                else:
                    logger.info(f"No vectors found for document {doc_id}")
                    
            except Exception as query_error:
                # If metadata filtering doesn't work, try alternative approach
                logger.warning(f"Metadata filtering failed: {query_error}")
                logger.info("Note: Document vectors may still exist in the database")
            
            return {
                "success": True,
                "document_id": doc_id,
                "message": "Delete operation completed"
            }
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Convenience functions for easier usage
async def process_document(pdf_url: str) -> Dict[str, Any]:
    """
    Process a document (convenience function).
    
    Args:
        pdf_url: URL of the PDF to process
        
    Returns:
        Processing results
    """
    system = DocumentIntelligenceSystem()
    return await system.process_document(pdf_url)


async def query_document(question: str, top_k: int = 5) -> Dict[str, Any]:
    """
    Query documents (convenience function).
    
    Args:
        question: Question to ask
        top_k: Number of relevant clauses to retrieve
        
    Returns:
        Query results with answer and sources
    """
    system = DocumentIntelligenceSystem()
    return await system.query_document(question, top_k)


if __name__ == "__main__":
    # Example usage
    async def main():
        # Initialize system
        system = DocumentIntelligenceSystem()
        
        # Example PDF URL (replace with actual PDF)
        pdf_url = "https://example.com/sample-document.pdf"
        
        # Process document
        print("Processing document...")
        result = await system.process_document(pdf_url)
        print(f"Processing result: {result}")
        
        if result["success"]:
            # Query the document
            print("\nQuerying document...")
            query_result = await system.query_document(
                "What are the main points discussed in the document?",
                top_k=3
            )
            
            print(f"Query result: {query_result}")
            
            if query_result["success"]:
                print(f"\nAnswer: {query_result['answer']}")
                print(f"\nSource clauses ({len(query_result['source_clauses'])}):")
                for i, clause in enumerate(query_result['source_clauses'], 1):
                    print(f"{i}. Score: {clause['similarity_score']:.3f}")
                    print(f"   Text: {clause['text'][:200]}...")
                    print()
    
    # Uncomment to run the example
    # asyncio.run(main())
