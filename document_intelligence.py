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
                # Convert Google Drive sharing URLs to direct download URLs
                if 'drive.google.com' in pdf_url and '/file/d/' in pdf_url:
                    # Extract file ID from sharing URL
                    file_id = pdf_url.split('/file/d/')[1].split('/')[0]
                    # Convert to direct download URL
                    pdf_url = f"https://drive.google.com/uc?export=download&id={file_id}"
                    logger.info(f"Converted Google Drive URL to direct download: {pdf_url}")
                
                # Download from URL
                logger.info(f"Downloading PDF from URL: {pdf_url}")
                
                # Set headers to avoid blocking
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                response = requests.get(pdf_url, stream=True, timeout=30, headers=headers, allow_redirects=True)
                response.raise_for_status()
                
                # Check if the response is actually a PDF
                content = response.content
                if not content.startswith(b'%PDF'):
                    # If it's not a PDF, it might be a HTML page (Google Drive confirmation)
                    if b'Google Drive - Virus scan warning' in content or b'virus scan' in content.lower():
                        logger.warning("Google Drive virus scan warning detected, trying to extract direct download link")
                        # Try to find the actual download URL in the HTML
                        content_str = content.decode('utf-8', errors='ignore')
                        import re
                        download_match = re.search(r'href="([^"]*uc\?export=download[^"]*)', content_str)
                        if download_match:
                            actual_url = download_match.group(1).replace('&amp;', '&')
                            logger.info(f"Found actual download URL: {actual_url}")
                            response = requests.get(actual_url, stream=True, timeout=30, headers=headers)
                            response.raise_for_status()
                            content = response.content
                        else:
                            raise Exception("Could not find direct download link in Google Drive response")
                    else:
                        raise Exception(f"Downloaded content is not a PDF. Content type: {response.headers.get('Content-Type', 'Unknown')}")
                
                return content
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
    
    def _chunk_text_into_clauses(self, text: str, max_chunk_size: int = 800) -> List[str]:
        """
        Intelligently chunk text into meaningful clauses/paragraphs with overlap for better accuracy.
        
        Args:
            text: The text to chunk
            max_chunk_size: Maximum size of each chunk
            
        Returns:
            List of text chunks with strategic overlap
        """
        # Clean and normalize the text
        text = re.sub(r'\s+', ' ', text.strip())
        text = re.sub(r'\n+', '\n', text)  # Normalize line breaks
        
        # Enhanced clause/paragraph separators for better semantic chunking
        clause_separators = [
            r'\n\n+',  # Double newlines (clear paragraph breaks)
            r'\n(?=[A-Z][^a-z]*:)',  # Section headers (e.g., "SECTION:", "COVERAGE:")
            r'\n(?=\d+\.\s)',  # Numbered lists
            r'\n(?=[A-Z][^a-z]*\s[A-Z])',  # All caps section headers
            r'\.\s+(?=[A-Z][a-z])',  # Sentence endings followed by capital letters
            r';\s+(?=[A-Z])',  # Semicolons followed by capital letters
            r':\s+(?=[A-Z][a-z])',  # Colons followed by capital letters (not headers)
        ]
        
        chunks = [text]
        
        # Apply separators in order of priority
        for separator in clause_separators:
            new_chunks = []
            for chunk in chunks:
                if len(chunk) > max_chunk_size * 1.5:  # Only split if chunk is significantly large
                    split_chunks = re.split(separator, chunk)
                    new_chunks.extend([c.strip() for c in split_chunks if c.strip()])
                else:
                    new_chunks.append(chunk)
            chunks = new_chunks
        
        # Smart merging with overlap for context preservation
        final_chunks = []
        current_chunk = ""
        overlap_size = 100  # Characters to overlap between chunks
        
        for chunk in chunks:
            # If adding this chunk exceeds max size
            if len(current_chunk + " " + chunk) > max_chunk_size and current_chunk:
                final_chunks.append(current_chunk.strip())
                
                # Create overlap from the end of current chunk
                words = current_chunk.split()
                if len(words) > 10:  # Only overlap if there are enough words
                    overlap = ' '.join(words[-10:])  # Last 10 words as context
                    current_chunk = overlap + " " + chunk
                else:
                    current_chunk = chunk
            else:
                current_chunk = (current_chunk + " " + chunk).strip()
        
        # Add the last chunk
        if current_chunk:
            final_chunks.append(current_chunk)
        
        # Enhanced filtering: remove very short chunks and duplicates
        filtered_chunks = []
        seen_chunks = set()
        
        for chunk in final_chunks:
            chunk = chunk.strip()
            if len(chunk) >= 100:  # Increased minimum size for better context
                # Simple deduplication based on first 50 characters
                chunk_key = chunk[:50].lower()
                if chunk_key not in seen_chunks:
                    filtered_chunks.append(chunk)
                    seen_chunks.add(chunk_key)
        
        return filtered_chunks
    
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
    
    async def _generate_embeddings_batch(self, texts: List[str], batch_size: int = 16) -> List[List[float]]:
        """Generate embeddings for multiple texts in batches for better performance."""
        try:
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                # Generate embeddings for batch
                response = self.azure_client_embeddings.embeddings.create(
                    input=batch,
                    model=self.embedding_model
                )
                
                # Extract embeddings from response
                batch_embeddings = [data.embedding for data in response.data]
                all_embeddings.extend(batch_embeddings)
                
                # Small delay to prevent API rate limiting
                if i > 0 and i % (batch_size * 3) == 0:
                    await asyncio.sleep(0.1)
            
            logger.info(f"Generated {len(all_embeddings)} embeddings in batches")
            return all_embeddings
            
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
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
            
            # Generate embeddings for all chunks using batch processing for better performance
            logger.info(f"Generating embeddings for {len(chunks)} chunks using batch processing...")
            embeddings = await self._generate_embeddings_batch(chunks)
            
            vectors_to_upsert = []
            clauses_data = []  # For database storage
            
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
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
    
    async def query_document(self, question: str, top_k: int = 12) -> Dict[str, Any]:
        """
        Optimized query processing with improved accuracy and speed.
        
        Args:
            question: The question to ask
            top_k: Number of top relevant clauses to retrieve (increased default for better accuracy)
            
        Returns:
            Dictionary with answer and source clauses
        """
        logger.info(f"Processing query: {question}")
        
        try:
            # Generate embedding for the question
            question_embedding = await self._generate_embedding(question)
            
            # Enhanced Pinecone search with better parameters for accuracy
            search_results = self.pinecone_index.query(
                vector=question_embedding,
                top_k=min(top_k * 3, 30),  # Get even more results for better accuracy
                include_metadata=True
            )
            
            if not search_results.matches:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information to answer your question.",
                    "source_clauses": [],
                    "confidence": 0.0
                }
            
            # Filter and rank results by relevance - LOWER threshold for better recall
            filtered_matches = []
            for match in search_results.matches:
                # Much lower threshold to catch more relevant content
                if match.score >= 0.70:  # Lower threshold for better recall
                    filtered_matches.append(match)
            
            # Always use at least top_k results for better accuracy
            if len(filtered_matches) < top_k:
                filtered_matches = search_results.matches[:top_k]
            else:
                # Use the best filtered results
                filtered_matches = filtered_matches[:top_k]
            
            # Extract relevant clauses with enhanced context preparation
            source_clauses = []
            context_chunks = []
            total_context_length = 0
            max_context_length = 6000  # Increased context for better accuracy
            
            for match in filtered_matches:
                clause_text = match.metadata["text"]
                
                # Skip if adding this would exceed our context limit
                if total_context_length + len(clause_text) > max_context_length:
                    break
                
                clause_info = {
                    "text": clause_text,
                    "document_url": match.metadata["document_url"],
                    "chunk_index": match.metadata["chunk_index"],
                    "similarity_score": float(match.score)
                }
                source_clauses.append(clause_info)
                context_chunks.append(f"[Clause {len(context_chunks) + 1}]\n{clause_text}")
                total_context_length += len(clause_text)
            
            # Create well-structured context
            context_text = "\n\n".join(context_chunks)
            
            # Generate answer using optimized prompt
            answer = await self._generate_answer_optimized(question, context_text, source_clauses)
            
            # Calculate weighted confidence score (gives more weight to higher-scoring matches)
            if source_clauses:
                weighted_confidence = sum(
                    clause["similarity_score"] * (1.0 / (i + 1))  # Higher weight for earlier (more relevant) matches
                    for i, clause in enumerate(source_clauses)
                ) / len(source_clauses)
            else:
                weighted_confidence = 0.0
            
            return {
                "success": True,
                "answer": answer,
                "source_clauses": source_clauses,
                "confidence": weighted_confidence,
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
    
    async def _generate_answer_optimized(self, question: str, context: str, source_clauses: List[Dict]) -> str:
        """Generate an optimized answer with better prompt engineering for hackathon accuracy."""
        try:
            # Analyze question type to tailor the prompt
            question_lower = question.lower()
            is_specific_value = any(word in question_lower for word in ['how much', 'how many', 'what is the', 'amount', 'cost', 'price', 'coverage'])
            is_list_question = any(word in question_lower for word in ['what are', 'list', 'benefits', 'features', 'types'])
            is_definition = any(word in question_lower for word in ['what does', 'define', 'meaning', 'explain'])
            
            # Create specialized system prompt based on question type
            if is_specific_value:
                system_prompt = """You are a precise document analyst. Extract specific values, amounts, and factual information from the provided context. 
                Be exact and concise. If you find a specific number, amount, or value, state it clearly. 
                If the exact information isn't available, say so explicitly rather than guessing."""
            elif is_list_question:
                system_prompt = """You are a document analyst focused on extracting comprehensive lists and enumerations. 
                When asked for lists, provide complete, organized information from the context. 
                Use bullet points or numbered lists when appropriate for clarity."""
            elif is_definition:
                system_prompt = """You are a document analyst specializing in definitions and explanations. 
                Provide clear, accurate definitions based on the context. 
                Quote directly from the source when possible for maximum accuracy."""
            else:
                system_prompt = """You are a highly accurate document analysis assistant specialized in insurance and policy documents. 
                Provide precise, factual answers based solely on the provided context. 
                When uncertain, acknowledge limitations rather than speculating."""
            
            # Enhanced user prompt with structure
            user_prompt = f"""DOCUMENT CONTEXT:
{context}

QUESTION: {question}

INSTRUCTIONS:
1. Answer based ONLY on the provided context
2. Be specific and precise
3. If the context contains exact values, amounts, or lists, include them
4. If information is not available in the context, state "This information is not available in the provided context"
5. Keep your answer focused and direct

ANSWER:"""
            
            # Optimized parameters for better accuracy
            response = self.azure_client_chat.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=800,  # Shorter for more focused answers
                temperature=0.1,  # Lower temperature for more consistent answers
                top_p=0.9  # Focused sampling
            )
            
            answer = response.choices[0].message.content
            
            # Post-process answer to improve accuracy
            answer = answer.strip()
            
            # Remove common AI assistant prefixes that might reduce accuracy scores
            prefixes_to_remove = [
                "Based on the provided context, ",
                "According to the document, ",
                "From the context provided, ",
                "The document states that "
            ]
            
            for prefix in prefixes_to_remove:
                if answer.startswith(prefix):
                    answer = answer[len(prefix):]
                    break
            
            return answer
            
        except Exception as e:
            logger.error(f"Failed to generate optimized answer: {e}")
            # Fallback to basic answer generation
            return await self._generate_answer(question, context)
    
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
