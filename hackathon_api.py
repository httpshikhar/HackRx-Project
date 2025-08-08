"""
Production-ready FastAPI application for Enhanced Document Intelligence System.

EXACT HACKATHON COMPLIANCE:
- Endpoint: POST /hackrx/run
- Request: {"documents": "blob_url", "questions": ["q1", "q2", ...]}
- Response: {"answers": ["answer1", "answer2", ...]}
- No authentication required
- Async processing with PostgreSQL integration
- Advanced o3-mini conflict detection
"""

import os
import asyncio
import hashlib
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
import uvicorn

# Local imports
from document_intelligence import DocumentIntelligenceSystem
from database import DatabaseManager, db_manager, init_database, close_database
from conflict_detector import ConflictDetector
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Pydantic models for exact hackathon compliance
class HackathonRequest(BaseModel):
    """Exact request format required by hackathon."""
    documents: str = Field(..., description="Blob URL of the document to process")
    questions: List[str] = Field(..., description="List of questions to answer")
    
    @validator('questions')
    def validate_questions(cls, v):
        if not v or len(v) == 0:
            raise ValueError("At least one question must be provided")
        if len(v) > 20:  # Rate limiting
            raise ValueError("Maximum 20 questions allowed per request")
        for question in v:
            if not question.strip():
                raise ValueError("Questions cannot be empty")
        return v


class HackathonResponse(BaseModel):
    """Exact response format required by hackathon."""
    answers: List[str] = Field(..., description="List of answers corresponding to questions")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    timestamp: str
    database_connected: bool
    system_stats: Optional[Dict[str, Any]] = None


class ProductionDocumentIntelligenceSystem(DocumentIntelligenceSystem):
    """Production-ready enhanced document intelligence system with database integration."""
    
    def __init__(self):
        """Initialize with database integration."""
        super().__init__()
        self.conflict_detector = ConflictDetector()
        self.processing_locks = {}  # To prevent duplicate processing
        
    async def process_document_with_db(self, document_url: str) -> Dict[str, Any]:
        """Process document with database integration and caching."""
        start_time = time.time()
        
        try:
            # Check if document exists and is already processed
            document, is_new = await db_manager.create_or_get_document(
                document_url, 
                original_filename=os.path.basename(document_url)
            )
            
            # If document is already completed, return success
            if document.processing_status == 'completed':
                logger.info(f"Document already processed: {document.id}")
                return {
                    "success": True,
                    "document_id": document.id,
                    "cached": True,
                    "chunks_processed": document.total_clauses,
                    "processing_time_ms": 0
                }
            
            # If document is currently being processed by another request
            if document.processing_status == 'processing':
                # Wait a bit and check again (simple distributed lock mechanism)
                for _ in range(30):  # Wait up to 30 seconds
                    await asyncio.sleep(1)
                    doc_check, _ = await db_manager.create_or_get_document(document_url)
                    if doc_check.processing_status == 'completed':
                        return {
                            "success": True,
                            "document_id": doc_check.id,
                            "cached": True,
                            "chunks_processed": doc_check.total_clauses,
                            "processing_time_ms": int((time.time() - start_time) * 1000)
                        }
                
                # If still processing, proceed anyway (might be stuck)
                logger.warning(f"Document {document.id} appears stuck in processing state")
            
            # Update status to processing
            await db_manager.update_document_status(document.id, 'processing')
            
            # Process the document using parent class method
            result = await super().process_document(document_url)
            
            if result["success"]:
                # Store clauses in database
                if "clauses_data" in result:  # We'll need to modify parent to return clause data
                    # Update clause data with the correct document UUID
                    for clause_data in result["clauses_data"]:
                        clause_data["document_id"] = document.id
                    await db_manager.bulk_insert_clauses(result["clauses_data"])
                
                # Update document status to completed
                await db_manager.update_document_status(
                    document.id, 
                    'completed',
                    total_clauses=result["chunks_processed"],
                    total_characters=result["total_characters"]
                )
                
                # Log analytics
                processing_time = int((time.time() - start_time) * 1000)
                await db_manager.log_analytics_event(
                    'document_processed',
                    document_id=document.id,
                    response_time_ms=processing_time,
                    metadata={"chunks": result["chunks_processed"], "characters": result["total_characters"]}
                )
                
                result["processing_time_ms"] = processing_time
                result["document_id"] = document.id
                
            else:
                # Update document status to failed
                await db_manager.update_document_status(
                    document.id,
                    'failed', 
                    error=result.get("error", "Unknown processing error")
                )
                
                # Log error analytics
                await db_manager.log_analytics_event(
                    'document_processing_failed',
                    document_id=document.id,
                    error_occurred=True,
                    error_message=result.get("error", "Unknown error"),
                    response_time_ms=int((time.time() - start_time) * 1000)
                )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in document processing with database: {e}")
            
            # Try to update document status to failed if we have document_id
            try:
                if 'document' in locals():
                    await db_manager.update_document_status(
                        document.id, 
                        'failed', 
                        error=str(e)
                    )
            except:
                pass
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }
    
    async def query_document_with_cache(self, question: str, document_id: str) -> Dict[str, Any]:
        """Query document with intelligent caching."""
        start_time = time.time()
        
        try:
            # Check cache first
            cached_result = await db_manager.get_cached_query(question, document_id)
            if cached_result:
                logger.info(f"Cache hit for query on document {document_id}")
                
                # Log cache hit analytics
                await db_manager.log_analytics_event(
                    'query_cache_hit',
                    document_id=document_id,
                    query_text=question,
                    cache_hit=True,
                    response_time_ms=int((time.time() - start_time) * 1000)
                )
                
                return {
                    "success": True,
                    "answer": cached_result.enhanced_answer or cached_result.standard_answer,
                    "source_clauses": cached_result.source_clauses,
                    "confidence": cached_result.confidence_level,
                    "conflicts_detected": cached_result.conflicts_detected,
                    "conflict_analysis": cached_result.conflict_analysis,
                    "requires_human_review": cached_result.requires_human_review,
                    "cached": True,
                    "processing_time_ms": int((time.time() - start_time) * 1000)
                }
            
            # Cache miss - perform enhanced query with conflict detection
            result = await super().query_document_with_conflict_detection(question, top_k=10)
            
            if result["success"]:
                processing_time = int((time.time() - start_time) * 1000)
                
                # Prepare cache data
                cache_data = {
                    "standard_answer": result.get("standard_answer", result.get("answer", "")),
                    "enhanced_answer": result.get("final_answer"),
                    "source_clauses": result.get("source_clauses", []),
                    "conflicts_detected": result.get("conflicts_detected", 0),
                    "conflict_analysis": result.get("conflict_analysis", []),
                    "confidence_level": result.get("confidence", "medium"),
                    "requires_human_review": result.get("requires_human_review", False),
                    "processing_time_ms": processing_time,
                    "model_versions": {
                        "embedding_model": self.embedding_model,
                        "chat_model": self.chat_model,
                        "reasoning_model": "o3-mini"
                    }
                }
                
                # Cache the result
                await db_manager.cache_query_result(question, document_id, cache_data)
                
                # Log analytics
                await db_manager.log_analytics_event(
                    'query_processed',
                    document_id=document_id,
                    query_text=question,
                    response_time_ms=processing_time,
                    api_calls_made=2 if result.get("conflicts_detected", 0) > 0 else 1,  # Estimate
                    cache_hit=False,
                    metadata={
                        "confidence": result.get("confidence"),
                        "conflicts": result.get("conflicts_detected", 0)
                    }
                )
                
                # Return the final answer
                result["answer"] = result.get("final_answer") or result.get("standard_answer") or result.get("answer", "")
                result["cached"] = False
                result["processing_time_ms"] = processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Error in cached query: {e}")
            
            # Log error
            await db_manager.log_analytics_event(
                'query_failed',
                document_id=document_id,
                query_text=question,
                error_occurred=True,
                error_message=str(e),
                response_time_ms=int((time.time() - start_time) * 1000)
            )
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000)
            }


# Global system instance
production_system: Optional[ProductionDocumentIntelligenceSystem] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan - startup and shutdown."""
    # Startup
    logger.info("Starting Enhanced Document Intelligence API...")
    
    try:
        # Initialize database
        await init_database()
        logger.info("✅ Database connection pool initialized")
        
        # Initialize production system
        global production_system
        production_system = ProductionDocumentIntelligenceSystem()
        logger.info("✅ Production system initialized")
        
        # Clean up expired cache on startup
        cleaned = await db_manager.cleanup_expired_cache()
        if cleaned > 0:
            logger.info(f"✅ Cleaned up {cleaned} expired cache entries")
        
        logger.info("🚀 API ready for requests")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down Enhanced Document Intelligence API...")
    
    try:
        await close_database()
        logger.info("✅ Database connections closed")
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
    
    logger.info("👋 API shutdown complete")


# Initialize FastAPI app with lifespan management
app = FastAPI(
    title="Enhanced Document Intelligence API",
    description="Production-ready document intelligence with o3-mini conflict detection",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_client_info(request: Request) -> Dict[str, Any]:
    """Extract client information for analytics."""
    return {
        "client_ip": request.client.host,
        "user_agent": request.headers.get("user-agent")
    }


# MAIN HACKATHON ENDPOINT
@app.post("/hackrx/run", response_model=HackathonResponse)
async def hackathon_endpoint(
    request: HackathonRequest,
    background_tasks: BackgroundTasks,
    http_request: Request
):
    """
    EXACT HACKATHON COMPLIANCE ENDPOINT
    
    Request Format: {"documents": "blob_url", "questions": ["q1", "q2", ...]}
    Response Format: {"answers": ["answer1", "answer2", ...]}
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())
    client_info = get_client_info(http_request)
    
    logger.info(f"🎯 Hackathon request {request_id}: {len(request.questions)} questions for {request.documents}")
    
    try:
        # Create API request log
        api_log_id = await db_manager.create_api_request_log({
            "request_id": request_id,
            "client_ip": client_info["client_ip"],
            "user_agent": client_info["user_agent"],
            "document_url": request.documents,
            "questions": request.questions
        })
        
        # Process document (with caching)
        logger.info(f"📄 Processing document: {request.documents}")
        processing_result = await production_system.process_document_with_db(request.documents)
        
        if not processing_result["success"]:
            await db_manager.update_api_request_log(api_log_id, {
                "processing_status": "failed",
                "errors_encountered": 1,
                "error_details": {"document_processing_error": processing_result.get("error")},
                "total_processing_time_ms": int((time.time() - start_time) * 1000)
            })
            
            raise HTTPException(
                status_code=500,
                detail=f"Document processing failed: {processing_result.get('error', 'Unknown error')}"
            )
        
        document_id = processing_result["document_id"]
        logger.info(f"✅ Document processed: {document_id} ({processing_result.get('chunks_processed', 0)} clauses)")
        
        # Process all questions in parallel for efficiency
        logger.info(f"🤔 Processing {len(request.questions)} questions in parallel...")
        
        async def process_single_question(question: str) -> str:
            """Process a single question and return the answer."""
            try:
                result = await production_system.query_document_with_cache(question, document_id)
                if result["success"]:
                    return result["answer"]
                else:
                    logger.error(f"Question processing failed: {result.get('error')}")
                    return f"Error processing question: {result.get('error', 'Unknown error')}"
            except Exception as e:
                logger.error(f"Exception in question processing: {e}")
                return f"Error processing question: {str(e)}"
        
        # Create tasks for parallel processing
        question_tasks = [process_single_question(q) for q in request.questions]
        
        # Process questions with timeout protection
        try:
            answers = await asyncio.wait_for(
                asyncio.gather(*question_tasks), 
                timeout=300  # 5 minute timeout
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=408, 
                detail="Request timeout - questions took too long to process"
            )
        
        # Calculate metrics
        total_time_ms = int((time.time() - start_time) * 1000)
        cache_hits = sum(1 for _ in range(len(request.questions)))  # Simplified - would track actual cache hits
        
        logger.info(f"✅ Processed {len(answers)} answers in {total_time_ms}ms")
        
        # Update API request log
        background_tasks.add_task(
            db_manager.update_api_request_log,
            api_log_id,
            {
                "processing_status": "completed",
                "total_processing_time_ms": total_time_ms,
                "cache_hits": cache_hits,
                "api_calls_made": len(request.questions) * 2,  # Estimate
                "response_data": {"answers_count": len(answers)}
            }
        )
        
        # Return exact hackathon format
        return HackathonResponse(answers=answers)
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"❌ Unexpected error in hackathon endpoint: {e}")
        
        # Update API log with error
        try:
            await db_manager.update_api_request_log(api_log_id, {
                "processing_status": "failed",
                "errors_encountered": 1,
                "error_details": {"unexpected_error": str(e)},
                "total_processing_time_ms": int((time.time() - start_time) * 1000)
            })
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


# HEALTH CHECK ENDPOINT
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint with system statistics."""
    try:
        # Test database connection
        stats = await db_manager.get_system_stats()
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            database_connected=True,
            system_stats=stats
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            database_connected=False
        )


# SYSTEM STATUS ENDPOINT
@app.get("/status")
async def system_status():
    """Detailed system status for monitoring."""
    try:
        stats = await db_manager.get_system_stats()
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0",
            "features": {
                "document_processing": True,
                "conflict_detection": True,
                "query_caching": True,
                "parallel_processing": True,
                "postgresql_integration": True
            },
            "system_stats": stats
        }
    except Exception as e:
        logger.error(f"Status endpoint error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )


# CACHE MANAGEMENT ENDPOINT
@app.post("/admin/cache/clean")
async def clean_cache():
    """Clean expired cache entries."""
    try:
        cleaned = await db_manager.cleanup_expired_cache()
        return {
            "status": "success",
            "cleaned_entries": cleaned,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Cache cleanup error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ERROR HANDLERS
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """General exception handler."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )


if __name__ == "__main__":
    # Run the server
    uvicorn.run(
        "hackathon_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        log_level="info",
        access_log=True,
        reload=False  # Set to True for development
    )
