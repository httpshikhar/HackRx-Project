"""
Advanced Conflict Detection System for Legal/Insurance Document Analysis

This module provides sophisticated conflict detection capabilities using o3-mini's
reasoning abilities to identify contradictions, overlaps, and ambiguities in 
legal/insurance documents.
"""

import os
import json
import hashlib
import asyncio
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
from datetime import datetime, timedelta

# Third-party imports
from openai import AzureOpenAI
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()


class ConflictType(Enum):
    """Types of conflicts that can be detected in legal/insurance documents."""
    DIRECT_CONTRADICTION = "direct_contradiction"
    AMOUNT_DISCREPANCY = "amount_discrepancy"
    CONDITION_MISMATCH = "condition_mismatch"
    COVERAGE_OVERLAP = "coverage_overlap"
    TEMPORAL_CONFLICT = "temporal_conflict"
    SCOPE_AMBIGUITY = "scope_ambiguity"
    ELIGIBILITY_CONTRADICTION = "eligibility_contradiction"


class ConflictSeverity(Enum):
    """Severity levels for detected conflicts."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFORMATIONAL = "informational"


@dataclass
class Clause:
    """Represents a document clause with metadata."""
    id: str
    text: str
    document_url: str
    chunk_index: int
    similarity_score: float
    embedding: Optional[List[float]] = None


@dataclass
class ConflictAnalysis:
    """Represents a detected conflict with analysis details."""
    id: str
    type: ConflictType
    severity: ConflictSeverity
    clauses: List[Clause]
    reasoning: str
    resolution: str
    confidence_score: float
    requires_human_review: bool
    detected_at: datetime


class ConflictDetector:
    """
    Advanced conflict detection system that uses o3-mini reasoning capabilities
    to identify and analyze conflicts in legal/insurance documents.
    """
    
    def __init__(self):
        """Initialize the conflict detection system."""
        self._setup_clients()
        self._conflict_cache = {}
        self._similarity_threshold = 0.75  # Threshold for clause pairing
        self._max_clauses_for_analysis = 25  # Limit for cost control
        
    def _setup_clients(self):
        """Setup Azure OpenAI client for o3-mini reasoning."""
        self.azure_client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_O3_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
        )
        
        # Model configuration
        self.reasoning_model = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", "o3-mini")
        logger.info("Conflict detection system initialized with o3-mini reasoning")
    
    def _generate_conflict_id(self, clause1: Clause, clause2: Clause) -> str:
        """Generate a unique ID for a conflict analysis."""
        combined = f"{clause1.id}_{clause2.id}"
        return hashlib.md5(combined.encode()).hexdigest()[:12]
    
    def _calculate_clause_similarity(self, clause1: Clause, clause2: Clause) -> float:
        """Calculate semantic similarity between two clauses using embeddings."""
        if not clause1.embedding or not clause2.embedding:
            return 0.0
        
        # Calculate cosine similarity
        emb1 = np.array(clause1.embedding)
        emb2 = np.array(clause2.embedding)
        
        dot_product = np.dot(emb1, emb2)
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _get_smart_clause_pairs(self, clauses: List[Clause]) -> List[Tuple[Clause, Clause]]:
        """
        Generate smart clause pairs for analysis based on semantic similarity.
        Avoids expensive O(n²) analysis by only pairing similar clauses.
        """
        pairs = []
        
        # Limit clauses for cost control
        limited_clauses = clauses[:self._max_clauses_for_analysis]
        
        for i, clause1 in enumerate(limited_clauses):
            for j, clause2 in enumerate(limited_clauses[i+1:], i+1):
                # Calculate similarity
                similarity = self._calculate_clause_similarity(clause1, clause2)
                
                # Only analyze pairs with high semantic similarity
                if similarity >= self._similarity_threshold:
                    pairs.append((clause1, clause2))
        
        logger.info(f"Generated {len(pairs)} clause pairs for conflict analysis")
        return pairs
    
    async def _analyze_clause_pair_for_conflicts(self, clause1: Clause, clause2: Clause) -> Optional[ConflictAnalysis]:
        """
        Use o3-mini to analyze a pair of clauses for potential conflicts.
        """
        # Check cache first
        conflict_id = self._generate_conflict_id(clause1, clause2)
        if conflict_id in self._conflict_cache:
            return self._conflict_cache[conflict_id]
        
        # Prepare advanced legal reasoning prompt for o3-mini
        system_prompt = """You are an expert legal document analyst specializing in insurance and contract law. 
        Your task is to analyze pairs of clauses for potential conflicts, contradictions, and ambiguities.
        
        ANALYSIS FRAMEWORK:
        1. Direct Contradictions: Clauses that state opposing requirements or conditions
        2. Amount Discrepancies: Different monetary amounts, percentages, or limits for same item
        3. Condition Mismatches: Different conditions for similar coverage or eligibility
        4. Coverage Overlaps: Multiple clauses covering same scenario with different terms
        5. Temporal Conflicts: Different time periods, waiting periods, or effective dates
        6. Scope Ambiguities: Unclear boundaries between clause applications
        7. Eligibility Contradictions: Different eligibility criteria for similar benefits
        
        SEVERITY ASSESSMENT:
        - CRITICAL: Complete contradiction that makes policy unenforceable
        - HIGH: Significant conflict affecting coverage or obligations
        - MEDIUM: Ambiguity that could lead to disputes
        - LOW: Minor inconsistency that doesn't affect core terms
        - INFORMATIONAL: Overlap without conflict
        
        RESPONSE FORMAT: Return JSON with structured analysis."""
        
        user_prompt = f"""Analyze these two insurance/legal clauses for conflicts:

CLAUSE 1 (ID: {clause1.id}):
{clause1.text}

CLAUSE 2 (ID: {clause2.id}):
{clause2.text}

Provide analysis in JSON format:
{{
    "conflict_detected": boolean,
    "conflict_type": "one of: direct_contradiction, amount_discrepancy, condition_mismatch, coverage_overlap, temporal_conflict, scope_ambiguity, eligibility_contradiction",
    "severity": "one of: critical, high, medium, low, informational",
    "reasoning": "Step-by-step legal analysis of why this is or isn't a conflict",
    "specific_issues": ["list of specific conflicting elements"],
    "resolution_approach": "Legal principle-based recommendation for resolving the conflict",
    "confidence": 0.0-1.0,
    "requires_human_review": boolean,
    "legal_precedence_rule": "Which clause should take precedence and why"
}}"""

        try:
            response = await self._call_o3_mini_reasoning(system_prompt, user_prompt)
            
            # Parse JSON response
            analysis_data = json.loads(response)
            
            if analysis_data.get("conflict_detected", False):
                # Create conflict analysis object
                conflict = ConflictAnalysis(
                    id=conflict_id,
                    type=ConflictType(analysis_data["conflict_type"]),
                    severity=ConflictSeverity(analysis_data["severity"]),
                    clauses=[clause1, clause2],
                    reasoning=analysis_data["reasoning"],
                    resolution=analysis_data["resolution_approach"],
                    confidence_score=analysis_data["confidence"],
                    requires_human_review=analysis_data["requires_human_review"],
                    detected_at=datetime.now()
                )
                
                # Cache the result
                self._conflict_cache[conflict_id] = conflict
                return conflict
            
            # No conflict detected, cache negative result
            self._conflict_cache[conflict_id] = None
            return None
            
        except Exception as e:
            logger.error(f"Error analyzing clause pair {conflict_id}: {e}")
            return None
    
    async def _call_o3_mini_reasoning(self, system_prompt: str, user_prompt: str) -> str:
        """Make a call to o3-mini for advanced reasoning analysis."""
        try:
            response = self.azure_client.chat.completions.create(
                model=self.reasoning_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=2000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Error calling o3-mini reasoning: {e}")
            raise
    
    async def detect_conflicts_for_query(self, query: str, relevant_clauses: List[Dict[str, Any]]) -> List[ConflictAnalysis]:
        """
        Detect conflicts among clauses relevant to a specific query.
        
        Args:
            query: The user's query
            relevant_clauses: List of clause data from RAG system
            
        Returns:
            List of detected conflicts
        """
        logger.info(f"Starting conflict detection for query: {query}")
        
        # Convert clause data to Clause objects
        clauses = []
        for clause_data in relevant_clauses:
            clause = Clause(
                id=f"{clause_data.get('document_url', 'unknown')}_{clause_data.get('chunk_index', 0)}",
                text=clause_data["text"],
                document_url=clause_data.get("document_url", ""),
                chunk_index=clause_data.get("chunk_index", 0),
                similarity_score=clause_data.get("similarity_score", 0.0),
                embedding=clause_data.get("embedding")
            )
            clauses.append(clause)
        
        # Generate smart clause pairs
        clause_pairs = self._get_smart_clause_pairs(clauses)
        
        # Analyze pairs for conflicts (with async processing)
        conflicts = []
        analysis_tasks = []
        
        for clause1, clause2 in clause_pairs:
            task = self._analyze_clause_pair_for_conflicts(clause1, clause2)
            analysis_tasks.append(task)
        
        # Process in batches to avoid overwhelming the API
        batch_size = 5
        for i in range(0, len(analysis_tasks), batch_size):
            batch = analysis_tasks[i:i + batch_size]
            batch_results = await asyncio.gather(*batch, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, ConflictAnalysis):
                    conflicts.append(result)
                elif isinstance(result, Exception):
                    logger.error(f"Error in conflict analysis: {result}")
        
        logger.info(f"Detected {len(conflicts)} conflicts for query")
        return conflicts
    
    async def resolve_conflicts_with_reasoning(self, conflicts: List[ConflictAnalysis], user_query: str) -> Dict[str, Any]:
        """
        Use o3-mini reasoning to provide resolution recommendations for detected conflicts.
        """
        if not conflicts:
            return {
                "resolution_needed": False,
                "message": "No conflicts detected that require resolution."
            }
        
        # Prepare comprehensive resolution prompt
        system_prompt = """You are a senior legal counsel specializing in insurance law and contract interpretation.
        You have been presented with conflicts detected in an insurance policy document.
        
        Your task is to:
        1. Analyze the overall pattern of conflicts
        2. Apply legal precedence rules and insurance law principles
        3. Provide a coherent resolution strategy
        4. Determine if human expert review is absolutely necessary
        5. Assess the overall policy integrity
        
        LEGAL PRINCIPLES TO APPLY:
        - Specific provisions override general ones
        - Later amendments override earlier provisions
        - Ambiguities are typically resolved in favor of the insured
        - Coverage provisions are broadly interpreted
        - Exclusions are narrowly interpreted
        - Industry standard interpretations
        
        RESPONSE REQUIREMENTS:
        - Clear, actionable resolution recommendations
        - Legal reasoning for each recommendation
        - Risk assessment for the insured
        - Priority ranking of issues to address"""
        
        conflicts_summary = []
        for conflict in conflicts:
            conflicts_summary.append({
                "type": conflict.type.value,
                "severity": conflict.severity.value,
                "reasoning": conflict.reasoning,
                "resolution": conflict.resolution
            })
        
        user_prompt = f"""CONTEXT: User asked: "{user_query}"

DETECTED CONFLICTS:
{json.dumps(conflicts_summary, indent=2)}

Provide comprehensive resolution analysis in JSON format:
{{
    "overall_assessment": "Summary of policy integrity and conflict severity",
    "resolution_strategy": "High-level approach to resolving all conflicts",
    "specific_recommendations": [
        {{
            "conflict_id": "ID of conflict being addressed",
            "recommendation": "Specific action to take",
            "legal_basis": "Legal principle or precedent supporting this recommendation",
            "priority": "high/medium/low",
            "risk_if_unresolved": "Potential consequences"
        }}
    ],
    "final_answer_to_user": "Clear answer to user's original question considering all conflicts",
    "confidence_level": "high/medium/low",
    "requires_human_review": boolean,
    "recommended_next_steps": ["List of actions to take"],
    "legal_disclaimer": "Standard legal disclaimer about professional advice"
}}"""
        
        try:
            response = await self._call_o3_mini_reasoning(system_prompt, user_prompt)
            resolution_analysis = json.loads(response)
            
            return {
                "resolution_needed": True,
                "analysis": resolution_analysis,
                "conflicts_count": len(conflicts),
                "high_severity_count": len([c for c in conflicts if c.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH]])
            }
            
        except Exception as e:
            logger.error(f"Error in conflict resolution analysis: {e}")
            return {
                "resolution_needed": True,
                "error": f"Failed to generate resolution analysis: {str(e)}",
                "conflicts_count": len(conflicts)
            }
    
    async def analyze_document_integrity(self, all_clauses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform a comprehensive document integrity analysis.
        """
        logger.info("Starting comprehensive document integrity analysis")
        
        # Convert to Clause objects
        clauses = []
        for clause_data in all_clauses:
            clause = Clause(
                id=f"{clause_data.get('document_url', 'unknown')}_{clause_data.get('chunk_index', 0)}",
                text=clause_data["text"],
                document_url=clause_data.get("document_url", ""),
                chunk_index=clause_data.get("chunk_index", 0),
                similarity_score=clause_data.get("similarity_score", 1.0),
                embedding=clause_data.get("embedding")
            )
            clauses.append(clause)
        
        # Perform conflict analysis on all clauses
        clause_pairs = self._get_smart_clause_pairs(clauses)
        
        conflicts = []
        batch_size = 3  # Smaller batch for comprehensive analysis
        
        for i in range(0, len(clause_pairs), batch_size):
            batch_pairs = clause_pairs[i:i + batch_size]
            batch_tasks = [
                self._analyze_clause_pair_for_conflicts(clause1, clause2) 
                for clause1, clause2 in batch_pairs
            ]
            
            batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
            
            for result in batch_results:
                if isinstance(result, ConflictAnalysis):
                    conflicts.append(result)
        
        # Analyze overall document health
        conflict_types = {}
        severity_counts = {}
        
        for conflict in conflicts:
            # Count by type
            conflict_types[conflict.type.value] = conflict_types.get(conflict.type.value, 0) + 1
            # Count by severity
            severity_counts[conflict.severity.value] = severity_counts.get(conflict.severity.value, 0) + 1
        
        return {
            "total_conflicts": len(conflicts),
            "conflict_types": conflict_types,
            "severity_distribution": severity_counts,
            "document_health_score": self._calculate_document_health_score(conflicts, len(clauses)),
            "critical_issues": [c for c in conflicts if c.severity == ConflictSeverity.CRITICAL],
            "requires_immediate_attention": len([c for c in conflicts if c.severity in [ConflictSeverity.CRITICAL, ConflictSeverity.HIGH]]) > 0,
            "analysis_timestamp": datetime.now().isoformat()
        }
    
    def _calculate_document_health_score(self, conflicts: List[ConflictAnalysis], total_clauses: int) -> float:
        """Calculate a health score for the document (0.0 to 1.0)."""
        if not conflicts:
            return 1.0
        
        # Weight conflicts by severity
        severity_weights = {
            ConflictSeverity.CRITICAL: 1.0,
            ConflictSeverity.HIGH: 0.7,
            ConflictSeverity.MEDIUM: 0.4,
            ConflictSeverity.LOW: 0.2,
            ConflictSeverity.INFORMATIONAL: 0.1
        }
        
        total_penalty = sum(severity_weights.get(conflict.severity, 0.5) for conflict in conflicts)
        
        # Normalize by total possible clause pairs
        max_possible_conflicts = (total_clauses * (total_clauses - 1)) / 2
        normalized_penalty = total_penalty / max(max_possible_conflicts * 0.1, 1)  # Assume 10% conflict rate would be score 0
        
        health_score = max(0.0, 1.0 - normalized_penalty)
        return round(health_score, 3)


# Enhanced RAG integration functions
async def enhance_rag_with_conflict_detection(
    standard_answer: str,
    relevant_clauses: List[Dict[str, Any]], 
    user_query: str,
    conflict_detector: ConflictDetector
) -> Dict[str, Any]:
    """
    Enhance standard RAG response with conflict detection analysis.
    """
    logger.info("Enhancing RAG response with conflict detection")
    
    # Detect conflicts in relevant clauses
    conflicts = await conflict_detector.detect_conflicts_for_query(user_query, relevant_clauses)
    
    # Get resolution analysis if conflicts found
    resolution_analysis = None
    if conflicts:
        resolution_analysis = await conflict_detector.resolve_conflicts_with_reasoning(conflicts, user_query)
    
    # Calculate overall confidence
    has_critical_conflicts = any(c.severity == ConflictSeverity.CRITICAL for c in conflicts)
    has_high_conflicts = any(c.severity == ConflictSeverity.HIGH for c in conflicts)
    
    if has_critical_conflicts:
        confidence = "low"
    elif has_high_conflicts:
        confidence = "medium"
    elif conflicts:
        confidence = "medium-high"
    else:
        confidence = "high"
    
    # Determine final answer
    final_answer = standard_answer
    if resolution_analysis and resolution_analysis.get("resolution_needed"):
        analysis = resolution_analysis.get("analysis", {})
        final_answer = analysis.get("final_answer_to_user", standard_answer)
    
    return {
        "standard_answer": standard_answer,
        "conflicts_detected": len(conflicts),
        "conflict_analysis": [
            {
                "id": c.id,
                "type": c.type.value,
                "severity": c.severity.value,
                "clauses": [{"id": cl.id, "text": cl.text[:200] + "..." if len(cl.text) > 200 else cl.text} for cl in c.clauses],
                "reasoning": c.reasoning,
                "resolution": c.resolution,
                "confidence": c.confidence_score,
                "requires_human_review": c.requires_human_review
            }
            for c in conflicts
        ],
        "resolution_analysis": resolution_analysis,
        "final_answer": final_answer,
        "confidence": confidence,
        "requires_human_review": any(c.requires_human_review for c in conflicts) or has_critical_conflicts,
        "analysis_metadata": {
            "total_clauses_analyzed": len(relevant_clauses),
            "analysis_timestamp": datetime.now().isoformat(),
            "model_used": "o3-mini"
        }
    }


if __name__ == "__main__":
    # Example usage
    async def test_conflict_detector():
        detector = ConflictDetector()
        
        # Mock clause data
        sample_clauses = [
            {
                "text": "Coverage is provided for pre-existing conditions after 36 months waiting period.",
                "document_url": "policy.pdf",
                "chunk_index": 1,
                "similarity_score": 0.9
            },
            {
                "text": "Pre-existing conditions are excluded for the first 24 months of coverage.",
                "document_url": "policy.pdf", 
                "chunk_index": 5,
                "similarity_score": 0.85
            }
        ]
        
        conflicts = await detector.detect_conflicts_for_query(
            "What is the waiting period for pre-existing conditions?",
            sample_clauses
        )
        
        print(f"Detected {len(conflicts)} conflicts")
        for conflict in conflicts:
            print(f"Conflict: {conflict.type.value} - {conflict.severity.value}")
            print(f"Reasoning: {conflict.reasoning}")
    
    # Uncomment to run test
    # asyncio.run(test_conflict_detector())
