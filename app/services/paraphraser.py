"""
Enhanced Paraphrasing Service
Handles text paraphrasing using multiple AI models and methods with enhanced capabilities.
"""
import time
import uuid
import logging
from datetime import datetime
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod
import asyncio

import torch
import spacy
import nltk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.document import Document, ParaphraseSession, ParaphraseMethod, DocumentStatus
from app.services.document_processor import document_processor
from app.services.enhanced_document_processor import enhanced_document_processor
from app.services.indonesian_nlp_pipeline import indonesian_nlp_pipeline, DocumentAnalysis

# Initialize logger first
logger = logging.getLogger(__name__)

try:
    from app.services.enhanced_indot5_paraphraser import enhanced_indot5_paraphraser, ParaphraseResult
except Exception as e:
    logger.warning(f"Failed to import enhanced_indot5_paraphraser: {e}")
    enhanced_indot5_paraphraser = None
    ParaphraseResult = None
from app.services.rule_based_paraphraser import rule_based_paraphraser, ParaphraseQuality


class BaseParaphraser(ABC):
    """Abstract base class for paraphrasing models."""
    
    @abstractmethod
    async def paraphrase(self, text: str) -> str:
        """Paraphrase the given text."""
        pass
    
    @abstractmethod
    def calculate_similarity(self, original: str, paraphrased: str) -> float:
        """Calculate similarity score between original and paraphrased text."""
        pass


class IndoT5Paraphraser(BaseParaphraser):
    """Indonesian T5 model paraphraser."""
    
    def __init__(self):
        self.model_name = "Wikidepia/IndoT5-base-paraphrase"
        self.model = None
        self.tokenizer = None
        self._load_model()
    
    def _load_model(self):
        """Load the IndoT5 model and tokenizer."""
        try:
            logger.info("Loading IndoT5 model...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name)
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("IndoT5 model loaded on GPU")
            else:
                logger.info("IndoT5 model loaded on CPU")
                
        except Exception as e:
            logger.error(f"Failed to load IndoT5 model: {str(e)}")
            raise
    
    async def paraphrase(self, text: str) -> str:
        """
        Paraphrase text using IndoT5 model.
        
        Args:
            text: Text to paraphrase
            
        Returns:
            str: Paraphrased text
        """
        try:
            # Prepare input
            input_text = f"paraphrase: {text}"
            inputs = self.tokenizer.encode(
                input_text, 
                return_tensors="pt", 
                max_length=512, 
                truncation=True
            )
            
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            # Generate paraphrase
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=512,
                    num_beams=4,
                    temperature=0.7,
                    do_sample=True,
                    early_stopping=True
                )
            
            # Decode output
            paraphrased = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return paraphrased
            
        except Exception as e:
            logger.error(f"IndoT5 paraphrasing failed: {str(e)}")
            raise
    
    def calculate_similarity(self, original: str, paraphrased: str) -> float:
        """Calculate similarity using simple token overlap."""
        # Simple implementation - can be enhanced with more sophisticated methods
        original_tokens = set(original.lower().split())
        paraphrased_tokens = set(paraphrased.lower().split())
        
        if not original_tokens:
            return 0.0
        
        intersection = original_tokens.intersection(paraphrased_tokens)
        union = original_tokens.union(paraphrased_tokens)
        
        return len(intersection) / len(union) if union else 0.0


class GeminiParaphraser(BaseParaphraser):
    """Google Gemini API paraphraser."""
    
    def __init__(self):
        self.api_key = settings.gemini_api_key
        if not self.api_key:
            logger.warning("Gemini API key not provided")
    
    async def paraphrase(self, text: str) -> str:
        """
        Paraphrase text using Gemini API.
        
        Args:
            text: Text to paraphrase
            
        Returns:
            str: Paraphrased text
        """
        if not self.api_key:
            raise ValueError("Gemini API key not configured")
        
        # TODO: Implement Gemini API integration
        # This is a placeholder implementation
        logger.info("Gemini paraphrasing (placeholder implementation)")
        return f"[Gemini Paraphrased] {text}"
    
    def calculate_similarity(self, original: str, paraphrased: str) -> float:
        """Calculate similarity using semantic analysis."""
        # Placeholder implementation
        return 0.8


class HybridParaphraser(BaseParaphraser):
    """Hybrid paraphraser combining multiple methods."""
    
    def __init__(self):
        self.indot5 = IndoT5Paraphraser()
        self.gemini = GeminiParaphraser()
        
        # Load spaCy model for NLP tasks
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
            self.nlp = None
    
    async def paraphrase(self, text: str) -> str:
        """
        Paraphrase using hybrid approach.
        
        Args:
            text: Text to paraphrase
            
        Returns:
            str: Paraphrased text
        """
        try:
            # Use IndoT5 as primary method
            paraphrased = await self.indot5.paraphrase(text)
            
            # TODO: Add post-processing and quality checks
            # Could include grammar correction, coherence checking, etc.
            
            return paraphrased
            
        except Exception as e:
            logger.error(f"Hybrid paraphrasing failed: {str(e)}")
            # Fallback to simple sentence restructuring
            return self._simple_paraphrase(text)
    
    def _simple_paraphrase(self, text: str) -> str:
        """Simple rule-based paraphrasing as fallback."""
        # Basic sentence restructuring
        sentences = text.split(". ")
        paraphrased_sentences = []
        
        for sentence in sentences:
            # Simple transformations
            if sentence.startswith("The "):
                sentence = sentence.replace("The ", "A ", 1)
            elif sentence.startswith("A "):
                sentence = sentence.replace("A ", "The ", 1)
            
            paraphrased_sentences.append(sentence)
        
        return ". ".join(paraphrased_sentences)
    
    def calculate_similarity(self, original: str, paraphrased: str) -> float:
        """Calculate similarity using multiple metrics."""
        # Combine different similarity measures
        token_similarity = self.indot5.calculate_similarity(original, paraphrased)
        
        # TODO: Add semantic similarity using sentence embeddings
        semantic_similarity = 0.8  # Placeholder
        
        # Weighted combination
        return (token_similarity * 0.4) + (semantic_similarity * 0.6)


class EnhancedParaphrasingService:
    """Enhanced service for handling comprehensive document paraphrasing requests."""
    
    def __init__(self):
        # Legacy paraphrasers for backward compatibility
        self.paraphrasers = {
            ParaphraseMethod.INDOT5: IndoT5Paraphraser(),
            ParaphraseMethod.GEMINI: GeminiParaphraser(),
            ParaphraseMethod.HYBRID: HybridParaphraser(),
        }
        
        # Enhanced paraphrasers
        self.enhanced_indot5 = enhanced_indot5_paraphraser if enhanced_indot5_paraphraser else None
        self.rule_based = rule_based_paraphraser
        self.nlp_pipeline = indonesian_nlp_pipeline
    
    async def paraphrase_document_enhanced(
        self,
        document_id: uuid.UUID,
        method: ParaphraseMethod,
        db: Session,
        use_nlp_analysis: bool = True,
        preserve_academic_terms: bool = True,
        preserve_citations: bool = True,
        num_variants: int = 3,
        quality_threshold: float = 0.6
    ) -> Optional[ParaphraseSession]:
        """
        Enhanced document paraphrasing with comprehensive analysis.
        
        Args:
            document_id: Document UUID
            method: Paraphrasing method to use
            db: Database session
            use_nlp_analysis: Whether to use NLP analysis for prioritization
            preserve_academic_terms: Whether to preserve academic terminology
            preserve_citations: Whether to preserve citation formats
            num_variants: Number of variants to generate
            quality_threshold: Minimum quality threshold for acceptance
            
        Returns:
            ParaphraseSession: Enhanced session record
        """
        # Get document
        document = document_processor.get_document(document_id, db)
        if not document:
            logger.error(f"Document {document_id} not found")
            return None
        
        # Update document status
        document_processor.update_document_status(
            document_id, DocumentStatus.PROCESSING, db
        )
        
        start_time = time.time()
        
        try:
            # Perform NLP analysis if requested
            analysis = None
            if use_nlp_analysis:
                logger.info(f"Performing NLP analysis for document {document_id}")
                analysis = await self.nlp_pipeline.analyze_document(document.original_content)
            
            # Choose paraphrasing strategy based on method
            if method == ParaphraseMethod.INDOT5:
                paraphrased_text, metadata = await self._paraphrase_with_enhanced_indot5(
                    document.original_content, analysis, num_variants, quality_threshold
                )
            elif method == ParaphraseMethod.HYBRID:
                paraphrased_text, metadata = await self._paraphrase_with_hybrid_approach(
                    document.original_content, analysis, preserve_academic_terms, 
                    preserve_citations, num_variants, quality_threshold
                )
            else:
                # Fallback to legacy paraphraser
                logger.info(f"Using legacy paraphraser for method {method}")
                paraphraser = self.paraphrasers[method]
                paraphrased_text = await paraphraser.paraphrase(document.original_content)
                metadata = {"method": "legacy", "variants_generated": 1}
            
            # Calculate enhanced similarity
            similarity_score = self._calculate_enhanced_similarity(
                document.original_content, paraphrased_text
            )
            
            processing_time = int(time.time() - start_time)
            
            # Create enhanced session record
            session = ParaphraseSession(
                document_id=document_id,
                method_used=method,
                similarity_score=similarity_score,
                processing_time=processing_time,
                token_usage={
                    "enhanced_processing": True,
                    "nlp_analysis_used": use_nlp_analysis,
                    "variants_generated": metadata.get("variants_generated", 1),
                    "quality_threshold": quality_threshold,
                    **metadata
                }
            )
            
            db.add(session)
            
            # Update document with paraphrased content
            document_processor.update_document_status(
                document_id, 
                DocumentStatus.COMPLETED, 
                db,
                paraphrased_content=paraphrased_text
            )
            
            db.commit()
            db.refresh(session)
            
            logger.info(
                f"Enhanced paraphrasing completed for document {document_id}. "
                f"Similarity: {similarity_score:.3f}, Time: {processing_time}s"
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Enhanced paraphrasing failed for document {document_id}: {str(e)}")
            
            # Update document status to failed
            document_processor.update_document_status(
                document_id, DocumentStatus.FAILED, db
            )
            
            # Create session record for tracking
            processing_time = int(time.time() - start_time)
            session = ParaphraseSession(
                document_id=document_id,
                method_used=method,
                processing_time=processing_time,
                token_usage={"error": str(e), "enhanced_processing": True}
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            return session
    
    async def _paraphrase_with_enhanced_indot5(
        self, 
        text: str, 
        analysis: Optional[DocumentAnalysis], 
        num_variants: int,
        quality_threshold: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Paraphrase using enhanced IndoT5 with analysis-based optimization."""
        
        if analysis:
            # Use analysis to determine optimal paraphrasing strategy
            high_priority_sentences = self.nlp_pipeline.extract_sentences_for_paraphrasing(
                analysis, min_priority=0.6, max_sentences=10
            )
            
            if high_priority_sentences:
                # Paraphrase high-priority sentences
                results = []
                for idx, sentence in high_priority_sentences:
                    try:
                        if self.enhanced_indot5:
                            result = await self.enhanced_indot5.paraphrase_single(
                                sentence, num_variants
                            )
                            results.append((idx, result))
                        else:
                            logger.warning("Enhanced IndoT5 paraphraser not available, skipping sentence")
                    except Exception as e:
                        logger.warning(f"Failed to paraphrase sentence {idx}: {e}")
                
                # Reconstruct document with paraphrased sentences
                paraphrased_text = self._reconstruct_document_with_paraphrases(
                    text, results, analysis
                )
                
                metadata = {
                    "strategy": "sentence_prioritization",
                    "sentences_paraphrased": len(results),
                    "total_sentences": len(analysis.sentences),
                    "variants_generated": sum(len(r[1].paraphrased_variants) for r in results)
                }
            else:
                # Paraphrase entire document
                if self.enhanced_indot5:
                    result = await self.enhanced_indot5.paraphrase_single(text, num_variants)
                    paraphrased_text = result.best_variant
                else:
                    logger.warning("Enhanced IndoT5 paraphraser not available, returning original text")
                    paraphrased_text = text
                    result = type('MockResult', (), {
                        'paraphrased_variants': [text],
                        'quality_scores': [0.5],
                        'processing_time': 0.0
                    })()
                
                metadata = {
                    "strategy": "full_document",
                    "variants_generated": len(result.paraphrased_variants),
                    "quality_scores": result.quality_scores,
                    "processing_time": result.processing_time
                }
        else:
            # Simple enhanced paraphrasing without analysis
            if self.enhanced_indot5:
                result = await self.enhanced_indot5.paraphrase_single(text, num_variants)
                paraphrased_text = result.best_variant
            else:
                logger.warning("Enhanced IndoT5 paraphraser not available, returning original text")
                paraphrased_text = text
                result = type('MockResult', (), {
                    'paraphrased_variants': [text],
                    'quality_scores': [0.5],
                    'processing_time': 0.0
                })()
            
            metadata = {
                "strategy": "simple_enhanced",
                "variants_generated": len(result.paraphrased_variants),
                "quality_scores": result.quality_scores
            }
        
        return paraphrased_text, metadata
    
    async def _paraphrase_with_hybrid_approach(
        self, 
        text: str, 
        analysis: Optional[DocumentAnalysis],
        preserve_academic_terms: bool,
        preserve_citations: bool,
        num_variants: int,
        quality_threshold: float
    ) -> Tuple[str, Dict[str, Any]]:
        """Paraphrase using hybrid rule-based and model-based approach."""
        
        # Step 1: Rule-based paraphrasing
        rule_based_results = await self.rule_based.paraphrase(
            text, preserve_academic_terms, preserve_citations, num_variants
        )
        
        best_rule_based = None
        if rule_based_results:
            best_rule_based = max(rule_based_results, key=lambda x: x[1].overall_score)
        
        # Step 2: Enhanced IndoT5 paraphrasing
        indot5_result = None
        if self.enhanced_indot5:
            indot5_result = await self.enhanced_indot5.paraphrase_single(text, num_variants)
        else:
            logger.warning("Enhanced IndoT5 paraphraser not available for hybrid approach")
        
        # Step 3: Compare and select best result
        candidates = []
        
        if best_rule_based and best_rule_based[1].overall_score >= quality_threshold:
            candidates.append(("rule_based", best_rule_based[0], best_rule_based[1].overall_score))
        
        if indot5_result and indot5_result.quality_scores:
            max_quality = max(indot5_result.quality_scores)
            if max_quality >= quality_threshold:
                best_idx = indot5_result.quality_scores.index(max_quality)
                best_variant = indot5_result.paraphrased_variants[best_idx]
                candidates.append(("indot5", best_variant, max_quality))
        
        # Select best candidate
        if candidates:
            best_method, paraphrased_text, score = max(candidates, key=lambda x: x[2])
        else:
            # Fallback to original if no candidate meets threshold
            paraphrased_text = text
            best_method = "fallback"
            score = 0.0
        
        metadata = {
            "strategy": "hybrid_comparison",
            "best_method": best_method,
            "rule_based_variants": len(rule_based_results),
            "indot5_variants": len(indot5_result.paraphrased_variants),
            "candidates_count": len(candidates),
            "final_quality_score": score,
            "quality_threshold": quality_threshold
        }
        
        return paraphrased_text, metadata
    
    def _reconstruct_document_with_paraphrases(
        self, 
        original_text: str, 
        paraphrase_results: List[Tuple[int, Any]],
        analysis: DocumentAnalysis
    ) -> str:
        """Reconstruct document by replacing specific sentences with their paraphrases."""
        
        sentences = [s.text for s in analysis.sentences]
        
        # Replace sentences with their paraphrases
        for sentence_idx, result in paraphrase_results:
            if sentence_idx < len(sentences) and result.best_variant:
                sentences[sentence_idx] = result.best_variant
        
        # Reconstruct document
        return ' '.join(sentences)
    
    def _calculate_enhanced_similarity(self, original: str, paraphrased: str) -> float:
        """Calculate enhanced similarity using multiple metrics."""
        # Use NLP pipeline for semantic similarity
        try:
            similarity = self.nlp_pipeline.calculate_text_similarity(original, paraphrased)
            return similarity
        except Exception as e:
            logger.warning(f"Enhanced similarity calculation failed: {e}")
            # Fallback to simple token similarity
            return self._simple_similarity(original, paraphrased)
    
    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Simple token-based similarity as fallback."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0

    async def paraphrase_document(
        self,
        document_id: uuid.UUID,
        method: ParaphraseMethod,
        db: Session
    ) -> Optional[ParaphraseSession]:
        """
        Paraphrase a document using the specified method.
        
        Args:
            document_id: Document UUID
            method: Paraphrasing method to use
            db: Database session
            
        Returns:
            ParaphraseSession: Created session record
        """
        # Get document
        document = document_processor.get_document(document_id, db)
        if not document:
            logger.error(f"Document {document_id} not found")
            return None
        
        # Update document status
        document_processor.update_document_status(
            document_id, DocumentStatus.PROCESSING, db
        )
        
        start_time = time.time()
        
        try:
            # Get paraphraser
            paraphraser = self.paraphrasers[method]
            
            # Perform paraphrasing
            logger.info(f"Starting paraphrasing for document {document_id} using {method}")
            paraphrased_text = await paraphraser.paraphrase(document.original_content)
            
            # Calculate similarity
            similarity_score = paraphraser.calculate_similarity(
                document.original_content, 
                paraphrased_text
            )
            
            processing_time = int(time.time() - start_time)
            
            # Create session record
            session = ParaphraseSession(
                document_id=document_id,
                method_used=method,
                similarity_score=similarity_score,
                processing_time=processing_time,
                token_usage={"method": method.value}  # Placeholder for token usage
            )
            
            db.add(session)
            
            # Update document with paraphrased content
            document_processor.update_document_status(
                document_id, 
                DocumentStatus.COMPLETED, 
                db,
                paraphrased_content=paraphrased_text
            )
            
            db.commit()
            db.refresh(session)
            
            logger.info(
                f"Paraphrasing completed for document {document_id}. "
                f"Similarity: {similarity_score:.3f}, Time: {processing_time}s"
            )
            
            return session
            
        except Exception as e:
            logger.error(f"Paraphrasing failed for document {document_id}: {str(e)}")
            
            # Update document status to failed
            document_processor.update_document_status(
                document_id, DocumentStatus.FAILED, db
            )
            
            # Still create session record for tracking
            processing_time = int(time.time() - start_time)
            session = ParaphraseSession(
                document_id=document_id,
                method_used=method,
                processing_time=processing_time,
                token_usage={"error": str(e)}
            )
            
            db.add(session)
            db.commit()
            db.refresh(session)
            
            return session
    
    def get_session(self, session_id: uuid.UUID, db: Session) -> Optional[ParaphraseSession]:
        """
        Get paraphrase session by ID.
        
        Args:
            session_id: Session UUID
            db: Database session
            
        Returns:
            ParaphraseSession or None if not found
        """
        return db.query(ParaphraseSession).filter(
            ParaphraseSession.id == session_id
        ).first()
    
    def get_document_sessions(
        self, 
        document_id: uuid.UUID, 
        db: Session
    ) -> List[ParaphraseSession]:
        """
        Get all paraphrase sessions for a document.
        
        Args:
            document_id: Document UUID
            db: Database session
            
        Returns:
            List of ParaphraseSession records
        """
        return db.query(ParaphraseSession).filter(
            ParaphraseSession.document_id == document_id
        ).order_by(ParaphraseSession.created_at.desc()).all()


# Global instances
paraphrasing_service = EnhancedParaphrasingService()  # Legacy service
enhanced_paraphrasing_service = EnhancedParaphrasingService()  # Enhanced service
