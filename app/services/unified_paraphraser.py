"""
Unified Paraphrasing Service
Consolidated paraphrasing engine integrating all existing methodologies
with custom synonyms dictionary integration and comprehensive quality assessment.
"""
import json
import logging
import asyncio
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path
import numpy as np
from enum import Enum

from app.services.enhanced_indot5_paraphraser import EnhancedIndoT5Paraphraser, ParaphraseResult
from app.services.rule_based_paraphraser import HybridRuleBasedParaphraser
from app.services.indonesian_nlp_pipeline import IndonesianNLPPipeline, DocumentAnalysis
from app.services.enhanced_document_processor import EnhancedDocumentProcessor

logger = logging.getLogger(__name__)


class FormalityLevel(str, Enum):
    """Formality levels for paraphrasing."""
    ACADEMIC = "academic"
    FORMAL = "formal"
    NEUTRAL = "neutral"


class ComplexityTarget(str, Enum):
    """Target complexity levels."""
    SIMPLIFY = "simplify"
    MAINTAIN = "maintain"
    ENHANCE = "enhance"


@dataclass
class UnifiedInput:
    """Input data structure for unified paraphrasing."""
    text: str
    options: 'UnifiedOptions'
    quality_criteria: 'QualityCriteria'


@dataclass
class UnifiedOptions:
    """Configuration options for unified paraphrasing."""
    num_variants: int = 3
    preserve_academic_terms: bool = True
    preserve_citations: bool = True
    formality_level: FormalityLevel = FormalityLevel.ACADEMIC
    target_complexity: ComplexityTarget = ComplexityTarget.MAINTAIN
    
    # Method-specific options
    use_indot5: bool = True
    use_rule_based: bool = True
    use_custom_synonyms: bool = True
    use_structural_transformation: bool = True


@dataclass
class QualityCriteria:
    """Quality criteria for paraphrase selection."""
    quality_threshold: float = 0.7
    min_semantic_similarity: float = 0.3
    max_semantic_similarity: float = 0.9
    min_grammar_score: float = 0.6
    require_academic_tone: bool = True


@dataclass
class VariantCandidate:
    """A candidate paraphrase variant with metadata."""
    text: str
    source_method: str
    confidence: float
    quality_scores: Dict[str, float]
    metadata: Dict[str, Any]


@dataclass
class ProtectedContent:
    """Content with protected elements marked."""
    protected_text: str
    protection_map: Dict[str, str]
    academic_terms: List[str]
    citations: List[str]


@dataclass
class QualityAssessmentResult:
    """Comprehensive quality assessment result."""
    overall_score: float
    dimension_scores: Dict[str, float]
    confidence_level: float
    recommendations: List[str]
    meets_threshold: bool


@dataclass
class UnifiedResult:
    """Final result from unified paraphrasing."""
    original_text: str
    best_variant: str
    all_variants: List[VariantCandidate]
    quality_assessment: QualityAssessmentResult
    processing_time: float
    method_contributions: Dict[str, int]
    metadata: Dict[str, Any]


class UnifiedQualityAssessment:
    """Unified quality assessment framework combining all existing metrics."""
    
    def __init__(self):
        self.weights = {
            'semantic_similarity': 0.25,
            'grammar_correctness': 0.20,
            'academic_tone_preservation': 0.20,
            'readability_score': 0.15,
            'structural_diversity': 0.10,
            'context_appropriateness': 0.10
        }
    
    def assess_comprehensive_quality(
        self, 
        original: str, 
        variant: str, 
        analysis: DocumentAnalysis
    ) -> QualityAssessmentResult:
        """
        Multi-dimensional quality assessment consolidating all existing metrics.
        """
        quality_dimensions = {
            'semantic_similarity': self._calculate_semantic_similarity(original, variant),
            'grammar_correctness': self._assess_grammar_quality(variant),
            'academic_tone_preservation': self._evaluate_academic_tone(variant),
            'readability_score': self._calculate_readability(variant),
            'structural_diversity': self._measure_structural_changes(original, variant),
            'context_appropriateness': self._evaluate_context_fit(variant, analysis)
        }
        
        # Weighted comprehensive scoring
        overall_score = sum(
            quality_dimensions[dimension] * self.weights[dimension]
            for dimension in quality_dimensions
        )
        
        confidence_level = self._calculate_confidence(quality_dimensions)
        recommendations = self._generate_quality_recommendations(quality_dimensions)
        
        return QualityAssessmentResult(
            overall_score=overall_score,
            dimension_scores=quality_dimensions,
            confidence_level=confidence_level,
            recommendations=recommendations,
            meets_threshold=overall_score >= 0.7  # Default threshold
        )
    
    def _calculate_semantic_similarity(self, original: str, variant: str) -> float:
        """Calculate semantic similarity using multiple approaches."""
        # Simple token overlap as baseline
        orig_words = set(original.lower().split())
        var_words = set(variant.lower().split())
        
        if not orig_words:
            return 0.0
        
        intersection = orig_words.intersection(var_words)
        union = orig_words.union(var_words)
        
        jaccard_similarity = len(intersection) / len(union) if union else 0.0
        
        # Enhanced with character-level similarity
        from difflib import SequenceMatcher
        char_similarity = SequenceMatcher(None, original, variant).ratio()
        
        # Combine both metrics
        return (jaccard_similarity * 0.6 + char_similarity * 0.4)
    
    def _assess_grammar_quality(self, text: str) -> float:
        """Assess grammar quality using simple heuristics."""
        words = text.split()
        
        if len(words) < 2:
            return 0.3
        
        # Check basic sentence structure
        has_capital_start = text[0].isupper() if text else False
        has_punctuation_end = text[-1] in '.!?' if text else False
        
        # Check for repeated words
        word_set = set(words)
        repetition_ratio = len(words) / len(word_set) if word_set else 1.0
        
        # Basic scoring
        structure_score = 0.5 + (0.25 if has_capital_start else 0) + (0.25 if has_punctuation_end else 0)
        repetition_score = max(0, 2.0 - repetition_ratio) / 2.0
        
        return (structure_score + repetition_score) / 2.0
    
    def _evaluate_academic_tone(self, text: str) -> float:
        """Evaluate academic tone preservation."""
        academic_indicators = [
            'penelitian', 'analisis', 'metode', 'hasil', 'kesimpulan',
            'research', 'analysis', 'method', 'results', 'conclusion',
            'berdasarkan', 'menunjukkan', 'mengindikasikan', 'menjelaskan'
        ]
        
        text_lower = text.lower()
        academic_count = sum(1 for indicator in academic_indicators if indicator in text_lower)
        
        # Normalize by text length
        words = text.split()
        if not words:
            return 0.5
        
        density = academic_count / len(words)
        return min(1.0, density * 10)  # Scale appropriately
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score."""
        words = text.split()
        sentences = text.split('.')
        
        if not words or not sentences:
            return 0.5
        
        avg_sentence_length = len(words) / len(sentences)
        
        # Simple readability heuristic
        if avg_sentence_length < 10:
            return 0.9  # Very readable
        elif avg_sentence_length < 20:
            return 0.7  # Good readability
        elif avg_sentence_length < 30:
            return 0.5  # Moderate readability
        else:
            return 0.3  # Poor readability
    
    def _measure_structural_changes(self, original: str, variant: str) -> float:
        """Measure structural diversity between original and variant."""
        orig_sentences = original.split('.')
        var_sentences = variant.split('.')
        
        # Compare sentence count
        sentence_diff = abs(len(orig_sentences) - len(var_sentences))
        sentence_score = max(0, 1.0 - sentence_diff * 0.2)
        
        # Compare average sentence length
        orig_avg = np.mean([len(s.split()) for s in orig_sentences if s.strip()])
        var_avg = np.mean([len(s.split()) for s in var_sentences if s.strip()])
        
        length_diff = abs(orig_avg - var_avg) / max(orig_avg, 1)
        length_score = max(0, 1.0 - length_diff)
        
        return (sentence_score + length_score) / 2.0
    
    def _evaluate_context_fit(self, variant: str, analysis: DocumentAnalysis) -> float:
        """Evaluate how well the variant fits the original context."""
        # For now, return a moderate score
        # This could be enhanced with more sophisticated analysis
        return 0.75
    
    def _calculate_confidence(self, quality_dimensions: Dict[str, float]) -> float:
        """Calculate confidence level based on dimension consistency."""
        scores = list(quality_dimensions.values())
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        # High confidence if scores are consistent and high
        consistency = max(0, 1.0 - std_score)
        quality = mean_score
        
        return (consistency * 0.4 + quality * 0.6)
    
    def _generate_quality_recommendations(self, quality_dimensions: Dict[str, float]) -> List[str]:
        """Generate recommendations based on quality scores."""
        recommendations = []
        
        if quality_dimensions['semantic_similarity'] < 0.3:
            recommendations.append("Increase semantic similarity to original text")
        elif quality_dimensions['semantic_similarity'] > 0.9:
            recommendations.append("Reduce similarity to create more variation")
        
        if quality_dimensions['grammar_correctness'] < 0.6:
            recommendations.append("Improve grammar and sentence structure")
        
        if quality_dimensions['academic_tone_preservation'] < 0.5:
            recommendations.append("Maintain academic tone and terminology")
        
        if quality_dimensions['readability_score'] < 0.5:
            recommendations.append("Improve readability and sentence clarity")
        
        return recommendations


class UnifiedParaphraser:
    """
    Consolidated paraphrasing engine integrating all existing methodologies
    with custom synonyms dictionary integration and comprehensive quality assessment.
    """
    
    def __init__(self, custom_synonyms_path: Optional[str] = None):
        # Initialize ALL existing components
        self.indot5_engine = None  # Lazy loading
        self.rule_based_engine = None  # Lazy loading
        self.nlp_pipeline = IndonesianNLPPipeline()
        self.document_processor = EnhancedDocumentProcessor()
        
        # Load and integrate custom synonyms dictionary
        self.custom_synonyms = {}
        if custom_synonyms_path:
            self.custom_synonyms = self._load_custom_synonyms(custom_synonyms_path)
        
        # Initialize unified quality assessment framework
        self.quality_assessor = UnifiedQualityAssessment()
        
        # Performance tracking
        self.stats = {
            'total_requests': 0,
            'method_usage': {
                'indot5': 0,
                'rule_based': 0,
                'custom_synonyms': 0,
                'structural': 0
            },
            'average_processing_time': 0.0
        }
    
    def _load_custom_synonyms(self, synonyms_path: str) -> Dict[str, Any]:
        """Load custom synonyms.json with validation."""
        try:
            synonyms_file = Path(synonyms_path)
            if not synonyms_file.exists():
                logger.warning(f"Custom synonyms file not found: {synonyms_path}")
                return {}
            
            with open(synonyms_file, 'r', encoding='utf-8') as f:
                synonyms_data = json.load(f)
            
            # Validate structure
            validated_synonyms = {}
            for word, data in synonyms_data.items():
                if isinstance(data, dict) and 'sinonim' in data:
                    validated_synonyms[word] = {
                        'tag': data.get('tag', 'UNKNOWN'),
                        'sinonim': data['sinonim'],
                        'confidence': 0.95  # High priority for custom synonyms
                    }
            
            logger.info(f"Loaded {len(validated_synonyms)} custom synonyms")
            return validated_synonyms
            
        except Exception as e:
            logger.error(f"Failed to load custom synonyms: {e}")
            return {}
    
    def _get_indot5_engine(self) -> Optional[EnhancedIndoT5Paraphraser]:
        """Lazy load IndoT5 engine."""
        if self.indot5_engine is None:
            try:
                self.indot5_engine = EnhancedIndoT5Paraphraser()
                logger.info("IndoT5 engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize IndoT5 engine: {e}")
                return None
        return self.indot5_engine
    
    def _get_rule_based_engine(self) -> Optional[HybridRuleBasedParaphraser]:
        """Lazy load rule-based engine."""
        if self.rule_based_engine is None:
            try:
                self.rule_based_engine = HybridRuleBasedParaphraser()
                # Merge custom synonyms into rule-based engine
                if self.custom_synonyms:
                    self._merge_synonym_databases()
                logger.info("Rule-based engine initialized")
            except Exception as e:
                logger.error(f"Failed to initialize rule-based engine: {e}")
                return None
        return self.rule_based_engine
    
    def _merge_synonym_databases(self):
        """Merge custom synonyms with existing Indonesian synonym databases."""
        if not self.rule_based_engine or not self.custom_synonyms:
            return
        
        try:
            # Access the synonym dictionary from rule-based engine
            if hasattr(self.rule_based_engine, 'synonym_dict'):
                existing_dict = self.rule_based_engine.synonym_dict
                
                # Merge custom synonyms with high priority
                for word, data in self.custom_synonyms.items():
                    existing_dict.add_synonym_entry(
                        word=word,
                        synonyms=data['sinonim'],
                        pos_tag=data['tag'],
                        confidence=data['confidence']
                    )
                
                logger.info("Custom synonyms merged with existing database")
                
        except Exception as e:
            logger.error(f"Failed to merge synonym databases: {e}")
    
    async def unified_paraphrase(self, input_data: UnifiedInput) -> UnifiedResult:
        """
        Master paraphrasing method consolidating all existing approaches.
        """
        start_time = time.time()
        self.stats['total_requests'] += 1
        
        try:
            # PHASE 1: Comprehensive Text Analysis
            analysis_result = await self._comprehensive_analysis(input_data.text)
            
            # PHASE 2: Multi-Layer Protection Strategy
            protected_content = await self._apply_comprehensive_protection(
                input_data.text, analysis_result, input_data.options
            )
            
            # PHASE 3: Multi-Method Variant Generation
            variant_candidates = await self._generate_unified_variants(
                protected_content, analysis_result, input_data.options
            )
            
            # PHASE 4: Advanced Quality Assessment & Selection
            optimized_result = await self._select_optimal_variant(
                variant_candidates, analysis_result, input_data.quality_criteria
            )
            
            # PHASE 5: Content Restoration & Formatting
            processing_time = time.time() - start_time
            final_output = await self._restore_and_format(
                optimized_result, 
                protected_content.protection_map,
                input_data.text,
                variant_candidates,
                analysis_result,
                processing_time
            )
            
            # Update statistics
            self.stats['average_processing_time'] = (
                (self.stats['average_processing_time'] * (self.stats['total_requests'] - 1) + processing_time)
                / self.stats['total_requests']
            )
            
            return final_output
            
        except Exception as e:
            logger.error(f"Unified paraphrasing failed: {e}")
            processing_time = time.time() - start_time
            
            # Return fallback result
            return UnifiedResult(
                original_text=input_data.text,
                best_variant=input_data.text,
                all_variants=[],
                quality_assessment=QualityAssessmentResult(
                    overall_score=0.0,
                    dimension_scores={},
                    confidence_level=0.0,
                    recommendations=["Processing failed, original text returned"],
                    meets_threshold=False
                ),
                processing_time=processing_time,
                method_contributions={},
                metadata={"error": str(e)}
            )
    
    async def _comprehensive_analysis(self, text: str) -> DocumentAnalysis:
        """Perform comprehensive text analysis using NLP pipeline."""
        try:
            analysis = await self.nlp_pipeline.analyze_document(text)
            logger.debug(f"Text analysis completed: {len(analysis.sentences)} sentences")
            return analysis
        except Exception as e:
            logger.error(f"Text analysis failed: {e}")
            # Return minimal analysis
            from app.services.indonesian_nlp_pipeline import DocumentAnalysis, SentenceInfo
            return DocumentAnalysis(
                sentences=[SentenceInfo(
                    text=text,
                    tokens=[],
                    academic_terms=[],
                    named_entities=[],
                    has_citations=False,
                    complexity_score=0.5,
                    readability_score=0.5,
                    priority_for_paraphrasing=0.5
                )],
                overall_readability=0.5,
                overall_complexity=0.5,
                academic_terms=[],
                named_entities=[],
                quality_metrics={}
            )
    
    async def _apply_comprehensive_protection(
        self, 
        text: str, 
        analysis: DocumentAnalysis,
        options: UnifiedOptions
    ) -> ProtectedContent:
        """Apply multi-layer protection strategy for important content."""
        protection_map = {}
        protected_text = text
        academic_terms = []
        citations = []
        
        try:
            # Protect academic terms if requested
            if options.preserve_academic_terms:
                academic_terms = analysis.academic_terms
                protected_text, term_map = self.nlp_pipeline.preserve_academic_terms(protected_text)
                protection_map.update(term_map)
            
            # Protect citations if requested
            if options.preserve_citations:
                # Extract and protect citation patterns
                import re
                citation_patterns = [
                    r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
                    r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2023]
                    r'(?:et al\.|dkk\.)',  # et al. or dkk.
                ]
                
                for i, pattern in enumerate(citation_patterns):
                    matches = list(re.finditer(pattern, protected_text))
                    for j, match in enumerate(matches):
                        placeholder = f"__CITATION_{i}_{j}__"
                        protection_map[placeholder] = match.group()
                        citations.append(match.group())
                        protected_text = protected_text.replace(match.group(), placeholder, 1)
            
            return ProtectedContent(
                protected_text=protected_text,
                protection_map=protection_map,
                academic_terms=academic_terms,
                citations=citations
            )
            
        except Exception as e:
            logger.error(f"Content protection failed: {e}")
            return ProtectedContent(
                protected_text=text,
                protection_map={},
                academic_terms=[],
                citations=[]
            )
    
    async def _generate_unified_variants(
        self, 
        protected_content: ProtectedContent, 
        analysis: DocumentAnalysis,
        options: UnifiedOptions
    ) -> List[VariantCandidate]:
        """
        Generate variants using ALL existing methods in parallel.
        """
        variant_pool = []
        
        # Method 1: Enhanced IndoT5 Neural Processing
        if options.use_indot5:
            indot5_variants = await self._generate_indot5_variants(
                protected_content.protected_text, options
            )
            variant_pool.extend(indot5_variants)
            self.stats['method_usage']['indot5'] += len(indot5_variants)
        
        # Method 2: Advanced Rule-Based Transformation
        if options.use_rule_based:
            rule_based_variants = await self._generate_rule_based_variants(
                protected_content.protected_text, options
            )
            variant_pool.extend(rule_based_variants)
            self.stats['method_usage']['rule_based'] += len(rule_based_variants)
        
        # Method 3: Custom Synonyms Application
        if options.use_custom_synonyms and self.custom_synonyms:
            custom_synonym_variants = await self._apply_custom_synonyms(
                protected_content.protected_text, analysis, options
            )
            variant_pool.extend(custom_synonym_variants)
            self.stats['method_usage']['custom_synonyms'] += len(custom_synonym_variants)
        
        # Method 4: Hybrid Structural Transformation
        if options.use_structural_transformation:
            structural_variants = await self._apply_structural_transformations(
                protected_content.protected_text, analysis
            )
            variant_pool.extend(structural_variants)
            self.stats['method_usage']['structural'] += len(structural_variants)
        
        return variant_pool
    
    async def _generate_indot5_variants(
        self, 
        text: str, 
        options: UnifiedOptions
    ) -> List[VariantCandidate]:
        """Generate variants using Enhanced IndoT5 model."""
        variants = []
        
        try:
            indot5_engine = self._get_indot5_engine()
            if not indot5_engine:
                return variants
            
            result = await indot5_engine.paraphrase_single(text, options.num_variants)
            
            for i, variant_text in enumerate(result.paraphrased_variants):
                variants.append(VariantCandidate(
                    text=variant_text,
                    source_method="indot5",
                    confidence=result.quality_scores[i] if i < len(result.quality_scores) else 0.5,
                    quality_scores={
                        "similarity": result.similarity_scores[i] if i < len(result.similarity_scores) else 0.5,
                        "quality": result.quality_scores[i] if i < len(result.quality_scores) else 0.5
                    },
                    metadata={"model": "enhanced_indot5", "processing_time": result.processing_time}
                ))
                
        except Exception as e:
            logger.error(f"IndoT5 variant generation failed: {e}")
        
        return variants
    
    async def _generate_rule_based_variants(
        self, 
        text: str, 
        options: UnifiedOptions
    ) -> List[VariantCandidate]:
        """Generate variants using rule-based approach."""
        variants = []
        
        try:
            rule_based_engine = self._get_rule_based_engine()
            if not rule_based_engine:
                return variants
            
            results = await rule_based_engine.paraphrase(
                text,
                preserve_academic_terms=options.preserve_academic_terms,
                preserve_citations=options.preserve_citations,
                num_variants=options.num_variants
            )
            
            for variant_text, quality in results:
                variants.append(VariantCandidate(
                    text=variant_text,
                    source_method="rule_based",
                    confidence=quality.overall_score,
                    quality_scores={
                        "grammar": quality.grammar_score,
                        "readability": quality.readability_score,
                        "academic_tone": quality.academic_tone_score,
                        "overall": quality.overall_score
                    },
                    metadata={"method": "rule_based_transformation"}
                ))
                
        except Exception as e:
            logger.error(f"Rule-based variant generation failed: {e}")
        
        return variants
    
    async def _apply_custom_synonyms(
        self, 
        text: str, 
        analysis: DocumentAnalysis,
        options: UnifiedOptions
    ) -> List[VariantCandidate]:
        """Apply custom synonyms to generate variants."""
        variants = []
        
        try:
            if not self.custom_synonyms:
                return variants
            
            words = text.split()
            modified_texts = []
            
            # Generate multiple variants by applying different synonym combinations
            for attempt in range(min(options.num_variants, 3)):
                modified_words = words.copy()
                modifications = 0
                
                for i, word in enumerate(words):
                    word_lower = word.lower().strip('.,!?;:')
                    
                    if word_lower in self.custom_synonyms:
                        synonyms = self.custom_synonyms[word_lower]['sinonim']
                        if synonyms and modifications < 3:  # Limit modifications per variant
                            # Select synonym based on attempt number
                            synonym_idx = attempt % len(synonyms)
                            selected_synonym = synonyms[synonym_idx]
                            
                            # Preserve capitalization
                            if word[0].isupper():
                                selected_synonym = selected_synonym.capitalize()
                            
                            modified_words[i] = word.replace(word_lower, selected_synonym)
                            modifications += 1
                
                if modifications > 0:
                    modified_text = ' '.join(modified_words)
                    modified_texts.append((modified_text, modifications))
            
            # Create variant candidates
            for modified_text, modifications in modified_texts:
                confidence = min(0.9, 0.7 + (modifications * 0.1))
                
                variants.append(VariantCandidate(
                    text=modified_text,
                    source_method="custom_synonyms",
                    confidence=confidence,
                    quality_scores={
                        "synonym_density": modifications / len(words),
                        "custom_confidence": confidence
                    },
                    metadata={"modifications": modifications, "method": "custom_synonyms"}
                ))
                
        except Exception as e:
            logger.error(f"Custom synonyms application failed: {e}")
        
        return variants
    
    async def _apply_structural_transformations(
        self, 
        text: str, 
        analysis: DocumentAnalysis
    ) -> List[VariantCandidate]:
        """Apply structural transformations to generate variants."""
        variants = []
        
        try:
            # Basic structural transformations
            sentences = text.split('.')
            
            if len(sentences) > 2:
                # Sentence reordering
                reordered_text = '. '.join(sentences[1:] + [sentences[0]])
                if reordered_text.strip():
                    variants.append(VariantCandidate(
                        text=reordered_text,
                        source_method="structural",
                        confidence=0.6,
                        quality_scores={"structural_change": 0.8},
                        metadata={"transformation": "sentence_reordering"}
                    ))
            
            # Clause combination/separation
            if ', ' in text:
                combined_text = text.replace(', ', ' dan ')
                variants.append(VariantCandidate(
                    text=combined_text,
                    source_method="structural",
                    confidence=0.7,
                    quality_scores={"structural_change": 0.6},
                    metadata={"transformation": "clause_combination"}
                ))
                
        except Exception as e:
            logger.error(f"Structural transformation failed: {e}")
        
        return variants
    
    async def _select_optimal_variant(
        self, 
        variant_candidates: List[VariantCandidate],
        analysis: DocumentAnalysis,
        quality_criteria: QualityCriteria
    ) -> VariantCandidate:
        """Advanced quality assessment and selection of optimal variant."""
        if not variant_candidates:
            logger.warning("No variant candidates available for selection")
            return None
        
        try:
            # Assess quality for each candidate
            scored_candidates = []
            
            for candidate in variant_candidates:
                quality_assessment = self.quality_assessor.assess_comprehensive_quality(
                    analysis.sentences[0].text if analysis.sentences else "",
                    candidate.text,
                    analysis
                )
                
                # Combine candidate confidence with quality assessment
                final_score = (
                    candidate.confidence * 0.3 + 
                    quality_assessment.overall_score * 0.7
                )
                
                if quality_assessment.overall_score >= quality_criteria.quality_threshold:
                    scored_candidates.append((candidate, final_score, quality_assessment))
            
            # Select best candidate
            if scored_candidates:
                best_candidate, best_score, best_quality = max(
                    scored_candidates, key=lambda x: x[1]
                )
                logger.info(f"Selected variant with score {best_score:.3f} from {best_candidate.source_method}")
                return best_candidate
            else:
                # Fallback to highest confidence candidate
                fallback_candidate = max(variant_candidates, key=lambda x: x.confidence)
                logger.warning(f"No candidate met quality threshold, using fallback: {fallback_candidate.source_method}")
                return fallback_candidate
                
        except Exception as e:
            logger.error(f"Variant selection failed: {e}")
            # Return first candidate as ultimate fallback
            return variant_candidates[0] if variant_candidates else None
    
    async def _restore_and_format(
        self, 
        selected_variant: VariantCandidate,
        protection_map: Dict[str, str],
        original_text: str,
        all_variants: List[VariantCandidate],
        analysis: DocumentAnalysis,
        processing_time: float
    ) -> UnifiedResult:
        """Restore protected content and format final output."""
        if not selected_variant:
            logger.error("No variant selected for restoration")
            # Return fallback result
            return UnifiedResult(
                original_text=original_text,
                best_variant=original_text,
                all_variants=[],
                quality_assessment=QualityAssessmentResult(
                    overall_score=0.0,
                    dimension_scores={},
                    confidence_level=0.0,
                    recommendations=["No variant could be generated"],
                    meets_threshold=False
                ),
                processing_time=processing_time,
                method_contributions={},
                metadata={"error": "No variant selected"}
            )
        
        try:
            # Restore protected content
            final_text = selected_variant.text
            
            for placeholder, original in protection_map.items():
                final_text = final_text.replace(placeholder, original)
            
            # Create comprehensive result
            method_contributions = {}
            for candidate in all_variants:
                method = candidate.source_method
                method_contributions[method] = method_contributions.get(method, 0) + 1
            
            # Get quality assessment for best variant
            quality_assessment = self.quality_assessor.assess_comprehensive_quality(
                original_text,
                final_text,
                analysis
            )
            
            return UnifiedResult(
                original_text=original_text,
                best_variant=final_text,
                all_variants=all_variants,
                quality_assessment=quality_assessment,
                processing_time=processing_time,
                method_contributions=method_contributions,
                metadata=selected_variant.metadata
            )
            
        except Exception as e:
            logger.error(f"Content restoration failed: {e}")
            # Return fallback result
            return UnifiedResult(
                original_text=original_text,
                best_variant=original_text,
                all_variants=all_variants,
                quality_assessment=QualityAssessmentResult(
                    overall_score=0.0,
                    dimension_scores={},
                    confidence_level=0.0,
                    recommendations=[f"Restoration failed: {str(e)}"],
                    meets_threshold=False
                ),
                processing_time=processing_time,
                method_contributions={},
                metadata={"error": str(e)}
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        return self.stats.copy()


# Global instance for the unified paraphraser
unified_paraphraser = None


def get_unified_paraphraser(custom_synonyms_path: Optional[str] = None) -> UnifiedParaphraser:
    """Get global unified paraphraser instance with lazy initialization."""
    global unified_paraphraser
    
    if unified_paraphraser is None:
        unified_paraphraser = UnifiedParaphraser(custom_synonyms_path)
        logger.info("Unified paraphraser initialized")
    
    return unified_paraphraser
