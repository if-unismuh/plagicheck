"""
Indonesian NLP Pipeline Service
Comprehensive NLP preprocessing pipeline using spaCy for Indonesian language.
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
import asyncio

import spacy
from spacy.lang.id import Indonesian
import spacy_udpipe
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from textstat import flesch_reading_ease, coleman_liau_index
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
import numpy as np

logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    logger.warning("Failed to download NLTK data")


@dataclass
class TokenInfo:
    """Information about a token."""
    text: str
    lemma: str
    pos: str
    tag: str
    is_academic: bool
    is_named_entity: bool
    entity_type: Optional[str]
    dependency: str
    head_text: str


@dataclass
class SentenceInfo:
    """Information about a sentence."""
    text: str
    tokens: List[TokenInfo]
    complexity_score: float
    readability_score: float
    has_citations: bool
    has_academic_terms: bool
    named_entities: List[Dict[str, Any]]
    sentiment: Optional[str]
    priority_for_paraphrasing: float


@dataclass
class DocumentAnalysis:
    """Complete document analysis."""
    sentences: List[SentenceInfo]
    overall_readability: float
    overall_complexity: float
    academic_terms: Set[str]
    named_entities: Set[str]
    quality_metrics: Dict[str, Any]
    paraphrasing_priorities: List[int]  # Sentence indices sorted by priority


class IndonesianNLPPipeline:
    """Comprehensive NLP pipeline for Indonesian language processing."""
    
    def __init__(self):
        self.nlp_id = None
        self.nlp_en = None
        self.sentence_model = None
        self._initialize_models()
        
        # Academic terms patterns for Indonesian
        self.academic_terms_id = {
            'penelitian', 'riset', 'analisis', 'metode', 'metodologi',
            'hipotesis', 'teori', 'konsep', 'kerangka', 'framework',
            'variabel', 'sampel', 'populasi', 'data', 'dataset',
            'observasi', 'eksperimen', 'survei', 'wawancara', 'kuesioner',
            'statistik', 'signifikan', 'korelasi', 'regresi', 'validitas',
            'reliabilitas', 'instrumen', 'teknik', 'prosedur', 'hasil',
            'temuan', 'kesimpulan', 'rekomendasi', 'implikasi', 'saran',
            'literatur', 'referensi', 'sitasi', 'daftar pustaka',
            'abstrak', 'pendahuluan', 'tinjauan', 'pembahasan', 'penutup'
        }
        
        # Academic terms patterns for English (fallback)
        self.academic_terms_en = {
            'research', 'study', 'analysis', 'method', 'methodology',
            'hypothesis', 'theory', 'concept', 'framework', 'model',
            'variable', 'sample', 'population', 'data', 'dataset',
            'observation', 'experiment', 'survey', 'interview', 'questionnaire',
            'statistics', 'significant', 'correlation', 'regression', 'validity',
            'reliability', 'instrument', 'technique', 'procedure', 'results',
            'findings', 'conclusion', 'recommendation', 'implication', 'suggestion',
            'literature', 'reference', 'citation', 'bibliography',
            'abstract', 'introduction', 'review', 'discussion', 'conclusion'
        }
        
        # Indonesian stopwords
        self.stopwords_id = {
            'yang', 'dan', 'di', 'ke', 'dari', 'dalam', 'untuk', 'pada',
            'dengan', 'adalah', 'ini', 'itu', 'akan', 'telah', 'dapat',
            'ada', 'tidak', 'atau', 'juga', 'seperti', 'karena', 'jika',
            'maka', 'sehingga', 'namun', 'tetapi', 'sedangkan', 'bahwa'
        }
        
        # Citation patterns
        self.citation_patterns = [
            r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
            r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2023]
            r'(?:et al\.|dkk\.)',  # et al. or dkk.
            r'\b(?:ibid|op\. cit|loc\. cit)\b',  # Latin abbreviations
        ]
    
    def _initialize_models(self):
        """Initialize NLP models with fallback options."""
        # Try to load Indonesian spaCy model
        try:
            # First try UDPipe-based Indonesian model
            self.nlp_id = spacy_udpipe.load("id")
            logger.info("Loaded UDPipe Indonesian model")
        except Exception as e:
            logger.warning(f"Failed to load UDPipe Indonesian model: {e}")
            
            # Fallback to basic Indonesian language class
            try:
                self.nlp_id = Indonesian()
                logger.info("Loaded basic Indonesian language model")
            except Exception as e:
                logger.warning(f"Failed to load Indonesian model: {e}")
        
        # Load English model as fallback
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
            logger.info("Loaded English spaCy model")
        except OSError:
            logger.warning("English spaCy model not found. Install with: python -m spacy download en_core_web_sm")
        
        # Load sentence transformer for semantic similarity
        try:
            self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            logger.info("Loaded multilingual sentence transformer")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
    
    async def analyze_document(self, text: str) -> DocumentAnalysis:
        """
        Perform comprehensive document analysis.
        
        Args:
            text: Document text to analyze
            
        Returns:
            DocumentAnalysis: Complete analysis results
        """
        # Sentence segmentation
        sentences = await self._segment_sentences(text)
        
        # Analyze each sentence
        sentence_infos = []
        all_academic_terms = set()
        all_named_entities = set()
        
        for sentence in sentences:
            sentence_info = await self._analyze_sentence(sentence)
            sentence_infos.append(sentence_info)
            
            # Collect academic terms and entities
            if sentence_info.has_academic_terms:
                academic_terms = self._extract_academic_terms(sentence)
                all_academic_terms.update(academic_terms)
            
            for entity in sentence_info.named_entities:
                all_named_entities.add(entity['text'])
        
        # Calculate overall metrics
        overall_readability = self._calculate_overall_readability(text)
        overall_complexity = self._calculate_overall_complexity(sentence_infos)
        
        # Determine paraphrasing priorities
        priorities = self._calculate_paraphrasing_priorities(sentence_infos)
        
        # Quality metrics
        quality_metrics = {
            "total_sentences": len(sentences),
            "avg_sentence_length": np.mean([len(s.text.split()) for s in sentence_infos]),
            "academic_terms_density": len(all_academic_terms) / len(text.split()) if text.split() else 0,
            "citation_count": sum(1 for s in sentence_infos if s.has_citations),
            "complex_sentences_count": sum(1 for s in sentence_infos if s.complexity_score > 0.7),
            "readability_distribution": self._get_readability_distribution(sentence_infos)
        }
        
        return DocumentAnalysis(
            sentences=sentence_infos,
            overall_readability=overall_readability,
            overall_complexity=overall_complexity,
            academic_terms=all_academic_terms,
            named_entities=all_named_entities,
            quality_metrics=quality_metrics,
            paraphrasing_priorities=priorities
        )
    
    async def _segment_sentences(self, text: str) -> List[str]:
        """Segment text into sentences."""
        # Clean text first
        text = self._clean_text(text)
        
        # Use NLTK for sentence tokenization (works well for Indonesian)
        try:
            sentences = sent_tokenize(text)
        except Exception as e:
            logger.warning(f"NLTK sentence tokenization failed: {e}")
            # Fallback to simple period-based splitting
            sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        # Filter out very short sentences
        sentences = [s for s in sentences if len(s.split()) >= 3]
        
        return sentences
    
    async def _analyze_sentence(self, sentence: str) -> SentenceInfo:
        """Analyze a single sentence comprehensively."""
        # Choose appropriate NLP model
        nlp = self.nlp_id if self.nlp_id else self.nlp_en
        
        tokens = []
        named_entities = []
        
        if nlp:
            try:
                doc = nlp(sentence)
                
                # Extract token information
                for token in doc:
                    token_info = TokenInfo(
                        text=token.text,
                        lemma=token.lemma_ if hasattr(token, 'lemma_') else token.text,
                        pos=token.pos_ if hasattr(token, 'pos_') else 'UNKNOWN',
                        tag=token.tag_ if hasattr(token, 'tag_') else 'UNKNOWN',
                        is_academic=self._is_academic_term(token.text),
                        is_named_entity=token.ent_type_ != '' if hasattr(token, 'ent_type_') else False,
                        entity_type=token.ent_type_ if hasattr(token, 'ent_type_') else None,
                        dependency=token.dep_ if hasattr(token, 'dep_') else 'UNKNOWN',
                        head_text=token.head.text if hasattr(token, 'head') else token.text
                    )
                    tokens.append(token_info)
                
                # Extract named entities
                if hasattr(doc, 'ents'):
                    for ent in doc.ents:
                        named_entities.append({
                            'text': ent.text,
                            'label': ent.label_,
                            'start': ent.start_char,
                            'end': ent.end_char
                        })
            
            except Exception as e:
                logger.warning(f"NLP analysis failed for sentence: {e}")
                # Create basic token info
                words = sentence.split()
                tokens = [
                    TokenInfo(
                        text=word,
                        lemma=word.lower(),
                        pos='UNKNOWN',
                        tag='UNKNOWN',
                        is_academic=self._is_academic_term(word),
                        is_named_entity=False,
                        entity_type=None,
                        dependency='UNKNOWN',
                        head_text=word
                    ) for word in words
                ]
        
        # Calculate metrics
        complexity_score = self._calculate_sentence_complexity(sentence, tokens)
        readability_score = self._calculate_sentence_readability(sentence)
        has_citations = self._has_citations(sentence)
        has_academic_terms = any(token.is_academic for token in tokens)
        priority = self._calculate_sentence_priority(
            sentence, complexity_score, readability_score, has_academic_terms
        )
        
        return SentenceInfo(
            text=sentence,
            tokens=tokens,
            complexity_score=complexity_score,
            readability_score=readability_score,
            has_citations=has_citations,
            has_academic_terms=has_academic_terms,
            named_entities=named_entities,
            sentiment=None,  # Can be enhanced with sentiment analysis
            priority_for_paraphrasing=priority
        )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Fix common encoding issues
        text = text.replace('\u2019', "'").replace('\u201c', '"').replace('\u201d', '"')
        text = text.replace('\u2013', '-').replace('\u2014', '-')
        
        return text.strip()
    
    def _is_academic_term(self, word: str) -> bool:
        """Check if word is an academic term."""
        word_lower = word.lower()
        return (word_lower in self.academic_terms_id or 
                word_lower in self.academic_terms_en)
    
    def _extract_academic_terms(self, sentence: str) -> Set[str]:
        """Extract academic terms from sentence."""
        words = set(word.lower() for word in sentence.split())
        academic_terms = words.intersection(
            self.academic_terms_id.union(self.academic_terms_en)
        )
        return academic_terms
    
    def _has_citations(self, sentence: str) -> bool:
        """Check if sentence contains citations."""
        for pattern in self.citation_patterns:
            if re.search(pattern, sentence, re.IGNORECASE):
                return True
        return False
    
    def _calculate_sentence_complexity(self, sentence: str, tokens: List[TokenInfo]) -> float:
        """Calculate sentence complexity score (0-1)."""
        # Factors that contribute to complexity:
        # 1. Sentence length
        # 2. Number of clauses (approximated by conjunctions)
        # 3. Academic terms density
        # 4. Passive voice (approximated)
        
        words = sentence.split()
        word_count = len(words)
        
        # Length factor (normalize to 0-1)
        length_factor = min(word_count / 30.0, 1.0)  # 30+ words = max complexity
        
        # Conjunction count (indicates complex structure)
        conjunctions = ['dan', 'atau', 'tetapi', 'namun', 'karena', 'jika', 'sehingga', 
                       'and', 'or', 'but', 'however', 'because', 'if', 'so']
        conjunction_count = sum(1 for word in words if word.lower() in conjunctions)
        conjunction_factor = min(conjunction_count / 3.0, 1.0)
        
        # Academic terms density
        academic_count = sum(1 for token in tokens if token.is_academic)
        academic_factor = min(academic_count / max(word_count * 0.3, 1), 1.0)
        
        # Passive voice indicators (Indonesian)
        passive_indicators = ['di-', 'ter-', 'ke-an']
        passive_count = sum(1 for word in words 
                           if any(word.lower().startswith(indicator) for indicator in passive_indicators))
        passive_factor = min(passive_count / max(word_count * 0.2, 1), 1.0)
        
        # Weighted combination
        complexity = (length_factor * 0.3 + 
                     conjunction_factor * 0.3 + 
                     academic_factor * 0.2 + 
                     passive_factor * 0.2)
        
        return min(complexity, 1.0)
    
    def _calculate_sentence_readability(self, sentence: str) -> float:
        """Calculate sentence readability score."""
        try:
            # Use Flesch Reading Ease as base
            score = flesch_reading_ease(sentence)
            # Normalize to 0-1 (higher = more readable)
            return max(0, min(score / 100.0, 1.0))
        except:
            # Fallback calculation based on word and syllable count
            words = sentence.split()
            avg_word_length = np.mean([len(word) for word in words]) if words else 0
            # Rough readability estimate (inverse of complexity)
            return max(0, 1.0 - (avg_word_length / 15.0))
    
    def _calculate_sentence_priority(
        self, 
        sentence: str, 
        complexity: float, 
        readability: float, 
        has_academic_terms: bool
    ) -> float:
        """Calculate priority for paraphrasing (0-1, higher = more priority)."""
        # High complexity sentences should be prioritized
        complexity_factor = complexity
        
        # Low readability sentences should be prioritized  
        readability_factor = 1.0 - readability
        
        # Sentences with academic terms might need careful handling
        academic_factor = 0.3 if has_academic_terms else 0.0
        
        # Very short or very long sentences might need attention
        word_count = len(sentence.split())
        length_factor = 0.0
        if word_count < 5 or word_count > 40:
            length_factor = 0.2
        
        priority = (complexity_factor * 0.4 + 
                   readability_factor * 0.4 + 
                   academic_factor * 0.1 + 
                   length_factor * 0.1)
        
        return min(priority, 1.0)
    
    def _calculate_overall_readability(self, text: str) -> float:
        """Calculate overall document readability."""
        try:
            return max(0, flesch_reading_ease(text) / 100.0)
        except:
            # Fallback calculation
            words = text.split()
            sentences = self._segment_sentences(text)
            
            if not sentences or not words:
                return 0.5
            
            avg_sentence_length = len(words) / len(sentences)
            avg_word_length = np.mean([len(word) for word in words])
            
            # Simple readability estimate
            readability = 1.0 - min((avg_sentence_length / 25.0 + avg_word_length / 8.0) / 2.0, 1.0)
            return max(0, readability)
    
    def _calculate_overall_complexity(self, sentence_infos: List[SentenceInfo]) -> float:
        """Calculate overall document complexity."""
        if not sentence_infos:
            return 0.0
        
        return np.mean([s.complexity_score for s in sentence_infos])
    
    def _calculate_paraphrasing_priorities(self, sentence_infos: List[SentenceInfo]) -> List[int]:
        """Calculate paraphrasing priorities and return sorted indices."""
        priorities = [(i, sentence.priority_for_paraphrasing) 
                     for i, sentence in enumerate(sentence_infos)]
        
        # Sort by priority (descending)
        priorities.sort(key=lambda x: x[1], reverse=True)
        
        return [i for i, _ in priorities]
    
    def _get_readability_distribution(self, sentence_infos: List[SentenceInfo]) -> Dict[str, int]:
        """Get distribution of readability scores."""
        if not sentence_infos:
            return {"easy": 0, "medium": 0, "hard": 0}
        
        readability_scores = [s.readability_score for s in sentence_infos]
        
        easy = sum(1 for score in readability_scores if score > 0.7)
        medium = sum(1 for score in readability_scores if 0.3 <= score <= 0.7)
        hard = sum(1 for score in readability_scores if score < 0.3)
        
        return {"easy": easy, "medium": medium, "hard": hard}
    
    def calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.sentence_model:
            # Fallback to string similarity
            return fuzz.ratio(text1, text2) / 100.0
        
        try:
            # Use sentence transformer for semantic similarity
            embeddings = self.sentence_model.encode([text1, text2])
            similarity = np.dot(embeddings[0], embeddings[1]) / (
                np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
            )
            return max(0, min(similarity, 1.0))
        except Exception as e:
            logger.warning(f"Semantic similarity calculation failed: {e}")
            return fuzz.ratio(text1, text2) / 100.0
    
    def extract_sentences_for_paraphrasing(
        self, 
        analysis: DocumentAnalysis, 
        min_priority: float = 0.5,
        max_sentences: Optional[int] = None
    ) -> List[Tuple[int, str]]:
        """
        Extract sentences that should be prioritized for paraphrasing.
        
        Args:
            analysis: Document analysis results
            min_priority: Minimum priority threshold
            max_sentences: Maximum number of sentences to return
            
        Returns:
            List of (index, sentence) tuples
        """
        candidates = []
        
        for i, sentence_info in enumerate(analysis.sentences):
            if sentence_info.priority_for_paraphrasing >= min_priority:
                candidates.append((i, sentence_info.text))
        
        # Sort by priority
        candidates.sort(key=lambda x: analysis.sentences[x[0]].priority_for_paraphrasing, reverse=True)
        
        if max_sentences:
            candidates = candidates[:max_sentences]
        
        return candidates
    
    def preserve_academic_terms(self, text: str) -> Tuple[str, Dict[str, str]]:
        """
        Identify and mark academic terms for preservation during paraphrasing.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (processed_text, replacement_map)
        """
        replacement_map = {}
        processed_text = text
        
        # Find all academic terms
        all_terms = self.academic_terms_id.union(self.academic_terms_en)
        
        for i, term in enumerate(all_terms):
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.finditer(pattern, processed_text, re.IGNORECASE)
            
            for match in matches:
                original_term = match.group()
                placeholder = f"__ACADEMIC_TERM_{i}__"
                replacement_map[placeholder] = original_term
                processed_text = processed_text.replace(original_term, placeholder, 1)
        
        return processed_text, replacement_map
    
    def restore_academic_terms(self, text: str, replacement_map: Dict[str, str]) -> str:
        """Restore academic terms after paraphrasing."""
        restored_text = text
        
        for placeholder, original_term in replacement_map.items():
            restored_text = restored_text.replace(placeholder, original_term)
        
        return restored_text


# Global instance
indonesian_nlp_pipeline = IndonesianNLPPipeline()
