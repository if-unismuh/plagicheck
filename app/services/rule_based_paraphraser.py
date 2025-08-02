"""
Hybrid Rule-Based Paraphraser Service
Comprehensive rule-based paraphrasing with lexical substitution and syntactic transformation.
"""
import re
import logging
import random
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import asyncio

import spacy
from spacy.tokens import Doc, Token
import nltk
from nltk.corpus import wordnet
from textstat import flesch_reading_ease, flesch_kincaid_grade
import language_tool_python
from fuzzywuzzy import fuzz

logger = logging.getLogger(__name__)


@dataclass
class TransformationRule:
    """Rule for syntactic transformation."""
    name: str
    pattern: str
    replacement: str
    conditions: List[str]
    weight: float
    preserve_meaning: bool


@dataclass
class SynonymRule:
    """Rule for lexical substitution."""
    word: str
    synonyms: List[str]
    pos_tags: List[str]
    context_words: List[str]
    confidence: float


@dataclass
class ParaphraseQuality:
    """Quality assessment of paraphrased text."""
    similarity_score: float
    grammar_score: float
    readability_score: float
    academic_tone_score: float
    overall_score: float
    issues: List[str]


class IndonesianSynonymDictionary:
    """Indonesian synonym dictionary with context awareness."""
    
    def __init__(self):
        # Indonesian synonyms organized by part of speech
        self.synonyms = {
            'noun': {
                'penelitian': ['riset', 'kajian', 'studi', 'investigasi'],
                'metode': ['cara', 'teknik', 'prosedur', 'pendekatan'],
                'data': ['informasi', 'fakta', 'keterangan', 'bahan'],
                'analisis': ['kajian', 'penelaahan', 'pembahasan', 'telaah'],
                'hasil': ['outcome', 'temuan', 'produk', 'konsekuensi'],
                'sampel': ['contoh', 'specimen', 'model', 'representasi'],
                'populasi': ['kelompok', 'komunitas', 'kumpulan', 'massa'],
                'variabel': ['faktor', 'unsur', 'elemen', 'komponen'],
                'hipotesis': ['dugaan', 'asumsi', 'prediksi', 'perkiraan'],
                'teori': ['konsep', 'gagasan', 'pemikiran', 'doktrin'],
                'konsep': ['ide', 'gagasan', 'pemahaman', 'pengertian'],
                'framework': ['kerangka', 'struktur', 'rangka', 'bingkai'],
            },
            'verb': {
                'menganalisis': ['mengkaji', 'menelaah', 'meneliti', 'mempelajari'],
                'menunjukkan': ['memperlihatkan', 'menampilkan', 'menyajikan', 'mengindikasikan'],
                'menjelaskan': ['menerangkan', 'menguraikan', 'menjabarkan', 'memaparka'],
                'mengidentifikasi': ['menemukan', 'mengenali', 'menentukan', 'menetapkan'],
                'menggunakan': ['memakai', 'memanfaatkan', 'menerapkan', 'mengaplikasikan'],
                'mengembangkan': ['membangun', 'menciptakan', 'membuat', 'menyusun'],
                'membuktikan': ['mendemonstrasikan', 'menyatakan', 'menegaskan', 'memvalidasi'],
                'menghasilkan': ['memproduksi', 'menghasilkan', 'menimbulkan', 'menciptakan'],
            },
            'adjective': {
                'penting': ['krusial', 'vital', 'esensial', 'fundamental'],
                'signifikan': ['bermakna', 'berarti', 'substansial', 'material'],
                'efektif': ['berhasil guna', 'manjur', 'ampuh', 'berdaya guna'],
                'relevan': ['berkaitan', 'sesuai', 'tepat', 'cocok'],
                'komprehensif': ['menyeluruh', 'lengkap', 'detail', 'mendalam'],
                'sistematis': ['teratur', 'metodis', 'berurutan', 'tertib'],
                'objektif': ['netral', 'tidak bias', 'adil', 'imparsial'],
                'valid': ['sah', 'absah', 'legitimate', 'dapat diterima'],
            },
            'adverb': {
                'secara': ['dengan cara', 'melalui', 'lewat', 'via'],
                'sangat': ['amat', 'betul-betul', 'benar-benar', 'sungguh'],
                'lebih': ['bertambah', 'semakin', 'makin', 'kian'],
                'hanya': ['cuma', 'semata', 'sekadar', 'melulu'],
                'juga': ['pula', 'pun', 'serta', 'dan'],
            }
        }
        
        # Academic connectors and transitions
        self.academic_connectors = {
            'sebagai akibat': ['akibatnya', 'konsekuensinya', 'hasilnya'],
            'oleh karena itu': ['maka dari itu', 'dengan demikian', 'karena hal tersebut'],
            'di sisi lain': ['sebaliknya', 'namun demikian', 'akan tetapi'],
            'selain itu': ['di samping itu', 'lebih dari itu', 'terlebih lagi'],
            'dengan kata lain': ['artinya', 'maksudnya', 'yaitu'],
            'sebagai contoh': ['misalnya', 'contohnya', 'seperti'],
        }
    
    def get_synonyms(self, word: str, pos: str, context: List[str] = None) -> List[str]:
        """Get synonyms for a word based on POS and context."""
        word_lower = word.lower()
        
        # Check direct synonyms
        for pos_category, words in self.synonyms.items():
            if word_lower in words:
                return words[word_lower]
        
        # Check academic connectors
        for connector, alternatives in self.academic_connectors.items():
            if word_lower in connector:
                return alternatives
        
        return []
    
    def is_academic_term(self, word: str) -> bool:
        """Check if word is an academic term that should be preserved."""
        academic_terms = {
            'penelitian', 'riset', 'analisis', 'metode', 'metodologi',
            'hipotesis', 'teori', 'data', 'sampel', 'populasi', 'variabel',
            'research', 'analysis', 'method', 'methodology', 'hypothesis',
            'theory', 'data', 'sample', 'population', 'variable'
        }
        return word.lower() in academic_terms


class SyntacticTransformer:
    """Handles syntactic transformations while preserving meaning."""
    
    def __init__(self):
        # Indonesian transformation patterns
        self.transformation_rules = [
            TransformationRule(
                name="active_to_passive",
                pattern=r"(\w+)\s+(me\w+)\s+(.+)",
                replacement=r"\3 di\2 oleh \1",
                conditions=["has_object", "is_transitive"],
                weight=0.8,
                preserve_meaning=True
            ),
            TransformationRule(
                name="passive_to_active", 
                pattern=r"(.+)\s+di(\w+)\s+oleh\s+(\w+)",
                replacement=r"\3 me\2 \1",
                conditions=["is_passive"],
                weight=0.7,
                preserve_meaning=True
            ),
            TransformationRule(
                name="clause_reordering",
                pattern=r"(.+),\s*(karena|sebab|oleh karena)\s+(.+)",
                replacement=r"\2 \3, \1",
                conditions=["has_causal_clause"],
                weight=0.6,
                preserve_meaning=True
            ),
            TransformationRule(
                name="sentence_combination",
                pattern=r"(.+)\.\s*(.+yang.+)",
                replacement=r"\1 \2",
                conditions=["consecutive_related"],
                weight=0.5,
                preserve_meaning=True
            )
        ]
    
    def transform_sentence(self, sentence: str, nlp_doc: Doc = None) -> List[str]:
        """Apply syntactic transformations to generate variants."""
        variants = []
        
        for rule in self.transformation_rules:
            if self._check_conditions(sentence, rule.conditions, nlp_doc):
                try:
                    transformed = re.sub(rule.pattern, rule.replacement, sentence)
                    if transformed != sentence and self._is_grammatical(transformed):
                        variants.append(transformed)
                except Exception as e:
                    logger.debug(f"Transformation rule {rule.name} failed: {e}")
        
        return variants
    
    def _check_conditions(self, sentence: str, conditions: List[str], nlp_doc: Doc = None) -> bool:
        """Check if transformation conditions are met."""
        for condition in conditions:
            if condition == "has_object":
                # Simple heuristic: check for common object patterns
                if not re.search(r'(me\w+)\s+\w+', sentence):
                    return False
            elif condition == "is_transitive":
                # Check for transitive verb patterns
                if not re.search(r'me\w+', sentence):
                    return False
            elif condition == "is_passive":
                if not re.search(r'di\w+', sentence):
                    return False
            elif condition == "has_causal_clause":
                if not re.search(r'(karena|sebab|oleh karena)', sentence):
                    return False
            elif condition == "consecutive_related":
                # Simple check for related consecutive sentences
                if not re.search(r'yang', sentence):
                    return False
        
        return True
    
    def _is_grammatical(self, sentence: str) -> bool:
        """Basic grammaticality check."""
        # Simple heuristics for Indonesian grammar
        words = sentence.split()
        
        # Check minimum length
        if len(words) < 3:
            return False
        
        # Check for basic sentence structure
        has_verb = any(word.startswith(('me', 'di', 'ter', 'ber')) for word in words)
        
        return has_verb
    
    def reorder_clauses(self, sentence: str) -> List[str]:
        """Reorder clauses while maintaining meaning."""
        variants = []
        
        # Split by common conjunctions
        conjunctions = ['karena', 'sebab', 'sehingga', 'namun', 'tetapi', 'dan', 'atau']
        
        for conj in conjunctions:
            if conj in sentence:
                parts = sentence.split(conj, 1)
                if len(parts) == 2:
                    # Reorder: second part + conjunction + first part
                    reordered = f"{parts[1].strip()} {conj} {parts[0].strip()}"
                    if reordered != sentence:
                        variants.append(reordered)
        
        return variants


class GrammarChecker:
    """Grammar checking and correction for Indonesian."""
    
    def __init__(self):
        self.tool = None
        self._initialize_grammar_tool()
    
    def _initialize_grammar_tool(self):
        """Initialize grammar checking tool."""
        try:
            # Try to use LanguageTool for Indonesian
            self.tool = language_tool_python.LanguageTool('id')
            logger.info("Indonesian grammar checker initialized")
        except Exception as e:
            logger.warning(f"Failed to initialize Indonesian grammar checker: {e}")
            try:
                # Fallback to English
                self.tool = language_tool_python.LanguageTool('en')
                logger.info("English grammar checker initialized as fallback")
            except Exception as e2:
                logger.warning(f"Failed to initialize any grammar checker: {e2}")
                self.tool = None
    
    def check_grammar(self, text: str) -> Tuple[float, List[str]]:
        """Check grammar and return score and issues."""
        if not self.tool:
            return 0.8, []  # Default acceptable score
        
        try:
            matches = self.tool.check(text)
            
            # Calculate grammar score
            words = len(text.split())
            error_count = len(matches)
            grammar_score = max(0, 1.0 - (error_count / max(words * 0.1, 1)))
            
            # Extract issues
            issues = [match.message for match in matches[:5]]  # Top 5 issues
            
            return grammar_score, issues
            
        except Exception as e:
            logger.warning(f"Grammar check failed: {e}")
            return 0.8, []
    
    def correct_text(self, text: str) -> str:
        """Attempt to correct grammar issues."""
        if not self.tool:
            return text
        
        try:
            corrected = self.tool.correct(text)
            return corrected
        except Exception as e:
            logger.warning(f"Grammar correction failed: {e}")
            return text


class HybridRuleBasedParaphraser:
    """Comprehensive rule-based paraphrasing system."""
    
    def __init__(self):
        self.synonym_dict = IndonesianSynonymDictionary()
        self.transformer = SyntacticTransformer()
        self.grammar_checker = GrammarChecker()
        self.nlp = None
        
        # Quality thresholds
        self.min_similarity = 0.3
        self.max_similarity = 0.9
        self.min_grammar_score = 0.6
        self.min_readability_score = 0.4
        self.min_academic_tone = 0.5
        
        # Initialize NLP models
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP models."""
        try:
            import spacy
            self.nlp = spacy.load("en_core_web_sm")  # Fallback to English
            logger.info("SpaCy model loaded for rule-based paraphrasing")
        except Exception as e:
            logger.warning(f"Failed to load SpaCy model: {e}")
    
    async def paraphrase(
        self, 
        text: str, 
        preserve_academic_terms: bool = True,
        preserve_citations: bool = True,
        num_variants: int = 3
    ) -> List[Tuple[str, ParaphraseQuality]]:
        """
        Generate paraphrases using rule-based methods.
        
        Args:
            text: Text to paraphrase
            preserve_academic_terms: Whether to preserve academic terminology
            preserve_citations: Whether to preserve citation formats
            num_variants: Number of variants to generate
            
        Returns:
            List of (paraphrased_text, quality) tuples
        """
        # Preprocess text
        protected_text, protection_map = self._protect_elements(
            text, preserve_academic_terms, preserve_citations
        )
        
        # Generate variants using different strategies
        variants = []
        
        # Strategy 1: Lexical substitution
        lexical_variants = await self._lexical_substitution(protected_text)
        variants.extend(lexical_variants)
        
        # Strategy 2: Syntactic transformation
        syntactic_variants = await self._syntactic_transformation(protected_text)
        variants.extend(syntactic_variants)
        
        # Strategy 3: Hybrid approach
        hybrid_variants = await self._hybrid_transformation(protected_text)
        variants.extend(hybrid_variants)
        
        # Remove duplicates
        unique_variants = list(set(variants))
        
        # Restore protected elements
        restored_variants = []
        for variant in unique_variants:
            restored = self._restore_protected_elements(variant, protection_map)
            restored_variants.append(restored)
        
        # Assess quality
        quality_variants = []
        for variant in restored_variants:
            quality = await self._assess_quality(text, variant)
            if quality.overall_score >= 0.5:  # Quality threshold
                quality_variants.append((variant, quality))
        
        # Sort by quality and return top variants
        quality_variants.sort(key=lambda x: x[1].overall_score, reverse=True)
        
        return quality_variants[:num_variants]
    
    def _protect_elements(
        self, 
        text: str, 
        preserve_academic: bool, 
        preserve_citations: bool
    ) -> Tuple[str, Dict[str, str]]:
        """Protect academic terms and citations from modification."""
        protection_map = {}
        protected_text = text
        placeholder_counter = 0
        
        if preserve_academic:
            # Protect academic terms
            for pos_category, words in self.synonym_dict.synonyms.items():
                for term in words.keys():
                    if term in protected_text.lower():
                        placeholder = f"__ACADEMIC_TERM_{placeholder_counter}__"
                        protection_map[placeholder] = term
                        protected_text = re.sub(
                            r'\b' + re.escape(term) + r'\b',
                            placeholder,
                            protected_text,
                            flags=re.IGNORECASE
                        )
                        placeholder_counter += 1
        
        if preserve_citations:
            # Protect citations
            citation_patterns = [
                r'\([^)]*\d{4}[^)]*\)',  # (Author, 2023)
                r'\[[^\]]*\d{4}[^\]]*\]',  # [Author, 2023]
                r'(?:et al\.|dkk\.)',  # et al. or dkk.
            ]
            
            for pattern in citation_patterns:
                matches = re.finditer(pattern, protected_text)
                for match in matches:
                    placeholder = f"__CITATION_{placeholder_counter}__"
                    protection_map[placeholder] = match.group()
                    protected_text = protected_text.replace(match.group(), placeholder, 1)
                    placeholder_counter += 1
        
        return protected_text, protection_map
    
    def _restore_protected_elements(self, text: str, protection_map: Dict[str, str]) -> str:
        """Restore protected elements after paraphrasing."""
        restored_text = text
        
        for placeholder, original in protection_map.items():
            restored_text = restored_text.replace(placeholder, original)
        
        return restored_text
    
    async def _lexical_substitution(self, text: str) -> List[str]:
        """Generate variants through lexical substitution."""
        variants = []
        
        # Analyze text with NLP if available
        doc = None
        if self.nlp:
            try:
                doc = self.nlp(text)
            except:
                pass
        
        words = text.split()
        
        # Generate substitution variants
        for i in range(min(3, len(words) // 3)):  # Multiple substitution attempts
            variant_words = words.copy()
            substitution_count = 0
            
            for j, word in enumerate(words):
                if substitution_count >= 3:  # Limit substitutions per variant
                    break
                
                # Get POS tag if available
                pos = 'noun'  # Default
                if doc and j < len(doc):
                    pos = doc[j].pos_.lower()
                
                # Get synonyms
                synonyms = self.synonym_dict.get_synonyms(word, pos)
                
                if synonyms and not self.synonym_dict.is_academic_term(word):
                    # Random selection for variety
                    synonym = random.choice(synonyms)
                    variant_words[j] = synonym
                    substitution_count += 1
            
            if substitution_count > 0:
                variant = ' '.join(variant_words)
                if variant != text:
                    variants.append(variant)
        
        return variants
    
    async def _syntactic_transformation(self, text: str) -> List[str]:
        """Generate variants through syntactic transformation."""
        variants = []
        
        # Analyze with NLP if available
        doc = None
        if self.nlp:
            try:
                doc = self.nlp(text)
            except:
                pass
        
        # Apply transformation rules
        transformed = self.transformer.transform_sentence(text, doc)
        variants.extend(transformed)
        
        # Apply clause reordering
        reordered = self.transformer.reorder_clauses(text)
        variants.extend(reordered)
        
        # Sentence structure modifications
        structure_variants = await self._modify_sentence_structure(text)
        variants.extend(structure_variants)
        
        return variants
    
    async def _modify_sentence_structure(self, text: str) -> List[str]:
        """Modify sentence structure while preserving meaning."""
        variants = []
        
        # Split complex sentences
        if len(text.split()) > 20:
            # Try to split at conjunctions
            for conj in [', dan ', ', serta ', ', namun ', ', tetapi ']:
                if conj in text:
                    parts = text.split(conj, 1)
                    if len(parts) == 2:
                        # Create two sentences
                        variant = f"{parts[0].strip()}. {parts[1].strip()}"
                        variants.append(variant)
        
        # Combine short sentences (if applicable)
        sentences = text.split('. ')
        if len(sentences) > 1:
            for i in range(len(sentences) - 1):
                if len(sentences[i].split()) < 8 and len(sentences[i+1].split()) < 8:
                    # Combine with conjunction
                    combined = f"{sentences[i]}, dan {sentences[i+1]}"
                    variants.append(combined)
        
        return variants
    
    async def _hybrid_transformation(self, text: str) -> List[str]:
        """Combine lexical and syntactic transformations."""
        variants = []
        
        # First apply lexical substitution
        lexical_variants = await self._lexical_substitution(text)
        
        # Then apply syntactic transformation to lexical variants
        for variant in lexical_variants[:2]:  # Limit to avoid explosion
            syntactic = await self._syntactic_transformation(variant)
            variants.extend(syntactic[:1])  # Take best syntactic variant
        
        return variants
    
    async def _assess_quality(self, original: str, paraphrased: str) -> ParaphraseQuality:
        """Comprehensive quality assessment."""
        # Similarity score
        similarity_score = fuzz.ratio(original, paraphrased) / 100.0
        
        # Grammar score
        grammar_score, grammar_issues = self.grammar_checker.check_grammar(paraphrased)
        
        # Readability score
        try:
            readability_score = max(0, flesch_reading_ease(paraphrased) / 100.0)
        except:
            readability_score = 0.5
        
        # Academic tone score
        academic_tone_score = self._assess_academic_tone(paraphrased)
        
        # Overall quality (weighted combination)
        overall_score = (
            similarity_score * 0.25 +
            grammar_score * 0.3 +
            readability_score * 0.2 +
            academic_tone_score * 0.25
        )
        
        # Collect issues
        issues = []
        if similarity_score < self.min_similarity:
            issues.append("Too different from original")
        elif similarity_score > self.max_similarity:
            issues.append("Too similar to original")
        
        if grammar_score < self.min_grammar_score:
            issues.extend(grammar_issues)
        
        if readability_score < self.min_readability_score:
            issues.append("Low readability")
        
        if academic_tone_score < self.min_academic_tone:
            issues.append("Non-academic tone")
        
        return ParaphraseQuality(
            similarity_score=similarity_score,
            grammar_score=grammar_score,
            readability_score=readability_score,
            academic_tone_score=academic_tone_score,
            overall_score=overall_score,
            issues=issues
        )
    
    def _assess_academic_tone(self, text: str) -> float:
        """Assess academic tone of the text."""
        academic_indicators = [
            'penelitian', 'analisis', 'metode', 'hasil', 'kesimpulan',
            'research', 'analysis', 'method', 'results', 'conclusion',
            'berdasarkan', 'menunjukkan', 'mengindikasikan', 'menjelaskan',
            'based on', 'shows', 'indicates', 'explains'
        ]
        
        words = text.lower().split()
        academic_count = sum(1 for word in words if word in academic_indicators)
        
        # Score based on academic term density
        academic_density = academic_count / len(words) if words else 0
        
        # Formal language indicators
        formal_patterns = [
            r'secara\s+\w+',  # secara sistematis, secara empiris, etc.
            r'berdasarkan\s+\w+',  # berdasarkan hasil, berdasarkan data, etc.
            r'dalam\s+hal\s+ini',  # formal connecting phrases
            r'sebagaimana\s+\w+',
        ]
        
        formal_count = sum(1 for pattern in formal_patterns if re.search(pattern, text))
        formal_score = min(formal_count / 3.0, 1.0)
        
        # Combine scores
        tone_score = (academic_density * 3 + formal_score) / 2
        return min(tone_score, 1.0)
    
    def update_synonym_dictionary(self, word: str, synonyms: List[str], pos: str = 'noun'):
        """Update the synonym dictionary with new entries."""
        if pos in self.synonym_dict.synonyms:
            self.synonym_dict.synonyms[pos][word.lower()] = synonyms
            logger.info(f"Updated synonyms for {word}: {synonyms}")
    
    def get_transformation_stats(self) -> Dict[str, Any]:
        """Get statistics about transformation operations."""
        return {
            "synonym_dictionary_size": sum(
                len(words) for words in self.synonym_dict.synonyms.values()
            ),
            "transformation_rules_count": len(self.transformer.transformation_rules),
            "grammar_checker_available": self.grammar_checker.tool is not None,
            "nlp_model_available": self.nlp is not None,
            "quality_thresholds": {
                "min_similarity": self.min_similarity,
                "max_similarity": self.max_similarity,
                "min_grammar_score": self.min_grammar_score,
                "min_readability_score": self.min_readability_score,
                "min_academic_tone": self.min_academic_tone
            }
        }


# Global instance
rule_based_paraphraser = HybridRuleBasedParaphraser()
