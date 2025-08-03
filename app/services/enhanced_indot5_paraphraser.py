"""
Enhanced IndoT5 Local Paraphraser Service
Advanced IndoT5 paraphrasing with GPU optimization, batch processing, and quality filtering.
"""
import os
import gc
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import time
from concurrent.futures import ThreadPoolExecutor

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoConfig
from transformers import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import psutil

logger = logging.getLogger(__name__)


@dataclass
class ParaphraseResult:
    """Result of paraphrasing operation."""
    original_text: str
    paraphrased_variants: List[str]
    similarity_scores: List[float]
    quality_scores: List[float]
    processing_time: float
    best_variant: str
    metadata: Dict[str, Any]


@dataclass
class ModelConfig:
    """Configuration for IndoT5 model."""
    model_name: str = "Wikidepia/IndoT5-base-paraphrase"
    max_length: int = 512
    num_beams: int = 5
    num_return_sequences: int = 3
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    repetition_penalty: float = 1.2
    do_sample: bool = True
    early_stopping: bool = True
    batch_size: int = 4
    use_gpu: bool = True
    precision: str = "fp16"  # fp16, fp32


class EnhancedIndoT5Paraphraser:
    """Enhanced IndoT5 paraphraser with advanced features."""
    
    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = None
        self.tokenizer = None
        self.similarity_model = None
        self.device = None
        self.model_loaded = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        
        # Performance tracking
        self.total_requests = 0
        self.total_processing_time = 0.0
        self.cache = {}
        self.max_cache_size = 1000
        
        # Quality thresholds
        self.min_similarity = 0.3  # Minimum similarity to original
        self.max_similarity = 0.9  # Maximum similarity (avoid too similar)
        self.min_quality_score = 0.5
        
        # Don't initialize models immediately - use lazy loading
    
    async def _initialize_models(self):
        """Initialize all models with optimizations."""
        logger.info("Initializing Enhanced IndoT5 Paraphraser...")
        
        try:
            # Setup device
            self._setup_device()
            
            # Load main model
            await self._load_indot5_model()
            
            # Load similarity model
            await self._load_similarity_model()
            
            # Optimize model for inference
            self._optimize_model()
            
            self.model_loaded = True
            logger.info("Enhanced IndoT5 Paraphraser initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Enhanced IndoT5 Paraphraser: {e}")
            raise
    
    def _setup_device(self):
        """Setup computation device with optimization."""
        if self.config.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            
            # GPU memory optimization
            torch.cuda.empty_cache()
            if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
                torch.cuda.set_per_process_memory_fraction(0.8)
            
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
        else:
            self.device = torch.device("cpu")
            logger.info("Using CPU for computation")
            
            # CPU optimization
            torch.set_num_threads(min(psutil.cpu_count(), 8))
    
    async def _load_indot5_model(self):
        """Load IndoT5 model with optimization."""
        logger.info(f"Loading IndoT5 model: {self.config.model_name}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model_name,
                local_files_only=False
            )
            
            # Load model configuration
            model_config = AutoConfig.from_pretrained(self.config.model_name)
            
            # Load model with precision settings
            if self.config.precision == "fp16" and self.device.type == "cuda":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    config=model_config,
                    torch_dtype=torch.float16
                )
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    self.config.model_name,
                    config=model_config
                )
            
            # Move to device
            self.model = self.model.to(self.device)
            
            # Set evaluation mode
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except OSError as e:
            if "No space left on device" in str(e):
                logger.error("Insufficient disk space to load IndoT5 model. Please free up space or use a different paraphrasing method.")
                raise RuntimeError("Insufficient disk space to load IndoT5 model. Please free up space or use a different paraphrasing method.")
            logger.error(f"Failed to load IndoT5 model: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to load IndoT5 model: {e}")
            raise
    
    async def _load_similarity_model(self):
        """Load sentence similarity model."""
        try:
            self.similarity_model = SentenceTransformer(
                'paraphrase-multilingual-MiniLM-L12-v2',
                device=self.device.type
            )
            logger.info("Similarity model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load similarity model: {e}")
            self.similarity_model = None
    
    def _optimize_model(self):
        """Apply optimization techniques to the model."""
        if self.device.type == "cuda":
            # Enable mixed precision if available
            try:
                self.model = self.model.half()
                logger.info("Enabled FP16 optimization")
            except:
                logger.warning("FP16 optimization not available")
        
        # Compile model for faster inference (PyTorch 2.0+)
        try:
            if hasattr(torch, 'compile'):
                self.model = torch.compile(self.model)
                logger.info("Model compiled for optimization")
        except:
            logger.warning("Model compilation not available")
    
    async def paraphrase_single(
        self, 
        text: str, 
        num_variants: Optional[int] = None
    ) -> ParaphraseResult:
        """
        Paraphrase a single text with multiple variants.
        
        Args:
            text: Text to paraphrase
            num_variants: Number of variants to generate
            
        Returns:
            ParaphraseResult: Paraphrasing results
        """
        if not self.model_loaded:
            await self._initialize_models()
        
        start_time = time.time()
        num_variants = num_variants or self.config.num_return_sequences
        
        # Check cache
        cache_key = f"{text}_{num_variants}"
        if cache_key in self.cache:
            logger.debug("Returning cached result")
            return self.cache[cache_key]
        
        try:
            # Generate variants
            variants = await self._generate_variants(text, num_variants)
            
            # Calculate quality scores
            similarity_scores = self._calculate_similarity_scores(text, variants)
            quality_scores = await self._calculate_quality_scores(text, variants)
            
            # Filter and rank variants
            filtered_variants = self._filter_variants(
                variants, similarity_scores, quality_scores
            )
            
            # Select best variant
            best_variant = self._select_best_variant(
                text, filtered_variants, similarity_scores, quality_scores
            )
            
            processing_time = time.time() - start_time
            
            # Create result
            result = ParaphraseResult(
                original_text=text,
                paraphrased_variants=filtered_variants,
                similarity_scores=similarity_scores,
                quality_scores=quality_scores,
                processing_time=processing_time,
                best_variant=best_variant,
                metadata={
                    "num_requested": num_variants,
                    "num_generated": len(variants),
                    "num_filtered": len(filtered_variants),
                    "device": str(self.device),
                    "model_config": self.config.__dict__
                }
            )
            
            # Cache result
            self._cache_result(cache_key, result)
            
            # Update statistics
            self.total_requests += 1
            self.total_processing_time += processing_time
            
            return result
            
        except Exception as e:
            logger.error(f"Single paraphrasing failed: {e}")
            raise
    
    async def paraphrase_batch(
        self, 
        texts: List[str], 
        num_variants: Optional[int] = None,
        show_progress: bool = True
    ) -> List[ParaphraseResult]:
        """
        Paraphrase multiple texts in batch for efficiency.
        
        Args:
            texts: List of texts to paraphrase
            num_variants: Number of variants per text
            show_progress: Whether to show progress bar
            
        Returns:
            List of ParaphraseResult objects
        """
        if not self.model_loaded:
            await self._initialize_models()
        
        results = []
        batch_size = self.config.batch_size
        
        # Process in batches
        batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
        
        progress_bar = tqdm(batches, desc="Paraphrasing batches") if show_progress else batches
        
        for batch in progress_bar:
            try:
                batch_results = await self._process_batch(batch, num_variants)
                results.extend(batch_results)
                
                # Clear GPU cache periodically
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Add error results for failed batch
                for text in batch:
                    error_result = ParaphraseResult(
                        original_text=text,
                        paraphrased_variants=[text],  # Return original as fallback
                        similarity_scores=[1.0],
                        quality_scores=[0.0],
                        processing_time=0.0,
                        best_variant=text,
                        metadata={"error": str(e)}
                    )
                    results.append(error_result)
        
        return results
    
    async def _generate_variants(self, text: str, num_variants: int) -> List[str]:
        """Generate paraphrase variants using the model."""
        # Prepare input
        input_text = f"paraphrase: {text}"
        
        try:
            # Tokenize
            inputs = self.tokenizer.encode(
                input_text,
                return_tensors="pt",
                max_length=self.config.max_length,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate with multiple variants
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=self.config.max_length,
                    num_beams=self.config.num_beams,
                    num_return_sequences=min(num_variants, self.config.num_beams),
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k,
                    repetition_penalty=self.config.repetition_penalty,
                    do_sample=self.config.do_sample,
                    early_stopping=self.config.early_stopping,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode outputs
            variants = []
            for output in outputs:
                decoded = self.tokenizer.decode(output, skip_special_tokens=True)
                if decoded and decoded != text:  # Avoid identical outputs
                    variants.append(decoded)
            
            # Remove duplicates while preserving order
            unique_variants = []
            seen = set()
            for variant in variants:
                if variant not in seen:
                    unique_variants.append(variant)
                    seen.add(variant)
            
            return unique_variants
            
        except Exception as e:
            logger.error(f"Variant generation failed: {e}")
            return [text]  # Return original as fallback
    
    def _calculate_similarity_scores(self, original: str, variants: List[str]) -> List[float]:
        """Calculate similarity scores between original and variants."""
        if not self.similarity_model:
            # Fallback to simple token-based similarity
            return [self._token_similarity(original, variant) for variant in variants]
        
        try:
            # Encode texts
            texts = [original] + variants
            embeddings = self.similarity_model.encode(texts)
            
            # Calculate cosine similarities
            original_embedding = embeddings[0]
            similarities = []
            
            for i in range(1, len(embeddings)):
                similarity = np.dot(original_embedding, embeddings[i]) / (
                    np.linalg.norm(original_embedding) * np.linalg.norm(embeddings[i])
                )
                similarities.append(max(0, min(similarity, 1.0)))
            
            return similarities
            
        except Exception as e:
            logger.warning(f"Similarity calculation failed: {e}")
            return [self._token_similarity(original, variant) for variant in variants]
    
    def _token_similarity(self, text1: str, text2: str) -> float:
        """Simple token-based similarity calculation."""
        tokens1 = set(text1.lower().split())
        tokens2 = set(text2.lower().split())
        
        if not tokens1 and not tokens2:
            return 1.0
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1.intersection(tokens2)
        union = tokens1.union(tokens2)
        
        return len(intersection) / len(union) if union else 0.0
    
    async def _calculate_quality_scores(self, original: str, variants: List[str]) -> List[float]:
        """Calculate quality scores for variants."""
        scores = []
        
        for variant in variants:
            # Basic quality metrics
            length_ratio = len(variant.split()) / max(len(original.split()), 1)
            length_score = 1.0 - abs(1.0 - length_ratio) * 0.5  # Prefer similar lengths
            
            # Grammar and fluency (simplified)
            grammar_score = self._assess_grammar_quality(variant)
            
            # Diversity score (how different from original)
            diversity_score = 1.0 - self._token_similarity(original, variant)
            diversity_score = min(diversity_score * 2, 1.0)  # Boost diversity
            
            # Combined quality score
            quality = (length_score * 0.3 + grammar_score * 0.4 + diversity_score * 0.3)
            scores.append(max(0, min(quality, 1.0)))
        
        return scores
    
    def _assess_grammar_quality(self, text: str) -> float:
        """Simple grammar quality assessment."""
        # Basic heuristics for grammar quality
        words = text.split()
        
        if len(words) < 2:
            return 0.3
        
        # Check for basic sentence structure
        has_capital_start = text[0].isupper() if text else False
        has_punctuation_end = text[-1] in '.!?' if text else False
        
        # Check for repeated words
        word_set = set(words)
        repetition_ratio = len(words) / len(word_set) if word_set else 1.0
        
        # Combine factors
        structure_score = 0.5 + (0.25 if has_capital_start else 0) + (0.25 if has_punctuation_end else 0)
        repetition_score = max(0, 2.0 - repetition_ratio) / 2.0
        
        return (structure_score + repetition_score) / 2.0
    
    def _filter_variants(
        self, 
        variants: List[str], 
        similarities: List[float], 
        qualities: List[float]
    ) -> List[str]:
        """Filter variants based on quality and similarity thresholds."""
        filtered = []
        
        for i, variant in enumerate(variants):
            similarity = similarities[i] if i < len(similarities) else 0.0
            quality = qualities[i] if i < len(qualities) else 0.0
            
            # Apply filters
            if (self.min_similarity <= similarity <= self.max_similarity and 
                quality >= self.min_quality_score):
                filtered.append(variant)
        
        return filtered
    
    def _select_best_variant(
        self, 
        original: str, 
        variants: List[str], 
        similarities: List[float], 
        qualities: List[float]
    ) -> str:
        """Select the best variant based on combined scoring."""
        if not variants:
            return original
        
        best_variant = variants[0]
        best_score = 0.0
        
        for i, variant in enumerate(variants):
            similarity = similarities[i] if i < len(similarities) else 0.0
            quality = qualities[i] if i < len(qualities) else 0.0
            
            # Combined score favoring balanced similarity and quality
            combined_score = (similarity * 0.4 + quality * 0.6)
            
            if combined_score > best_score:
                best_score = combined_score
                best_variant = variant
        
        return best_variant
    
    async def _process_batch(self, batch: List[str], num_variants: Optional[int]) -> List[ParaphraseResult]:
        """Process a batch of texts efficiently."""
        # For now, process sequentially with optimizations
        # Future enhancement: true batch processing
        results = []
        
        for text in batch:
            try:
                result = await self.paraphrase_single(text, num_variants)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process text in batch: {e}")
                # Add error result
                error_result = ParaphraseResult(
                    original_text=text,
                    paraphrased_variants=[text],
                    similarity_scores=[1.0],
                    quality_scores=[0.0],
                    processing_time=0.0,
                    best_variant=text,
                    metadata={"error": str(e)}
                )
                results.append(error_result)
        
        return results
    
    def _cache_result(self, key: str, result: ParaphraseResult):
        """Cache result with size management."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entries
            keys_to_remove = list(self.cache.keys())[:self.max_cache_size // 4]
            for k in keys_to_remove:
                del self.cache[k]
        
        self.cache[key] = result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        avg_time = (self.total_processing_time / self.total_requests 
                   if self.total_requests > 0 else 0.0)
        
        memory_info = {}
        if self.device.type == "cuda":
            memory_info = {
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
                "gpu_memory_cached": torch.cuda.memory_reserved() / 1e9,
                "gpu_max_memory": torch.cuda.max_memory_allocated() / 1e9
            }
        else:
            memory_info = {
                "cpu_memory_percent": psutil.virtual_memory().percent
            }
        
        return {
            "total_requests": self.total_requests,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": avg_time,
            "cache_size": len(self.cache),
            "model_loaded": self.model_loaded,
            "device": str(self.device),
            **memory_info
        }
    
    def clear_cache(self):
        """Clear the result cache."""
        self.cache.clear()
        logger.info("Result cache cleared")
    
    def update_config(self, **kwargs):
        """Update model configuration."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated config: {key} = {value}")
    
    def cleanup(self):
        """Cleanup resources."""
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
        
        self.clear_cache()
        
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
        
        # Force garbage collection
        gc.collect()
        
        logger.info("Enhanced IndoT5 Paraphraser cleaned up")


# Global instance with lazy initialization
enhanced_indot5_paraphraser = None

def get_enhanced_indot5_paraphraser():
    """Get or create the enhanced IndoT5 paraphraser instance."""
    global enhanced_indot5_paraphraser
    if enhanced_indot5_paraphraser is None:
        enhanced_indot5_paraphraser = EnhancedIndoT5Paraphraser()
    return enhanced_indot5_paraphraser
