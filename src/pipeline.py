"""
pipeline.py - Main Pipeline Orchestrator

This module orchestrates the full NLP pipeline:
1. Preprocess reviews (clean, split sentences, mark negations)
2. Detect aspects per sentence
3. Analyze sentiment per aspect
4. Aggregate to listing level with temporal weighting
5. Compute confidence scores

The pipeline is designed to be:
- Modular: Each step can be run independently
- Explainable: Intermediate results preserved for debugging
- Efficient: Batch processing where possible
"""

import uuid
from datetime import datetime
from typing import List, Dict, Optional
from collections import defaultdict

from src.schemas import (
    RawReview, ProcessedSentence, AspectMatch,
    ListingIntelligence, Aspect
)
from src.preprocessing import preprocess_review
from src.aspect_detection import detect_aspects_in_sentence
from src.sentiment_analysis import analyze_aspects_sentiments
from src.aggregation import (
    aggregate_by_listing, group_sentences_by_listing,
    compute_weights_for_reviews, collect_aspect_data_points
)
from src.confidence import enhance_listing_intelligence


# =============================================================================
# PIPELINE STAGES
# =============================================================================

def stage_preprocess(reviews: List[RawReview]) -> Dict[str, List[str]]:
    """
    Stage 1: Preprocess all reviews.
    
    Returns:
        Dict mapping review_id to list of processed sentences
    """
    results = {}
    for review in reviews:
        preprocessed = preprocess_review(review.review_text)
        results[review.review_id] = preprocessed.sentences
    return results


def stage_detect_and_analyze(
    sentences_by_review: Dict[str, List[str]]
) -> List[ProcessedSentence]:
    """
    Stage 2 & 3: Detect aspects and analyze sentiment for all sentences.
    
    Returns:
        List of ProcessedSentence objects with detected aspects and sentiment
    """
    all_sentences = []
    sentence_counter = 0
    
    for review_id, sentences in sentences_by_review.items():
        for sentence in sentences:
            sentence_counter += 1
            sentence_id = f"sent_{sentence_counter:06d}"
            
            # Detect aspects in this sentence
            aspect_matches = detect_aspects_in_sentence(sentence)
            
            # Analyze sentiment for each detected aspect
            if aspect_matches:
                aspect_matches = analyze_aspects_sentiments(sentence, aspect_matches)
            
            processed = ProcessedSentence(
                sentence_id=sentence_id,
                review_id=review_id,
                original_text=sentence,  # Already processed with _NEG markers
                processed_text=sentence,
                detected_aspects=aspect_matches
            )
            all_sentences.append(processed)
    
    return all_sentences


def stage_aggregate(
    sentences: List[ProcessedSentence],
    reviews: List[RawReview],
    reference_date: Optional[datetime] = None
) -> Dict[str, ListingIntelligence]:
    """
    Stage 4 & 5: Aggregate to listing level with confidence scoring.
    
    Returns:
        Dict mapping listing_id to ListingIntelligence
    """
    # Group by listing
    grouped = group_sentences_by_listing(sentences, reviews)
    
    results = {}
    for listing_id, (listing_sentences, listing_reviews) in grouped.items():
        # Basic aggregation
        intelligence = aggregate_by_listing(
            sentences=listing_sentences,
            reviews=listing_reviews,
            listing_id=listing_id,
            reference_date=reference_date
        )
        
        # Collect scores per aspect for confidence computation
        review_weights = compute_weights_for_reviews(
            listing_reviews,
            reference_date or datetime.now()
        )
        aspect_data = collect_aspect_data_points(
            listing_sentences, listing_reviews, review_weights
        )
        
        aspect_scores: Dict[Aspect, List[float]] = {}
        for aspect, data_points in aspect_data.items():
            aspect_scores[aspect] = [dp.sentiment_score for dp in data_points]
        
        # Enhance with confidence scores
        enhanced = enhance_listing_intelligence(intelligence, aspect_scores)
        results[listing_id] = enhanced
    
    return results


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pipeline(
    reviews: List[RawReview],
    reference_date: Optional[datetime] = None,
    verbose: bool = False
) -> Dict[str, ListingIntelligence]:
    """
    Run the complete Guest Review Intelligence pipeline.
    
    Pipeline stages:
    1. Preprocess: Clean text, split sentences, mark negations
    2. Detect: Find aspects in each sentence using keyword lexicons
    3. Analyze: Compute sentiment per aspect using lexicon + context
    4. Aggregate: Roll up to listing level with temporal weighting
    5. Confidence: Compute variance, disagreement, and confidence scores
    
    Args:
        reviews: List of RawReview objects to analyze
        reference_date: Reference date for temporal weighting (default: now)
        verbose: If True, print progress information
        
    Returns:
        Dict mapping listing_id to ListingIntelligence
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    if verbose:
        print(f"Processing {len(reviews)} reviews...")
    
    # Stage 1: Preprocess
    if verbose:
        print("Stage 1: Preprocessing reviews...")
    sentences_by_review = stage_preprocess(reviews)
    
    total_sentences = sum(len(s) for s in sentences_by_review.values())
    if verbose:
        print(f"  - Extracted {total_sentences} sentences")
    
    # Stage 2 & 3: Detect aspects and analyze sentiment
    if verbose:
        print("Stage 2-3: Detecting aspects and analyzing sentiment...")
    processed_sentences = stage_detect_and_analyze(sentences_by_review)
    
    aspects_found = sum(
        len(s.detected_aspects) for s in processed_sentences
    )
    if verbose:
        print(f"  - Found {aspects_found} aspect mentions")
    
    # Stage 4 & 5: Aggregate with confidence
    if verbose:
        print("Stage 4-5: Aggregating and computing confidence...")
    results = stage_aggregate(processed_sentences, reviews, reference_date)
    
    if verbose:
        print(f"  - Generated intelligence for {len(results)} listings")
        print("Pipeline complete!")
    
    return results


def run_pipeline_with_details(
    reviews: List[RawReview],
    reference_date: Optional[datetime] = None
) -> Dict:
    """
    Run pipeline and return detailed intermediate results for debugging.
    
    Returns:
        Dict with:
        - listings: Dict[listing_id, ListingIntelligence]
        - sentences: List[ProcessedSentence]
        - stats: Pipeline statistics
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    # Stage 1
    sentences_by_review = stage_preprocess(reviews)
    
    # Stage 2 & 3
    processed_sentences = stage_detect_and_analyze(sentences_by_review)
    
    # Stage 4 & 5
    results = stage_aggregate(processed_sentences, reviews, reference_date)
    
    # Compute statistics
    total_sentences = len(processed_sentences)
    sentences_with_aspects = sum(
        1 for s in processed_sentences if s.detected_aspects
    )
    total_aspect_mentions = sum(
        len(s.detected_aspects) for s in processed_sentences
    )
    
    # Count by aspect
    aspect_counts: Dict[str, int] = defaultdict(int)
    for sent in processed_sentences:
        for am in sent.detected_aspects:
            aspect_counts[am.aspect.value] += 1
    
    stats = {
        'total_reviews': len(reviews),
        'total_sentences': total_sentences,
        'sentences_with_aspects': sentences_with_aspects,
        'total_aspect_mentions': total_aspect_mentions,
        'aspect_coverage_pct': round(100 * sentences_with_aspects / max(1, total_sentences), 1),
        'aspect_mention_counts': dict(aspect_counts),
        'listings_analyzed': len(results)
    }
    
    return {
        'listings': results,
        'sentences': processed_sentences,
        'stats': stats
    }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_single_review(
    review: RawReview,
    reference_date: Optional[datetime] = None
) -> List[ProcessedSentence]:
    """
    Analyze a single review and return sentence-level results.
    
    Useful for debugging or real-time analysis.
    """
    preprocessed = preprocess_review(review.review_text)
    
    results = []
    for i, sentence in enumerate(preprocessed.sentences):
        sentence_id = f"{review.review_id}_sent_{i}"
        
        aspect_matches = detect_aspects_in_sentence(sentence)
        if aspect_matches:
            aspect_matches = analyze_aspects_sentiments(sentence, aspect_matches)
        
        processed = ProcessedSentence(
            sentence_id=sentence_id,
            review_id=review.review_id,
            original_text=sentence,
            processed_text=sentence,
            detected_aspects=aspect_matches
        )
        results.append(processed)
    
    return results


def format_results_as_json(
    results: Dict[str, ListingIntelligence],
    indent: int = 2
) -> str:
    """Convert pipeline results to JSON string."""
    import json
    
    output = {}
    for listing_id, intelligence in results.items():
        output[listing_id] = intelligence.to_dict()
    
    return json.dumps(output, indent=indent)
