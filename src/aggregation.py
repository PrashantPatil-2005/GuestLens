"""
aggregation.py - Temporal Weighting and Listing-Level Aggregation

Design Decisions:
1. Exponential Decay: Recent reviews weighted higher using exponential decay
   with configurable half-life (default 180 days = 6 months)

2. Weighted Average: Temporal weights applied to sentiment scores when
   computing listing-level aggregates

3. Trend Detection: Compare recent period sentiment to historical average
   to detect improvement/decline patterns

4. Per-Aspect Aggregation: Each aspect gets its own aggregation statistics
   to enable targeted operational insights
"""

import math
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

from src.schemas import (
    Aspect, AspectMatch, ProcessedSentence, RawReview,
    AspectAggregation, ListingIntelligence, TrendDirection
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Half-life for temporal decay in days
# After HALF_LIFE_DAYS, a review has half the weight of a review from today
DEFAULT_HALF_LIFE_DAYS = 180

# Minimum days difference to compute trend (need enough separation)
TREND_MINIMUM_DAYS = 30

# Proportion of reviews considered "recent" for trend analysis
RECENT_PROPORTION = 0.3


# =============================================================================
# TEMPORAL WEIGHTING
# =============================================================================

def compute_temporal_weight(
    review_date: datetime,
    reference_date: datetime,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS
) -> float:
    """
    Compute temporal weight using exponential decay.
    
    Formula: weight = 2^(-days_ago / half_life)
    
    At days_ago = 0: weight = 1.0
    At days_ago = half_life: weight = 0.5
    At days_ago = 2*half_life: weight = 0.25
    
    Args:
        review_date: When the review was posted
        reference_date: The reference point (usually analysis date = today)
        half_life_days: Days after which weight is halved
        
    Returns:
        Weight between 0 and 1
    """
    days_ago = (reference_date - review_date).days
    
    # Clamp to non-negative (future dates would have negative days_ago)
    days_ago = max(0, days_ago)
    
    # Exponential decay: weight = 2^(-days_ago / half_life)
    weight = math.pow(2, -days_ago / half_life_days)
    
    return weight


def compute_weights_for_reviews(
    reviews: List[RawReview],
    reference_date: Optional[datetime] = None,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS
) -> Dict[str, float]:
    """
    Compute temporal weights for a list of reviews.
    
    Args:
        reviews: List of RawReview objects
        reference_date: Reference point for weight calculation (default: now)
        half_life_days: Half-life for decay
        
    Returns:
        Dict mapping review_id to temporal weight
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    weights = {}
    for review in reviews:
        weights[review.review_id] = compute_temporal_weight(
            review.review_date,
            reference_date,
            half_life_days
        )
    
    return weights


# =============================================================================
# DATA COLLECTION
# =============================================================================

@dataclass
class AspectDataPoint:
    """Single data point for aspect aggregation."""
    review_id: str
    sentence_id: str
    review_date: datetime
    sentiment_score: float
    confidence: float
    temporal_weight: float


def collect_aspect_data_points(
    sentences: List[ProcessedSentence],
    reviews: List[RawReview],
    review_weights: Dict[str, float]
) -> Dict[Aspect, List[AspectDataPoint]]:
    """
    Collect all aspect mentions with their metadata for aggregation.
    
    Args:
        sentences: Processed sentences with detected aspects
        reviews: Original reviews for date lookup
        review_weights: Pre-computed temporal weights per review_id
        
    Returns:
        Dict mapping each Aspect to its list of data points
    """
    # Create lookup for review dates
    review_dates = {r.review_id: r.review_date for r in reviews}
    
    aspect_data: Dict[Aspect, List[AspectDataPoint]] = defaultdict(list)
    
    for sentence in sentences:
        review_id = sentence.review_id
        review_date = review_dates.get(review_id)
        temporal_weight = review_weights.get(review_id, 1.0)
        
        if review_date is None:
            continue
        
        for aspect_match in sentence.detected_aspects:
            data_point = AspectDataPoint(
                review_id=review_id,
                sentence_id=sentence.sentence_id,
                review_date=review_date,
                sentiment_score=aspect_match.sentiment_score,
                confidence=aspect_match.confidence,
                temporal_weight=temporal_weight
            )
            aspect_data[aspect_match.aspect].append(data_point)
    
    return aspect_data


# =============================================================================
# TREND DETECTION
# =============================================================================

def detect_trend(
    data_points: List[AspectDataPoint],
    date_range_days: int
) -> TrendDirection:
    """
    Detect trend by comparing recent sentiment to historical average.
    
    Strategy:
    1. Sort data points by date
    2. Split into "recent" (last RECENT_PROPORTION) and "historical"
    3. Compare average sentiments
    
    Args:
        data_points: List of aspect data points sorted or unsorted
        date_range_days: Total span of days in the data
        
    Returns:
        TrendDirection enum value
    """
    if len(data_points) < 3:
        return TrendDirection.INSUFFICIENT_DATA
    
    if date_range_days < TREND_MINIMUM_DAYS:
        return TrendDirection.INSUFFICIENT_DATA
    
    # Sort by date
    sorted_points = sorted(data_points, key=lambda x: x.review_date)
    
    # Split: last RECENT_PROPORTION as "recent"
    split_index = max(1, int(len(sorted_points) * (1 - RECENT_PROPORTION)))
    historical = sorted_points[:split_index]
    recent = sorted_points[split_index:]
    
    if not historical or not recent:
        return TrendDirection.INSUFFICIENT_DATA
    
    # Compute averages
    historical_avg = sum(dp.sentiment_score for dp in historical) / len(historical)
    recent_avg = sum(dp.sentiment_score for dp in recent) / len(recent)
    
    # Determine trend based on difference
    diff = recent_avg - historical_avg
    
    # Threshold for meaningful change
    THRESHOLD = 0.15
    
    if diff > THRESHOLD:
        return TrendDirection.IMPROVING
    elif diff < -THRESHOLD:
        return TrendDirection.DECLINING
    else:
        return TrendDirection.STABLE


# =============================================================================
# AGGREGATION
# =============================================================================

def aggregate_aspect(
    aspect: Aspect,
    data_points: List[AspectDataPoint],
    date_range_days: int
) -> AspectAggregation:
    """
    Aggregate all data points for a single aspect.
    
    Computes:
    - weighted_sentiment: Temporal-weighted mean
    - raw_sentiment_mean: Simple arithmetic mean
    - sentiment_variance: Statistical variance
    - disagreement_score: Computed in confidence.py (placeholder 0)
    - confidence_score: Computed in confidence.py (placeholder 0)
    - mention_count: Number of mentions
    - recent_trend: Based on time-ordered sentiment analysis
    
    Args:
        aspect: The aspect being aggregated
        data_points: All data points for this aspect
        date_range_days: Total span of days in data
        
    Returns:
        AspectAggregation object
    """
    if not data_points:
        return AspectAggregation(
            aspect=aspect,
            weighted_sentiment=0.0,
            raw_sentiment_mean=0.0,
            sentiment_variance=0.0,
            disagreement_score=0.0,  # Will be computed in confidence.py
            confidence_score=0.0,    # Will be computed in confidence.py
            mention_count=0,
            recent_trend=TrendDirection.INSUFFICIENT_DATA
        )
    
    scores = [dp.sentiment_score for dp in data_points]
    weights = [dp.temporal_weight for dp in data_points]
    
    # Raw mean
    raw_mean = sum(scores) / len(scores)
    
    # Weighted mean
    total_weight = sum(weights)
    if total_weight > 0:
        weighted_mean = sum(s * w for s, w in zip(scores, weights)) / total_weight
    else:
        weighted_mean = raw_mean
    
    # Variance: sum((x - mean)^2) / n
    variance = sum((s - raw_mean) ** 2 for s in scores) / len(scores)
    
    # Trend detection
    trend = detect_trend(data_points, date_range_days)
    
    return AspectAggregation(
        aspect=aspect,
        weighted_sentiment=weighted_mean,
        raw_sentiment_mean=raw_mean,
        sentiment_variance=variance,
        disagreement_score=0.0,  # Will be filled by confidence module
        confidence_score=0.0,   # Will be filled by confidence module
        mention_count=len(data_points),
        recent_trend=trend
    )


def aggregate_by_listing(
    sentences: List[ProcessedSentence],
    reviews: List[RawReview],
    listing_id: str,
    reference_date: Optional[datetime] = None,
    half_life_days: int = DEFAULT_HALF_LIFE_DAYS
) -> ListingIntelligence:
    """
    Aggregate all sentence-level analysis to listing-level intelligence.
    
    Args:
        sentences: Processed sentences for this listing
        reviews: Original reviews for this listing
        listing_id: ID of the listing
        reference_date: Reference date for temporal weighting (default: now)
        half_life_days: Half-life for temporal decay
        
    Returns:
        ListingIntelligence object with per-aspect aggregations
    """
    if reference_date is None:
        reference_date = datetime.now()
    
    if not reviews:
        return ListingIntelligence(
            listing_id=listing_id,
            analysis_timestamp=reference_date,
            aspect_aggregations={},
            total_reviews=0,
            total_sentences=0,
            date_range_start=reference_date,
            date_range_end=reference_date
        )
    
    # Compute temporal weights
    review_weights = compute_weights_for_reviews(reviews, reference_date, half_life_days)
    
    # Collect data points per aspect
    aspect_data = collect_aspect_data_points(sentences, reviews, review_weights)
    
    # Date range
    review_dates = [r.review_date for r in reviews]
    date_range_start = min(review_dates)
    date_range_end = max(review_dates)
    date_range_days = (date_range_end - date_range_start).days
    
    # Aggregate each aspect
    aggregations = {}
    for aspect in Aspect:
        data_points = aspect_data.get(aspect, [])
        aggregations[aspect.value] = aggregate_aspect(aspect, data_points, date_range_days)
    
    return ListingIntelligence(
        listing_id=listing_id,
        analysis_timestamp=reference_date,
        aspect_aggregations=aggregations,
        total_reviews=len(reviews),
        total_sentences=len(sentences),
        date_range_start=date_range_start,
        date_range_end=date_range_end
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def group_sentences_by_listing(
    sentences: List[ProcessedSentence],
    reviews: List[RawReview]
) -> Dict[str, Tuple[List[ProcessedSentence], List[RawReview]]]:
    """
    Group sentences and reviews by listing_id for batch processing.
    
    Returns:
        Dict mapping listing_id to (sentences, reviews) tuple
    """
    # Build review_id -> listing_id lookup
    review_to_listing = {r.review_id: r.listing_id for r in reviews}
    
    # Group sentences
    listing_sentences: Dict[str, List[ProcessedSentence]] = defaultdict(list)
    for sent in sentences:
        listing_id = review_to_listing.get(sent.review_id, "unknown")
        listing_sentences[listing_id].append(sent)
    
    # Group reviews
    listing_reviews: Dict[str, List[RawReview]] = defaultdict(list)
    for review in reviews:
        listing_reviews[review.listing_id].append(review)
    
    # Combine
    result = {}
    all_listing_ids = set(listing_sentences.keys()) | set(listing_reviews.keys())
    
    for listing_id in all_listing_ids:
        result[listing_id] = (
            listing_sentences.get(listing_id, []),
            listing_reviews.get(listing_id, [])
        )
    
    return result
