"""
confidence.py - Variance, Disagreement, and Confidence Scoring

Design Decisions:
1. Confidence = f(volume, consistency, temporal coverage)
   - More mentions = higher confidence
   - Lower variance = higher confidence  
   - Wider date range = higher confidence

2. Disagreement Detection: Using bimodality analysis
   - High disagreement when reviews are polarized (both positive and negative)
   - Low disagreement when sentiment is consistent

3. Volume Normalization: Logarithmic scaling for mention counts
   - Prevents very popular listings from dominating
   - Accounts for diminishing returns of additional reviews

4. All scores normalized to 0-1 range for easy interpretation
"""

import math
from typing import List, Dict
from dataclasses import replace

from src.schemas import Aspect, AspectAggregation, ListingIntelligence


# =============================================================================
# CONFIGURATION
# =============================================================================

# Minimum mentions for reasonable confidence
MIN_MENTIONS_FOR_CONFIDENCE = 3

# Optimal number of mentions (beyond this, diminishing returns)
OPTIMAL_MENTIONS = 20

# Variance threshold (below this = highly consistent)
LOW_VARIANCE_THRESHOLD = 0.1

# Variance threshold (above this = highly inconsistent)
HIGH_VARIANCE_THRESHOLD = 0.4


# =============================================================================
# VARIANCE COMPUTATION
# =============================================================================

def compute_variance(scores: List[float]) -> float:
    """
    Compute statistical variance of sentiment scores.
    
    Variance = sum((x - mean)^2) / n
    
    Args:
        scores: List of sentiment scores (-1 to 1)
        
    Returns:
        Variance (0 = identical scores, max ~1 for fully polarized)
    """
    if len(scores) < 2:
        return 0.0
    
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)
    
    return variance


def compute_standard_deviation(scores: List[float]) -> float:
    """Compute standard deviation."""
    return math.sqrt(compute_variance(scores))


# =============================================================================
# DISAGREEMENT SCORING
# =============================================================================

def compute_disagreement(scores: List[float]) -> float:
    """
    Measure disagreement/polarization in reviews.
    
    High disagreement when:
    - Scores are spread across both positive and negative
    - Score distribution is bimodal (clusters at extremes)
    
    Low disagreement when:
    - Scores cluster together regardless of whether positive/negative
    
    Approach:
    1. Count positive vs negative mentions
    2. Check for balance (50/50 split = max disagreement)
    3. Factor in spread of scores
    
    Args:
        scores: List of sentiment scores (-1 to 1)
        
    Returns:
        Disagreement score 0-1 (0 = consensus, 1 = highly polarized)
    """
    if len(scores) < 2:
        return 0.0
    
    # Count positive, negative, neutral
    positive = sum(1 for s in scores if s > 0.2)
    negative = sum(1 for s in scores if s < -0.2)
    total_opinionated = positive + negative
    
    if total_opinionated == 0:
        # All neutral - no disagreement
        return 0.0
    
    # Balance factor: how evenly split between positive and negative
    # Max when 50/50, min when all one side
    min_side = min(positive, negative)
    balance = (2 * min_side) / total_opinionated  # 0 to 1
    
    # Spread factor: variance contributes to disagreement
    variance = compute_variance(scores)
    spread = min(1.0, variance / HIGH_VARIANCE_THRESHOLD)
    
    # Combine: high balance AND high spread = high disagreement
    # Using geometric mean to require both conditions
    disagreement = math.sqrt(balance * spread)
    
    return min(1.0, disagreement)


def detect_polarization_pattern(scores: List[float]) -> Dict:
    """
    Analyze the pattern of polarization for explainability.
    
    Returns:
        Dict with:
        - pattern: "unanimous_positive", "unanimous_negative", "polarized", "mixed", "neutral"
        - positive_pct: percentage of positive scores
        - negative_pct: percentage of negative scores
        - neutral_pct: percentage of neutral scores
    """
    if not scores:
        return {"pattern": "no_data", "positive_pct": 0, "negative_pct": 0, "neutral_pct": 0}
    
    n = len(scores)
    positive = sum(1 for s in scores if s > 0.2)
    negative = sum(1 for s in scores if s < -0.2)
    neutral = n - positive - negative
    
    pos_pct = round(100 * positive / n, 1)
    neg_pct = round(100 * negative / n, 1)
    neu_pct = round(100 * neutral / n, 1)
    
    # Determine pattern
    if pos_pct >= 80:
        pattern = "unanimous_positive"
    elif neg_pct >= 80:
        pattern = "unanimous_negative"
    elif neu_pct >= 60:
        pattern = "neutral"
    elif pos_pct >= 30 and neg_pct >= 30:
        pattern = "polarized"
    else:
        pattern = "mixed"
    
    return {
        "pattern": pattern,
        "positive_pct": pos_pct,
        "negative_pct": neg_pct,
        "neutral_pct": neu_pct
    }


# =============================================================================
# CONFIDENCE SCORING
# =============================================================================

def compute_confidence(
    mention_count: int,
    variance: float,
    date_range_days: int = 0,
    total_reviews: int = 0
) -> float:
    """
    Compute confidence score based on data quality signals.
    
    Confidence factors:
    1. Volume: More mentions = higher confidence (log-scaled)
    2. Consistency: Lower variance = higher confidence
    3. Coverage: Higher mention rate relative to reviews = higher confidence
    
    Args:
        mention_count: Number of times aspect was mentioned
        variance: Variance in sentiment scores
        date_range_days: Total span of days (for coverage bonus)
        total_reviews: Total reviews analyzed (for mention rate)
        
    Returns:
        Confidence score 0-1
    """
    if mention_count == 0:
        return 0.0
    
    # Factor 1: Volume (log-scaled)
    # Reaches ~0.7 at MIN_MENTIONS_FOR_CONFIDENCE, ~1.0 at OPTIMAL_MENTIONS
    volume_factor = min(1.0, math.log(mention_count + 1) / math.log(OPTIMAL_MENTIONS + 1))
    
    # Penalize very low mention counts
    if mention_count < MIN_MENTIONS_FOR_CONFIDENCE:
        volume_factor *= mention_count / MIN_MENTIONS_FOR_CONFIDENCE
    
    # Factor 2: Consistency (inverse of variance)
    # Low variance = high consistency = high confidence
    if variance <= LOW_VARIANCE_THRESHOLD:
        consistency_factor = 1.0
    elif variance >= HIGH_VARIANCE_THRESHOLD:
        consistency_factor = 0.5
    else:
        # Linear interpolation
        normalized_var = (variance - LOW_VARIANCE_THRESHOLD) / (HIGH_VARIANCE_THRESHOLD - LOW_VARIANCE_THRESHOLD)
        consistency_factor = 1.0 - (0.5 * normalized_var)
    
    # Factor 3: Coverage (mention rate among reviews)
    if total_reviews > 0:
        mention_rate = mention_count / total_reviews
        # Normalize: 30% mention rate is "good coverage"
        coverage_factor = min(1.0, mention_rate / 0.3)
    else:
        coverage_factor = 0.5  # Neutral if we don't know total reviews
    
    # Combine factors with weights
    # Volume is most important, then consistency, then coverage
    confidence = (
        0.45 * volume_factor +
        0.35 * consistency_factor +
        0.20 * coverage_factor
    )
    
    return min(1.0, max(0.0, confidence))


def get_confidence_level(confidence: float) -> str:
    """
    Convert numeric confidence to categorical label.
    
    Returns: "very_low", "low", "medium", "high", "very_high"
    """
    if confidence < 0.2:
        return "very_low"
    elif confidence < 0.4:
        return "low"
    elif confidence < 0.6:
        return "medium"
    elif confidence < 0.8:
        return "high"
    else:
        return "very_high"


# =============================================================================
# ENHANCEMENT FUNCTIONS
# =============================================================================

def enhance_aggregation_with_confidence(
    aggregation: AspectAggregation,
    scores: List[float],
    total_reviews: int,
    date_range_days: int = 0
) -> AspectAggregation:
    """
    Enhance an AspectAggregation with computed disagreement and confidence.
    
    This is called after basic aggregation to fill in the confidence fields.
    
    Args:
        aggregation: AspectAggregation with basic stats computed
        scores: List of all sentiment scores for this aspect
        total_reviews: Total number of reviews for the listing
        date_range_days: Span of days in the data
        
    Returns:
        Updated AspectAggregation with disagreement_score and confidence_score
    """
    disagreement = compute_disagreement(scores)
    confidence = compute_confidence(
        mention_count=aggregation.mention_count,
        variance=aggregation.sentiment_variance,
        date_range_days=date_range_days,
        total_reviews=total_reviews
    )
    
    return replace(
        aggregation,
        disagreement_score=disagreement,
        confidence_score=confidence
    )


def enhance_listing_intelligence(
    intelligence: ListingIntelligence,
    aspect_scores: Dict[Aspect, List[float]]
) -> ListingIntelligence:
    """
    Enhance ListingIntelligence with confidence scores for all aspects.
    
    Args:
        intelligence: ListingIntelligence with basic aggregations
        aspect_scores: Dict mapping Aspect to list of sentiment scores
        
    Returns:
        Enhanced ListingIntelligence
    """
    date_range_days = (intelligence.date_range_end - intelligence.date_range_start).days
    
    enhanced_aggregations = {}
    for aspect_name, agg in intelligence.aspect_aggregations.items():
        aspect = Aspect(aspect_name)
        scores = aspect_scores.get(aspect, [])
        
        enhanced_agg = enhance_aggregation_with_confidence(
            aggregation=agg,
            scores=scores,
            total_reviews=intelligence.total_reviews,
            date_range_days=date_range_days
        )
        enhanced_aggregations[aspect_name] = enhanced_agg
    
    return replace(intelligence, aspect_aggregations=enhanced_aggregations)
