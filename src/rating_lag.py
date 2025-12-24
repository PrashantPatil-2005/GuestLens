"""
rating_lag.py - Rating vs Review Mismatch Detection

Detects when the numeric star rating doesn't match the sentiment
expressed in review text. This is "rating lag" - guests rate based
on expectations or social pressure rather than actual experience.

Note: This module requires external rating data not available in Phase-1.
The rating is an optional input to the risk pipeline.
"""

from typing import Optional, Tuple

from src.schemas import ListingIntelligence
from src.risk_schemas import FlagType, RiskDriver, DriverSeverity


# =============================================================================
# CONFIGURATION
# =============================================================================

# Mapping from sentiment to expected star rating (1-5 scale)
# sentiment -1.0 to 1.0 → rating 1.0 to 5.0
SENTIMENT_TO_RATING_MAPPING = [
    (-1.0, 1.0),   # Very negative → 1 star
    (-0.6, 2.0),   # Negative → 2 stars
    (-0.2, 3.0),   # Slightly negative → 3 stars
    (0.2, 3.5),    # Neutral → 3.5 stars
    (0.5, 4.0),    # Positive → 4 stars
    (0.8, 4.5),    # Very positive → 4.5 stars
    (1.0, 5.0),    # Excellent → 5 stars
]

# Minimum mismatch to consider significant
RATING_LAG_THRESHOLD = 0.75  # 0.75 star difference


# =============================================================================
# CONVERSION FUNCTIONS
# =============================================================================

def sentiment_to_expected_rating(sentiment: float) -> float:
    """
    Convert overall weighted sentiment to expected star rating.
    
    Uses linear interpolation between defined mapping points.
    
    Args:
        sentiment: Overall sentiment score (-1 to 1)
        
    Returns:
        Expected rating (1.0 to 5.0)
    """
    # Clamp sentiment to valid range
    sentiment = max(-1.0, min(1.0, sentiment))
    
    # Find the two mapping points to interpolate between
    for i in range(len(SENTIMENT_TO_RATING_MAPPING) - 1):
        s1, r1 = SENTIMENT_TO_RATING_MAPPING[i]
        s2, r2 = SENTIMENT_TO_RATING_MAPPING[i + 1]
        
        if s1 <= sentiment <= s2:
            # Linear interpolation
            ratio = (sentiment - s1) / (s2 - s1) if s2 != s1 else 0
            return r1 + ratio * (r2 - r1)
    
    # Fallback (shouldn't reach here with clamped input)
    if sentiment < SENTIMENT_TO_RATING_MAPPING[0][0]:
        return SENTIMENT_TO_RATING_MAPPING[0][1]
    return SENTIMENT_TO_RATING_MAPPING[-1][1]


def rating_to_expected_sentiment(rating: float) -> float:
    """
    Convert star rating to expected sentiment.
    Inverse of sentiment_to_expected_rating.
    
    Args:
        rating: Star rating (1 to 5)
        
    Returns:
        Expected sentiment (-1 to 1)
    """
    # Clamp rating
    rating = max(1.0, min(5.0, rating))
    
    # Find the two mapping points to interpolate between
    for i in range(len(SENTIMENT_TO_RATING_MAPPING) - 1):
        s1, r1 = SENTIMENT_TO_RATING_MAPPING[i]
        s2, r2 = SENTIMENT_TO_RATING_MAPPING[i + 1]
        
        if r1 <= rating <= r2:
            # Linear interpolation (inverted)
            ratio = (rating - r1) / (r2 - r1) if r2 != r1 else 0
            return s1 + ratio * (s2 - s1)
    
    # Fallback
    if rating < SENTIMENT_TO_RATING_MAPPING[0][1]:
        return SENTIMENT_TO_RATING_MAPPING[0][0]
    return SENTIMENT_TO_RATING_MAPPING[-1][0]


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def compute_overall_sentiment(intelligence: ListingIntelligence) -> float:
    """
    Compute overall weighted sentiment across all aspects.
    
    Uses the same aspect weights as risk scoring.
    """
    from src.risk_scoring import ASPECT_WEIGHTS
    
    total_weighted_sentiment = 0.0
    total_weight = 0.0
    
    for aspect_name, agg in intelligence.aspect_aggregations.items():
        if agg.mention_count > 0:
            weight = ASPECT_WEIGHTS.get(aspect_name, 1.0)
            total_weighted_sentiment += agg.weighted_sentiment * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return total_weighted_sentiment / total_weight


def detect_rating_mismatch(
    intelligence: ListingIntelligence,
    actual_rating: Optional[float] = None
) -> Tuple[Optional[FlagType], Optional[RiskDriver], dict]:
    """
    Detect mismatch between actual star rating and review sentiment.
    
    Args:
        intelligence: Phase-1 listing intelligence
        actual_rating: Optional actual star rating (1-5)
        
    Returns:
        Tuple of (flag or None, driver or None, metadata dict)
    """
    metadata = {}
    
    # Compute overall sentiment
    overall_sentiment = compute_overall_sentiment(intelligence)
    expected_rating = sentiment_to_expected_rating(overall_sentiment)
    
    metadata['computed_overall_sentiment'] = round(overall_sentiment, 3)
    metadata['expected_rating'] = round(expected_rating, 2)
    
    if actual_rating is None:
        # No actual rating provided, can't detect mismatch
        metadata['rating_available'] = False
        return None, None, metadata
    
    metadata['rating_available'] = True
    metadata['actual_rating'] = actual_rating
    
    # Calculate mismatch
    mismatch = actual_rating - expected_rating
    metadata['rating_mismatch'] = round(mismatch, 2)
    
    if abs(mismatch) >= RATING_LAG_THRESHOLD:
        flag = FlagType.RATING_LAG
        
        if mismatch > 0:
            # Rating higher than sentiment suggests
            description = (
                f"Rating ({actual_rating:.1f}) is higher than sentiment suggests "
                f"(expected {expected_rating:.1f}). Guests may rate positively despite issues."
            )
            severity = DriverSeverity.MEDIUM
        else:
            # Rating lower than sentiment suggests
            description = (
                f"Rating ({actual_rating:.1f}) is lower than sentiment suggests "
                f"(expected {expected_rating:.1f}). Reviews are more positive than scores indicate."
            )
            severity = DriverSeverity.LOW  # Less concerning
        
        driver = RiskDriver(
            aspect=None,
            driver_type="rating_sentiment_mismatch",
            severity=severity,
            description=description,
            value=mismatch
        )
        
        return flag, driver, metadata
    
    return None, None, metadata


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def analyze_rating_pattern(
    ratings_with_sentiments: list
) -> dict:
    """
    Analyze patterns in rating vs sentiment over multiple reviews/listings.
    
    Args:
        ratings_with_sentiments: List of (actual_rating, sentiment) tuples
        
    Returns:
        Analysis dict with correlation and patterns
    """
    if len(ratings_with_sentiments) < 3:
        return {'sufficient_data': False}
    
    ratings = [r for r, s in ratings_with_sentiments]
    sentiments = [s for r, s in ratings_with_sentiments]
    
    # Simple correlation (Pearson-like)
    n = len(ratings)
    mean_r = sum(ratings) / n
    mean_s = sum(sentiments) / n
    
    numerator = sum((r - mean_r) * (s - mean_s) for r, s in ratings_with_sentiments)
    denom_r = sum((r - mean_r) ** 2 for r in ratings) ** 0.5
    denom_s = sum((s - mean_s) ** 2 for s in sentiments) ** 0.5
    
    if denom_r * denom_s == 0:
        correlation = 0
    else:
        correlation = numerator / (denom_r * denom_s)
    
    return {
        'sufficient_data': True,
        'correlation': round(correlation, 3),
        'avg_rating': round(mean_r, 2),
        'avg_sentiment': round(mean_s, 3),
        'sample_size': n
    }
