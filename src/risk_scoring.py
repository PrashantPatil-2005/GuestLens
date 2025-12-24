"""
risk_scoring.py - Risk Score Computation

Scoring Formula:
- Aspect Risk = 50% sentiment + 25% variance + 25% trend
- Overall Risk = Weighted average of aspect risks

Design Decisions:
1. Sentiment inverted: positive sentiment → low risk, negative → high risk
2. Variance directly contributes: high variance = inconsistent = risky
3. Trend penalty: declining trend adds to risk
4. Confidence discount: low confidence → reduced risk weight
5. Configurable weights for different operational priorities
"""

from typing import Dict, List, Tuple
from dataclasses import replace

from src.schemas import Aspect, AspectAggregation, ListingIntelligence, TrendDirection
from src.risk_schemas import (
    RiskLevel, AspectRisk, RiskDriver, DriverSeverity,
    score_to_risk_level
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Weights for combining aspect risks into overall risk
# Higher weight = more important for overall risk
ASPECT_WEIGHTS: Dict[str, float] = {
    "safety": 1.5,        # Safety issues are critical
    "cleanliness": 1.2,   # High guest impact
    "host_behavior": 1.1, # Service quality
    "noise": 1.0,         # Experience quality
    "amenities": 0.9,     # Can sometimes be subjective
    "location": 0.7       # Harder to change, often known upfront
}

# Risk component weights (must sum to 100)
SENTIMENT_WEIGHT = 50
VARIANCE_WEIGHT = 25
TREND_WEIGHT = 25

# Trend penalties (out of TREND_WEIGHT points)
TREND_PENALTIES = {
    TrendDirection.DECLINING: 25,       # Full penalty
    TrendDirection.INSUFFICIENT_DATA: 10,  # Partial (unknown = some risk)
    TrendDirection.STABLE: 0,
    TrendDirection.IMPROVING: -5        # Bonus (reduces risk)
}

# Thresholds for driver generation
HIGH_VARIANCE_THRESHOLD = 0.25
HIGH_DISAGREEMENT_THRESHOLD = 0.4
NEGATIVE_SENTIMENT_THRESHOLD = 0.0
VERY_NEGATIVE_THRESHOLD = -0.3


# =============================================================================
# NORMALIZATION
# =============================================================================

def normalize_sentiment_to_risk(sentiment: float) -> float:
    """
    Convert sentiment (-1 to 1) to risk contribution (0 to 1).
    
    Inversion: positive sentiment → low risk
    - sentiment = 1.0 → risk = 0.0
    - sentiment = 0.0 → risk = 0.5
    - sentiment = -1.0 → risk = 1.0
    """
    # Linear mapping: risk = (1 - sentiment) / 2
    return (1 - sentiment) / 2


def clamp(value: float, min_val: float = 0, max_val: float = 100) -> float:
    """Clamp value to range."""
    return max(min_val, min(max_val, value))


# =============================================================================
# ASPECT-LEVEL RISK
# =============================================================================

def compute_aspect_risk(
    aspect: Aspect,
    aggregation: AspectAggregation
) -> Tuple[AspectRisk, List[RiskDriver]]:
    """
    Compute risk score for a single aspect.
    
    Formula:
        base_risk = (sentiment_risk * 50) + (variance_risk * 25) + trend_penalty
        final_risk = base_risk * confidence_factor
    
    Args:
        aspect: The aspect being scored
        aggregation: Phase-1 aggregation for this aspect
        
    Returns:
        Tuple of (AspectRisk, list of RiskDrivers)
    """
    drivers = []
    
    # Component 1: Sentiment (50 points max)
    sentiment_risk = normalize_sentiment_to_risk(aggregation.weighted_sentiment)
    sentiment_contribution = sentiment_risk * SENTIMENT_WEIGHT
    
    # Generate driver for negative sentiment
    if aggregation.weighted_sentiment < VERY_NEGATIVE_THRESHOLD:
        drivers.append(RiskDriver(
            aspect=aspect.value,
            driver_type="very_negative_sentiment",
            severity=DriverSeverity.HIGH,
            description=f"Strongly negative sentiment ({aggregation.weighted_sentiment:.2f})",
            value=aggregation.weighted_sentiment
        ))
    elif aggregation.weighted_sentiment < NEGATIVE_SENTIMENT_THRESHOLD:
        drivers.append(RiskDriver(
            aspect=aspect.value,
            driver_type="negative_sentiment",
            severity=DriverSeverity.MEDIUM,
            description=f"Negative sentiment ({aggregation.weighted_sentiment:.2f})",
            value=aggregation.weighted_sentiment
        ))
    
    # Component 2: Variance (25 points max)
    # Cap variance contribution at 25 points
    variance_contribution = min(aggregation.sentiment_variance * 100, VARIANCE_WEIGHT)
    
    if aggregation.sentiment_variance > HIGH_VARIANCE_THRESHOLD:
        drivers.append(RiskDriver(
            aspect=aspect.value,
            driver_type="high_variance",
            severity=DriverSeverity.MEDIUM,
            description=f"Inconsistent guest experiences (variance: {aggregation.sentiment_variance:.2f})",
            value=aggregation.sentiment_variance
        ))
    
    # Component 3: Trend (25 points max, can be negative for bonus)
    trend_penalty = TREND_PENALTIES.get(aggregation.recent_trend, 0)
    trend_contribution = max(0, trend_penalty)  # Floor at 0 for display
    
    if aggregation.recent_trend == TrendDirection.DECLINING:
        drivers.append(RiskDriver(
            aspect=aspect.value,
            driver_type="declining_trend",
            severity=DriverSeverity.HIGH,
            description="Sentiment is declining over time",
            value=None
        ))
    
    # High disagreement driver
    if aggregation.disagreement_score > HIGH_DISAGREEMENT_THRESHOLD:
        drivers.append(RiskDriver(
            aspect=aspect.value,
            driver_type="polarized_opinions",
            severity=DriverSeverity.MEDIUM,
            description=f"Highly polarized reviews (disagreement: {aggregation.disagreement_score:.2f})",
            value=aggregation.disagreement_score
        ))
    
    # Calculate base risk
    base_risk = sentiment_contribution + variance_contribution + trend_penalty
    base_risk = clamp(base_risk, 0, 100)
    
    # Apply confidence factor
    # Low confidence → reduce the risk score (we're less certain)
    confidence_factor = aggregation.confidence_score
    if confidence_factor < 0.3:
        # Very low confidence - heavily discount the risk
        confidence_factor = max(0.3, confidence_factor)
        drivers.append(RiskDriver(
            aspect=aspect.value,
            driver_type="low_confidence",
            severity=DriverSeverity.LOW,
            description=f"Limited data, risk score may be unreliable",
            value=aggregation.confidence_score
        ))
    
    final_risk = base_risk * confidence_factor
    final_risk = clamp(final_risk)
    
    # Determine risk level and create driver list for AspectRisk
    risk_level = score_to_risk_level(final_risk)
    driver_types = [d.driver_type for d in drivers]
    
    aspect_risk = AspectRisk(
        aspect=aspect,
        risk_score=final_risk,
        risk_level=risk_level,
        drivers=driver_types,
        sentiment_contribution=sentiment_contribution,
        variance_contribution=variance_contribution,
        trend_contribution=trend_contribution
    )
    
    return aspect_risk, drivers


def compute_all_aspect_risks(
    intelligence: ListingIntelligence
) -> Tuple[Dict[str, AspectRisk], List[RiskDriver]]:
    """
    Compute risk for all aspects in a listing.
    
    Returns:
        Tuple of (aspect_risks dict, all risk drivers)
    """
    aspect_risks = {}
    all_drivers = []
    
    for aspect_name, aggregation in intelligence.aspect_aggregations.items():
        try:
            aspect = Aspect(aspect_name)
        except ValueError:
            continue
            
        risk, drivers = compute_aspect_risk(aspect, aggregation)
        aspect_risks[aspect_name] = risk
        all_drivers.extend(drivers)
    
    return aspect_risks, all_drivers


# =============================================================================
# OVERALL RISK
# =============================================================================

def compute_overall_risk(
    aspect_risks: Dict[str, AspectRisk]
) -> float:
    """
    Compute weighted average of aspect risks.
    
    Uses ASPECT_WEIGHTS to prioritize important aspects.
    """
    if not aspect_risks:
        return 0.0
    
    total_weighted_risk = 0.0
    total_weight = 0.0
    
    for aspect_name, risk in aspect_risks.items():
        weight = ASPECT_WEIGHTS.get(aspect_name, 1.0)
        
        # Only include aspects with mentions (non-zero contribution)
        if risk.risk_score > 0:
            total_weighted_risk += risk.risk_score * weight
            total_weight += weight
    
    if total_weight == 0:
        return 0.0
    
    return total_weighted_risk / total_weight


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_highest_risk_aspects(
    aspect_risks: Dict[str, AspectRisk],
    top_n: int = 3
) -> List[Tuple[str, float]]:
    """Get the top N highest risk aspects."""
    sorted_aspects = sorted(
        aspect_risks.items(),
        key=lambda x: x[1].risk_score,
        reverse=True
    )
    return [(name, risk.risk_score) for name, risk in sorted_aspects[:top_n]]


def count_declining_aspects(intelligence: ListingIntelligence) -> int:
    """Count how many aspects have declining trends."""
    count = 0
    for agg in intelligence.aspect_aggregations.values():
        if agg.recent_trend == TrendDirection.DECLINING:
            count += 1
    return count
