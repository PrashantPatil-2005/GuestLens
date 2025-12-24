"""
contradiction_detection.py - Detect Contradictions and Anomalies

Detects patterns that indicate problems or inconsistencies:
1. High variance: Guests have very different experiences
2. Polarization: Reviews split between positive and negative
3. Trend conflicts: Aspects moving in opposite directions
4. Multi-aspect decline: Multiple areas degrading simultaneously
"""

from typing import List, Dict, Tuple
from dataclasses import dataclass

from src.schemas import ListingIntelligence, AspectAggregation, TrendDirection
from src.risk_schemas import FlagType, RiskDriver, DriverSeverity


# =============================================================================
# CONFIGURATION
# =============================================================================

# Thresholds for contradiction detection
HIGH_VARIANCE_THRESHOLD = 0.30
HIGH_DISAGREEMENT_THRESHOLD = 0.50
MIN_MENTIONS_FOR_VARIANCE_FLAG = 5
MULTI_DECLINE_THRESHOLD = 2  # Number of declining aspects to trigger flag


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def detect_high_variance(
    intelligence: ListingIntelligence
) -> Tuple[List[FlagType], List[RiskDriver]]:
    """
    Detect aspects with high variance in sentiment.
    
    High variance = guests have inconsistent experiences
    (some love it, some hate it, not just polarized)
    
    Returns:
        Tuple of (flags, drivers)
    """
    flags = []
    drivers = []
    
    high_variance_aspects = []
    
    for aspect_name, agg in intelligence.aspect_aggregations.items():
        # Only flag if we have enough data to be confident
        if agg.mention_count < MIN_MENTIONS_FOR_VARIANCE_FLAG:
            continue
            
        if agg.sentiment_variance > HIGH_VARIANCE_THRESHOLD:
            high_variance_aspects.append(aspect_name)
            drivers.append(RiskDriver(
                aspect=aspect_name,
                driver_type="inconsistent_experience",
                severity=DriverSeverity.MEDIUM,
                description=f"High sentiment variance in {aspect_name} ({agg.sentiment_variance:.2f})",
                value=agg.sentiment_variance
            ))
    
    if high_variance_aspects:
        flags.append(FlagType.HIGH_VARIANCE)
    
    return flags, drivers


def detect_polarization(
    intelligence: ListingIntelligence
) -> Tuple[List[FlagType], List[RiskDriver]]:
    """
    Detect aspects with polarized opinions (bimodal distribution).
    
    Polarization = reviews split between positive and negative
    (different from variance: polarization is specifically disagreement)
    
    Returns:
        Tuple of (flags, drivers)
    """
    flags = []
    drivers = []
    
    polarized_aspects = []
    
    for aspect_name, agg in intelligence.aspect_aggregations.items():
        if agg.mention_count < MIN_MENTIONS_FOR_VARIANCE_FLAG:
            continue
            
        if agg.disagreement_score > HIGH_DISAGREEMENT_THRESHOLD:
            polarized_aspects.append(aspect_name)
            drivers.append(RiskDriver(
                aspect=aspect_name,
                driver_type="polarized_reviews",
                severity=DriverSeverity.HIGH,
                description=f"Reviews strongly disagree on {aspect_name} (disagreement: {agg.disagreement_score:.2f})",
                value=agg.disagreement_score
            ))
    
    if polarized_aspects:
        flags.append(FlagType.POLARIZED)
    
    return flags, drivers


def detect_multi_aspect_decline(
    intelligence: ListingIntelligence
) -> Tuple[List[FlagType], List[RiskDriver]]:
    """
    Detect when multiple aspects are declining simultaneously.
    
    This is a strong negative signal - the listing is degrading
    across multiple dimensions.
    
    Returns:
        Tuple of (flags, drivers)
    """
    flags = []
    drivers = []
    
    declining_aspects = []
    
    for aspect_name, agg in intelligence.aspect_aggregations.items():
        if agg.recent_trend == TrendDirection.DECLINING:
            declining_aspects.append(aspect_name)
    
    if len(declining_aspects) >= MULTI_DECLINE_THRESHOLD:
        flags.append(FlagType.MULTI_ASPECT_DECLINE)
        flags.append(FlagType.DECLINING_TREND)
        
        drivers.append(RiskDriver(
            aspect=None,  # Overall, not aspect-specific
            driver_type="multi_aspect_decline",
            severity=DriverSeverity.HIGH,
            description=f"Multiple aspects declining: {', '.join(declining_aspects)}",
            value=float(len(declining_aspects))
        ))
    elif len(declining_aspects) == 1:
        # Single decline is still noteworthy
        flags.append(FlagType.DECLINING_TREND)
        drivers.append(RiskDriver(
            aspect=declining_aspects[0],
            driver_type="declining_trend",
            severity=DriverSeverity.MEDIUM,
            description=f"Declining sentiment in {declining_aspects[0]}",
            value=None
        ))
    
    return flags, drivers


def detect_trend_conflicts(
    intelligence: ListingIntelligence
) -> Tuple[List[FlagType], List[RiskDriver]]:
    """
    Detect when important aspects are moving in opposite directions.
    
    Example: cleanliness improving but safety declining
    (less common, but worth flagging for investigation)
    
    Returns:
        Tuple of (flags, drivers)
    """
    drivers = []
    
    improving = []
    declining = []
    
    for aspect_name, agg in intelligence.aspect_aggregations.items():
        if agg.recent_trend == TrendDirection.IMPROVING:
            improving.append(aspect_name)
        elif agg.recent_trend == TrendDirection.DECLINING:
            declining.append(aspect_name)
    
    # Flag if we have both improving and declining (mixed signals)
    if improving and declining:
        drivers.append(RiskDriver(
            aspect=None,
            driver_type="trend_conflict",
            severity=DriverSeverity.LOW,
            description=f"Mixed trends: {', '.join(improving)} improving while {', '.join(declining)} declining",
            value=None
        ))
    
    return [], drivers  # No special flag, just a driver


def detect_low_confidence(
    intelligence: ListingIntelligence
) -> Tuple[List[FlagType], List[RiskDriver]]:
    """
    Detect when overall confidence is low due to limited data.
    
    Returns:
        Tuple of (flags, drivers)
    """
    flags = []
    drivers = []
    
    # Check overall review count
    if intelligence.total_reviews < 5:
        flags.append(FlagType.LOW_CONFIDENCE)
        drivers.append(RiskDriver(
            aspect=None,
            driver_type="insufficient_reviews",
            severity=DriverSeverity.LOW,
            description=f"Only {intelligence.total_reviews} reviews - risk assessment may be unreliable",
            value=float(intelligence.total_reviews)
        ))
    
    # Check for aspects with very low confidence
    low_conf_aspects = []
    for aspect_name, agg in intelligence.aspect_aggregations.items():
        if agg.mention_count >= 3 and agg.confidence_score < 0.3:
            low_conf_aspects.append(aspect_name)
    
    if low_conf_aspects and FlagType.LOW_CONFIDENCE not in flags:
        flags.append(FlagType.LOW_CONFIDENCE)
    
    return flags, drivers


def detect_safety_concerns(
    intelligence: ListingIntelligence
) -> Tuple[List[FlagType], List[RiskDriver]]:
    """
    Special handling for safety aspect due to its critical nature.
    
    Any negative safety sentiment gets elevated attention.
    
    Returns:
        Tuple of (flags, drivers)
    """
    flags = []
    drivers = []
    
    safety_agg = intelligence.aspect_aggregations.get('safety')
    
    if safety_agg and safety_agg.mention_count > 0:
        # Any negative safety sentiment is concerning
        if safety_agg.weighted_sentiment < -0.2:
            flags.append(FlagType.SAFETY_CONCERN)
            drivers.append(RiskDriver(
                aspect='safety',
                driver_type="safety_concern",
                severity=DriverSeverity.HIGH,
                description=f"Negative safety feedback detected (sentiment: {safety_agg.weighted_sentiment:.2f})",
                value=safety_agg.weighted_sentiment
            ))
        # Even neutral safety with declining trend is concerning
        elif safety_agg.recent_trend == TrendDirection.DECLINING:
            flags.append(FlagType.SAFETY_CONCERN)
            drivers.append(RiskDriver(
                aspect='safety',
                driver_type="declining_safety",
                severity=DriverSeverity.HIGH,
                description="Safety sentiment is declining over time",
                value=None
            ))
    
    return flags, drivers


# =============================================================================
# MAIN DETECTION FUNCTION
# =============================================================================

def detect_all_contradictions(
    intelligence: ListingIntelligence
) -> Tuple[List[FlagType], List[RiskDriver]]:
    """
    Run all contradiction/anomaly detectors on a listing.
    
    Returns:
        Tuple of (all flags, all drivers)
    """
    all_flags = []
    all_drivers = []
    
    # Run all detectors
    detectors = [
        detect_high_variance,
        detect_polarization,
        detect_multi_aspect_decline,
        detect_trend_conflicts,
        detect_low_confidence,
        detect_safety_concerns,
    ]
    
    for detector in detectors:
        flags, drivers = detector(intelligence)
        all_flags.extend(flags)
        all_drivers.extend(drivers)
    
    # Deduplicate flags
    all_flags = list(set(all_flags))
    
    return all_flags, all_drivers
