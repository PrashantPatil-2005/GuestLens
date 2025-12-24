"""
mock_data.py - Mock Phase-2 outputs for API development

Generates realistic mock data for testing the dashboard.
In production, this would be replaced with actual Phase-2 pipeline calls.
"""

from datetime import datetime
import random
from typing import Dict, Optional

from api.schemas import (
    ListingAssessment, AspectData, RiskLevel, RatingTrust,
    ActionType, TrendDirection
)


# =============================================================================
# MOCK DATA STORE
# =============================================================================

# Pre-defined mock listings with different risk profiles
MOCK_LISTINGS: Dict[str, dict] = {
    "listing_001": {
        "risk_profile": "low",
        "description": "Well-maintained property with consistent positive reviews"
    },
    "listing_002": {
        "risk_profile": "medium",
        "description": "Some concerns about noise and cleanliness, monitoring needed"
    },
    "listing_003": {
        "risk_profile": "high",
        "description": "Safety concerns and polarized reviews, needs urgent review"
    },
    "listing_004": {
        "risk_profile": "low",
        "description": "New listing with limited reviews but positive trend"
    },
    "listing_005": {
        "risk_profile": "medium",
        "description": "Host responsiveness issues, declining trend"
    },
}


def _sentiment_to_label(sentiment: float) -> str:
    """Convert sentiment score to readable label."""
    if sentiment > 0.3:
        return "Positive"
    elif sentiment < -0.3:
        return "Negative"
    return "Neutral"


def _variance_to_label(variance: float) -> str:
    """Convert variance to readable label."""
    if variance < 0.15:
        return "Consistent"
    elif variance > 0.35:
        return "Inconsistent"
    return "Mixed"


def _generate_aspect_data(aspect_name: str, risk_profile: str) -> AspectData:
    """Generate mock aspect data based on risk profile."""
    
    # Base values by risk profile
    if risk_profile == "low":
        sentiment = random.uniform(0.5, 0.9)
        variance = random.uniform(0.05, 0.15)
        trend_weights = [0.3, 0.1, 0.6]  # improving, declining, stable
    elif risk_profile == "medium":
        sentiment = random.uniform(0.0, 0.5)
        variance = random.uniform(0.15, 0.35)
        trend_weights = [0.2, 0.4, 0.4]
    else:  # high
        sentiment = random.uniform(-0.5, 0.2)
        variance = random.uniform(0.25, 0.5)
        trend_weights = [0.1, 0.6, 0.3]
    
    # Special cases for certain aspects
    if aspect_name == "safety" and risk_profile == "high":
        sentiment = random.uniform(-0.7, -0.2)
    
    trend = random.choices(
        [TrendDirection.IMPROVING, TrendDirection.DECLINING, TrendDirection.STABLE],
        weights=trend_weights
    )[0]
    
    return AspectData(
        name=aspect_name,
        sentiment=round(sentiment, 2),
        sentiment_label=_sentiment_to_label(sentiment),
        trend=trend,
        variance=round(variance, 2),
        variance_label=_variance_to_label(variance),
        mention_count=random.randint(5, 30)
    )


def generate_mock_assessment(listing_id: str) -> Optional[ListingAssessment]:
    """
    Generate a complete mock assessment for a listing.
    
    Returns None if listing_id not found.
    """
    if listing_id not in MOCK_LISTINGS:
        return None
    
    profile = MOCK_LISTINGS[listing_id]
    risk_profile = profile["risk_profile"]
    
    # Determine overall risk level and score
    if risk_profile == "low":
        overall_risk = RiskLevel.LOW
        risk_score = random.uniform(10, 30)
        confidence = random.uniform(0.7, 0.95)
        rating_trust = RatingTrust.RELIABLE
        action = ActionType.IGNORE
        risk_drivers = []
    elif risk_profile == "medium":
        overall_risk = RiskLevel.MEDIUM
        risk_score = random.uniform(35, 55)
        confidence = random.uniform(0.6, 0.85)
        rating_trust = random.choice([RatingTrust.RELIABLE, RatingTrust.UNRELIABLE])
        action = ActionType.MONITOR
        risk_drivers = random.sample([
            "Declining trend in cleanliness",
            "Moderate variance in guest experiences",
            "Some negative host behavior feedback",
            "Noise complaints increasing"
        ], k=2)
    else:  # high
        overall_risk = RiskLevel.HIGH
        risk_score = random.uniform(60, 85)
        confidence = random.uniform(0.7, 0.9)
        rating_trust = RatingTrust.UNRELIABLE
        action = ActionType.FLAG
        risk_drivers = random.sample([
            "Safety concerns reported by multiple guests",
            "Highly polarized reviews (disagreement: 0.7+)",
            "Multiple aspects declining simultaneously",
            "Rating significantly higher than sentiment suggests",
            "Cleanliness issues mentioned repeatedly"
        ], k=3)
    
    # Generate aspect data
    aspects = [
        _generate_aspect_data("cleanliness", risk_profile),
        _generate_aspect_data("noise", risk_profile),
        _generate_aspect_data("location", risk_profile),
        _generate_aspect_data("host_behavior", risk_profile),
        _generate_aspect_data("amenities", risk_profile),
        _generate_aspect_data("safety", risk_profile),
    ]
    
    return ListingAssessment(
        listing_id=listing_id,
        overall_risk=overall_risk,
        risk_score=round(risk_score, 1),
        confidence=round(confidence, 2),
        rating_trust=rating_trust,
        aspects=aspects,
        risk_drivers=risk_drivers,
        recommended_action=action,
        review_count=random.randint(15, 50),
        assessment_date=datetime.now().isoformat()
    )


def get_available_listings() -> list:
    """Return list of available mock listing IDs."""
    return list(MOCK_LISTINGS.keys())
