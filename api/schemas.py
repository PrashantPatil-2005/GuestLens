"""
schemas.py - Pydantic models for API responses

These define the exact JSON structure returned by the API.
Designed for quick decision-making by ops/host managers.
"""

from pydantic import BaseModel
from typing import List, Dict, Optional
from enum import Enum


class RiskLevel(str, Enum):
    """Overall risk classification - should be instantly readable."""
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"


class RatingTrust(str, Enum):
    """Whether the star rating can be trusted vs review sentiment."""
    RELIABLE = "Reliable"
    UNRELIABLE = "Unreliable"


class ActionType(str, Enum):
    """Clear action recommendation for the decision-maker."""
    FLAG = "flag"
    MONITOR = "monitor"
    IGNORE = "ignore"


class TrendDirection(str, Enum):
    """Trend for aspect sentiment over time."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"


class AspectData(BaseModel):
    """
    Per-aspect intelligence summary.
    Shows: what guests think, is it consistent, is it changing?
    """
    name: str
    sentiment: float          # -1 to 1, how positive/negative
    sentiment_label: str      # "Positive", "Neutral", "Negative"
    trend: TrendDirection     # Is it getting better/worse?
    variance: float           # 0-1, how consistent are opinions?
    variance_label: str       # "Consistent", "Mixed", "Inconsistent"
    mention_count: int        # How many reviews mentioned this?


class ListingAssessment(BaseModel):
    """
    Complete Phase-2 output for a single listing.
    Everything a decision-maker needs in one view.
    """
    listing_id: str
    
    # Top-level status (immediate attention grabbers)
    overall_risk: RiskLevel
    risk_score: float         # 0-100
    confidence: float         # 0-1, how reliable is this assessment?
    rating_trust: RatingTrust
    
    # Per-aspect breakdown
    aspects: List[AspectData]
    
    # Explainability (WHY is this flagged?)
    risk_drivers: List[str]
    
    # Clear action
    recommended_action: ActionType
    
    # Metadata
    review_count: int
    assessment_date: str


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: Optional[str] = None
