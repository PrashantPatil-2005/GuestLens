"""
risk_schemas.py - Data Models for Phase-2 Risk Assessment

Design Decisions:
- Extends Phase-1 schemas without modifying them
- Clear separation between risk levels and actions
- Risk drivers provide explainability
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Dict, Optional, Any
import json

from src.schemas import Aspect


class RiskLevel(str, Enum):
    """
    Categorical risk level based on score thresholds.
    
    Thresholds:
    - LOW: 0-30
    - MODERATE: 31-50
    - HIGH: 51-70  
    - CRITICAL: 71-100
    """
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class ActionType(str, Enum):
    """
    Recommended action based on risk assessment.
    Maps 1:1 with RiskLevel but semantically different.
    """
    IGNORE = "ignore"      # No action needed
    MONITOR = "monitor"    # Track for changes
    FLAG = "flag"          # Requires attention
    URGENT = "urgent"      # Immediate review needed


class DriverSeverity(str, Enum):
    """Severity level for risk drivers."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class FlagType(str, Enum):
    """Types of special flags detected."""
    HIGH_VARIANCE = "high_variance"
    POLARIZED = "polarized"
    DECLINING_TREND = "declining_trend"
    MULTI_ASPECT_DECLINE = "multi_aspect_decline"
    RATING_LAG = "rating_lag"
    LOW_CONFIDENCE = "low_confidence"
    SAFETY_CONCERN = "safety_concern"


# =============================================================================
# RISK DRIVER SCHEMA
# =============================================================================

@dataclass
class RiskDriver:
    """
    Explains why a listing or aspect is flagged.
    Provides explainability for risk scores.
    
    Attributes:
        aspect: Which aspect this driver relates to (None for overall)
        driver_type: What triggered this driver
        severity: How severe this risk signal is
        description: Human-readable explanation
        value: The numeric value that triggered this (optional)
    """
    aspect: Optional[str]
    driver_type: str
    severity: DriverSeverity
    description: str
    value: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'aspect': self.aspect,
            'driver_type': self.driver_type,
            'severity': self.severity.value,
            'description': self.description,
            'value': round(self.value, 3) if self.value is not None else None
        }


# =============================================================================
# ASPECT-LEVEL RISK
# =============================================================================

@dataclass
class AspectRisk:
    """
    Risk assessment for a single aspect.
    
    Attributes:
        aspect: The aspect being assessed
        risk_score: Numeric risk score (0-100)
        risk_level: Categorical risk level
        drivers: List of specific risk drivers for this aspect
        sentiment_contribution: How much sentiment contributed to risk
        variance_contribution: How much variance contributed
        trend_contribution: How much trend contributed
    """
    aspect: Aspect
    risk_score: float
    risk_level: RiskLevel
    drivers: List[str]
    sentiment_contribution: float
    variance_contribution: float
    trend_contribution: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'aspect': self.aspect.value,
            'risk_score': round(self.risk_score, 1),
            'risk_level': self.risk_level.value,
            'drivers': self.drivers,
            'contributions': {
                'sentiment': round(self.sentiment_contribution, 1),
                'variance': round(self.variance_contribution, 1),
                'trend': round(self.trend_contribution, 1)
            }
        }


# =============================================================================
# LISTING-LEVEL RISK ASSESSMENT
# =============================================================================

@dataclass
class ListingRiskAssessment:
    """
    Complete risk assessment for a listing.
    This is the final output of Phase-2 pipeline.
    
    Attributes:
        listing_id: ID of the assessed listing
        assessment_timestamp: When this assessment was generated
        overall_risk_score: Weighted aggregate risk score (0-100)
        risk_level: Overall categorical risk level
        recommended_action: What action should be taken
        aspect_risks: Per-aspect risk assessments
        flags: Special condition flags detected
        risk_drivers: Detailed explanations of risk factors
        metadata: Additional context (review count, date range, etc.)
    """
    listing_id: str
    assessment_timestamp: datetime
    overall_risk_score: float
    risk_level: RiskLevel
    recommended_action: ActionType
    aspect_risks: Dict[str, AspectRisk]
    flags: List[FlagType]
    risk_drivers: List[RiskDriver]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'listing_id': self.listing_id,
            'assessment_timestamp': self.assessment_timestamp.isoformat(),
            'overall_risk_score': round(self.overall_risk_score, 1),
            'risk_level': self.risk_level.value,
            'recommended_action': self.recommended_action.value,
            'aspect_risks': {k: v.to_dict() for k, v in self.aspect_risks.items()},
            'flags': [f.value for f in self.flags],
            'risk_drivers': [d.to_dict() for d in self.risk_drivers],
            'metadata': self.metadata
        }
    
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def score_to_risk_level(score: float) -> RiskLevel:
    """Convert numeric risk score to categorical level."""
    if score <= 30:
        return RiskLevel.LOW
    elif score <= 50:
        return RiskLevel.MODERATE
    elif score <= 70:
        return RiskLevel.HIGH
    else:
        return RiskLevel.CRITICAL


def risk_level_to_action(level: RiskLevel) -> ActionType:
    """Map risk level to recommended action."""
    mapping = {
        RiskLevel.LOW: ActionType.IGNORE,
        RiskLevel.MODERATE: ActionType.MONITOR,
        RiskLevel.HIGH: ActionType.FLAG,
        RiskLevel.CRITICAL: ActionType.URGENT
    }
    return mapping[level]
