"""
test_risk_pipeline.py - Tests for Phase-2 Risk Assessment

Tests cover:
1. Risk scoring calculations
2. Contradiction detection
3. Rating lag detection
4. Action mapping
5. Override rules
"""

import pytest
from datetime import datetime
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import (
    Aspect, TrendDirection, AspectAggregation, ListingIntelligence
)
from src.risk_schemas import (
    RiskLevel, ActionType, FlagType,
    score_to_risk_level, risk_level_to_action
)
from src.risk_scoring import (
    normalize_sentiment_to_risk, compute_aspect_risk, compute_overall_risk
)
from src.contradiction_detection import (
    detect_high_variance, detect_polarization, detect_safety_concerns
)
from src.rating_lag import sentiment_to_expected_rating, detect_rating_mismatch
from src.action_mapper import (
    score_to_action, upgrade_action, apply_safety_override
)
from src.risk_pipeline import assess_listing_risk


# =============================================================================
# HELPERS
# =============================================================================

def create_test_aggregation(
    aspect: Aspect,
    sentiment: float = 0.5,
    variance: float = 0.1,
    disagreement: float = 0.1,
    confidence: float = 0.8,
    trend: TrendDirection = TrendDirection.STABLE,
    mentions: int = 10
) -> AspectAggregation:
    """Create a test AspectAggregation."""
    return AspectAggregation(
        aspect=aspect,
        weighted_sentiment=sentiment,
        raw_sentiment_mean=sentiment,
        sentiment_variance=variance,
        disagreement_score=disagreement,
        confidence_score=confidence,
        mention_count=mentions,
        recent_trend=trend
    )


def create_test_intelligence(
    aspects: dict = None,
    total_reviews: int = 20
) -> ListingIntelligence:
    """Create a test ListingIntelligence."""
    now = datetime.now()
    
    if aspects is None:
        aspects = {
            'cleanliness': create_test_aggregation(Aspect.CLEANLINESS),
            'safety': create_test_aggregation(Aspect.SAFETY),
        }
    
    return ListingIntelligence(
        listing_id="test_listing",
        analysis_timestamp=now,
        aspect_aggregations=aspects,
        total_reviews=total_reviews,
        total_sentences=total_reviews * 3,
        date_range_start=now,
        date_range_end=now
    )


# =============================================================================
# RISK SCORING TESTS
# =============================================================================

class TestRiskScoring:
    """Tests for risk scoring module."""
    
    def test_sentiment_normalization(self):
        """Test sentiment to risk conversion."""
        # Positive sentiment → low risk
        assert normalize_sentiment_to_risk(1.0) == 0.0
        # Negative sentiment → high risk
        assert normalize_sentiment_to_risk(-1.0) == 1.0
        # Neutral → medium risk
        assert normalize_sentiment_to_risk(0.0) == 0.5
    
    def test_positive_sentiment_low_risk(self):
        """Test that positive sentiment produces low risk."""
        agg = create_test_aggregation(
            Aspect.CLEANLINESS,
            sentiment=0.8,
            variance=0.05,
            confidence=0.9
        )
        risk, drivers = compute_aspect_risk(Aspect.CLEANLINESS, agg)
        assert risk.risk_score < 30  # Should be low risk
    
    def test_negative_sentiment_high_risk(self):
        """Test that negative sentiment produces high risk."""
        agg = create_test_aggregation(
            Aspect.CLEANLINESS,
            sentiment=-0.6,
            variance=0.1,
            confidence=0.9
        )
        risk, drivers = compute_aspect_risk(Aspect.CLEANLINESS, agg)
        assert risk.risk_score > 50  # Should be higher risk
    
    def test_declining_trend_adds_risk(self):
        """Test that declining trend increases risk."""
        stable_agg = create_test_aggregation(
            Aspect.CLEANLINESS,
            sentiment=0.3,
            trend=TrendDirection.STABLE
        )
        declining_agg = create_test_aggregation(
            Aspect.CLEANLINESS,
            sentiment=0.3,
            trend=TrendDirection.DECLINING
        )
        
        stable_risk, _ = compute_aspect_risk(Aspect.CLEANLINESS, stable_agg)
        declining_risk, _ = compute_aspect_risk(Aspect.CLEANLINESS, declining_agg)
        
        assert declining_risk.risk_score > stable_risk.risk_score
    
    def test_low_confidence_reduces_risk(self):
        """Test that low confidence discounts risk score."""
        high_conf = create_test_aggregation(
            Aspect.CLEANLINESS,
            sentiment=-0.5,
            confidence=0.9
        )
        low_conf = create_test_aggregation(
            Aspect.CLEANLINESS,
            sentiment=-0.5,
            confidence=0.3
        )
        
        high_risk, _ = compute_aspect_risk(Aspect.CLEANLINESS, high_conf)
        low_risk, _ = compute_aspect_risk(Aspect.CLEANLINESS, low_conf)
        
        assert low_risk.risk_score < high_risk.risk_score


# =============================================================================
# CONTRADICTION DETECTION TESTS
# =============================================================================

class TestContradictionDetection:
    """Tests for contradiction detection."""
    
    def test_high_variance_detection(self):
        """Test high variance flag generation."""
        intelligence = create_test_intelligence({
            'cleanliness': create_test_aggregation(
                Aspect.CLEANLINESS,
                variance=0.5,  # High variance
                mentions=10
            )
        })
        
        flags, drivers = detect_high_variance(intelligence)
        assert FlagType.HIGH_VARIANCE in flags
    
    def test_polarization_detection(self):
        """Test polarized reviews flag generation."""
        intelligence = create_test_intelligence({
            'noise': create_test_aggregation(
                Aspect.NOISE,
                disagreement=0.7,  # High disagreement
                mentions=10
            )
        })
        
        flags, drivers = detect_polarization(intelligence)
        assert FlagType.POLARIZED in flags
    
    def test_safety_concern_detection(self):
        """Test safety concern flag for negative safety sentiment."""
        intelligence = create_test_intelligence({
            'safety': create_test_aggregation(
                Aspect.SAFETY,
                sentiment=-0.5,  # Negative safety
                mentions=5
            )
        })
        
        flags, drivers = detect_safety_concerns(intelligence)
        assert FlagType.SAFETY_CONCERN in flags


# =============================================================================
# RATING LAG TESTS
# =============================================================================

class TestRatingLag:
    """Tests for rating lag detection."""
    
    def test_sentiment_to_rating_mapping(self):
        """Test sentiment to rating conversion."""
        # Very positive sentiment → high rating
        assert sentiment_to_expected_rating(0.9) >= 4.5
        # Very negative sentiment → low rating
        assert sentiment_to_expected_rating(-0.9) <= 1.5
        # Neutral → middle rating
        assert 3.0 <= sentiment_to_expected_rating(0.0) <= 3.5
    
    def test_rating_mismatch_detection(self):
        """Test detection of rating lag."""
        intelligence = create_test_intelligence({
            'cleanliness': create_test_aggregation(
                Aspect.CLEANLINESS,
                sentiment=-0.5  # Negative
            ),
            'noise': create_test_aggregation(
                Aspect.NOISE,
                sentiment=-0.4  # Negative
            )
        })
        
        # Rating much higher than sentiment suggests
        flag, driver, meta = detect_rating_mismatch(intelligence, actual_rating=4.8)
        
        assert flag == FlagType.RATING_LAG


# =============================================================================
# ACTION MAPPING TESTS  
# =============================================================================

class TestActionMapping:
    """Tests for action mapping."""
    
    def test_score_thresholds(self):
        """Test that score thresholds map to correct actions."""
        assert score_to_action(20) == ActionType.IGNORE
        assert score_to_action(40) == ActionType.MONITOR
        assert score_to_action(60) == ActionType.FLAG
        assert score_to_action(80) == ActionType.URGENT
    
    def test_action_upgrade(self):
        """Test action upgrade function."""
        assert upgrade_action(ActionType.IGNORE) == ActionType.MONITOR
        assert upgrade_action(ActionType.MONITOR) == ActionType.FLAG
        assert upgrade_action(ActionType.FLAG) == ActionType.URGENT
        assert upgrade_action(ActionType.URGENT) == ActionType.URGENT  # Can't go higher
    
    def test_safety_override(self):
        """Test that high safety risk triggers URGENT."""
        from src.risk_schemas import AspectRisk
        
        aspect_risks = {
            'safety': AspectRisk(
                aspect=Aspect.SAFETY,
                risk_score=75,  # Above threshold
                risk_level=RiskLevel.CRITICAL,
                drivers=['negative_sentiment'],
                sentiment_contribution=50,
                variance_contribution=15,
                trend_contribution=10
            )
        }
        
        action, reasons = apply_safety_override(aspect_risks, ActionType.MONITOR)
        assert action == ActionType.URGENT


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestRiskPipeline:
    """Integration tests for full risk pipeline."""
    
    def test_pipeline_runs_successfully(self):
        """Test that pipeline completes without error."""
        intelligence = create_test_intelligence()
        assessment = assess_listing_risk(intelligence)
        
        assert assessment is not None
        assert assessment.listing_id == "test_listing"
    
    def test_output_structure(self):
        """Test that output has correct structure."""
        intelligence = create_test_intelligence()
        assessment = assess_listing_risk(intelligence)
        
        assert hasattr(assessment, 'overall_risk_score')
        assert hasattr(assessment, 'recommended_action')
        assert hasattr(assessment, 'aspect_risks')
        assert hasattr(assessment, 'flags')
        assert hasattr(assessment, 'risk_drivers')
    
    def test_json_serialization(self):
        """Test that output can be serialized to JSON."""
        intelligence = create_test_intelligence()
        assessment = assess_listing_risk(intelligence)
        
        json_str = assessment.to_json()
        import json
        parsed = json.loads(json_str)
        
        assert 'listing_id' in parsed
        assert 'overall_risk_score' in parsed
        assert 'recommended_action' in parsed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
