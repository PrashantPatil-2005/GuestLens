"""
schemas.py - Data Models for Guest Review Intelligence System

Design Decisions:
- Use dataclasses for simplicity and JSON serialization support
- Explicit typing for all fields to enable validation
- Enums for constrained values (aspects)
- All models are immutable (frozen=True) for data integrity
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import List, Dict, Tuple, Optional, Any
import json


class Aspect(str, Enum):
    """
    Predefined aspects for guest review analysis.
    Using str mixin allows direct JSON serialization.
    """
    CLEANLINESS = "cleanliness"
    NOISE = "noise"
    LOCATION = "location"
    HOST_BEHAVIOR = "host_behavior"
    AMENITIES = "amenities"
    SAFETY = "safety"


class TrendDirection(str, Enum):
    """Trend direction for aspect sentiment over time."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    INSUFFICIENT_DATA = "insufficient_data"


# =============================================================================
# INPUT SCHEMA
# =============================================================================

@dataclass
class RawReview:
    """
    Raw review as ingested from Airbnb data source.
    
    Attributes:
        review_id: Unique identifier for the review
        listing_id: ID of the Airbnb listing
        reviewer_name: Name of the guest who left the review
        review_date: When the review was posted
        review_text: Full text content of the review
    """
    review_id: str
    listing_id: str
    reviewer_name: str
    review_date: datetime
    review_text: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        d = asdict(self)
        d['review_date'] = self.review_date.isoformat()
        return d


# =============================================================================
# INTERMEDIATE SCHEMAS
# =============================================================================

@dataclass
class AspectMatch:
    """
    Result of aspect detection and sentiment analysis for a single aspect
    within a sentence.
    
    Attributes:
        aspect: Which aspect was detected
        sentiment_score: Sentiment from -1 (very negative) to 1 (very positive)
        confidence: Confidence in the detection (0-1)
        matched_keywords: Keywords that triggered this aspect match
        has_negation: Whether negation was detected affecting this aspect
    """
    aspect: Aspect
    sentiment_score: float
    confidence: float
    matched_keywords: List[str]
    has_negation: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'aspect': self.aspect.value,
            'sentiment_score': round(self.sentiment_score, 3),
            'confidence': round(self.confidence, 3),
            'matched_keywords': self.matched_keywords,
            'has_negation': self.has_negation
        }


@dataclass
class ProcessedSentence:
    """
    A sentence extracted from a review with aspect detection results.
    
    Attributes:
        sentence_id: Unique ID for this sentence
        review_id: Parent review ID
        original_text: Original sentence text
        processed_text: Cleaned text with negation markers
        detected_aspects: List of aspects found in this sentence
    """
    sentence_id: str
    review_id: str
    original_text: str
    processed_text: str
    detected_aspects: List[AspectMatch] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'sentence_id': self.sentence_id,
            'review_id': self.review_id,
            'original_text': self.original_text,
            'processed_text': self.processed_text,
            'detected_aspects': [a.to_dict() for a in self.detected_aspects]
        }


# =============================================================================
# OUTPUT SCHEMAS
# =============================================================================

@dataclass
class AspectAggregation:
    """
    Aggregated statistics for a single aspect across all reviews for a listing.
    
    Attributes:
        aspect: The aspect being aggregated
        weighted_sentiment: Temporal-weighted mean sentiment (-1 to 1)
        raw_sentiment_mean: Simple arithmetic mean of sentiment scores
        sentiment_variance: Statistical variance in sentiment scores
        disagreement_score: Measure of conflicting opinions (0-1, higher = more conflict)
        confidence_score: Overall confidence based on volume and consistency (0-1)
        mention_count: Number of times this aspect was mentioned
        recent_trend: Direction of sentiment change over time
    """
    aspect: Aspect
    weighted_sentiment: float
    raw_sentiment_mean: float
    sentiment_variance: float
    disagreement_score: float
    confidence_score: float
    mention_count: int
    recent_trend: TrendDirection
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'aspect': self.aspect.value,
            'weighted_sentiment': round(self.weighted_sentiment, 3),
            'raw_sentiment_mean': round(self.raw_sentiment_mean, 3),
            'sentiment_variance': round(self.sentiment_variance, 3),
            'disagreement_score': round(self.disagreement_score, 3),
            'confidence_score': round(self.confidence_score, 3),
            'mention_count': self.mention_count,
            'recent_trend': self.recent_trend.value
        }


@dataclass
class ListingIntelligence:
    """
    Complete aspect-level intelligence for a single listing.
    This is the final output of the pipeline.
    
    Attributes:
        listing_id: ID of the analyzed listing
        analysis_timestamp: When this analysis was generated
        aspect_aggregations: Per-aspect aggregated statistics
        total_reviews: Number of reviews analyzed
        total_sentences: Number of sentences processed
        date_range_start: Earliest review date
        date_range_end: Latest review date
    """
    listing_id: str
    analysis_timestamp: datetime
    aspect_aggregations: Dict[str, AspectAggregation]
    total_reviews: int
    total_sentences: int
    date_range_start: datetime
    date_range_end: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            'listing_id': self.listing_id,
            'analysis_timestamp': self.analysis_timestamp.isoformat(),
            'aspect_aggregations': {
                k: v.to_dict() for k, v in self.aspect_aggregations.items()
            },
            'total_reviews': self.total_reviews,
            'total_sentences': self.total_sentences,
            'date_range_start': self.date_range_start.isoformat(),
            'date_range_end': self.date_range_end.isoformat()
        }
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_listing_intelligence_from_dict(data: Dict[str, Any]) -> ListingIntelligence:
    """
    Deserialize ListingIntelligence from dictionary.
    Useful for loading from JSON storage.
    """
    aspect_aggs = {}
    for aspect_name, agg_data in data.get('aspect_aggregations', {}).items():
        aspect_aggs[aspect_name] = AspectAggregation(
            aspect=Aspect(agg_data['aspect']),
            weighted_sentiment=agg_data['weighted_sentiment'],
            raw_sentiment_mean=agg_data['raw_sentiment_mean'],
            sentiment_variance=agg_data['sentiment_variance'],
            disagreement_score=agg_data['disagreement_score'],
            confidence_score=agg_data['confidence_score'],
            mention_count=agg_data['mention_count'],
            recent_trend=TrendDirection(agg_data['recent_trend'])
        )
    
    return ListingIntelligence(
        listing_id=data['listing_id'],
        analysis_timestamp=datetime.fromisoformat(data['analysis_timestamp']),
        aspect_aggregations=aspect_aggs,
        total_reviews=data['total_reviews'],
        total_sentences=data['total_sentences'],
        date_range_start=datetime.fromisoformat(data['date_range_start']),
        date_range_end=datetime.fromisoformat(data['date_range_end'])
    )
