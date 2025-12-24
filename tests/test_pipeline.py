"""
test_pipeline.py - Integration Tests for Guest Review Intelligence Pipeline

These tests verify:
1. End-to-end pipeline functionality
2. Negation handling correctness
3. Temporal weighting behavior
4. Disagreement detection
5. Output structure validity
"""

import pytest
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.schemas import RawReview, Aspect
from src.preprocessing import preprocess_review, mark_negations, expand_contractions
from src.aspect_detection import detect_aspects_in_sentence
from src.sentiment_analysis import analyze_aspect_sentiment, compute_sentence_sentiment
from src.aggregation import compute_temporal_weight, aggregate_by_listing
from src.confidence import compute_disagreement, compute_confidence
from src.pipeline import run_pipeline, analyze_single_review
from tests.synthetic_data import generate_synthetic_dataset, generate_negation_test_reviews


class TestPreprocessing:
    """Tests for text preprocessing module."""
    
    def test_contraction_expansion(self):
        """Test that contractions are properly expanded."""
        assert "was not" in expand_contractions("wasn't")
        assert "is not" in expand_contractions("isn't")
        assert "do not" in expand_contractions("don't")
    
    def test_negation_marking(self):
        """Test that negations are properly marked."""
        result = mark_negations("the room was not clean at all")
        assert "_NEG" in result
        assert "clean_NEG" in result
    
    def test_sentence_splitting(self):
        """Test sentence splitting."""
        text = "Great location. Clean room. Would recommend!"
        result = preprocess_review(text)
        assert len(result.sentences) == 3
    
    def test_preserves_context(self):
        """Test that original text is preserved."""
        text = "The host wasn't helpful."
        result = preprocess_review(text)
        assert result.original == text


class TestAspectDetection:
    """Tests for aspect detection module."""
    
    def test_detects_cleanliness(self):
        """Test cleanliness aspect detection."""
        sentence = "the apartment was spotless and very clean"
        aspects = detect_aspects_in_sentence(sentence)
        aspect_names = [a.aspect for a in aspects]
        assert Aspect.CLEANLINESS in aspect_names
    
    def test_detects_location(self):
        """Test location aspect detection."""
        sentence = "great location near the metro station"
        aspects = detect_aspects_in_sentence(sentence)
        aspect_names = [a.aspect for a in aspects]
        assert Aspect.LOCATION in aspect_names
    
    def test_detects_host_behavior(self):
        """Test host behavior detection."""
        sentence = "the host was very responsive and helpful"
        aspects = detect_aspects_in_sentence(sentence)
        aspect_names = [a.aspect for a in aspects]
        assert Aspect.HOST_BEHAVIOR in aspect_names
    
    def test_multiple_aspects(self):
        """Test detection of multiple aspects in one sentence."""
        sentence = "clean apartment in a quiet neighborhood"
        aspects = detect_aspects_in_sentence(sentence)
        assert len(aspects) >= 2


class TestSentimentAnalysis:
    """Tests for sentiment analysis module."""
    
    def test_positive_sentiment(self):
        """Test positive sentiment scoring."""
        score, _ = compute_sentence_sentiment("amazing excellent wonderful")
        assert score > 0.7
    
    def test_negative_sentiment(self):
        """Test negative sentiment scoring."""
        score, _ = compute_sentence_sentiment("terrible awful disgusting")
        assert score < -0.7
    
    def test_negation_flips_sentiment(self):
        """Test that negation markers flip sentiment."""
        # Without negation
        score_pos, _ = compute_sentence_sentiment("clean")
        # With negation
        score_neg, _ = compute_sentence_sentiment("clean_NEG")
        
        assert score_pos > 0
        assert score_neg < 0


class TestTemporalWeighting:
    """Tests for temporal weighting."""
    
    def test_recent_weight_higher(self):
        """Test that recent reviews have higher weight."""
        now = datetime.now()
        recent = now - timedelta(days=7)
        old = now - timedelta(days=365)
        
        weight_recent = compute_temporal_weight(recent, now)
        weight_old = compute_temporal_weight(old, now)
        
        assert weight_recent > weight_old
    
    def test_today_weight_is_one(self):
        """Test that today's weight is 1.0."""
        now = datetime.now()
        weight = compute_temporal_weight(now, now)
        assert weight == pytest.approx(1.0)
    
    def test_half_life_correct(self):
        """Test that weight is 0.5 at half-life."""
        now = datetime.now()
        half_life = now - timedelta(days=180)
        weight = compute_temporal_weight(half_life, now, half_life_days=180)
        assert weight == pytest.approx(0.5, abs=0.01)


class TestConfidence:
    """Tests for confidence and disagreement scoring."""
    
    def test_low_disagreement_consensus(self):
        """Test low disagreement when all agree."""
        scores = [0.8, 0.7, 0.75, 0.85, 0.72]  # All positive
        disagreement = compute_disagreement(scores)
        assert disagreement < 0.3
    
    def test_high_disagreement_polarized(self):
        """Test high disagreement when polarized."""
        scores = [0.9, 0.8, -0.8, -0.9, 0.85, -0.75]  # Mixed
        disagreement = compute_disagreement(scores)
        assert disagreement > 0.4
    
    def test_confidence_increases_with_volume(self):
        """Test that more mentions increase confidence."""
        conf_low = compute_confidence(mention_count=2, variance=0.1)
        conf_high = compute_confidence(mention_count=20, variance=0.1)
        assert conf_high > conf_low


class TestPipelineIntegration:
    """Integration tests for full pipeline."""
    
    def test_pipeline_runs_successfully(self):
        """Test that pipeline completes without error."""
        reviews = generate_synthetic_dataset(n_listings=2, reviews_per_listing=10)
        results = run_pipeline(reviews)
        assert len(results) == 2
    
    def test_output_structure(self):
        """Test that output has correct structure."""
        reviews = generate_synthetic_dataset(n_listings=1, reviews_per_listing=5)
        results = run_pipeline(reviews)
        
        listing = list(results.values())[0]
        
        # Check required fields
        assert hasattr(listing, 'listing_id')
        assert hasattr(listing, 'aspect_aggregations')
        assert hasattr(listing, 'total_reviews')
        
        # Check aspects
        for aspect in Aspect:
            assert aspect.value in listing.aspect_aggregations
    
    def test_json_serialization(self):
        """Test that output can be serialized to JSON."""
        reviews = generate_synthetic_dataset(n_listings=1, reviews_per_listing=5)
        results = run_pipeline(reviews)
        
        listing = list(results.values())[0]
        json_str = listing.to_json()
        
        # Should not raise
        parsed = json.loads(json_str)
        assert 'listing_id' in parsed


class TestNegationHandling:
    """Specific tests for negation handling."""
    
    def test_not_clean_is_negative(self):
        """Test 'not clean' produces negative sentiment for cleanliness."""
        review = RawReview(
            review_id="test_neg_1",
            listing_id="test",
            reviewer_name="Tester",
            review_date=datetime.now(),
            review_text="The room was not clean at all."
        )
        
        sentences = analyze_single_review(review)
        
        for sent in sentences:
            for aspect in sent.detected_aspects:
                if aspect.aspect == Aspect.CLEANLINESS:
                    assert aspect.sentiment_score < 0
    
    def test_wasnt_helpful_is_negative(self):
        """Test 'wasn't helpful' produces negative sentiment for host."""
        review = RawReview(
            review_id="test_neg_2",
            listing_id="test",
            reviewer_name="Tester",
            review_date=datetime.now(),
            review_text="The host wasn't helpful at all."
        )
        
        sentences = analyze_single_review(review)
        
        for sent in sentences:
            for aspect in sent.detected_aspects:
                if aspect.aspect == Aspect.HOST_BEHAVIOR:
                    assert aspect.sentiment_score < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
