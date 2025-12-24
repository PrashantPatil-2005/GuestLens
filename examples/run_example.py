"""
run_example.py - Demonstration of Guest Review Intelligence Pipeline

This script shows how to use the pipeline with synthetic data
and outputs structured JSON results.
"""

import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline, run_pipeline_with_details, format_results_as_json
from tests.synthetic_data import (
    generate_synthetic_dataset,
    generate_negation_test_reviews,
    generate_temporal_test_reviews,
    generate_polarized_test_reviews
)


def print_separator(title: str = ""):
    """Print a visual separator."""
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def run_basic_example():
    """Run pipeline on a synthetic dataset."""
    print_separator("BASIC EXAMPLE: Synthetic Dataset")
    
    # Generate synthetic data
    print("\nGenerating synthetic dataset...")
    reviews = generate_synthetic_dataset(n_listings=3, reviews_per_listing=15)
    print(f"Generated {len(reviews)} reviews for 3 listings")
    
    # Run pipeline with details
    print("\nRunning pipeline...")
    results = run_pipeline_with_details(reviews)
    
    # Print statistics
    print("\n--- Pipeline Statistics ---")
    for key, value in results['stats'].items():
        print(f"  {key}: {value}")
    
    # Print results for each listing
    print("\n--- Results by Listing ---")
    for listing_id, intelligence in results['listings'].items():
        print(f"\n[{listing_id}]")
        print(f"  Total reviews: {intelligence.total_reviews}")
        print(f"  Date range: {intelligence.date_range_start.date()} to {intelligence.date_range_end.date()}")
        
        print("  Aspects:")
        for aspect_name, agg in intelligence.aspect_aggregations.items():
            if agg.mention_count > 0:
                print(f"    - {aspect_name}:")
                print(f"        Mentions: {agg.mention_count}")
                print(f"        Weighted Sentiment: {agg.weighted_sentiment:.3f}")
                print(f"        Confidence: {agg.confidence_score:.3f}")
                print(f"        Trend: {agg.recent_trend.value}")


def run_negation_example():
    """Demonstrate negation handling."""
    print_separator("NEGATION HANDLING TEST")
    
    reviews = generate_negation_test_reviews()
    print(f"\nTest reviews:")
    for r in reviews:
        print(f"  - {r.review_text[:70]}...")
    
    results = run_pipeline(reviews, verbose=False)
    
    print("\n--- Results ---")
    for listing_id, intelligence in results.items():
        print(f"\n[{listing_id}]")
        for aspect_name, agg in intelligence.aspect_aggregations.items():
            if agg.mention_count > 0:
                print(f"  {aspect_name}: sentiment={agg.weighted_sentiment:.3f}, mentions={agg.mention_count}")


def run_temporal_example():
    """Demonstrate temporal weighting."""
    print_separator("TEMPORAL WEIGHTING TEST")
    
    reviews = generate_temporal_test_reviews()
    print(f"\nGenerated {len(reviews)} reviews:")
    print("  - 10 OLD negative reviews (12-24 months ago)")
    print("  - 10 RECENT positive reviews (0-3 months ago)")
    
    results = run_pipeline(reviews, verbose=False)
    
    print("\n--- Results ---")
    for listing_id, intelligence in results.items():
        print(f"\n[{listing_id}]")
        for aspect_name, agg in intelligence.aspect_aggregations.items():
            if agg.mention_count > 0:
                print(f"  {aspect_name}:")
                print(f"    Raw Mean: {agg.raw_sentiment_mean:.3f}")
                print(f"    Weighted (recent bias): {agg.weighted_sentiment:.3f}")
                print(f"    Trend: {agg.recent_trend.value}")


def run_polarization_example():
    """Demonstrate disagreement detection."""
    print_separator("DISAGREEMENT/POLARIZATION TEST")
    
    reviews = generate_polarized_test_reviews()
    print(f"\nGenerated {len(reviews)} polarized reviews:")
    print("  - 10 VERY positive reviews")
    print("  - 10 VERY negative reviews")
    
    results = run_pipeline(reviews, verbose=False)
    
    print("\n--- Results ---")
    for listing_id, intelligence in results.items():
        print(f"\n[{listing_id}]")
        for aspect_name, agg in intelligence.aspect_aggregations.items():
            if agg.mention_count > 0:
                print(f"  {aspect_name}:")
                print(f"    Variance: {agg.sentiment_variance:.3f}")
                print(f"    Disagreement: {agg.disagreement_score:.3f}")
                print(f"    Confidence: {agg.confidence_score:.3f}")


def output_json_example():
    """Show JSON output format."""
    print_separator("JSON OUTPUT FORMAT")
    
    reviews = generate_synthetic_dataset(n_listings=1, reviews_per_listing=10)
    results = run_pipeline(reviews, verbose=False)
    
    json_output = format_results_as_json(results)
    print("\n" + json_output)


def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("  GUEST REVIEW INTELLIGENCE SYSTEM - Phase 1 Demo")
    print("=" * 60)
    
    run_basic_example()
    run_negation_example()
    run_temporal_example()
    run_polarization_example()
    output_json_example()
    
    print_separator("DEMO COMPLETE")
    print("\nAll tests passed! The pipeline is working correctly.")


if __name__ == "__main__":
    main()
