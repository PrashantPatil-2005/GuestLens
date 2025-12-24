"""
run_risk_example.py - Phase-2 Risk Assessment Demonstration

Shows the complete flow from Phase-1 intelligence to Phase-2 risk assessment.
"""

import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline
from src.risk_pipeline import (
    assess_listing_risk,
    assess_listings_batch,
    sort_by_risk,
    format_assessment_report,
    summarize_assessment
)
from tests.synthetic_data import (
    generate_synthetic_dataset,
    generate_temporal_test_reviews,
    generate_polarized_test_reviews
)


def print_separator(title: str = ""):
    print("\n" + "=" * 60)
    if title:
        print(f"  {title}")
        print("=" * 60)


def run_basic_risk_example():
    """Demonstrate basic Phase-1 → Phase-2 flow."""
    print_separator("BASIC RISK ASSESSMENT")
    
    # Generate Phase-1 data
    print("\n1. Generating synthetic reviews...")
    reviews = generate_synthetic_dataset(n_listings=3, reviews_per_listing=15)
    
    # Run Phase-1 pipeline
    print("2. Running Phase-1 pipeline...")
    phase1_results = run_pipeline(reviews, verbose=False)
    
    # Run Phase-2 pipeline
    print("3. Running Phase-2 risk assessment...")
    phase2_results = assess_listings_batch(phase1_results, verbose=False)
    
    # Display results
    print("\n--- Risk Assessment Results ---")
    
    for listing_id, assessment in sort_by_risk(phase2_results):
        summary = summarize_assessment(assessment)
        print(f"\n[{listing_id}]")
        print(f"  Risk Score: {summary['risk_score']}/100")
        print(f"  Action: {summary['action'].upper()}")
        if summary['flags']:
            print(f"  Flags: {', '.join(summary['flags'])}")
        if summary['top_risks']:
            print(f"  Top Risks: {', '.join(f'{r['aspect']}({r['score']})' for r in summary['top_risks'])}")


def run_detailed_report_example():
    """Show detailed report format."""
    print_separator("DETAILED RISK REPORT")
    
    # Generate one listing
    reviews = generate_synthetic_dataset(n_listings=1, reviews_per_listing=20)
    phase1_results = run_pipeline(reviews, verbose=False)
    
    listing_id = list(phase1_results.keys())[0]
    intelligence = phase1_results[listing_id]
    
    # Assess with optional rating
    assessment = assess_listing_risk(
        intelligence=intelligence,
        actual_rating=4.2,  # Simulated actual rating
        verbose=False
    )
    
    # Print detailed report
    print(format_assessment_report(assessment))


def run_high_risk_example():
    """Demonstrate detection of high-risk patterns."""
    print_separator("HIGH-RISK DETECTION (Polarized Reviews)")
    
    # Generate polarized data (half positive, half negative)
    reviews = generate_polarized_test_reviews()
    print(f"Generated {len(reviews)} polarized reviews (50% positive, 50% negative)")
    
    phase1_results = run_pipeline(reviews, verbose=False)
    
    for listing_id, intelligence in phase1_results.items():
        assessment = assess_listing_risk(intelligence)
        
        print(f"\n[{listing_id}]")
        print(f"  Risk Score: {assessment.overall_risk_score:.1f}")
        print(f"  Action: {assessment.recommended_action.value.upper()}")
        print(f"  Flags: {[f.value for f in assessment.flags]}")
        
        # Show key drivers
        high_drivers = [d for d in assessment.risk_drivers if d.severity.value == 'high']
        if high_drivers:
            print(f"  Critical Drivers:")
            for d in high_drivers[:3]:
                print(f"    - {d.description}")


def run_temporal_risk_example():
    """Demonstrate trend detection in risk assessment."""
    print_separator("TREND DETECTION (Improving Listing)")
    
    # Generate temporal pattern (old negative, recent positive)
    reviews = generate_temporal_test_reviews()
    print("Generated reviews: old=negative, recent=positive")
    
    phase1_results = run_pipeline(reviews, verbose=False)
    
    for listing_id, intelligence in phase1_results.items():
        assessment = assess_listing_risk(intelligence)
        
        print(f"\n[{listing_id}]")
        print(f"  Risk Score: {assessment.overall_risk_score:.1f}")
        print(f"  Action: {assessment.recommended_action.value}")
        
        # Show trends
        print("  Aspect Trends:")
        for aspect_name, agg in intelligence.aspect_aggregations.items():
            if agg.mention_count > 0:
                aspect_risk = assessment.aspect_risks.get(aspect_name)
                risk_str = f" (risk: {aspect_risk.risk_score:.1f})" if aspect_risk else ""
                print(f"    - {aspect_name}: {agg.recent_trend.value}{risk_str}")


def show_json_output():
    """Show Phase-2 JSON output format."""
    print_separator("PHASE-2 JSON OUTPUT")
    
    reviews = generate_synthetic_dataset(n_listings=1, reviews_per_listing=10)
    phase1_results = run_pipeline(reviews, verbose=False)
    
    listing_id = list(phase1_results.keys())[0]
    intelligence = phase1_results[listing_id]
    
    assessment = assess_listing_risk(intelligence)
    
    print("\n" + assessment.to_json())


def show_input_output_comparison():
    """Show Phase-1 input and Phase-2 output side by side."""
    print_separator("PHASE-1 INPUT → PHASE-2 OUTPUT")
    
    reviews = generate_synthetic_dataset(n_listings=1, reviews_per_listing=12)
    phase1_results = run_pipeline(reviews, verbose=False)
    
    listing_id = list(phase1_results.keys())[0]
    intelligence = phase1_results[listing_id]
    
    print("\n--- PHASE-1 INPUT (ListingIntelligence) ---")
    print(json.dumps({
        'listing_id': intelligence.listing_id,
        'total_reviews': intelligence.total_reviews,
        'aspects': {
            name: {
                'weighted_sentiment': round(agg.weighted_sentiment, 3),
                'variance': round(agg.sentiment_variance, 3),
                'disagreement': round(agg.disagreement_score, 3),
                'confidence': round(agg.confidence_score, 3),
                'trend': agg.recent_trend.value,
                'mentions': agg.mention_count
            }
            for name, agg in intelligence.aspect_aggregations.items()
            if agg.mention_count > 0
        }
    }, indent=2))
    
    assessment = assess_listing_risk(intelligence)
    
    print("\n--- PHASE-2 OUTPUT (ListingRiskAssessment) ---")
    print(json.dumps({
        'listing_id': assessment.listing_id,
        'overall_risk_score': round(assessment.overall_risk_score, 1),
        'risk_level': assessment.risk_level.value,
        'recommended_action': assessment.recommended_action.value,
        'flags': [f.value for f in assessment.flags],
        'aspect_risks': {
            name: {
                'risk_score': round(risk.risk_score, 1),
                'risk_level': risk.risk_level.value,
                'drivers': risk.drivers
            }
            for name, risk in assessment.aspect_risks.items()
            if risk.risk_score > 0
        }
    }, indent=2))


def main():
    print("\n" + "=" * 60)
    print("  PHASE-2: DECISION & RISK LAYER - Demo")
    print("=" * 60)
    
    run_basic_risk_example()
    run_high_risk_example()
    run_temporal_risk_example()
    run_detailed_report_example()
    show_input_output_comparison()
    show_json_output()
    
    print_separator("DEMO COMPLETE")
    print("\nPhase-2 Risk Layer is working correctly!")


if __name__ == "__main__":
    main()
