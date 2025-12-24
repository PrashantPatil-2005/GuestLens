"""
risk_pipeline.py - Phase-2 Risk Assessment Pipeline

Main orchestrator that:
1. Consumes Phase-1 ListingIntelligence
2. Computes aspect and overall risk scores
3. Detects contradictions and anomalies
4. Optionally checks rating lag
5. Maps risk to recommended actions
6. Produces explainable ListingRiskAssessment
"""

from datetime import datetime
from typing import Dict, List, Optional, Tuple

from src.schemas import ListingIntelligence, Aspect
from src.risk_schemas import (
    ListingRiskAssessment, AspectRisk, RiskDriver, FlagType,
    RiskLevel, ActionType, DriverSeverity,
    score_to_risk_level
)
from src.risk_scoring import (
    compute_all_aspect_risks, compute_overall_risk
)
from src.contradiction_detection import detect_all_contradictions
from src.rating_lag import detect_rating_mismatch
from src.action_mapper import map_risk_to_action


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def assess_listing_risk(
    intelligence: ListingIntelligence,
    actual_rating: Optional[float] = None,
    assessment_time: Optional[datetime] = None,
    verbose: bool = False
) -> ListingRiskAssessment:
    """
    Run the complete Phase-2 risk assessment pipeline.
    
    Pipeline stages:
    1. Compute aspect-level risk scores
    2. Compute overall risk score (weighted average)
    3. Detect contradictions and anomalies
    4. Check rating lag (if rating provided)
    5. Map risk to recommended action
    6. Compile all into ListingRiskAssessment
    
    Args:
        intelligence: Phase-1 ListingIntelligence output
        actual_rating: Optional actual star rating (1-5) for rating lag detection
        assessment_time: Optional timestamp (default: now)
        verbose: If True, print progress
        
    Returns:
        ListingRiskAssessment with complete risk analysis
    """
    if assessment_time is None:
        assessment_time = datetime.now()
    
    if verbose:
        print(f"Assessing risk for listing: {intelligence.listing_id}")
    
    # Stage 1: Compute aspect risks
    if verbose:
        print("  Stage 1: Computing aspect risks...")
    aspect_risks, aspect_drivers = compute_all_aspect_risks(intelligence)
    
    # Stage 2: Compute overall risk
    if verbose:
        print("  Stage 2: Computing overall risk...")
    overall_risk_score = compute_overall_risk(aspect_risks)
    
    # Stage 3: Detect contradictions
    if verbose:
        print("  Stage 3: Detecting contradictions...")
    flags, contradiction_drivers = detect_all_contradictions(intelligence)
    
    # Combine all drivers
    all_drivers = aspect_drivers + contradiction_drivers
    
    # Stage 4: Rating lag detection (optional)
    metadata = {
        'total_reviews': intelligence.total_reviews,
        'total_sentences': intelligence.total_sentences,
        'date_range_start': intelligence.date_range_start.isoformat(),
        'date_range_end': intelligence.date_range_end.isoformat(),
    }
    
    if actual_rating is not None:
        if verbose:
            print("  Stage 4: Checking rating lag...")
        rating_flag, rating_driver, rating_meta = detect_rating_mismatch(
            intelligence, actual_rating
        )
        metadata.update(rating_meta)
        
        if rating_flag:
            flags.append(rating_flag)
        if rating_driver:
            all_drivers.append(rating_driver)
    
    # Stage 5: Map to action
    if verbose:
        print("  Stage 5: Mapping to action...")
    recommended_action, override_reasons = map_risk_to_action(
        overall_risk_score=overall_risk_score,
        aspect_risks=aspect_risks,
        flags=flags,
        intelligence=intelligence
    )
    
    if override_reasons:
        metadata['action_overrides'] = override_reasons
    
    # Compile assessment
    risk_level = score_to_risk_level(overall_risk_score)
    
    assessment = ListingRiskAssessment(
        listing_id=intelligence.listing_id,
        assessment_timestamp=assessment_time,
        overall_risk_score=overall_risk_score,
        risk_level=risk_level,
        recommended_action=recommended_action,
        aspect_risks=aspect_risks,
        flags=flags,
        risk_drivers=all_drivers,
        metadata=metadata
    )
    
    if verbose:
        print(f"  Complete. Risk: {overall_risk_score:.1f} ({risk_level.value}) → {recommended_action.value}")
    
    return assessment


def assess_listings_batch(
    intelligences: Dict[str, ListingIntelligence],
    ratings: Optional[Dict[str, float]] = None,
    verbose: bool = False
) -> Dict[str, ListingRiskAssessment]:
    """
    Assess risk for multiple listings.
    
    Args:
        intelligences: Dict mapping listing_id to ListingIntelligence
        ratings: Optional dict mapping listing_id to actual star rating
        verbose: If True, print progress
        
    Returns:
        Dict mapping listing_id to ListingRiskAssessment
    """
    if ratings is None:
        ratings = {}
    
    results = {}
    assessment_time = datetime.now()
    
    for listing_id, intelligence in intelligences.items():
        actual_rating = ratings.get(listing_id)
        results[listing_id] = assess_listing_risk(
            intelligence=intelligence,
            actual_rating=actual_rating,
            assessment_time=assessment_time,
            verbose=verbose
        )
    
    return results


# =============================================================================
# PRIORITY SORTING
# =============================================================================

def sort_by_risk(
    assessments: Dict[str, ListingRiskAssessment]
) -> List[Tuple[str, ListingRiskAssessment]]:
    """
    Sort assessments by risk score (highest first).
    
    Useful for prioritization dashboards.
    """
    return sorted(
        assessments.items(),
        key=lambda x: x[1].overall_risk_score,
        reverse=True
    )


def sort_by_action_priority(
    assessments: Dict[str, ListingRiskAssessment]
) -> List[Tuple[str, ListingRiskAssessment]]:
    """
    Sort assessments by action priority (most urgent first).
    
    Order: URGENT > FLAG > MONITOR > IGNORE
    Within same action, sort by risk score.
    """
    from src.action_mapper import get_action_priority
    
    return sorted(
        assessments.items(),
        key=lambda x: (
            get_action_priority(x[1].recommended_action),
            x[1].overall_risk_score
        ),
        reverse=True
    )


def get_urgent_listings(
    assessments: Dict[str, ListingRiskAssessment]
) -> List[str]:
    """Get listing IDs that require urgent action."""
    return [
        listing_id
        for listing_id, assessment in assessments.items()
        if assessment.recommended_action == ActionType.URGENT
    ]


def get_flagged_listings(
    assessments: Dict[str, ListingRiskAssessment]
) -> List[str]:
    """Get listing IDs that are flagged (not urgent but need attention)."""
    return [
        listing_id
        for listing_id, assessment in assessments.items()
        if assessment.recommended_action == ActionType.FLAG
    ]


# =============================================================================
# REPORTING UTILITIES
# =============================================================================

def summarize_assessment(assessment: ListingRiskAssessment) -> Dict:
    """
    Create a summary view of the assessment.
    
    Useful for quick review without full details.
    """
    # Get top risk drivers (high severity only)
    high_drivers = [
        d for d in assessment.risk_drivers
        if d.severity == DriverSeverity.HIGH
    ]
    
    # Get highest risk aspects
    sorted_aspects = sorted(
        assessment.aspect_risks.items(),
        key=lambda x: x[1].risk_score,
        reverse=True
    )
    top_risks = sorted_aspects[:3]
    
    return {
        'listing_id': assessment.listing_id,
        'risk_score': round(assessment.overall_risk_score, 1),
        'action': assessment.recommended_action.value,
        'flags': [f.value for f in assessment.flags],
        'top_risks': [
            {'aspect': name, 'score': round(risk.risk_score, 1)}
            for name, risk in top_risks
        ],
        'critical_drivers': [d.description for d in high_drivers[:3]]
    }


def format_assessment_report(assessment: ListingRiskAssessment) -> str:
    """
    Format assessment as human-readable report.
    """
    lines = [
        f"=" * 60,
        f"RISK ASSESSMENT: {assessment.listing_id}",
        f"=" * 60,
        f"",
        f"Overall Risk Score: {assessment.overall_risk_score:.1f}/100 ({assessment.risk_level.value})",
        f"Recommended Action: {assessment.recommended_action.value.upper()}",
        f"",
    ]
    
    if assessment.flags:
        lines.append(f"Flags: {', '.join(f.value for f in assessment.flags)}")
        lines.append("")
    
    lines.append("Aspect Risks:")
    for name, risk in sorted(
        assessment.aspect_risks.items(),
        key=lambda x: x[1].risk_score,
        reverse=True
    ):
        if risk.risk_score > 0:
            lines.append(f"  - {name}: {risk.risk_score:.1f} ({risk.risk_level.value})")
            if risk.drivers:
                lines.append(f"      Drivers: {', '.join(risk.drivers)}")
    
    lines.append("")
    
    if assessment.risk_drivers:
        lines.append("Key Risk Drivers:")
        for driver in assessment.risk_drivers:
            if driver.severity in [DriverSeverity.HIGH, DriverSeverity.MEDIUM]:
                aspect_prefix = f"[{driver.aspect}] " if driver.aspect else ""
                lines.append(f"  - {aspect_prefix}{driver.description}")
    
    lines.append("")
    lines.append(f"Assessment Time: {assessment.assessment_timestamp.isoformat()}")
    lines.append(f"Reviews Analyzed: {assessment.metadata.get('total_reviews', 'N/A')}")
    
    return "\n".join(lines)
