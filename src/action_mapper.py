"""
action_mapper.py - Map Risk Signals to Actions

Converts risk scores and flags into actionable recommendations:
- IGNORE: No action needed
- MONITOR: Track for changes
- FLAG: Requires attention
- URGENT: Immediate review needed

Special override rules ensure critical issues aren't missed.
"""

from typing import Dict, List, Tuple

from src.schemas import ListingIntelligence
from src.risk_schemas import (
    RiskLevel, ActionType, FlagType, AspectRisk,
    RiskDriver, DriverSeverity,
    score_to_risk_level, risk_level_to_action
)


# =============================================================================
# CONFIGURATION
# =============================================================================

# Base thresholds for action mapping (from risk score)
ACTION_THRESHOLDS = {
    (0, 30): ActionType.IGNORE,
    (31, 50): ActionType.MONITOR,
    (51, 70): ActionType.FLAG,
    (71, 100): ActionType.URGENT,
}

# Safety risk threshold for automatic URGENT upgrade
SAFETY_URGENT_THRESHOLD = 60

# Flags that upgrade action by one level
UPGRADING_FLAGS = {
    FlagType.SAFETY_CONCERN: 2,       # Upgrade by 2 levels
    FlagType.MULTI_ASPECT_DECLINE: 1,  # Upgrade by 1 level
    FlagType.POLARIZED: 1,            # Upgrade by 1 level (if high)
}

# Minimum reviews to trust action recommendations
MIN_REVIEWS_FOR_ACTION = 3


# =============================================================================
# ACTION MAPPING
# =============================================================================

def score_to_action(score: float) -> ActionType:
    """Convert risk score to action type."""
    for (low, high), action in ACTION_THRESHOLDS.items():
        if low <= score <= high:
            return action
    return ActionType.URGENT if score > 70 else ActionType.IGNORE


def upgrade_action(current: ActionType, levels: int = 1) -> ActionType:
    """
    Upgrade action severity by N levels.
    
    IGNORE → MONITOR → FLAG → URGENT
    """
    order = [ActionType.IGNORE, ActionType.MONITOR, ActionType.FLAG, ActionType.URGENT]
    current_idx = order.index(current)
    new_idx = min(len(order) - 1, current_idx + levels)
    return order[new_idx]


def downgrade_action(current: ActionType, levels: int = 1) -> ActionType:
    """
    Downgrade action severity by N levels.
    
    URGENT → FLAG → MONITOR → IGNORE
    """
    order = [ActionType.IGNORE, ActionType.MONITOR, ActionType.FLAG, ActionType.URGENT]
    current_idx = order.index(current)
    new_idx = max(0, current_idx - levels)
    return order[new_idx]


# =============================================================================
# OVERRIDE RULES
# =============================================================================

def apply_safety_override(
    aspect_risks: Dict[str, AspectRisk],
    current_action: ActionType
) -> Tuple[ActionType, List[str]]:
    """
    Check if safety risk triggers URGENT override.
    
    Rule: If safety risk > SAFETY_URGENT_THRESHOLD, upgrade to URGENT
    
    Returns:
        Tuple of (possibly upgraded action, list of override reasons)
    """
    reasons = []
    
    safety_risk = aspect_risks.get('safety')
    if safety_risk and safety_risk.risk_score > SAFETY_URGENT_THRESHOLD:
        if current_action != ActionType.URGENT:
            reasons.append(f"Safety risk ({safety_risk.risk_score:.0f}) exceeds threshold")
            return ActionType.URGENT, reasons
    
    return current_action, reasons


def apply_flag_overrides(
    flags: List[FlagType],
    current_action: ActionType
) -> Tuple[ActionType, List[str]]:
    """
    Apply action upgrades based on detected flags.
    
    Returns:
        Tuple of (possibly upgraded action, list of override reasons)
    """
    reasons = []
    action = current_action
    
    for flag in flags:
        if flag in UPGRADING_FLAGS:
            upgrade_levels = UPGRADING_FLAGS[flag]
            new_action = upgrade_action(action, upgrade_levels)
            if new_action != action:
                reasons.append(f"{flag.value} flag triggered upgrade")
                action = new_action
    
    return action, reasons


def apply_confidence_discount(
    intelligence: ListingIntelligence,
    current_action: ActionType
) -> Tuple[ActionType, List[str]]:
    """
    Downgrade action if confidence is too low.
    
    Rule: If very few reviews, reduce action severity
    (we're not confident enough to flag urgently)
    
    Returns:
        Tuple of (possibly downgraded action, list of reasons)
    """
    reasons = []
    
    if intelligence.total_reviews < MIN_REVIEWS_FOR_ACTION:
        if current_action in [ActionType.FLAG, ActionType.URGENT]:
            reasons.append(f"Low confidence ({intelligence.total_reviews} reviews)")
            return downgrade_action(current_action), reasons
    
    return current_action, reasons


# =============================================================================
# MAIN MAPPING FUNCTION
# =============================================================================

def map_risk_to_action(
    overall_risk_score: float,
    aspect_risks: Dict[str, AspectRisk],
    flags: List[FlagType],
    intelligence: ListingIntelligence,
    apply_overrides: bool = True
) -> Tuple[ActionType, List[str]]:
    """
    Map risk assessment to recommended action.
    
    Process:
    1. Convert overall risk score to base action
    2. Apply safety override if needed
    3. Apply flag-based upgrades
    4. Apply confidence discount if needed
    
    Args:
        overall_risk_score: The overall risk score (0-100)
        aspect_risks: Per-aspect risk assessments
        flags: Detected flags
        intelligence: Phase-1 intelligence for context
        apply_overrides: Whether to apply override rules
        
    Returns:
        Tuple of (final action, list of reasons for any overrides)
    """
    # Step 1: Base action from score
    action = score_to_action(overall_risk_score)
    override_reasons = []
    
    if not apply_overrides:
        return action, override_reasons
    
    # Step 2: Safety override
    action, reasons = apply_safety_override(aspect_risks, action)
    override_reasons.extend(reasons)
    
    # Step 3: Flag upgrades
    action, reasons = apply_flag_overrides(flags, action)
    override_reasons.extend(reasons)
    
    # Step 4: Confidence discount (can downgrade)
    action, reasons = apply_confidence_discount(intelligence, action)
    override_reasons.extend(reasons)
    
    return action, override_reasons


# =============================================================================
# ACTION EXPLANATION
# =============================================================================

def explain_action(
    action: ActionType,
    risk_score: float,
    flags: List[FlagType],
    override_reasons: List[str]
) -> str:
    """
    Generate human-readable explanation of the recommended action.
    """
    explanations = {
        ActionType.IGNORE: "No action needed. Listing is performing well.",
        ActionType.MONITOR: "Monitor for changes. Some areas could improve.",
        ActionType.FLAG: "Attention needed. Significant issues detected.",
        ActionType.URGENT: "Immediate review required. Critical issues present.",
    }
    
    base = explanations[action]
    
    details = [f"Risk score: {risk_score:.0f}/100"]
    
    if flags:
        details.append(f"Flags: {', '.join(f.value for f in flags)}")
    
    if override_reasons:
        details.append(f"Overrides: {'; '.join(override_reasons)}")
    
    return f"{base} ({'; '.join(details)})"


def get_action_priority(action: ActionType) -> int:
    """Get numeric priority for sorting (higher = more urgent)."""
    priorities = {
        ActionType.IGNORE: 1,
        ActionType.MONITOR: 2,
        ActionType.FLAG: 3,
        ActionType.URGENT: 4,
    }
    return priorities.get(action, 0)
