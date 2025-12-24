"""
aspect_detection.py - Rule-Based Aspect Detection

Design Decisions:
1. Keyword Lexicons: Each aspect has a curated list of keywords and phrases
   that strongly indicate that aspect is being discussed.

2. Context Windows: Instead of just matching keywords, we look at surrounding
   words to improve accuracy. "Location of the bathroom" shouldn't match location.

3. Weighted Keywords: Some keywords are stronger indicators than others.
   "Spotless" is a stronger cleanliness indicator than "dust".

4. Multi-word Phrases: Support for phrases like "walking distance" for location
   to capture more nuanced mentions.

5. Exclusion Patterns: Some words only indicate an aspect in certain contexts.
   Handle these with exclusion rules.
"""

import re
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass
from src.schemas import Aspect, AspectMatch


# =============================================================================
# ASPECT LEXICONS
# =============================================================================

# Each aspect maps to a dict of keyword -> weight (0-1)
# Higher weight = stronger indicator of the aspect

ASPECT_LEXICONS: Dict[Aspect, Dict[str, float]] = {
    Aspect.CLEANLINESS: {
        # Strong indicators (0.8-1.0)
        "spotless": 1.0,
        "immaculate": 1.0,
        "pristine": 1.0,
        "filthy": 1.0,
        "disgusting": 0.9,
        "cleanliness": 0.9,
        
        # Medium indicators (0.5-0.8)
        "clean": 0.8,
        "dirty": 0.8,
        "dust": 0.7,
        "dusty": 0.7,
        "stain": 0.7,
        "stains": 0.7,
        "stained": 0.7,
        "tidy": 0.7,
        "messy": 0.7,
        "hygiene": 0.7,
        "hygienic": 0.7,
        "sanitized": 0.7,
        "sanitary": 0.7,
        "mold": 0.8,
        "moldy": 0.8,
        "mould": 0.8,
        "smell": 0.6,
        "smelled": 0.6,
        "smells": 0.6,
        "odor": 0.7,
        "odour": 0.7,
        
        # Weak indicators (0.3-0.5)
        "fresh": 0.4,
        "sheets": 0.4,
        "towels": 0.4,
        "bathroom": 0.3,
        "toilet": 0.4,
        "hair": 0.5,
        "bugs": 0.6,
        "insects": 0.6,
        "cockroach": 0.8,
        "cockroaches": 0.8,
    },
    
    Aspect.NOISE: {
        # Strong indicators
        "quiet": 1.0,
        "noisy": 1.0,
        "noise": 0.9,
        "peaceful": 0.9,
        "silent": 0.8,
        "soundproof": 0.9,
        
        # Medium indicators
        "loud": 0.8,
        "sound": 0.6,
        "sounds": 0.6,
        "traffic": 0.7,
        "neighbors": 0.7,
        "neighbours": 0.7,
        "party": 0.6,
        "parties": 0.6,
        "music": 0.6,
        "barking": 0.7,
        "dogs": 0.5,
        "construction": 0.7,
        "earplugs": 0.9,
        "sleep": 0.4,
        "sleeping": 0.4,
        
        # Weak indicators
        "street": 0.4,
        "road": 0.3,
        "hear": 0.5,
        "heard": 0.5,
        "walls": 0.4,
        "thin": 0.3,
    },
    
    Aspect.LOCATION: {
        # Strong indicators
        "location": 1.0,
        "located": 0.9,
        "neighborhood": 0.9,
        "neighbourhood": 0.9,
        "central": 0.8,
        "downtown": 0.8,
        
        # Medium indicators
        "area": 0.7,
        "walking": 0.6,
        "distance": 0.5,
        "accessible": 0.7,
        "transport": 0.7,
        "transportation": 0.7,
        "subway": 0.7,
        "metro": 0.7,
        "bus": 0.5,
        "train": 0.5,
        "station": 0.5,
        "airport": 0.6,
        "beach": 0.5,
        "restaurants": 0.5,
        "shops": 0.5,
        "shopping": 0.5,
        "supermarket": 0.5,
        "grocery": 0.5,
        "convenient": 0.6,
        "conveniently": 0.6,
        
        # Weak indicators
        "minutes": 0.3,
        "walk": 0.4,
        "close": 0.4,
        "near": 0.4,
        "nearby": 0.5,
        "far": 0.4,
        "remote": 0.5,
    },
    
    Aspect.HOST_BEHAVIOR: {
        # Strong indicators
        "host": 1.0,
        "hosts": 1.0,
        "owner": 0.9,
        "responsive": 0.9,
        "unresponsive": 0.9,
        
        # Medium indicators
        "helpful": 0.7,
        "unhelpful": 0.8,
        "communication": 0.8,
        "communicate": 0.7,
        "communicated": 0.7,
        "response": 0.6,
        "responded": 0.6,
        "check-in": 0.8,
        "checkin": 0.8,
        "checkout": 0.7,
        "check-out": 0.7,
        "welcome": 0.6,
        "welcomed": 0.6,
        "welcoming": 0.6,
        "friendly": 0.6,
        "unfriendly": 0.7,
        "rude": 0.8,
        "kind": 0.5,
        "accommodating": 0.7,
        "flexible": 0.5,
        "inflexible": 0.6,
        
        # Weak indicators
        "message": 0.4,
        "messages": 0.4,
        "reply": 0.5,
        "replied": 0.5,
        "contact": 0.4,
        "instructions": 0.5,
        "tips": 0.4,
        "recommendations": 0.4,
    },
    
    Aspect.AMENITIES: {
        # Strong indicators
        "amenities": 1.0,
        "equipped": 0.8,
        "facilities": 0.8,
        
        # Specific amenities (medium to high)
        "wifi": 0.9,
        "wi-fi": 0.9,
        "internet": 0.8,
        "kitchen": 0.8,
        "bed": 0.7,
        "beds": 0.7,
        "mattress": 0.8,
        "towel": 0.6,
        "towels": 0.6,
        "parking": 0.8,
        "heating": 0.8,
        "heater": 0.7,
        "ac": 0.8,
        "air conditioning": 0.9,
        "aircon": 0.8,
        "bathroom": 0.6,
        "shower": 0.7,
        "bathtub": 0.7,
        "tv": 0.6,
        "television": 0.6,
        "washer": 0.7,
        "washing machine": 0.8,
        "dryer": 0.7,
        "dishwasher": 0.7,
        "fridge": 0.6,
        "refrigerator": 0.6,
        "microwave": 0.6,
        "stove": 0.6,
        "oven": 0.6,
        "coffee": 0.5,
        "kettle": 0.5,
        "utensils": 0.6,
        "cookware": 0.6,
        "linens": 0.6,
        "pillows": 0.6,
        "blanket": 0.5,
        "blankets": 0.5,
        "pool": 0.7,
        "gym": 0.6,
        "elevator": 0.6,
        "lift": 0.5,
        "balcony": 0.6,
        
        # Weak indicators
        "room": 0.3,
        "space": 0.3,
        "comfortable": 0.4,
        "cozy": 0.4,
    },
    
    Aspect.SAFETY: {
        # Strong indicators
        "safe": 0.9,
        "safety": 1.0,
        "secure": 0.9,
        "security": 0.9,
        "unsafe": 1.0,
        "dangerous": 1.0,
        "danger": 0.9,
        
        # Medium indicators
        "lock": 0.7,
        "locks": 0.7,
        "locked": 0.6,
        "alarm": 0.7,
        "camera": 0.6,
        "cameras": 0.6,
        "sketchy": 0.8,
        "shady": 0.7,
        "crime": 0.8,
        "theft": 0.8,
        "stolen": 0.8,
        "break-in": 0.9,
        "breakin": 0.9,
        
        # Trust/comfort related
        "trust": 0.5,
        "trusted": 0.5,
        "worry": 0.5,
        "worried": 0.5,
        "concerns": 0.5,
        "comfortable": 0.4,
        "uncomfortable": 0.5,
        
        # Weak indicators
        "night": 0.3,
        "dark": 0.4,
        "alone": 0.4,
        "stranger": 0.5,
        "strangers": 0.5,
    },
}

# Multi-word phrases that should be detected as single units
MULTI_WORD_PHRASES: Dict[Aspect, List[Tuple[str, float]]] = {
    Aspect.LOCATION: [
        ("walking distance", 0.8),
        ("public transport", 0.8),
        ("public transportation", 0.8),
        ("city center", 0.8),
        ("city centre", 0.8),
        ("old town", 0.6),
        ("main street", 0.5),
    ],
    Aspect.AMENITIES: [
        ("air conditioning", 0.9),
        ("washing machine", 0.8),
        ("coffee maker", 0.6),
        ("hair dryer", 0.6),
        ("hot water", 0.7),
        ("full kitchen", 0.8),
    ],
    Aspect.HOST_BEHAVIOR: [
        ("check in", 0.8),
        ("check out", 0.7),
        ("self check", 0.6),
    ],
    Aspect.CLEANLINESS: [
        ("deep clean", 0.9),
        ("freshly cleaned", 0.8),
    ],
}

# Words that when appearing near aspect keywords, negate the aspect match
# e.g., "location of the bathroom" shouldn't match LOCATION aspect
EXCLUSION_CONTEXTS: Dict[Aspect, Set[str]] = {
    Aspect.LOCATION: {"bathroom", "kitchen", "bedroom", "shower", "toilet"},
    Aspect.NOISE: {"make", "making"},  # "don't make noise" is instruction, not complaint
}


# =============================================================================
# DETECTION FUNCTIONS
# =============================================================================

def _find_multi_word_matches(text: str, aspect: Aspect) -> List[Tuple[str, float, int, int]]:
    """
    Find multi-word phrase matches in text.
    
    Returns:
        List of (phrase, weight, start_pos, end_pos) tuples
    """
    matches = []
    text_lower = text.lower()
    
    if aspect in MULTI_WORD_PHRASES:
        for phrase, weight in MULTI_WORD_PHRASES[aspect]:
            start = 0
            while True:
                pos = text_lower.find(phrase, start)
                if pos == -1:
                    break
                matches.append((phrase, weight, pos, pos + len(phrase)))
                start = pos + 1
    
    return matches


def _find_keyword_matches(text: str, aspect: Aspect) -> List[Tuple[str, float, int]]:
    """
    Find single keyword matches in text.
    
    Returns:
        List of (keyword, weight, position) tuples
    """
    matches = []
    words = text.lower().split()
    
    if aspect not in ASPECT_LEXICONS:
        return matches
    
    lexicon = ASPECT_LEXICONS[aspect]
    
    for i, word in enumerate(words):
        # Remove punctuation and _NEG marker for matching
        clean_word = re.sub(r'[^\w]', '', word.replace('_NEG', ''))
        
        if clean_word in lexicon:
            matches.append((clean_word, lexicon[clean_word], i))
    
    return matches


def _check_exclusion_context(text: str, aspect: Aspect, match_position: int) -> bool:
    """
    Check if match should be excluded based on surrounding context.
    
    Returns:
        True if match should be EXCLUDED, False if it's valid
    """
    if aspect not in EXCLUSION_CONTEXTS:
        return False
    
    words = text.lower().split()
    exclusions = EXCLUSION_CONTEXTS[aspect]
    
    # Check 3 words before and after the match position
    window_start = max(0, match_position - 3)
    window_end = min(len(words), match_position + 4)
    
    for i in range(window_start, window_end):
        clean_word = re.sub(r'[^\w]', '', words[i])
        if clean_word in exclusions:
            return True
    
    return False


def _check_negation_in_match(text: str, keyword: str) -> bool:
    """
    Check if the matched keyword is negated (has _NEG marker).
    
    Returns:
        True if keyword appears negated in text
    """
    # Look for keyword with _NEG suffix
    pattern = r'\b' + re.escape(keyword) + r'_NEG\b'
    return bool(re.search(pattern, text, re.IGNORECASE))


def detect_aspects_in_sentence(sentence: str) -> List[AspectMatch]:
    """
    Detect all aspects mentioned in a sentence.
    
    Algorithm:
    1. For each aspect, find multi-word phrase matches
    2. Find single keyword matches
    3. Apply exclusion context rules
    4. Combine and weight matches
    5. Return AspectMatch objects (without sentiment - that's handled separately)
    
    Args:
        sentence: Processed sentence (with negation markers)
        
    Returns:
        List of AspectMatch objects (sentiment_score set to 0, to be filled later)
    """
    results = []
    
    for aspect in Aspect:
        matched_keywords = []
        total_weight = 0.0
        has_negation = False
        
        # 1. Check multi-word phrases first
        multi_matches = _find_multi_word_matches(sentence, aspect)
        for phrase, weight, start, end in multi_matches:
            matched_keywords.append(phrase)
            total_weight += weight
        
        # 2. Check single keywords
        keyword_matches = _find_keyword_matches(sentence, aspect)
        
        for keyword, weight, position in keyword_matches:
            # Skip if in exclusion context
            if _check_exclusion_context(sentence, aspect, position):
                continue
            
            # Check if already covered by a multi-word phrase
            already_matched = any(keyword in phrase for phrase, _, _, _ in multi_matches)
            if already_matched:
                continue
            
            matched_keywords.append(keyword)
            total_weight += weight
            
            # Check for negation
            if _check_negation_in_match(sentence, keyword):
                has_negation = True
        
        # 3. If we found matches, create AspectMatch
        if matched_keywords:
            # Calculate confidence based on number of matches and their weights
            # More matches and higher weights = higher confidence
            confidence = min(1.0, total_weight / 2.0)  # Normalize
            
            results.append(AspectMatch(
                aspect=aspect,
                sentiment_score=0.0,  # To be filled by sentiment analysis
                confidence=confidence,
                matched_keywords=matched_keywords,
                has_negation=has_negation
            ))
    
    return results


def detect_aspects_batch(sentences: List[str]) -> List[List[AspectMatch]]:
    """
    Detect aspects in multiple sentences.
    
    Args:
        sentences: List of processed sentences
        
    Returns:
        List of AspectMatch lists, one per sentence
    """
    return [detect_aspects_in_sentence(sent) for sent in sentences]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def get_all_keywords_for_aspect(aspect: Aspect) -> Set[str]:
    """Get all keywords (single and multi-word) for an aspect."""
    keywords = set(ASPECT_LEXICONS.get(aspect, {}).keys())
    
    if aspect in MULTI_WORD_PHRASES:
        for phrase, _ in MULTI_WORD_PHRASES[aspect]:
            keywords.add(phrase)
    
    return keywords


def get_aspect_from_string(aspect_str: str) -> Optional[Aspect]:
    """Convert string to Aspect enum, case-insensitive."""
    try:
        return Aspect(aspect_str.lower())
    except ValueError:
        return None
