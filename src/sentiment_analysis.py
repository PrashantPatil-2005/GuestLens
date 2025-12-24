"""
sentiment_analysis.py - Lexicon-Based Sentiment Analysis

Design Decisions:
1. Lexicon-Based: Using a curated sentiment lexicon rather than ML models
   for full explainability. Each word has a known sentiment score.

2. Negation Handling: Words marked with _NEG suffix have their sentiment
   flipped (multiplied by -0.8, not -1.0, as negation often softens sentiment)

3. Aspect-Aware: Sentiment is computed per aspect, only considering words
   near the aspect keywords to maintain context relevance.

4. Intensity Modifiers: Words like "very", "extremely", "somewhat" modify
   the intensity of following sentiment words.

5. Aggregation Strategy: Multiple sentiment words in a sentence are combined
   using weighted average based on proximity to aspect keywords.
"""

import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, replace
from src.schemas import Aspect, AspectMatch


# =============================================================================
# SENTIMENT LEXICON
# =============================================================================

# Score range: -1.0 (very negative) to 1.0 (very positive)
# Organized by intensity for readability

SENTIMENT_LEXICON: Dict[str, float] = {
    # Extremely positive (0.9 to 1.0)
    "amazing": 1.0,
    "excellent": 1.0,
    "outstanding": 1.0,
    "exceptional": 1.0,
    "phenomenal": 1.0,
    "incredible": 0.95,
    "fantastic": 0.95,
    "wonderful": 0.95,
    "superb": 0.95,
    "brilliant": 0.9,
    "magnificent": 0.9,
    "marvelous": 0.9,
    "perfect": 1.0,
    "flawless": 1.0,
    "impeccable": 1.0,
    
    # Very positive (0.7 to 0.89)
    "great": 0.8,
    "lovely": 0.8,
    "beautiful": 0.8,
    "awesome": 0.85,
    "delightful": 0.8,
    "pleasant": 0.7,
    "comfortable": 0.7,
    "clean": 0.75,
    "spotless": 0.9,
    "immaculate": 0.9,
    "pristine": 0.9,
    "cozy": 0.7,
    "friendly": 0.75,
    "helpful": 0.75,
    "responsive": 0.75,
    "welcoming": 0.7,
    "convenient": 0.7,
    "peaceful": 0.75,
    "quiet": 0.7,
    "safe": 0.75,
    "secure": 0.75,
    
    # Moderately positive (0.4 to 0.69)
    "good": 0.6,
    "nice": 0.55,
    "fine": 0.4,
    "decent": 0.45,
    "adequate": 0.4,
    "satisfactory": 0.45,
    "functional": 0.4,
    "reasonable": 0.45,
    "acceptable": 0.4,
    "tidy": 0.5,
    "spacious": 0.6,
    "modern": 0.5,
    "stylish": 0.55,
    "equipped": 0.5,
    "central": 0.5,
    "accessible": 0.5,
    
    # Slightly positive (0.1 to 0.39)
    "okay": 0.2,
    "ok": 0.2,
    "alright": 0.2,
    "sufficient": 0.25,
    "basic": 0.15,
    "simple": 0.1,
    "standard": 0.2,
    "average": 0.1,
    
    # Neutral (around 0)
    "mixed": 0.0,
    "neutral": 0.0,
    
    # Slightly negative (-0.1 to -0.39)
    "disappointing": -0.35,
    "underwhelming": -0.3,
    "mediocre": -0.25,
    "lacking": -0.3,
    "limited": -0.2,
    "small": -0.15,
    "cramped": -0.35,
    "outdated": -0.3,
    "old": -0.15,
    "worn": -0.25,
    
    # Moderately negative (-0.4 to -0.69)
    "bad": -0.6,
    "poor": -0.55,
    "dirty": -0.65,
    "unclean": -0.6,
    "messy": -0.5,
    "dusty": -0.45,
    "stained": -0.5,
    "noisy": -0.55,
    "loud": -0.5,
    "uncomfortable": -0.55,
    "inconvenient": -0.5,
    "unfriendly": -0.55,
    "unhelpful": -0.55,
    "unresponsive": -0.6,
    "rude": -0.65,
    "slow": -0.4,
    "broken": -0.6,
    "faulty": -0.55,
    "cold": -0.4,
    "hot": -0.35,
    
    # Very negative (-0.7 to -0.89)
    "terrible": -0.85,
    "horrible": -0.85,
    "awful": -0.85,
    "dreadful": -0.8,
    "disgusting": -0.85,
    "filthy": -0.85,
    "nasty": -0.75,
    "smelly": -0.7,
    "moldy": -0.8,
    "gross": -0.75,
    "dangerous": -0.8,
    "unsafe": -0.85,
    "sketchy": -0.7,
    "scary": -0.7,
    
    # Extremely negative (-0.9 to -1.0)
    "worst": -1.0,
    "nightmare": -1.0,
    "unbearable": -0.95,
    "unacceptable": -0.9,
    "appalling": -0.95,
    "atrocious": -0.95,
    "abysmal": -0.95,
    "horrendous": -0.95,
    "revolting": -0.9,
    
    # Recommendation/experience words
    "recommend": 0.7,
    "recommended": 0.7,
    "love": 0.85,
    "loved": 0.85,
    "enjoy": 0.7,
    "enjoyed": 0.7,
    "hate": -0.85,
    "hated": -0.85,
    "regret": -0.7,
    "avoid": -0.75,
    "waste": -0.7,
    "worth": 0.6,
    "value": 0.5,
}


# =============================================================================
# INTENSITY MODIFIERS
# =============================================================================

# Multipliers for intensity words that precede sentiment words
INTENSITY_MODIFIERS: Dict[str, float] = {
    # Amplifiers (multiply sentiment by > 1)
    "very": 1.3,
    "really": 1.25,
    "extremely": 1.5,
    "incredibly": 1.5,
    "absolutely": 1.4,
    "completely": 1.35,
    "totally": 1.3,
    "super": 1.3,
    "highly": 1.25,
    "exceptionally": 1.4,
    "particularly": 1.2,
    "especially": 1.25,
    "remarkably": 1.3,
    "truly": 1.2,
    "so": 1.2,
    "such": 1.15,
    
    # Diminishers (multiply sentiment by < 1)
    "somewhat": 0.7,
    "slightly": 0.6,
    "fairly": 0.75,
    "rather": 0.8,
    "quite": 0.9,  # Can be either, defaulting to slight amplifier
    "a bit": 0.6,
    "a little": 0.6,
    "kind of": 0.6,
    "sort of": 0.6,
    "almost": 0.8,
    "mostly": 0.85,
    "generally": 0.8,
    "usually": 0.8,
}

# Negation flip factor (not -1.0 because negation often softens)
NEGATION_FLIP_FACTOR = -0.8

# Context window: how many words around aspect keywords to consider
SENTIMENT_CONTEXT_WINDOW = 5


# =============================================================================
# ANALYSIS FUNCTIONS
# =============================================================================

def _get_word_sentiment(word: str) -> Optional[float]:
    """
    Get sentiment score for a single word.
    
    Returns None if word not in lexicon.
    """
    clean_word = word.lower().replace("_neg", "")
    clean_word = re.sub(r'[^\w]', '', clean_word)
    return SENTIMENT_LEXICON.get(clean_word)


def _get_intensity_modifier(words: List[str], position: int) -> float:
    """
    Check if there's an intensity modifier before the given position.
    
    Returns the modifier multiplier (default 1.0 if none found).
    """
    if position <= 0:
        return 1.0
    
    # Check previous 1-2 words for modifiers
    for offset in [1, 2]:
        if position - offset >= 0:
            prev_word = words[position - offset].lower()
            prev_word = re.sub(r'[^\w\s]', '', prev_word)
            
            if prev_word in INTENSITY_MODIFIERS:
                return INTENSITY_MODIFIERS[prev_word]
            
            # Check two-word modifiers
            if offset == 1 and position - 2 >= 0:
                two_word = f"{words[position - 2]} {prev_word}".lower()
                if two_word in INTENSITY_MODIFIERS:
                    return INTENSITY_MODIFIERS[two_word]
    
    return 1.0


def _is_negated(word: str) -> bool:
    """Check if word has negation marker."""
    return "_NEG" in word or "_neg" in word


def compute_sentence_sentiment(sentence: str) -> Tuple[float, List[Tuple[str, float]]]:
    """
    Compute overall sentiment of a sentence.
    
    Returns:
        Tuple of (overall_score, list of (word, score) pairs for explainability)
    """
    words = sentence.split()
    sentiment_hits = []
    
    for i, word in enumerate(words):
        base_score = _get_word_sentiment(word)
        
        if base_score is not None:
            # Apply intensity modifier
            modifier = _get_intensity_modifier(words, i)
            modified_score = base_score * modifier
            
            # Apply negation flip if marked
            if _is_negated(word):
                modified_score *= NEGATION_FLIP_FACTOR
            
            # Clamp to [-1, 1]
            modified_score = max(-1.0, min(1.0, modified_score))
            
            clean_word = re.sub(r'[^\w]', '', word.replace("_NEG", ""))
            sentiment_hits.append((clean_word, modified_score))
    
    if not sentiment_hits:
        return 0.0, []
    
    # Compute average sentiment
    avg_score = sum(score for _, score in sentiment_hits) / len(sentiment_hits)
    return avg_score, sentiment_hits


def analyze_aspect_sentiment(
    sentence: str,
    aspect_match: AspectMatch
) -> AspectMatch:
    """
    Analyze sentiment specifically for an aspect mentioned in a sentence.
    
    Strategy:
    1. Find positions of aspect keywords in sentence
    2. Look at sentiment words within CONTEXT_WINDOW of keywords
    3. Weight sentiment by proximity to keywords
    4. Apply negation if aspect match indicates negation
    
    Args:
        sentence: The processed sentence
        aspect_match: AspectMatch with detected keywords (sentiment_score = 0)
        
    Returns:
        Updated AspectMatch with computed sentiment_score
    """
    words = sentence.lower().split()
    
    # Find positions of aspect keywords
    keyword_positions = []
    for keyword in aspect_match.matched_keywords:
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.replace("_neg", ""))
            if keyword.lower() in clean_word or clean_word in keyword.lower():
                keyword_positions.append(i)
    
    if not keyword_positions:
        # Fallback: use sentence-level sentiment
        overall_score, _ = compute_sentence_sentiment(sentence)
        return replace(aspect_match, sentiment_score=overall_score)
    
    # Collect sentiment scores weighted by proximity to keywords
    weighted_scores = []
    total_weight = 0.0
    
    for i, word in enumerate(words):
        base_score = _get_word_sentiment(word)
        
        if base_score is not None:
            # Calculate minimum distance to any aspect keyword
            min_distance = min(abs(i - pos) for pos in keyword_positions)
            
            # Only consider words within context window
            if min_distance > SENTIMENT_CONTEXT_WINDOW:
                continue
            
            # Weight inversely by distance (closer = higher weight)
            distance_weight = 1.0 / (1.0 + min_distance * 0.3)
            
            # Apply intensity modifier
            modifier = _get_intensity_modifier(words, i)
            modified_score = base_score * modifier
            
            # Apply negation if word is marked
            if _is_negated(word):
                modified_score *= NEGATION_FLIP_FACTOR
            
            weighted_scores.append(modified_score * distance_weight)
            total_weight += distance_weight
    
    # Calculate weighted average
    if weighted_scores and total_weight > 0:
        final_score = sum(weighted_scores) / total_weight
    else:
        # No sentiment words found near aspect - check if aspect keyword itself has sentiment
        # (e.g., "clean" is both aspect and sentiment)
        final_score = 0.0
        for keyword in aspect_match.matched_keywords:
            keyword_sentiment = _get_word_sentiment(keyword)
            if keyword_sentiment is not None:
                final_score = keyword_sentiment
                break
    
    # Apply overall negation if detected in aspect match
    if aspect_match.has_negation:
        final_score *= NEGATION_FLIP_FACTOR
    
    # Clamp final score
    final_score = max(-1.0, min(1.0, final_score))
    
    return replace(aspect_match, sentiment_score=final_score)


def analyze_aspects_sentiments(
    sentence: str,
    aspect_matches: List[AspectMatch]
) -> List[AspectMatch]:
    """
    Analyze sentiment for all aspects detected in a sentence.
    
    Args:
        sentence: Processed sentence
        aspect_matches: List of AspectMatch objects from aspect detection
        
    Returns:
        List of AspectMatch objects with sentiment_score filled in
    """
    return [analyze_aspect_sentiment(sentence, am) for am in aspect_matches]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def explain_sentiment(sentence: str) -> Dict:
    """
    Generate detailed explanation of sentiment analysis for debugging.
    
    Returns dict with:
    - overall_score: float
    - word_scores: list of {word, base_score, modifier, negated, final_score}
    """
    words = sentence.split()
    explanations = []
    
    for i, word in enumerate(words):
        base_score = _get_word_sentiment(word)
        
        if base_score is not None:
            modifier = _get_intensity_modifier(words, i)
            negated = _is_negated(word)
            
            final = base_score * modifier
            if negated:
                final *= NEGATION_FLIP_FACTOR
            
            explanations.append({
                'word': word.replace("_NEG", ""),
                'base_score': base_score,
                'modifier': modifier,
                'negated': negated,
                'final_score': round(final, 3)
            })
    
    overall = sum(e['final_score'] for e in explanations) / len(explanations) if explanations else 0.0
    
    return {
        'overall_score': round(overall, 3),
        'word_scores': explanations
    }


def get_sentiment_category(score: float) -> str:
    """
    Convert numeric score to categorical label.
    
    Useful for quick interpretation:
    - Very Negative: score < -0.6
    - Negative: -0.6 <= score < -0.2
    - Neutral: -0.2 <= score < 0.2
    - Positive: 0.2 <= score < 0.6
    - Very Positive: score >= 0.6
    """
    if score < -0.6:
        return "very_negative"
    elif score < -0.2:
        return "negative"
    elif score < 0.2:
        return "neutral"
    elif score < 0.6:
        return "positive"
    else:
        return "very_positive"
