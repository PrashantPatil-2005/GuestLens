"""
preprocessing.py - Text Preprocessing for Guest Reviews

Design Decisions:
1. Negation Preservation: Instead of removing negation words, we mark the 
   following 1-3 words with a _NEG suffix. This allows sentiment analysis
   to correctly flip scores for negated phrases like "not clean" → "clean_NEG"
   
2. Contraction Expansion: Expand contractions BEFORE negation marking so that
   "wasn't" becomes "was not" and then "not" triggers negation marking.
   
3. Rule-based Sentence Splitting: Using regex patterns rather than heavy NLP
   libraries for explainability and speed. Handles common edge cases like
   abbreviations and decimal numbers.
   
4. Context Preservation: Keep original text alongside processed version for
   debugging and explainability.
"""

import re
from typing import List, Tuple
from dataclasses import dataclass


# =============================================================================
# CONTRACTION MAPPINGS
# =============================================================================

CONTRACTIONS = {
    # Negation contractions (critical for negation detection)
    "n't": " not",
    "wasn't": "was not",
    "weren't": "were not",
    "isn't": "is not",
    "aren't": "are not",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "won't": "will not",
    "wouldn't": "would not",
    "couldn't": "could not",
    "shouldn't": "should not",
    "haven't": "have not",
    "hasn't": "has not",
    "hadn't": "had not",
    "can't": "cannot",
    "cannot": "can not",
    
    # Common contractions
    "'m": " am",
    "'re": " are",
    "'s": " is",  # Note: Also possessive, but simpler to expand
    "'ll": " will",
    "'ve": " have",
    "'d": " would",
    
    # Informal
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
}

# =============================================================================
# NEGATION PATTERNS
# =============================================================================

# Words that trigger negation of following words
NEGATION_TRIGGERS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "hardly", "barely", "scarcely", "seldom", "rarely",
    "without", "lack", "lacking", "lacks", "lacked",
    "except", "but"  # Context-dependent, but often negate following clause
}

# Words that terminate negation scope
NEGATION_TERMINATORS = {
    "but", "however", "although", "though", "yet", "still",
    ".", ",", ";", ":", "!", "?"
}

# Number of words after negation trigger to mark with _NEG
NEGATION_SCOPE = 3


# =============================================================================
# SENTENCE SPLITTING PATTERNS
# =============================================================================

# Abbreviations that shouldn't trigger sentence splits
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "vs", "etc", "inc", "ltd",
    "e.g", "i.e", "a.m", "p.m", "no", "nos", "approx", "apt", "dept",
    "est", "min", "max", "misc", "cont", "fig", "vol", "ref"
}


@dataclass
class PreprocessedText:
    """Container for original and processed text."""
    original: str
    processed: str
    sentences: List[str]


def expand_contractions(text: str) -> str:
    """
    Expand contractions to their full forms.
    
    This is done BEFORE negation marking so that contractions like "wasn't"
    become "was not" and the "not" can trigger proper negation handling.
    
    Args:
        text: Input text with potential contractions
        
    Returns:
        Text with contractions expanded
    """
    result = text.lower()
    
    # Sort by length (longest first) to handle overlapping patterns
    sorted_contractions = sorted(CONTRACTIONS.items(), key=lambda x: len(x[0]), reverse=True)
    
    for contraction, expansion in sorted_contractions:
        # Use word boundaries where appropriate
        if contraction.startswith("'"):
            # Suffix contractions like "'t", "'m", "'ll"
            result = result.replace(contraction, expansion)
        else:
            # Full word contractions
            pattern = r'\b' + re.escape(contraction) + r'\b'
            result = re.sub(pattern, expansion, result, flags=re.IGNORECASE)
    
    return result


def mark_negations(text: str) -> str:
    """
    Mark words affected by negation with _NEG suffix.
    
    Strategy:
    1. Identify negation trigger words
    2. Mark the following N words (within scope) with _NEG
    3. Stop marking at negation terminators or end of scope
    
    Example:
        "The room was not clean at all" → "The room was not clean_NEG at_NEG all_NEG"
        
    Args:
        text: Preprocessed text (contractions already expanded)
        
    Returns:
        Text with negation-affected words marked
    """
    words = text.split()
    result = []
    negation_remaining = 0  # Counter for remaining words to negate
    
    for word in words:
        # Clean word for comparison (remove punctuation)
        clean_word = re.sub(r'[^\w]', '', word.lower())
        
        # Check if this word terminates negation
        if clean_word in NEGATION_TERMINATORS or any(t in word for t in '.!?'):
            negation_remaining = 0
        
        # Check if this word triggers negation
        if clean_word in NEGATION_TRIGGERS:
            result.append(word)
            negation_remaining = NEGATION_SCOPE
            continue
        
        # Apply negation marking if within scope
        if negation_remaining > 0:
            # Don't mark very short words or punctuation
            if len(clean_word) > 2:
                # Preserve punctuation attached to word
                if word[-1] in '.,!?;:':
                    word = word[:-1] + "_NEG" + word[-1]
                else:
                    word = word + "_NEG"
            negation_remaining -= 1
        
        result.append(word)
    
    return ' '.join(result)


def clean_text(text: str) -> str:
    """
    Basic text cleaning while preserving important features.
    
    Operations:
    - Normalize whitespace
    - Preserve punctuation (needed for sentence splitting)
    - Remove excessive special characters
    - Normalize Unicode
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text
    """
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Remove excessive punctuation (keep single instances)
    text = re.sub(r'([!?.]){2,}', r'\1', text)
    
    # Remove emojis and special unicode (keep basic punctuation and letters)
    text = re.sub(r'[^\w\s.,!?;:\'-]', '', text)
    
    return text.strip()


def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences using rule-based approach.
    
    Handles edge cases:
    - Abbreviations (Dr., Mr., etc.)
    - Decimal numbers (3.5)
    - Ellipsis (...)
    - Multiple punctuation (!!)
    
    Args:
        text: Cleaned text to split
        
    Returns:
        List of sentence strings
    """
    # Step 1: Protect abbreviations from splitting
    protected = text
    for abbrev in ABBREVIATIONS:
        # Add placeholder to prevent split after abbreviation period
        pattern = r'\b' + re.escape(abbrev) + r'\.'
        protected = re.sub(pattern, abbrev.upper() + '<<<DOT>>>', protected, flags=re.IGNORECASE)
    
    # Step 2: Protect decimal numbers
    protected = re.sub(r'(\d)\.(\d)', r'\1<<<DOT>>>\2', protected)
    
    # Step 3: Split on sentence-ending punctuation
    # Match period, exclamation, or question mark followed by space and capital
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    
    # Step 4: Also split on just punctuation if no capital follows
    final_sentences = []
    for sent in sentences:
        # Split on punctuation followed by space
        sub_sents = re.split(r'(?<=[.!?])\s+', sent)
        final_sentences.extend(sub_sents)
    
    # Step 5: Restore protected markers
    result = []
    for sent in final_sentences:
        sent = sent.replace('<<<DOT>>>', '.')
        sent = sent.strip()
        if sent and len(sent) > 2:  # Filter very short fragments
            result.append(sent)
    
    return result


def preprocess_review(text: str) -> PreprocessedText:
    """
    Full preprocessing pipeline for a review.
    
    Pipeline:
    1. Clean text (normalize, remove noise)
    2. Split into sentences
    3. Expand contractions per sentence
    4. Mark negations per sentence
    
    Args:
        text: Raw review text
        
    Returns:
        PreprocessedText with original, processed, and sentence list
    """
    # Step 1: Clean
    cleaned = clean_text(text)
    
    # Step 2: Split into sentences
    sentences = split_sentences(cleaned)
    
    # Step 3 & 4: Process each sentence
    processed_sentences = []
    for sent in sentences:
        # Expand contractions
        expanded = expand_contractions(sent)
        # Mark negations
        negated = mark_negations(expanded)
        processed_sentences.append(negated)
    
    return PreprocessedText(
        original=text,
        processed=' '.join(processed_sentences),
        sentences=processed_sentences
    )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def has_negation_marker(word: str) -> bool:
    """Check if a word has negation marker."""
    return "_NEG" in word


def remove_negation_marker(word: str) -> str:
    """Remove negation marker from word if present."""
    return word.replace("_NEG", "")


def get_negation_aware_words(text: str) -> List[Tuple[str, bool]]:
    """
    Parse text into (word, is_negated) tuples.
    
    Args:
        text: Processed text with potential _NEG markers
        
    Returns:
        List of (clean_word, is_negated) tuples
    """
    words = text.split()
    result = []
    
    for word in words:
        # Remove punctuation and check negation
        clean = re.sub(r'[^\w_]', '', word)
        is_negated = has_negation_marker(clean)
        base_word = remove_negation_marker(clean).lower()
        
        if base_word:  # Skip empty strings
            result.append((base_word, is_negated))
    
    return result
