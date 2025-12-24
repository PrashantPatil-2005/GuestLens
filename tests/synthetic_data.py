"""
synthetic_data.py - Generate Synthetic Airbnb Reviews for Testing

This module generates realistic test data with known aspect/sentiment patterns
to validate the pipeline behavior.

Design approach:
- Templates with aspect-specific phrases
- Controlled sentiment (positive/negative/mixed)
- Varied review lengths and styles
- Realistic date distribution over 2 years
"""

import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import uuid

from src.schemas import RawReview, Aspect


# =============================================================================
# REVIEW TEMPLATES
# =============================================================================

# Positive phrases per aspect
POSITIVE_PHRASES: Dict[str, List[str]] = {
    "cleanliness": [
        "The apartment was spotless and very clean.",
        "Everything was immaculate, clearly freshly cleaned.",
        "Super tidy place, the bathroom was pristine.",
        "Really clean space, no dust anywhere.",
        "The cleanliness was exceptional, truly spotless.",
    ],
    "noise": [
        "Very quiet neighborhood, slept like a baby.",
        "Peaceful and silent, no traffic noise at all.",
        "The place was so quiet and relaxing.",
        "Loved how peaceful the area was.",
        "Incredibly quiet, you wouldn't know you're in the city.",
    ],
    "location": [
        "Perfect location, walking distance to everything.",
        "The location was amazing, very central.",
        "Great neighborhood with easy access to public transport.",
        "Excellent location near restaurants and shops.",
        "Conveniently located close to the metro station.",
    ],
    "host_behavior": [
        "The host was extremely responsive and helpful.",
        "Amazing host, great communication throughout.",
        "Very welcoming host, made check-in super easy.",
        "The host was incredibly friendly and accommodating.",
        "Quick responses from the host, very professional.",
    ],
    "amenities": [
        "Well equipped kitchen with everything you need.",
        "Great amenities, fast wifi and comfortable bed.",
        "Loved the fully stocked kitchen and modern appliances.",
        "Excellent amenities including parking and washer.",
        "The apartment had great facilities, AC worked perfectly.",
    ],
    "safety": [
        "Felt very safe in this neighborhood.",
        "Secure building with good locks.",
        "The area felt safe even at night.",
        "Very secure apartment, felt comfortable.",
        "Safe neighborhood, no concerns at all.",
    ],
}

# Negative phrases per aspect
NEGATIVE_PHRASES: Dict[str, List[str]] = {
    "cleanliness": [
        "The place was dirty and had dust everywhere.",
        "Not clean at all, found stains on the sheets.",
        "The bathroom was disgusting and moldy.",
        "Really messy, clearly not cleaned before arrival.",
        "Disappointing cleanliness, the kitchen was filthy.",
    ],
    "noise": [
        "Very noisy, couldn't sleep because of traffic.",
        "The neighbors were loud all night.",
        "Not quiet at all, constant noise from the street.",
        "Terrible noise levels, bring earplugs!",
        "The walls are thin, could hear everything.",
    ],
    "location": [
        "Bad location, far from everything.",
        "The neighborhood felt sketchy.",
        "Inconvenient location, no public transport nearby.",
        "Remote area, had to take taxis everywhere.",
        "Not a great area, wouldn't walk around at night.",
    ],
    "host_behavior": [
        "The host was unresponsive and unhelpful.",
        "Poor communication, never replied to messages.",
        "The host was rude and unwelcoming.",
        "Check-in was a nightmare, host wasn't available.",
        "Very unfriendly host, felt unwelcome.",
    ],
    "amenities": [
        "The wifi was broken and never fixed.",
        "Lacking basic amenities, no hot water.",
        "The bed was so uncomfortable, couldn't sleep.",
        "Kitchen was poorly equipped, missing utensils.",
        "AC didn't work, apartment was unbearably hot.",
    ],
    "safety": [
        "Didn't feel safe in this area.",
        "The locks on the door were broken.",
        "Sketchy neighborhood, wouldn't recommend.",
        "Felt unsafe walking around at night.",
        "Security concerns, the building wasn't secure.",
    ],
}

# Neutral/mixed phrases per aspect
NEUTRAL_PHRASES: Dict[str, List[str]] = {
    "cleanliness": [
        "The place was okay, could have been cleaner.",
        "It was clean enough, nothing special.",
        "Cleanliness was acceptable but not great.",
    ],
    "noise": [
        "Some noise from the street but manageable.",
        "It was quiet most of the time.",
        "Average noise levels for a city apartment.",
    ],
    "location": [
        "The location was convenient enough.",
        "Decent location, a bit far from the center.",
        "Location was fine, nothing remarkable.",
    ],
    "host_behavior": [
        "The host was okay, nothing exceptional.",
        "Communication was adequate.",
        "Host was fine, responded eventually.",
    ],
    "amenities": [
        "Basic amenities, had what we needed.",
        "The apartment had the essentials.",
        "Amenities were standard, nothing fancy.",
    ],
    "safety": [
        "The area seemed safe enough.",
        "No safety issues but nothing special.",
        "Felt okay about safety.",
    ],
}

# Generic positive/negative sentences (no specific aspect)
GENERIC_POSITIVE = [
    "Would definitely recommend!",
    "Great experience overall.",
    "Will definitely come back.",
    "Exceeded our expectations.",
    "Had a wonderful stay.",
    "Perfect for our trip.",
]

GENERIC_NEGATIVE = [
    "Would not recommend.",
    "Disappointing overall.",
    "Not worth the price.",
    "Expected much better.",
    "Left feeling unsatisfied.",
    "Won't be returning.",
]

# Reviewer names for realism
FIRST_NAMES = [
    "John", "Sarah", "Michael", "Emma", "David", "Lisa", "James", "Anna",
    "Robert", "Maria", "Chris", "Laura", "Kevin", "Jennifer", "Tom", "Emily",
    "Daniel", "Rachel", "Mark", "Jessica", "Paul", "Nicole", "Andrew", "Ashley"
]


# =============================================================================
# GENERATION FUNCTIONS
# =============================================================================

def generate_review_text(
    sentiment_profile: Dict[str, str],
    num_aspects: int = 3,
    include_generic: bool = True
) -> str:
    """
    Generate a review with specified sentiment per aspect.
    
    Args:
        sentiment_profile: Dict mapping aspect name to "positive", "negative", or "neutral"
        num_aspects: Number of aspects to mention
        include_generic: Whether to include generic opening/closing
        
    Returns:
        Generated review text
    """
    sentences = []
    
    # Select which aspects to include
    aspects = list(sentiment_profile.keys())
    if len(aspects) > num_aspects:
        aspects = random.sample(aspects, num_aspects)
    
    # Add aspect-specific sentences
    for aspect in aspects:
        sentiment = sentiment_profile.get(aspect, "neutral")
        
        if sentiment == "positive":
            phrases = POSITIVE_PHRASES.get(aspect, [])
        elif sentiment == "negative":
            phrases = NEGATIVE_PHRASES.get(aspect, [])
        else:
            phrases = NEUTRAL_PHRASES.get(aspect, [])
        
        if phrases:
            sentences.append(random.choice(phrases))
    
    # Add generic sentences
    if include_generic:
        # Check overall sentiment
        positive_count = sum(1 for s in sentiment_profile.values() if s == "positive")
        negative_count = sum(1 for s in sentiment_profile.values() if s == "negative")
        
        if positive_count > negative_count:
            sentences.append(random.choice(GENERIC_POSITIVE))
        elif negative_count > positive_count:
            sentences.append(random.choice(GENERIC_NEGATIVE))
    
    # Shuffle for variety
    random.shuffle(sentences)
    
    return " ".join(sentences)


def generate_review(
    listing_id: str,
    review_date: datetime,
    sentiment_profile: Dict[str, str],
    num_aspects: int = 3
) -> RawReview:
    """Generate a single review with specified characteristics."""
    review_id = f"review_{uuid.uuid4().hex[:12]}"
    reviewer_name = random.choice(FIRST_NAMES)
    review_text = generate_review_text(sentiment_profile, num_aspects)
    
    return RawReview(
        review_id=review_id,
        listing_id=listing_id,
        reviewer_name=reviewer_name,
        review_date=review_date,
        review_text=review_text
    )


def generate_date_sequence(
    start_date: datetime,
    end_date: datetime,
    n_reviews: int
) -> List[datetime]:
    """Generate a sequence of review dates, slightly clustered in recent months."""
    total_days = (end_date - start_date).days
    dates = []
    
    for _ in range(n_reviews):
        # Bias toward more recent dates (exponential distribution)
        # This creates a realistic pattern where recent months have more reviews
        random_factor = random.random() ** 0.7  # Skew toward 0 (more recent)
        days_ago = int(random_factor * total_days)
        review_date = end_date - timedelta(days=days_ago)
        dates.append(review_date)
    
    return sorted(dates)


def generate_listing_reviews(
    listing_id: str,
    n_reviews: int = 20,
    listing_quality: str = "good",
    end_date: Optional[datetime] = None,
    days_span: int = 730  # 2 years
) -> List[RawReview]:
    """
    Generate reviews for a single listing with consistent quality profile.
    
    Args:
        listing_id: ID of the listing
        n_reviews: Number of reviews to generate
        listing_quality: "excellent", "good", "average", "poor", or "mixed"
        end_date: Most recent review date (default: now)
        days_span: How many days to span reviews over
        
    Returns:
        List of RawReview objects
    """
    if end_date is None:
        end_date = datetime.now()
    
    start_date = end_date - timedelta(days=days_span)
    dates = generate_date_sequence(start_date, end_date, n_reviews)
    
    # Define sentiment profiles based on listing quality
    aspect_names = ["cleanliness", "noise", "location", "host_behavior", "amenities", "safety"]
    
    reviews = []
    for date in dates:
        # Generate sentiment profile based on listing quality
        profile = {}
        
        for aspect in aspect_names:
            if listing_quality == "excellent":
                # Mostly positive with occasional neutral
                sentiment = random.choices(
                    ["positive", "neutral", "negative"],
                    weights=[0.85, 0.12, 0.03]
                )[0]
            elif listing_quality == "good":
                # Mostly positive with some neutral and rare negative
                sentiment = random.choices(
                    ["positive", "neutral", "negative"],
                    weights=[0.70, 0.22, 0.08]
                )[0]
            elif listing_quality == "average":
                # Mix of all
                sentiment = random.choices(
                    ["positive", "neutral", "negative"],
                    weights=[0.40, 0.35, 0.25]
                )[0]
            elif listing_quality == "poor":
                # Mostly negative
                sentiment = random.choices(
                    ["positive", "neutral", "negative"],
                    weights=[0.15, 0.20, 0.65]
                )[0]
            else:  # mixed - polarized reviews
                # Either very positive or very negative
                sentiment = random.choices(
                    ["positive", "negative"],
                    weights=[0.5, 0.5]
                )[0]
            
            profile[aspect] = sentiment
        
        # Randomly select 2-4 aspects to mention
        num_aspects = random.randint(2, 4)
        
        review = generate_review(listing_id, date, profile, num_aspects)
        reviews.append(review)
    
    return reviews


def generate_synthetic_dataset(
    n_listings: int = 5,
    reviews_per_listing: int = 20,
    quality_distribution: Optional[Dict[str, int]] = None
) -> List[RawReview]:
    """
    Generate a complete synthetic dataset with multiple listings.
    
    Args:
        n_listings: Number of listings to generate
        reviews_per_listing: Reviews per listing (can vary ±5)
        quality_distribution: Dict specifying how many listings of each quality
                            e.g., {"excellent": 1, "good": 2, "average": 1, "poor": 1}
                            
    Returns:
        List of all generated RawReview objects
    """
    if quality_distribution is None:
        quality_distribution = {
            "excellent": max(1, n_listings // 5),
            "good": max(1, n_listings // 3),
            "average": max(1, n_listings // 4),
            "poor": max(1, n_listings // 5),
            "mixed": max(1, n_listings // 5)
        }
    
    # Build list of quality assignments
    qualities = []
    for quality, count in quality_distribution.items():
        qualities.extend([quality] * count)
    
    # Pad or trim to match n_listings
    while len(qualities) < n_listings:
        qualities.append(random.choice(["good", "average"]))
    qualities = qualities[:n_listings]
    random.shuffle(qualities)
    
    all_reviews = []
    
    for i in range(n_listings):
        listing_id = f"listing_{i+1:04d}"
        quality = qualities[i]
        
        # Vary number of reviews slightly
        n_reviews = reviews_per_listing + random.randint(-5, 5)
        n_reviews = max(5, n_reviews)
        
        reviews = generate_listing_reviews(listing_id, n_reviews, quality)
        all_reviews.extend(reviews)
    
    return all_reviews


# =============================================================================
# SPECIFIC TEST SCENARIOS
# =============================================================================

def generate_negation_test_reviews() -> List[RawReview]:
    """
    Generate reviews specifically testing negation handling.
    """
    now = datetime.now()
    
    test_cases = [
        # Negated positive → should be negative
        "The room was not clean at all. The bathroom wasn't sanitized.",
        # Negated negative → should be positive  
        "The area wasn't dangerous. The place was not dirty.",
        # Double negation
        "I wouldn't say the host wasn't helpful. Not a bad location.",
        # Negation with intensifier
        "The apartment was definitely not quiet. Really not clean.",
        # Standard positive (control)
        "The location was excellent and the host was very helpful.",
    ]
    
    reviews = []
    for i, text in enumerate(test_cases):
        review = RawReview(
            review_id=f"negation_test_{i+1}",
            listing_id="test_listing_001",
            reviewer_name="TestReviewer",
            review_date=now - timedelta(days=i*30),
            review_text=text
        )
        reviews.append(review)
    
    return reviews


def generate_temporal_test_reviews() -> List[RawReview]:
    """
    Generate reviews testing temporal weighting.
    
    Creates a pattern where:
    - Old reviews (>1 year): mostly negative
    - Recent reviews (<3 months): mostly positive
    
    This should result in positive weighted sentiment despite
    negative raw average.
    """
    now = datetime.now()
    reviews = []
    
    # Old negative reviews (12-24 months ago)
    for i in range(10):
        days_ago = random.randint(365, 730)
        review = RawReview(
            review_id=f"temporal_old_{i+1}",
            listing_id="temporal_test_listing",
            reviewer_name=random.choice(FIRST_NAMES),
            review_date=now - timedelta(days=days_ago),
            review_text=random.choice(NEGATIVE_PHRASES["cleanliness"]) + " " + 
                       random.choice(NEGATIVE_PHRASES["host_behavior"])
        )
        reviews.append(review)
    
    # Recent positive reviews (0-90 days ago)
    for i in range(10):
        days_ago = random.randint(0, 90)
        review = RawReview(
            review_id=f"temporal_recent_{i+1}",
            listing_id="temporal_test_listing",
            reviewer_name=random.choice(FIRST_NAMES),
            review_date=now - timedelta(days=days_ago),
            review_text=random.choice(POSITIVE_PHRASES["cleanliness"]) + " " +
                       random.choice(POSITIVE_PHRASES["host_behavior"])
        )
        reviews.append(review)
    
    return reviews


def generate_polarized_test_reviews() -> List[RawReview]:
    """
    Generate reviews testing disagreement detection.
    
    Creates a polarized pattern where half the reviews are
    very positive and half are very negative.
    """
    now = datetime.now()
    reviews = []
    
    # Very positive reviews
    for i in range(10):
        days_ago = random.randint(0, 365)
        review = RawReview(
            review_id=f"polar_positive_{i+1}",
            listing_id="polarized_test_listing",
            reviewer_name=random.choice(FIRST_NAMES),
            review_date=now - timedelta(days=days_ago),
            review_text="Amazing place! " + random.choice(POSITIVE_PHRASES["cleanliness"]) + 
                       " " + random.choice(POSITIVE_PHRASES["location"]) + " Highly recommend!"
        )
        reviews.append(review)
    
    # Very negative reviews
    for i in range(10):
        days_ago = random.randint(0, 365)
        review = RawReview(
            review_id=f"polar_negative_{i+1}",
            listing_id="polarized_test_listing",
            reviewer_name=random.choice(FIRST_NAMES),
            review_date=now - timedelta(days=days_ago),
            review_text="Terrible experience! " + random.choice(NEGATIVE_PHRASES["cleanliness"]) +
                       " " + random.choice(NEGATIVE_PHRASES["location"]) + " Avoid!"
        )
        reviews.append(review)
    
    return reviews
