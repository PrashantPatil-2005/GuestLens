"""
main.py - FastAPI backend for Guest Review Intelligence Dashboard

Endpoints:
- GET /api/listing/{listing_id} - Get Phase-2 assessment for a listing
- GET /api/listings - Get available listing IDs

This is a minimal decision-support API, not a full CRUD system.
Configured for Vercel serverless deployment.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List

from api.schemas import ListingAssessment, ErrorResponse
from api.mock_data import generate_mock_assessment, get_available_listings


# =============================================================================
# APP INITIALIZATION
# =============================================================================

app = FastAPI(
    title="Guest Review Intelligence API",
    description="Phase-2 risk assessment data for decision-support dashboard",
    version="1.0.0"
)

# CORS middleware - allow frontend to access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for Vercel deployment
    allow_credentials=True,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# =============================================================================
# ENDPOINTS (with /api prefix for Vercel routing)
# =============================================================================

@app.get("/")
@app.get("/api")
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Guest Review Intelligence API"}


@app.get("/api/listings", response_model=List[str])
def list_available_listings():
    """
    Get list of available listing IDs.
    
    Why:
    - Allows frontend to show a dropdown/selector
    - In production, would filter by user's access permissions
    """
    return get_available_listings()


@app.get(
    "/api/listing/{listing_id}",
    response_model=ListingAssessment,
    responses={404: {"model": ErrorResponse}}
)
def get_listing_assessment(listing_id: str):
    """
    Get Phase-2 risk assessment for a specific listing.
    
    Returns:
    - Overall risk level (High/Medium/Low)
    - Risk score (0-100)
    - Confidence score (0-1)
    - Rating trust (Reliable/Unreliable)
    - Per-aspect breakdown (sentiment, trend, variance)
    - Risk drivers (why is this flagged?)
    - Recommended action (flag/monitor/ignore)
    
    Why single endpoint:
    - Decision-makers need ALL this info at once
    - Minimizes API calls
    - Complete picture in one request
    """
    assessment = generate_mock_assessment(listing_id)
    
    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail=f"Listing '{listing_id}' not found. Available: {get_available_listings()}"
        )
    
    return assessment


# =============================================================================
# PRODUCTION HOOK (for future use)
# =============================================================================

def get_real_assessment(listing_id: str) -> ListingAssessment:
    """
    Hook for production: replace mock_data with real Phase-2 pipeline.
    
    Would involve:
    1. Query Phase-1 pipeline with listing's reviews
    2. Pass to Phase-2 risk pipeline
    3. Convert ListingRiskAssessment to API schema
    """
    # TODO: Integrate with actual pipeline
    # from src.pipeline import run_pipeline
    # from src.risk_pipeline import assess_listing_risk
    raise NotImplementedError("Real pipeline integration pending")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
