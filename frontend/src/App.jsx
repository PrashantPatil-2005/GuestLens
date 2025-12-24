import { useState, useEffect } from 'react'
import ListingInput from './components/ListingInput'
import StatusCard from './components/StatusCard'
import AspectsGrid from './components/AspectsGrid'
import RiskDrivers from './components/RiskDrivers'
import ActionBanner from './components/ActionBanner'

/**
 * App.jsx - Main dashboard component
 * 
 * Why single-page:
 * - Decision-makers need ONE view with everything
 * - No navigation means faster decisions
 * - Complete context without clicking around
 */

// Use relative path for Vercel deployment, fallback to localhost for dev
const API_BASE = import.meta.env.PROD ? '/api' : 'http://localhost:8000'

function App() {
    const [listings, setListings] = useState([])
    const [selectedListing, setSelectedListing] = useState('')
    const [assessment, setAssessment] = useState(null)
    const [loading, setLoading] = useState(false)
    const [error, setError] = useState(null)

    // Fetch available listings on mount
    useEffect(() => {
        fetch(`${API_BASE}/listings`)
            .then(res => res.json())
            .then(data => {
                setListings(data)
                if (data.length > 0) {
                    setSelectedListing(data[0])
                }
            })
            .catch(err => setError('Failed to load listings'))
    }, [])

    // Fetch assessment when listing changes
    useEffect(() => {
        if (!selectedListing) return

        setLoading(true)
        setError(null)

        fetch(`${API_BASE}/listing/${selectedListing}`)
            .then(res => {
                if (!res.ok) throw new Error('Listing not found')
                return res.json()
            })
            .then(data => {
                setAssessment(data)
                setLoading(false)
            })
            .catch(err => {
                setError(err.message)
                setLoading(false)
            })
    }, [selectedListing])

    return (
        <div className="app">
            <h1>🏠 Listing Intelligence Dashboard</h1>

            {/* 
        Listing selector - allows switching between listings
        In production: would filter by region, risk level, etc.
      */}
            <ListingInput
                listings={listings}
                selected={selectedListing}
                onChange={setSelectedListing}
            />

            {loading && <div className="loading">Loading assessment...</div>}
            {error && <div className="error">Error: {error}</div>}

            {assessment && !loading && (
                <>
                    {/* 
            Status Card - THE most important element
            Decision-maker should understand status in 2 seconds
          */}
                    <StatusCard
                        risk={assessment.overall_risk}
                        riskScore={assessment.risk_score}
                        confidence={assessment.confidence}
                        ratingTrust={assessment.rating_trust}
                    />

                    {/* 
            Aspect breakdown - details on each dimension
            Answers: "WHERE are the problems?"
          */}
                    <AspectsGrid aspects={assessment.aspects} />

                    {/* 
            Risk drivers - explicit explanation
            Answers: "WHY is this flagged?"
          */}
                    <RiskDrivers drivers={assessment.risk_drivers} />

                    {/* 
            Action banner - clear next step
            Answers: "WHAT should I do?"
          */}
                    <ActionBanner action={assessment.recommended_action} />

                    {/* Metadata footer */}
                    <div className="metadata">
                        {assessment.review_count} reviews analyzed •
                        Assessment: {new Date(assessment.assessment_date).toLocaleDateString()}
                    </div>
                </>
            )}
        </div>
    )
}

export default App
