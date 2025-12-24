/**
 * StatusCard.jsx - Overall status display
 * 
 * THE most important component on the page.
 * Decision-maker should understand listing status in 2 seconds.
 * 
 * Design choices:
 * - Color-coded border (red/yellow/green) for instant recognition
 * - Large, bold risk level text
 * - Supporting metrics (score, confidence, trust) below
 */

function StatusCard({ risk, riskScore, confidence, ratingTrust }) {
    // Determine CSS class based on risk level
    const riskClass = `risk-${risk.toLowerCase()}`
    const riskValueClass = risk.toLowerCase()

    return (
        <div className={`status-card ${riskClass}`}>
            <h2>Overall Status</h2>

            <div className="status-metrics">
                {/* Primary: Risk Level - the main headline */}
                <div className="metric">
                    <div className="metric-label">Risk Level</div>
                    <div className={`metric-value ${riskValueClass}`}>{risk}</div>
                </div>

                {/* Secondary: Numeric score for precision */}
                <div className="metric">
                    <div className="metric-label">Risk Score</div>
                    <div className="metric-value">{riskScore}/100</div>
                </div>

                {/* Tertiary: How much to trust this assessment */}
                <div className="metric">
                    <div className="metric-label">Confidence</div>
                    <div className="metric-value">{Math.round(confidence * 100)}%</div>
                </div>

                {/* Rating reliability indicator */}
                <div className="metric">
                    <div className="metric-label">Rating Trust</div>
                    <div className={`metric-value ${ratingTrust === 'Reliable' ? 'low' : 'high'}`}>
                        {ratingTrust}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default StatusCard
