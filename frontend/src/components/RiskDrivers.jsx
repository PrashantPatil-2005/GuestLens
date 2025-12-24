/**
 * RiskDrivers.jsx - Explainability component
 * 
 * Answers: "WHY is this flagged?"
 * 
 * Critical for trust and decision-making.
 * Without this, risk scores are black boxes.
 * 
 * Design choices:
 * - Simple bullet list (no complex visualizations)
 * - Clear, actionable language
 * - Positive state when no issues
 */

function RiskDrivers({ drivers }) {
    return (
        <div className="risk-drivers">
            <h2>⚠️ Risk Drivers</h2>

            {drivers.length === 0 ? (
                <p className="no-drivers">✓ No significant risk factors detected</p>
            ) : (
                <ul>
                    {drivers.map((driver, index) => (
                        <li key={index}>{driver}</li>
                    ))}
                </ul>
            )}
        </div>
    )
}

export default RiskDrivers
