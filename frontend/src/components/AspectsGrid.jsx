/**
 * AspectsGrid.jsx - Per-aspect breakdown
 * 
 * Answers: "WHERE are the problems?"
 * 
 * Each aspect card shows:
 * - Sentiment (positive/neutral/negative)
 * - Trend (improving/stable/declining)
 * - Variance (how consistent are opinions)
 * - Mention count (data volume)
 */

function AspectCard({ aspect }) {
    // Color-code trends for quick scanning
    const trendClass = `trend-${aspect.trend}`

    return (
        <div className="aspect-card">
            <h3>{aspect.name.replace('_', ' ')}</h3>

            <div className="aspect-row">
                <span>Sentiment</span>
                <span>{aspect.sentiment_label} ({aspect.sentiment.toFixed(2)})</span>
            </div>

            <div className="aspect-row">
                <span>Trend</span>
                <span className={trendClass}>
                    {aspect.trend === 'improving' && '↑ '}
                    {aspect.trend === 'declining' && '↓ '}
                    {aspect.trend}
                </span>
            </div>

            <div className="aspect-row">
                <span>Consistency</span>
                <span>{aspect.variance_label}</span>
            </div>

            <div className="aspect-row">
                <span>Mentions</span>
                <span>{aspect.mention_count}</span>
            </div>
        </div>
    )
}

function AspectsGrid({ aspects }) {
    return (
        <div className="aspects-grid">
            {aspects.map((aspect) => (
                <AspectCard key={aspect.name} aspect={aspect} />
            ))}
        </div>
    )
}

export default AspectsGrid
