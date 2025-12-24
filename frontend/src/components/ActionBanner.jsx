/**
 * ActionBanner.jsx - Clear action recommendation
 * 
 * Answers: "WHAT should I do?"
 * 
 * The final, most actionable element on the page.
 * Should be immediately clear and unambiguous.
 * 
 * Actions:
 * - FLAG (red): Needs immediate attention
 * - MONITOR (yellow): Track but don't panic
 * - IGNORE (green): All good, move on
 */

function ActionBanner({ action }) {
    // Map action to display text
    const actionConfig = {
        flag: {
            className: 'action-flag',
            text: '🚨 ACTION REQUIRED: Flag this listing for review'
        },
        monitor: {
            className: 'action-monitor',
            text: '👁️ Keep monitoring this listing'
        },
        ignore: {
            className: 'action-ignore',
            text: '✓ No action needed'
        }
    }

    const config = actionConfig[action] || actionConfig.ignore

    return (
        <div className={`action-banner ${config.className}`}>
            {config.text}
        </div>
    )
}

export default ActionBanner
