/**
 * ListingInput.jsx - Listing selector component
 * 
 * Why:
 * - Quick switching between listings for comparison
 * - Simple dropdown, no fancy autocomplete needed
 * - In production: would include search/filter
 */

function ListingInput({ listings, selected, onChange }) {
    return (
        <div className="listing-input">
            <label htmlFor="listing-select">Select Listing</label>
            <select
                id="listing-select"
                value={selected}
                onChange={(e) => onChange(e.target.value)}
            >
                {listings.map((id) => (
                    <option key={id} value={id}>
                        {id}
                    </option>
                ))}
            </select>
        </div>
    )
}

export default ListingInput
