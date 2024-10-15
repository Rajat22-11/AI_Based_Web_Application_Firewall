import React, { useState } from 'react';
import './App.css'; // Import the CSS for styles

function App() {
    const [query, setQuery] = useState('');
    const [result, setResult] = useState(null);
    const [error, setError] = useState(null);
    const [attackType, setAttackType] = useState('sqli'); // Default attack type

    const handlePredict = async () => {
        setError(null);
        setResult(null);

        try {
            const response = await fetch(`http://localhost:5000/predict/${attackType}`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ query }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.error || 'Something went wrong');
            }

            const data = await response.json();
            setResult(data);
        } catch (err) {
            setError(err.message);
        }
    };

    return (
        <div className="App">
            <h1>Attack Prediction System</h1>

            <div className="form-group">
                <label htmlFor="attackType">Select Attack Type:</label>
                <select
                    id="attackType"
                    value={attackType}
                    onChange={(e) => setAttackType(e.target.value)}
                >
                    <option value="sqli">SQL Injection</option>
                    <option value="xss">Cross-Site Scripting (XSS)</option>
                </select>
            </div>

            <div className="form-group">
                <label htmlFor="query">Enter Query:</label>
                <textarea
                    id="query"
                    rows="4"
                    cols="50"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    placeholder="Enter your query here..."
                />
            </div>

            <button onClick={handlePredict}>Predict</button>

            <div className="result-section">
                {result && (
                    <div className={`prediction ${result.Prediction === 1 ? 'alert' : 'safe'}`}>
                        <h3>Prediction:</h3>
                        <p>Type: {result.Type}</p>
                        <p>Confidence: {result.Prediction}</p>
                    </div>
                )}
                {error && (
                    <div className="error">
                        <h3>Error:</h3>
                        <p>{error}</p>
                    </div>
                )}
            </div>
        </div>
    );
}

export default App;
