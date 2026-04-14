// components/Recommendations.js
// Displays the top-10 recommendations returned by the Flask /recommend endpoint.

import React from "react";
import "../styles/Recommendations.css";

// Convert a 1-5 score to a visual bar width
function ScoreBar({ score }) {
  const pct = Math.round((score / 5) * 100);
  // Colour goes from yellow (low) → green (high)
  const colour =
    score >= 4.5 ? "#22c55e" :
    score >= 3.5 ? "#86efac" :
    score >= 2.5 ? "#fbbf24" : "#f87171";

  return (
    <div className="score-bar-wrap">
      <div
        className="score-bar-fill"
        style={{ width: `${pct}%`, background: colour }}
      />
    </div>
  );
}

function Recommendations({ recs, onBack }) {
  if (!recs || recs.length === 0) {
    return (
      <div className="recs-page empty">
        <h2>No recommendations yet.</h2>
        <button className="back-btn" onClick={onBack}>
          ← Go Back
        </button>
      </div>
    );
  }

  return (
    <div className="recs-page">
      {/* ── Header ── */}
      <div className="recs-header">
        <h2>🎉 Your Top {recs.length} Recommendations</h2>
        <p>Based on the movies you rated, our Item-CF model predicts you'll love:</p>
        <button className="back-btn" onClick={onBack}>
          ← Rate More Movies
        </button>
      </div>

      {/* ── Cards ── */}
      <div className="recs-list">
        {recs.map((movie, idx) => (
          <div key={movie.movie_id} className="rec-card">
            {/* Rank badge */}
            <div className="rec-rank">#{idx + 1}</div>

            <div className="rec-body">
              {/* Title & genres */}
              <h3 className="rec-title">{movie.title}</h3>
              <p className="rec-genres">
                {(movie.genres || "").replace(/\|/g, " · ")}
              </p>

              {/* Scores */}
              <div className="rec-scores">
                <div className="rec-score-row">
                  <span className="score-label">Predicted score</span>
                  <span className="score-value">
                    {movie.score.toFixed(2)} / 5
                  </span>
                </div>
                <ScoreBar score={movie.score} />

                <div className="rec-score-row" style={{ marginTop: 6 }}>
                  <span className="score-label">Avg community rating</span>
                  <span className="score-value">
                    {movie.avg_rating.toFixed(2)} / 5
                  </span>
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default Recommendations;
