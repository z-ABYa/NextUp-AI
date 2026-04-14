// components/MovieSelector.js
// Browse movies, give them star ratings, submit to /recommend.
// The user must rate at least 5 movies before submitting.

import React, { useState, useEffect, useCallback } from "react";
import "../styles/MovieSelector.css";

const GENRES = [
  "All", "Action", "Adventure", "Animation", "Children's",
  "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
  "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
  "Sci-Fi", "Thriller", "War", "Western",
];

// Simple star rating sub-component
function StarRating({ movieId, value, onChange }) {
  const [hover, setHover] = useState(0);

  return (
    <div className="stars">
      {[1, 2, 3, 4, 5].map((star) => (
        <button
          key={star}
          className={`star ${star <= (hover || value) ? "filled" : ""}`}
          onClick={() => onChange(movieId, star)}
          onMouseEnter={() => setHover(star)}
          onMouseLeave={() => setHover(0)}
          title={`${star} star${star > 1 ? "s" : ""}`}
        >
          ★
        </button>
      ))}
    </div>
  );
}

function MovieSelector({ api, onRecommend }) {
  // ── State ──────────────────────────────────────────────────────────────
  const [movies,    setMovies]    = useState([]);
  const [genre,     setGenre]     = useState("All");
  const [search,    setSearch]    = useState("");
  const [ratings,   setRatings]   = useState({});   // { movie_id: 1-5 }
  const [loading,   setLoading]   = useState(false);
  const [submitting,setSubmitting]= useState(false);
  const [error,     setError]     = useState("");
  const [page,      setPage]      = useState(1);
  const [total,     setTotal]     = useState(0);

  const LIMIT = 30;

  // ── Fetch movies ───────────────────────────────────────────────────────

  const fetchMovies = useCallback(async (selectedGenre, currentPage) => {
    setLoading(true);
    setError("");

    const g = selectedGenre === "All" ? "" : selectedGenre;
    const url = `${api}/movies?genre=${encodeURIComponent(g)}&page=${currentPage}&limit=${LIMIT}`;

    try {
      const res  = await fetch(url, { credentials: "include" });
      const data = await res.json();
      setMovies(data.movies || []);
      setTotal(data.total  || 0);
    } catch {
      setError("Could not load movies. Is Flask running?");
    } finally {
      setLoading(false);
    }
  }, [api]);

  // Re-fetch whenever genre or page changes
  useEffect(() => {
    fetchMovies(genre, page);
  }, [genre, page, fetchMovies]);

  // ── Handlers ───────────────────────────────────────────────────────────

  const handleGenreChange = (g) => {
    setGenre(g);
    setPage(1);
  };

  const handleRating = (movieId, stars) => {
    // Toggle: clicking the same rating removes it
    setRatings(prev => {
      if (prev[movieId] === stars) {
        const next = { ...prev };
        delete next[movieId];
        return next;
      }
      return { ...prev, [movieId]: stars };
    });
  };

  const handleSubmit = async () => {
    const count = Object.keys(ratings).length;
    if (count < 5) {
      setError(`Please rate at least 5 movies (you have ${count}).`);
      return;
    }

    setSubmitting(true);
    setError("");

    try {
      const res = await fetch(`${api}/recommend`, {
        method:      "POST",
        credentials: "include",
        headers:     { "Content-Type": "application/json" },
        body:        JSON.stringify({ ratings }),
      });

      const data = await res.json();
      if (!res.ok) {
        setError(data.error || "Recommendation failed.");
      } else {
        onRecommend(data.recommendations);
      }
    } catch {
      setError("Cannot connect to server.");
    } finally {
      setSubmitting(false);
    }
  };

  // ── Filtered display (client-side search on top of server-side genre) ──

  const displayed = search.trim()
    ? movies.filter(m =>
        m.title.toLowerCase().includes(search.toLowerCase())
      )
    : movies;

  const ratedCount   = Object.keys(ratings).length;
  const totalPages   = Math.ceil(total / LIMIT);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="selector-page">
      {/* ── Header ── */}
      <div className="selector-header">
        <h2>Rate Movies to Get Recommendations</h2>
        <p>Rate at least <strong>5 movies</strong> you have seen.</p>

        {/* Progress pill */}
        <div className={`progress-pill ${ratedCount >= 5 ? "ready" : ""}`}>
          {ratedCount} / 5+ rated
        </div>
      </div>

      {/* ── Controls ── */}
      <div className="controls">
        {/* Genre tabs */}
        <div className="genre-tabs">
          {GENRES.map(g => (
            <button
              key={g}
              className={genre === g ? "genre-tab active" : "genre-tab"}
              onClick={() => handleGenreChange(g)}
            >
              {g}
            </button>
          ))}
        </div>

        {/* Search */}
        <input
          className="search-box"
          type="text"
          placeholder="Search titles…"
          value={search}
          onChange={e => setSearch(e.target.value)}
        />
      </div>

      {/* ── Movie grid ── */}
      {loading ? (
        <div className="loader">Loading movies…</div>
      ) : (
        <>
          <div className="movie-grid">
            {displayed.map(movie => {
              const myRating = ratings[movie.movie_id] || 0;
              return (
                <div
                  key={movie.movie_id}
                  className={`movie-card ${myRating ? "rated" : ""}`}
                >
                  {/* Rating badge */}
                  {myRating > 0 && (
                    <span className="rated-badge">Rated {myRating}★</span>
                  )}

                  <div className="movie-info">
                    <h3 className="movie-title">{movie.title}</h3>
                    <p className="movie-genres">{movie.genres.replace(/\|/g, " · ")}</p>
                    <p className="movie-avg">
                      Avg rating: <strong>{movie.avg_rating}</strong>
                    </p>
                  </div>

                  <StarRating
                    movieId={movie.movie_id}
                    value={myRating}
                    onChange={handleRating}
                  />
                </div>
              );
            })}
          </div>

          {/* Pagination */}
          {!search && totalPages > 1 && (
            <div className="pagination">
              <button
                disabled={page <= 1}
                onClick={() => setPage(p => p - 1)}
              >
                ← Prev
              </button>
              <span>Page {page} of {totalPages}</span>
              <button
                disabled={page >= totalPages}
                onClick={() => setPage(p => p + 1)}
              >
                Next →
              </button>
            </div>
          )}
        </>
      )}

      {/* ── Error ── */}
      {error && <p className="selector-error">{error}</p>}

      {/* ── Submit bar ── */}
      <div className="submit-bar">
        <span className="submit-info">
          {ratedCount < 5
            ? `Rate ${5 - ratedCount} more movie${5 - ratedCount !== 1 ? "s" : ""} to continue`
            : `Great! You've rated ${ratedCount} movies.`}
        </span>
        <button
          className="submit-btn"
          onClick={handleSubmit}
          disabled={ratedCount < 5 || submitting}
        >
          {submitting ? "Finding recommendations…" : "Get My Recommendations →"}
        </button>
      </div>
    </div>
  );
}

export default MovieSelector;
