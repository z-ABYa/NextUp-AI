// components/Discovery.js
// Browse the movie catalogue filtered by genre AND/OR mood/vibe.

import React, { useState, useEffect, useCallback } from "react";
import "../styles/Discovery.css";

const GENRES = [
  "", "Action", "Adventure", "Animation", "Children's",
  "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
  "Film-Noir", "Horror", "Musical", "Mystery", "Romance",
  "Sci-Fi", "Thriller", "War", "Western",
];

const VIBES = [
  { value: "",         label: "Any vibe"  },
  { value: "happy",    label: "😄 Happy"   },
  { value: "dark",     label: "🌑 Dark"    },
  { value: "fun",      label: "🎉 Fun"     },
  { value: "romantic", label: "💕 Romantic"},
  { value: "epic",     label: "⚔️ Epic"    },
  { value: "classic",  label: "🎩 Classic" },
];

function Discovery({ api }) {
  // ── State ──────────────────────────────────────────────────────────────
  const [movies,  setMovies]  = useState([]);
  const [genre,   setGenre]   = useState("");
  const [vibe,    setVibe]    = useState("");
  const [loading, setLoading] = useState(false);
  const [error,   setError]   = useState("");
  const [page,    setPage]    = useState(1);
  const [total,   setTotal]   = useState(0);

  const LIMIT = 24;

  // ── Fetch ──────────────────────────────────────────────────────────────

  const fetchMovies = useCallback(async (g, v, p) => {
    setLoading(true);
    setError("");

    const url = new URL(`${api}/movies`);
    if (g) url.searchParams.set("genre", g);
    if (v) url.searchParams.set("vibe",  v);
    url.searchParams.set("page",  p);
    url.searchParams.set("limit", LIMIT);

    try {
      const res  = await fetch(url.toString(), { credentials: "include" });
      const data = await res.json();
      setMovies(data.movies || []);
      setTotal(data.total   || 0);
    } catch {
      setError("Could not load movies.");
    } finally {
      setLoading(false);
    }
  }, [api]);

  useEffect(() => {
    fetchMovies(genre, vibe, page);
  }, [genre, vibe, page, fetchMovies]);

  // ── Handlers ───────────────────────────────────────────────────────────

  const handleGenre = (e) => { setGenre(e.target.value); setPage(1); };
  const handleVibe  = (v)  => { setVibe(v === vibe ? "" : v); setPage(1); };

  const totalPages = Math.ceil(total / LIMIT);

  // ── Render ─────────────────────────────────────────────────────────────

  return (
    <div className="discovery-page">
      {/* ── Header ── */}
      <div className="discovery-header">
        <h2>🔍 Discover Movies</h2>
        <p>Browse the catalogue by genre and mood.</p>
      </div>

      {/* ── Filters ── */}
      <div className="filter-panel">
        {/* Genre dropdown */}
        <div className="filter-group">
          <label htmlFor="genre-select">Genre</label>
          <select
            id="genre-select"
            value={genre}
            onChange={handleGenre}
          >
            <option value="">All genres</option>
            {GENRES.filter(Boolean).map(g => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>

        {/* Vibe pills */}
        <div className="filter-group">
          <label>Vibe</label>
          <div className="vibe-pills">
            {VIBES.map(v => (
              <button
                key={v.value}
                className={`vibe-pill ${vibe === v.value && v.value ? "active" : ""}`}
                onClick={() => handleVibe(v.value)}
              >
                {v.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* ── Results count ── */}
      <p className="results-count">
        {loading ? "Searching…" : `${total} movie${total !== 1 ? "s" : ""} found`}
      </p>

      {/* ── Grid ── */}
      {error && <p className="disc-error">{error}</p>}

      {!loading && (
        <div className="discovery-grid">
          {movies.map(movie => (
            <div key={movie.movie_id} className="disc-card">
              {/* Star rating display */}
              <div className="disc-stars">
                {Array.from({ length: 5 }, (_, i) => (
                  <span
                    key={i}
                    className={i < Math.round(movie.avg_rating) ? "star on" : "star off"}
                  >
                    ★
                  </span>
                ))}
                <span className="disc-avg">{movie.avg_rating.toFixed(1)}</span>
              </div>

              <h3 className="disc-title">{movie.title}</h3>
              <p className="disc-genres">
                {(movie.genres || "").replace(/\|/g, " · ")}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* ── Pagination ── */}
      {totalPages > 1 && (
        <div className="disc-pagination">
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
    </div>
  );
}

export default Discovery;
