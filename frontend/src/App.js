// App.js
// Root component. Manages global auth state and renders the correct page.

import React, { useState, useEffect } from "react";
import Login from "./components/Login";
import Register from "./components/Register";
import MovieSelector from "./components/MovieSelector";
import Recommendations from "./components/Recommendations";
import Discovery from "./components/Discovery";
import "./styles/App.css";

const API = "http://127.0.0.1:5000";

function App() {
  // ── Auth state ──────────────────────────────────────────────────────────
  const [user, setUser]   = useState(null);   // null = not logged in
  const [page, setPage]   = useState("login"); // current page/view
  const [loading, setLoading] = useState(true);

  // Recommendations passed from MovieSelector → Recommendations page
  const [recs, setRecs] = useState([]);

  // On first load, ask the server if we already have a session
  useEffect(() => {
    fetch(`${API}/me`, { credentials: "include" })
      .then(r => r.json())
      .then(data => {
        if (data.user) {
          setUser(data.user);
          setPage("discover");
        }
      })
      .finally(() => setLoading(false));
  }, []);

  // ── Handlers ─────────────────────────────────────────────────────────────

  const handleLogin = (username) => {
    setUser(username);
    setPage("discover");
  };

  const handleLogout = () => {
    fetch(`${API}/logout`, { method: "POST", credentials: "include" });
    setUser(null);
    setPage("login");
    setRecs([]);
  };

  const handleRecommendations = (recommendations) => {
    setRecs(recommendations);
    setPage("recs");
  };

  // ── Render ────────────────────────────────────────────────────────────────

  if (loading) {
    return <div className="splash">Loading…</div>;
  }

  return (
    <div className="app">
      {/* ── Top nav bar (only when logged in) ── */}
      {user && (
        <nav className="navbar">
          <span className="nav-brand">🎬 NextUp AI</span>
          <div className="nav-links">
            <button
              className={page === "discover" ? "nav-btn active" : "nav-btn"}
              onClick={() => setPage("discover")}
            >
              Discover
            </button>
            <button
              className={page === "rate" ? "nav-btn active" : "nav-btn"}
              onClick={() => setPage("rate")}
            >
              Get Recommendations
            </button>
            {recs.length > 0 && (
              <button
                className={page === "recs" ? "nav-btn active" : "nav-btn"}
                onClick={() => setPage("recs")}
              >
                My Recs
              </button>
            )}
          </div>
          <div className="nav-user">
            <span>👤 {user}</span>
            <button className="logout-btn" onClick={handleLogout}>
              Log Out
            </button>
          </div>
        </nav>
      )}

      {/* ── Page routing ── */}
      <main className="main-content">
        {/* Auth pages */}
        {!user && page === "login" && (
          <Login
            api={API}
            onLogin={handleLogin}
            onSwitch={() => setPage("register")}
          />
        )}

        {!user && page === "register" && (
          <Register
            api={API}
            onRegistered={() => setPage("login")}
            onSwitch={() => setPage("login")}
          />
        )}

        {/* Protected pages */}
        {user && page === "discover" && (
          <Discovery api={API} />
        )}

        {user && page === "rate" && (
          <MovieSelector
            api={API}
            onRecommend={handleRecommendations}
          />
        )}

        {user && page === "recs" && (
          <Recommendations
            recs={recs}
            onBack={() => setPage("rate")}
          />
        )}
      </main>
    </div>
  );
}

export default App;
