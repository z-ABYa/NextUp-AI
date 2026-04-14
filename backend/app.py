"""
app.py
======
Flask backend for the Movie Recommendation App.

Startup:
    python app.py

Loads the pre-trained ItemCF model once at startup, then serves
JSON APIs consumed by the React frontend.
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sys,os

sys.path.append(os.path.abspath(".."))

from database import init_db, register_user, get_user
from model_loader import load_model, recommend_for_new_user, filter_movies

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
# Fix session issues with cross-origin requests (React <-> Flask)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Allow requests from the React dev server (localhost:3000)
# CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
CORS(app, supports_credentials=True, origins=[
    "http://127.0.0.1:3000", "http://localhost:3000"
    ])

# ---------------------------------------------------------------------------
# Load model & movie catalogue once at startup
# ---------------------------------------------------------------------------

print("Loading model and movie data …")
MODEL, MOVIES_DF = load_model("../recommender.pkl", "data/movies.dat")
print(f"Ready!  {len(MOVIES_DF)} movies loaded.")

# ---------------------------------------------------------------------------
# Initialise database
# ---------------------------------------------------------------------------

init_db()

# ===========================================================================
# Auth routes
# ===========================================================================

@app.route("/register", methods=["POST"])
def register():
    """
    Register a new user.
    Body: { "username": str, "password": str }
    """
    data = request.get_json()

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400

    success, msg = register_user(username, password)
    if not success:
        return jsonify({"error": msg}), 409   # 409 Conflict

    return jsonify({"message": "Registration successful!"}), 201


@app.route("/login", methods=["POST"])
def login():
    """
    Log in an existing user.
    Body: { "username": str, "password": str }
    Sets a server-side session cookie.
    """
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    user = get_user(username, password)
    if user is None:
        return jsonify({"error": "Invalid username or password."}), 401

    # Store the user_id and username in the server session
    session["user_id"]  = user["id"]
    session["username"] = user["username"]

    return jsonify({"message": "Logged in.", "username": user["username"]}), 200


@app.route("/logout", methods=["POST"])
def logout():
    """Clear the session (log out)."""
    session.clear()
    return jsonify({"message": "Logged out."}), 200


@app.route("/me", methods=["GET"])
def me():
    """Return current session user (used by React to restore login state)."""
    if "username" not in session:
        return jsonify({"user": None}), 200
    return jsonify({"user": session["username"]}), 200


# ===========================================================================
# Movie routes
# ===========================================================================

@app.route("/movies", methods=["GET"])
def movies():
    """
    GET /movies?genre=Action&vibe=dark&page=1&limit=50

    Returns a paginated list of movies, optionally filtered by:
      - genre : exact genre name (e.g. "Action", "Comedy")
      - vibe  : mood keyword mapped to one or more genres
                happy      → Comedy|Animation
                dark       → Thriller|Crime|Horror
                fun        → Comedy|Adventure|Children's
                romantic   → Romance
                epic       → Action|Adventure|Sci-Fi
                classic    → Drama|Film-Noir
    """
    genre = request.args.get("genre", "").strip()
    vibe  = request.args.get("vibe", "").strip()
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))

    result = filter_movies(MOVIES_DF, genre=genre, vibe=vibe)

    # Paginate
    total  = len(result)
    start  = (page - 1) * limit
    end    = start + limit
    page_data = result.iloc[start:end]

    return jsonify({
        "total":  total,
        "page":   page,
        "limit":  limit,
        "movies": page_data.to_dict(orient="records"),
    }), 200


# ===========================================================================
# Recommendation route
# ===========================================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    Body: {
        "ratings": { "<movie_id>": <rating 1-5>, ... }   // at least 5 entries
    }
    Returns top-10 recommended movies the user has NOT rated.

    Uses the pre-trained ItemCF model's item-item similarity matrix
    to score every unseen movie via weighted-neighbour aggregation.
    """
    print("SESSION:", dict(session))
    if "username" not in session:
        return jsonify({"error": "Please log in first."}), 401

    data = request.get_json()
    raw_ratings = data.get("ratings", {})

    # Validate input
    if len(raw_ratings) < 5:
        return jsonify({"error": "Please rate at least 5 movies."}), 400

    # Convert keys to ints (JSON keys are always strings)
    try:
        user_ratings = {int(k): float(v) for k, v in raw_ratings.items()}
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid rating data format."}), 400

    # Run cold-start recommendation using item-item similarity
    recs = recommend_for_new_user(MODEL, MOVIES_DF, user_ratings, n=10)

    """
app.py
======
Flask backend for the Movie Recommendation App.

Startup:
    python app.py

Loads the pre-trained ItemCF model once at startup, then serves
JSON APIs consumed by the React frontend.
"""

from flask import Flask, request, jsonify, session
from flask_cors import CORS
import sys,os

sys.path.append(os.path.abspath(".."))

from database import init_db, register_user, get_user
from model_loader import load_model, recommend_for_new_user, filter_movies

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = Flask(__name__)
# Fix session issues with cross-origin requests (React <-> Flask)
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False
app.secret_key = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-production")

# Allow requests from the React dev server (localhost:3000)
# CORS(app, supports_credentials=True, origins=["http://localhost:3000"])
CORS(app, supports_credentials=True, origins=["http://127.0.0.1:3000", "http://localhost:3000"])

# ---------------------------------------------------------------------------
# Load model & movie catalogue once at startup
# ---------------------------------------------------------------------------

print("Loading model and movie data …")
MODEL, MOVIES_DF = load_model("../recommender.pkl", "data/movies.dat")
print(f"Ready!  {len(MOVIES_DF)} movies loaded.")

# ---------------------------------------------------------------------------
# Initialise database
# ---------------------------------------------------------------------------

init_db()

# ===========================================================================
# Auth routes
# ===========================================================================

@app.route("/register", methods=["POST"])
def register():
    """
    Register a new user.
    Body: { "username": str, "password": str }
    """
    data = request.get_json()

    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required."}), 400

    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters."}), 400

    success, msg = register_user(username, password)
    if not success:
        return jsonify({"error": msg}), 409   # 409 Conflict

    return jsonify({"message": "Registration successful!"}), 201


@app.route("/login", methods=["POST"])
def login():
    """
    Log in an existing user.
    Body: { "username": str, "password": str }
    Sets a server-side session cookie.
    """
    data = request.get_json()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()

    user = get_user(username, password)
    if user is None:
        return jsonify({"error": "Invalid username or password."}), 401

    # Store the user_id and username in the server session
    session["user_id"]  = user["id"]
    session["username"] = user["username"]

    return jsonify({"message": "Logged in.", "username": user["username"]}), 200


@app.route("/logout", methods=["POST"])
def logout():
    """Clear the session (log out)."""
    session.clear()
    return jsonify({"message": "Logged out."}), 200


@app.route("/me", methods=["GET"])
def me():
    """Return current session user (used by React to restore login state)."""
    if "username" not in session:
        return jsonify({"user": None}), 200
    return jsonify({"user": session["username"]}), 200


# ===========================================================================
# Movie routes
# ===========================================================================

@app.route("/movies", methods=["GET"])
def movies():
    """
    GET /movies?genre=Action&vibe=dark&page=1&limit=50

    Returns a paginated list of movies, optionally filtered by:
      - genre : exact genre name (e.g. "Action", "Comedy")
      - vibe  : mood keyword mapped to one or more genres
                happy      → Comedy|Animation
                dark       → Thriller|Crime|Horror
                fun        → Comedy|Adventure|Children's
                romantic   → Romance
                epic       → Action|Adventure|Sci-Fi
                classic    → Drama|Film-Noir
    """
    genre = request.args.get("genre", "").strip()
    vibe  = request.args.get("vibe", "").strip()
    page  = int(request.args.get("page", 1))
    limit = int(request.args.get("limit", 50))

    result = filter_movies(MOVIES_DF, genre=genre, vibe=vibe)

    # Paginate
    total  = len(result)
    start  = (page - 1) * limit
    end    = start + limit
    page_data = result.iloc[start:end]

    return jsonify({
        "total":  total,
        "page":   page,
        "limit":  limit,
        "movies": page_data.to_dict(orient="records"),
    }), 200


# ===========================================================================
# Recommendation route
# ===========================================================================

@app.route("/recommend", methods=["POST"])
def recommend():
    """
    POST /recommend
    Body: {
        "ratings": { "<movie_id>": <rating 1-5>, ... }   // at least 5 entries
    }
    Returns top-10 recommended movies the user has NOT rated.

    Uses the pre-trained ItemCF model's item-item similarity matrix
    to score every unseen movie via weighted-neighbour aggregation.
    """
    print("SESSION:", dict(session))
    if "username" not in session:
        return jsonify({"error": "Please log in first."}), 401

    data = request.get_json()
    raw_ratings = data.get("ratings", {})

    # Validate input
    if len(raw_ratings) < 5:
        return jsonify({"error": "Please rate at least 5 movies."}), 400

    # Convert keys to ints (JSON keys are always strings)
    try:
        user_ratings = {int(k): float(v) for k, v in raw_ratings.items()}
    except (ValueError, TypeError):
        return jsonify({"error": "Invalid rating data format."}), 400

    # Run cold-start recommendation using item-item similarity
    recs = recommend_for_new_user(MODEL, MOVIES_DF, user_ratings, n=10)

    # Convert numpy types (e.g., int64, float32) to native Python types for JSON
    def convert(o):
        try:
            import numpy as np
            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
        except:
            pass
        return o

    # Apply conversion to each recommendation dict
    recs = [{k: convert(v) for k, v in rec.items()} for rec in recs]

    return jsonify({"recommendations": recs}), 200

# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
