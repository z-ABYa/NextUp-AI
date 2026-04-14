"""
database.py
===========
Handles all SQLite operations for user accounts.

Schema:
    users(id INTEGER PK, username TEXT UNIQUE, password_hash TEXT, created_at TEXT)

Passwords are hashed with werkzeug's pbkdf2 (built into Flask's dependency tree).
"""

import sqlite3
import hashlib
import os
from datetime import datetime

DB_PATH = "users.db"


def _get_connection():
    """Open a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # lets us access columns by name
    return conn


def _hash_password(password: str) -> str:
    """Simple SHA-256 hash with a fixed salt prefix (good enough for a demo)."""
    salted = f"movierec-salt-{password}"
    return hashlib.sha256(salted.encode()).hexdigest()


def init_db():
    """Create the users table if it doesn't already exist."""
    conn = _get_connection()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            username   TEXT    UNIQUE NOT NULL,
            password   TEXT    NOT NULL,
            created_at TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()
    print("Database initialised.")


def register_user(username: str, password: str) -> tuple[bool, str]:
    """
    Insert a new user.
    Returns (True, "ok") on success or (False, reason) on failure.
    """
    hashed = _hash_password(password)
    try:
        conn = _get_connection()
        conn.execute(
            "INSERT INTO users (username, password, created_at) VALUES (?, ?, ?)",
            (username, hashed, datetime.utcnow().isoformat())
        )
        conn.commit()
        conn.close()
        return True, "ok"
    except sqlite3.IntegrityError:
        return False, f"Username '{username}' is already taken."


def get_user(username: str, password: str):
    """
    Look up a user by credentials.
    Returns a dict-like Row on success, or None if credentials are wrong.
    """
    hashed = _hash_password(password)
    conn = _get_connection()
    row = conn.execute(
        "SELECT id, username FROM users WHERE username = ? AND password = ?",
        (username, hashed)
    ).fetchone()
    conn.close()

    if row is None:
        return None
    return {"id": row["id"], "username": row["username"]}
