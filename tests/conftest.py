"""
Pytest configuration and fixtures for SQL Engine test suite.
Automatically provisions minimal test database fixtures if Spider dataset is absent.
"""
import os
import sqlite3

import pytest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIXTURE_DIR = os.path.join(REPO_ROOT, "tests", "fixtures", "database", "concert_singer")
FIXTURE_DB_PATH = os.path.join(FIXTURE_DIR, "concert_singer.sqlite")


def init_mock_concert_singer(db_path: str) -> None:
    """Initialize a minimal concert_singer SQLite database with 6 sample rows."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS singer;")
    cur.execute("""
        CREATE TABLE singer (
            Singer_ID INTEGER PRIMARY KEY,
            Name TEXT,
            Country TEXT,
            Song_Name TEXT,
            Song_release_year TEXT,
            Age INTEGER,
            Is_male TEXT
        );
    """)
    rows = [
        (1, "Joe", "USA", "Song A", "2010", 25, "T"),
        (2, "Bob", "UK", "Song B", "2012", 30, "T"),
        (3, "Alice", "Canada", "Song C", "2015", 22, "F"),
        (4, "Dave", "USA", "Song D", "2018", 28, "T"),
        (5, "Eve", "France", "Song E", "2020", 21, "F"),
        (6, "Frank", "Germany", "Song F", "2021", 35, "T"),
    ]
    cur.executemany("INSERT INTO singer VALUES (?, ?, ?, ?, ?, ?, ?);", rows)
    conn.commit()
    conn.close()


@pytest.fixture(scope="session", autouse=True)
def ensure_test_fixtures():
    """Ensure minimal test fixtures exist for isolated execution."""
    real_path = os.path.join(REPO_ROOT, "data", "spider_data", "database", "concert_singer", "concert_singer.sqlite")
    if not os.path.exists(real_path) and not os.path.exists(FIXTURE_DB_PATH):
        init_mock_concert_singer(FIXTURE_DB_PATH)
    yield
