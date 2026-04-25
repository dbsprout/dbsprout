"""Integration test fixtures: an in-process SQLite DB seeded with a small e-commerce schema."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sqlalchemy as sa

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sqlite_db(tmp_path: Path) -> str:
    db_path = tmp_path / "fixture.db"
    url = f"sqlite:///{db_path}"
    engine = sa.create_engine(url)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT NOT NULL)"))
        conn.execute(
            sa.text("CREATE TABLE products (id INTEGER PRIMARY KEY, sku TEXT NOT NULL UNIQUE)")
        )
        conn.execute(
            sa.text(
                "CREATE TABLE orders ("
                "id INTEGER PRIMARY KEY, user_id INTEGER NOT NULL, "
                "FOREIGN KEY(user_id) REFERENCES users(id))"
            )
        )
        conn.execute(
            sa.text(
                "CREATE TABLE order_items ("
                "id INTEGER PRIMARY KEY, order_id INTEGER NOT NULL, "
                "product_id INTEGER NOT NULL, "
                "FOREIGN KEY(order_id) REFERENCES orders(id), "
                "FOREIGN KEY(product_id) REFERENCES products(id))"
            )
        )
        for i in range(1, 51):
            conn.execute(
                sa.text("INSERT INTO users (id, name) VALUES (:i, :n)"),
                {"i": i, "n": f"user{i}"},
            )
        for i in range(1, 21):
            conn.execute(
                sa.text("INSERT INTO products (id, sku) VALUES (:i, :s)"),
                {"i": i, "s": f"sku-{i:04d}"},
            )
        for i in range(1, 201):
            conn.execute(
                sa.text("INSERT INTO orders (id, user_id) VALUES (:i, :u)"),
                {"i": i, "u": (i % 50) + 1},
            )
        for i in range(1, 401):
            conn.execute(
                sa.text("INSERT INTO order_items (id, order_id, product_id) VALUES (:i, :o, :p)"),
                {"i": i, "o": (i % 200) + 1, "p": (i % 20) + 1},
            )
    engine.dispose()
    return url
