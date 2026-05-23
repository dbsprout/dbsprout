"""Heuristic generation engine — Mimesis + Faker + builtins.

Produces realistic row data based on ``GeneratorMapping`` from the
heuristic column mapper (S-012). FK and autoincrement PK columns
are skipped (set to ``None``).
"""

from __future__ import annotations

import random
import string
import uuid
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any

from mimesis import Address, Datetime, Finance, Internet, Payment, Person, Text
from mimesis.locales import Locale

if TYPE_CHECKING:
    from dbsprout.schema.models import TableSchema
    from dbsprout.spec.models import GeneratorMapping


class HeuristicEngine:
    """Generate rows using Mimesis providers and builtin fallbacks."""

    def __init__(self, locale: str = "en", seed: int = 42) -> None:
        self._seed = seed
        loc = Locale(locale) if locale != "en" else Locale.EN
        self._person = Person(loc)
        self._address = Address(loc)
        self._finance = Finance(loc)
        self._internet = Internet()
        self._text = Text(loc)
        self._dt = Datetime(loc)
        self._payment = Payment()

    def generate_table(
        self,
        table: TableSchema,
        mappings: dict[str, GeneratorMapping],
        num_rows: int,
    ) -> list[dict[str, Any]]:
        """Generate ``num_rows`` rows for a table.

        FK columns and autoincrement PKs are set to ``None``.
        """
        fk_cols = _fk_columns(table)
        auto_pk_cols = _autoincrement_pk_columns(table)

        # Column-oriented generation
        col_data: dict[str, list[Any]] = {}
        for col in table.columns:
            if col.name in fk_cols or col.name in auto_pk_cols:
                col_data[col.name] = [None] * num_rows
            else:
                mapping = mappings.get(col.name)
                from dbsprout.generate.deterministic import column_seed  # noqa: PLC0415

                col_seed = column_seed(self._seed, table.name, col.name)
                col_data[col.name] = self._generate_column(mapping, num_rows, col_seed)

        # Transpose to row-oriented
        return [{col_name: col_data[col_name][i] for col_name in col_data} for i in range(num_rows)]

    def _generate_column(
        self,
        mapping: GeneratorMapping | None,
        num_rows: int,
        col_seed: int = 42,
    ) -> list[Any]:
        """Generate values for a single column.

        A per-column ``random.Random`` instance is created from ``col_seed`` so
        that builtin generators are fully isolated from the process-global RNG.
        Mimesis providers own their own ``Random`` instance and are reseeded via
        ``provider.reseed(col_seed)``.
        """
        if mapping is None:
            return [None] * num_rows

        # Create a per-column Random instance — never touches the global RNG.
        rng = random.Random(col_seed)  # noqa: S311

        # Reseed every Mimesis provider with the column seed.
        for provider in (
            self._person,
            self._address,
            self._finance,
            self._internet,
            self._text,
            self._dt,
            self._payment,
        ):
            provider.reseed(col_seed)

        # Fast path: vectorized NumPy generation for eligible types
        from dbsprout.generate.vectorized import generate_vectorized  # noqa: PLC0415

        vec_result = generate_vectorized(
            mapping.generator_name, num_rows, seed=col_seed, params=mapping.params
        )
        if vec_result is not None:
            # Apply max_length truncation if needed
            max_length = mapping.params.get("max_length")
            if max_length is not None:
                vec_result = [v[:max_length] if isinstance(v, str) else v for v in vec_result]
            return vec_result

        gen = self._resolve_generator(mapping, rng)
        params = mapping.params
        max_length = params.get("max_length")

        values: list[Any] = []
        for _ in range(num_rows):
            val = gen(params)
            if max_length is not None and isinstance(val, str):
                val = val[:max_length]
            values.append(val)
        return values

    def _resolve_generator(
        self,
        mapping: GeneratorMapping,
        rng: random.Random,
    ) -> Any:
        """Resolve a GeneratorMapping to a callable (params) -> value."""
        name = mapping.generator_name

        # Mimesis providers
        mimesis_gen = self._mimesis_dispatch().get(name)
        if mimesis_gen is not None:
            return lambda _p, g=mimesis_gen: g()

        # Builtin generators — bind the per-column rng instance
        builtin_gen = _BUILTIN_DISPATCH.get(name)
        if builtin_gen is not None:
            return lambda p, g=builtin_gen, r=rng: g(p, r)

        # Final fallback — also bound to the per-column rng
        return lambda p, r=rng: _gen_random_string(p, r)

    def _mimesis_dispatch(self) -> dict[str, Any]:
        """Lazy dispatch table for Mimesis provider methods."""
        return {
            "email": self._person.email,
            "first_name": self._person.first_name,
            "last_name": self._person.last_name,
            "full_name": self._person.full_name,
            "username": self._person.username,
            "password": self._person.password,
            "phone": self._person.telephone,
            "gender": self._person.gender,
            "city": self._address.city,
            "state": self._address.state,
            "street_address": self._address.street_name,
            "address": self._address.address,
            "zip_code": self._address.zip_code,
            "country": self._address.country,
            "country_code": self._address.country_code,
            "latitude": self._address.latitude,
            "longitude": self._address.longitude,
            "url": self._internet.url,
            "avatar_url": self._internet.url,
            "image_url": self._internet.url,
            "slug": self._internet.slug,
            "ip_address": self._internet.ip_v4,
            "mac_address": self._internet.mac_address,
            "user_agent": self._internet.user_agent,
            "title": self._text.title,
            "text": self._text.text,
            "word": self._text.word,
            "hex_color": self._text.hex_color,
            "datetime": self._dt.datetime,
            "date_of_birth": self._dt.date,
            "price": self._finance.price,
            "currency_code": self._finance.currency_iso_code,
            "credit_card": self._payment.credit_card_number,
        }


# ── Builtin generators ──────────────────────────────────────────────────


def _gen_random_int(params: dict[str, Any], rng: random.Random) -> int:
    lo = params.get("min", 0)
    hi = params.get("max", 10000)
    if lo > hi:
        lo, hi = hi, lo
    return rng.randint(lo, hi)


def _gen_random_float(params: dict[str, Any], rng: random.Random) -> float:
    lo = params.get("min", 0.0)
    hi = params.get("max", 10000.0)
    return round(rng.uniform(lo, hi), 2)


def _gen_random_decimal(params: dict[str, Any], rng: random.Random) -> float:
    precision = params.get("precision", 10)
    scale = params.get("scale", 2)
    max_val = max(10 ** (precision - scale) - 1, 1)
    return float(round(rng.uniform(0, max_val), scale))


def _gen_random_bool(_params: dict[str, Any], rng: random.Random) -> bool:
    return rng.choice([True, False])


def _gen_random_string(params: dict[str, Any], rng: random.Random) -> str:
    length = min(params.get("max_length", 20), 100)
    return "".join(rng.choices(string.ascii_lowercase, k=length))


def _gen_random_text(_params: dict[str, Any], rng: random.Random) -> str:
    words = rng.randint(5, 20)
    return " ".join(
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 10))) for _ in range(words)
    )


def _gen_random_datetime(_params: dict[str, Any], rng: random.Random) -> datetime:
    base = datetime(2020, 1, 1, tzinfo=timezone.utc)
    offset = rng.randint(0, 365 * 5 * 24 * 3600)
    return base + timedelta(seconds=offset)


def _gen_random_date(_params: dict[str, Any], rng: random.Random) -> date:
    return _gen_random_datetime(_params, rng).date()


def _gen_random_time(_params: dict[str, Any], rng: random.Random) -> time:
    return _gen_random_datetime(_params, rng).time()


def _gen_uuid4(_params: dict[str, Any], _rng: random.Random) -> str:
    return str(uuid.uuid4())


def _gen_random_choice(params: dict[str, Any], rng: random.Random) -> Any:
    values = params.get("enum_values", ["a", "b", "c"])
    if not values:
        return None
    return rng.choice(values)


def _gen_random_bytes(_params: dict[str, Any], rng: random.Random) -> bytes:
    return bytes(rng.getrandbits(8) for _ in range(16))


def _gen_random_json(_params: dict[str, Any], rng: random.Random) -> dict[str, Any]:
    return {"key": _gen_random_string({"max_length": 10}, rng), "value": rng.randint(0, 100)}


def _gen_random_list(_params: dict[str, Any], rng: random.Random) -> list[Any]:
    return [rng.randint(0, 100) for _ in range(rng.randint(1, 5))]


def _gen_ssn(rng: random.Random) -> str:
    a = rng.randint(100, 999)
    b = rng.randint(10, 99)
    c = rng.randint(1000, 9999)
    return f"{a:03d}-{b:02d}-{c:04d}"


_BUILTIN_DISPATCH: dict[str, Any] = {
    "random_int": _gen_random_int,
    "random_float": _gen_random_float,
    "random_decimal": _gen_random_decimal,
    "random_bool": _gen_random_bool,
    "random_string": _gen_random_string,
    "random_text": _gen_random_text,
    "random_datetime": _gen_random_datetime,
    "random_date": _gen_random_date,
    "random_time": _gen_random_time,
    "uuid4": _gen_uuid4,
    "random_choice": _gen_random_choice,
    "random_bytes": _gen_random_bytes,
    "random_json": _gen_random_json,
    "random_list": _gen_random_list,
    "age": lambda _p, r: r.randint(18, 90),
    "ssn": lambda _p, r: _gen_ssn(r),
    "version": lambda _p, r: f"{r.randint(0, 9)}.{r.randint(0, 99)}.{r.randint(0, 99)}",
    "sku": lambda _p, r: f"SKU-{r.randint(10000, 99999)}",
    "reference_code": lambda _p, _r: f"REF-{uuid.uuid4().hex[:8].upper()}",
    "token": lambda _p, _r: uuid.uuid4().hex,
    "hash": lambda _p, _r: uuid.uuid4().hex + uuid.uuid4().hex[:32],
    "cvv": lambda _p, r: f"{r.randint(100, 999)}",
    "credit_card_expiry": lambda _p, r: f"{r.randint(1, 12):02d}/{r.randint(25, 30)}",
    "filename": lambda _p, r: f"file_{r.randint(1, 9999)}.txt",
    "status": lambda _p, r: r.choice(["active", "inactive", "pending", "archived"]),
    "category": lambda _p, r: r.choice(["general", "premium", "basic", "enterprise"]),
    "role": lambda _p, r: r.choice(["admin", "user", "editor", "viewer"]),
    "priority": lambda _p, r: r.choice(["low", "medium", "high", "critical"]),
    "locale": lambda _p, r: r.choice(["en_US", "en_GB", "de_DE", "fr_FR", "es_ES"]),
    "timezone": lambda _p, r: r.choice(["UTC", "US/Eastern", "US/Pacific", "Europe/London"]),
    "national_id": lambda _p, r: f"ID-{r.randint(100000, 999999)}",
    "mime_type": lambda _p, r: r.choice(["text/plain", "application/json", "image/png"]),
}


# ── Helpers ──────────────────────────────────────────────────────────────


def _fk_columns(table: TableSchema) -> set[str]:
    """Get column names that are FK columns."""
    return {col for fk in table.foreign_keys for col in fk.columns}


def _autoincrement_pk_columns(table: TableSchema) -> set[str]:
    """Get PK column names that are autoincrement."""
    return {col.name for col in table.columns if col.primary_key and col.autoincrement}
