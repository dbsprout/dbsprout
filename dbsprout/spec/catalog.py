"""Generator catalogue — machine-derived source of truth for the Studio picker.

The catalogue surfaces every ``(provider, method)`` pair that a user can
pick on a column's method pill. It is derived from two existing
registries:

* :data:`dbsprout.spec.patterns.PATTERNS` — the regex-driven heuristic
  registry that maps column names to ``(generator_name, provider)``.
* :data:`dbsprout.spec.heuristics._TYPE_FALLBACKS` — the type-driven
  fallback map keyed by :class:`~dbsprout.schema.models.ColumnType`.

Deriving the catalogue (rather than hand-rolling a list of method names)
means deleting a pattern auto-shrinks the catalogue and adding one auto-
exposes the new method on the UI. This closes the S-120 "no hand-
maintained string list" acceptance criterion.

The module is pure (no I/O, no globals beyond module-level constants) and
import-light — only depends on :mod:`dbsprout.schema.models` for the
:class:`ColumnType` enum and the two heuristic registries cited above.

Dtype filter
------------

Each method advertises the set of :class:`ColumnType` values it sensibly
applies to. The filter is rule-based, mirroring the shape of
``_TYPE_FALLBACKS`` plus a few well-known textual / temporal carve-outs:

* Numeric methods (e.g. ``random_int``) → INTEGER, BIGINT, SMALLINT.
* Float methods → FLOAT, DECIMAL.
* Bool methods → BOOLEAN.
* Datetime methods → DATETIME, TIMESTAMP, DATE, TIME (per-method).
* Most string-shaped methods → VARCHAR, TEXT.
* ``uuid4`` is the special case that applies to both UUID and VARCHAR
  (legacy schemas stringify ids in VARCHAR columns).

Param keys
----------

Per-method param keys come from how ``_build_params`` in
:mod:`dbsprout.spec.heuristics` populates ``params`` (``max_length``,
``precision``, ``scale``, ``enum_values``) plus the per-pattern ``params``
defaults (``min`` / ``max``). Future per-method Pydantic models can
replace this without changing the public catalogue API.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from dbsprout.schema.models import ColumnType
from dbsprout.spec.heuristics import _TYPE_FALLBACKS
from dbsprout.spec.patterns import PATTERNS

# ── dtype groups ──────────────────────────────────────────────────────────

#: Columns DBSprout treats as plain integers (no precision/scale).
_INT_DTYPES: frozenset[ColumnType] = frozenset(
    {ColumnType.INTEGER, ColumnType.BIGINT, ColumnType.SMALLINT}
)

#: Columns DBSprout treats as floats / fixed-point.
_FLOAT_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.FLOAT, ColumnType.DECIMAL})

#: Numeric umbrella — sum of ints + floats. Methods like ``random_decimal``
#: only fit floats; ``random_int`` only fits ints; ``age`` fits both.
_NUMERIC_DTYPES: frozenset[ColumnType] = _INT_DTYPES | _FLOAT_DTYPES

#: Columns DBSprout treats as strings (VARCHAR + TEXT).
_STRING_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.VARCHAR, ColumnType.TEXT})

#: Date-shape only (no time-of-day).
_DATE_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.DATE})

#: Time-of-day only.
_TIME_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.TIME})

#: Combined date+time.
_DATETIME_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.DATETIME, ColumnType.TIMESTAMP})

#: All temporal shapes — useful as a fallback for methods we know only
#: by name (e.g. ``datetime`` covers datetime+timestamp; ``date_of_birth``
#: covers date+datetime+timestamp).
_TEMPORAL_DTYPES: frozenset[ColumnType] = _DATE_DTYPES | _TIME_DTYPES | _DATETIME_DTYPES

#: Methods that apply to ``json`` / ``binary`` / ``array``-only-ish columns.
_JSON_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.JSON})
_BINARY_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.BINARY})
_ARRAY_DTYPES: frozenset[ColumnType] = frozenset({ColumnType.ARRAY})


# ── per-method dtype rules ────────────────────────────────────────────────

#: Hand-curated dtype set per *method name* (provider-agnostic). When a
#: method does not appear here we fall back to a rule-of-thumb based on
#: the name (``random_*``) and ultimately default to "string-ish".
#:
#: Keep entries terse — each row is one method.
_METHOD_DTYPES: dict[str, frozenset[ColumnType]] = {
    # Numeric.
    "random_int": _INT_DTYPES,
    "random_float": _FLOAT_DTYPES,
    "random_decimal": _FLOAT_DTYPES,
    "age": _INT_DTYPES,
    "latitude": _FLOAT_DTYPES,
    "longitude": _FLOAT_DTYPES,
    "price": _FLOAT_DTYPES,
    "version": _STRING_DTYPES,
    # Bool.
    "random_bool": frozenset({ColumnType.BOOLEAN}),
    # Strings.
    "random_string": _STRING_DTYPES,
    "random_text": _STRING_DTYPES,
    "text": _STRING_DTYPES,
    "title": _STRING_DTYPES,
    "slug": _STRING_DTYPES,
    "word": _STRING_DTYPES,
    "username": _STRING_DTYPES,
    "password": _STRING_DTYPES,
    "first_name": _STRING_DTYPES,
    "last_name": _STRING_DTYPES,
    "full_name": _STRING_DTYPES,
    "email": _STRING_DTYPES,
    "phone": _STRING_DTYPES,
    "url": _STRING_DTYPES,
    "address": _STRING_DTYPES,
    "street_address": _STRING_DTYPES,
    "city": _STRING_DTYPES,
    "state": _STRING_DTYPES,
    "zip_code": _STRING_DTYPES,
    "country": _STRING_DTYPES,
    "country_code": _STRING_DTYPES,
    "currency_code": _STRING_DTYPES,
    "credit_card": _STRING_DTYPES,
    "credit_card_expiry": _STRING_DTYPES,
    "cvv": _STRING_DTYPES,
    "ssn": _STRING_DTYPES,
    "national_id": _STRING_DTYPES,
    "avatar_url": _STRING_DTYPES,
    "image_url": _STRING_DTYPES,
    "filename": _STRING_DTYPES,
    "mime_type": _STRING_DTYPES,
    "ip_address": _STRING_DTYPES,
    "mac_address": _STRING_DTYPES,
    "user_agent": _STRING_DTYPES,
    "hex_color": _STRING_DTYPES,
    "locale": _STRING_DTYPES,
    "timezone": _STRING_DTYPES,
    "gender": _STRING_DTYPES,
    "category": _STRING_DTYPES,
    "role": _STRING_DTYPES,
    "status": _STRING_DTYPES,
    "priority": _STRING_DTYPES,
    "sku": _STRING_DTYPES,
    "reference_code": _STRING_DTYPES,
    "token": _STRING_DTYPES,
    "hash": _STRING_DTYPES,
    # Temporal.
    "random_date": _DATE_DTYPES | _DATETIME_DTYPES,
    "random_datetime": _DATETIME_DTYPES,
    "random_time": _TIME_DTYPES,
    "datetime": _DATETIME_DTYPES,
    "date_of_birth": _DATE_DTYPES | _DATETIME_DTYPES,
    # Special: uuid4 fits both UUID *and* VARCHAR (legacy stringified ids).
    "uuid4": frozenset({ColumnType.UUID}) | _STRING_DTYPES,
    # Choice over an enum_values list — works for ENUM but also any
    # narrowed string column.
    "random_choice": frozenset({ColumnType.ENUM}) | _STRING_DTYPES,
    # JSON / binary / array fallbacks.
    "random_json": _JSON_DTYPES,
    "random_bytes": _BINARY_DTYPES,
    "random_list": _ARRAY_DTYPES,
}


# ── per-method param-key rules ────────────────────────────────────────────

#: Param keys per method — small, hand-curated, drawn from how
#: :func:`dbsprout.spec.heuristics._build_params` and the per-pattern
#: ``params`` defaults populate the dict at heuristic time. ``frozenset``
#: so the API returns a stable list shape (sorted at serialization time).
_METHOD_PARAMS: dict[str, frozenset[str]] = {
    "random_int": frozenset({"min", "max"}),
    "random_float": frozenset({"min", "max"}),
    "random_decimal": frozenset({"min", "max", "precision", "scale"}),
    "random_string": frozenset({"max_length"}),
    "random_text": frozenset({"max_length"}),
    "random_choice": frozenset({"enum_values"}),
    # Patterns that supply ``min`` / ``max`` defaults inline.
    "age": frozenset({"min", "max"}),
    "price": frozenset({"min", "max"}),
}


# ── per-method human description ──────────────────────────────────────────

#: Curated one-liners surfaced on the Studio picker tooltip + method pill
#: ``title=``. Keyed by **method name only** because the description is
#: provider-agnostic in practice (``mimesis.email`` and any plugin's
#: ``email`` mean the same thing to the user). When a method is missing
#: from this map we fall back to a topical default below.
_METHOD_DESCRIPTIONS: dict[str, str] = {
    # Numeric.
    "random_int": "Random integer drawn uniformly between min and max.",
    "random_float": "Random float drawn uniformly between min and max.",
    "random_decimal": "Random fixed-point decimal with given precision and scale.",
    "age": "Realistic human age as an integer (0 to 120).",
    "latitude": "Random latitude in degrees (-90 to 90).",
    "longitude": "Random longitude in degrees (-180 to 180).",
    "price": "Random monetary amount as a float.",
    "version": "Semantic version string (e.g. '1.4.2').",
    # Bool.
    "random_bool": "Random boolean (True or False with equal probability).",
    # Strings.
    "random_string": "Random alphanumeric string up to max_length.",
    "random_text": "Multi-word random text, suitable for descriptions.",
    "text": "Lorem-ipsum style sentence or paragraph.",
    "title": "Short title-cased phrase suitable for headings.",
    "slug": "URL-safe slug derived from words (lowercase, hyphenated).",
    "word": "Single lowercase word from the locale dictionary.",
    "username": "Plausible username (letters, digits, dots/underscores).",
    "password": "Random password (mixed case + digits + symbols).",
    "first_name": "Person's first / given name.",
    "last_name": "Person's last / family name.",
    "full_name": "Person's full name (first + last).",
    "email": "Valid-looking email address (e.g. 'user@example.com').",
    "phone": "Phone number in a locale-aware format.",
    "url": "Random HTTPS URL with realistic host and path.",
    "address": "Multi-line postal address.",
    "street_address": "Street number + street name.",
    "city": "City name from the configured locale.",
    "state": "State / province / region name.",
    "zip_code": "Postal / ZIP code in a locale-aware format.",
    "country": "Country name in English.",
    "country_code": "ISO 3166-1 alpha-2 country code (e.g. 'US').",
    "currency_code": "ISO 4217 currency code (e.g. 'USD').",
    "credit_card": "Plausible (but invalid) credit-card number.",
    "credit_card_expiry": "Card expiry month / year string.",
    "cvv": "3- or 4-digit card verification value.",
    "ssn": "US Social Security Number-shaped string.",
    "national_id": "Locale-aware national identifier string.",
    "avatar_url": "Random avatar image URL.",
    "image_url": "Random image URL pointing at a placeholder service.",
    "filename": "Plausible filename with extension.",
    "mime_type": "MIME content-type (e.g. 'application/json').",
    "ip_address": "Random IPv4 / IPv6 address.",
    "mac_address": "Random hardware MAC address.",
    "user_agent": "Browser User-Agent header value.",
    "hex_color": "Random hex colour code (e.g. '#1aff4d').",
    "locale": "BCP 47 locale tag (e.g. 'en_US').",
    "timezone": "IANA timezone name (e.g. 'Europe/London').",
    "gender": "Gender label (locale-aware vocabulary).",
    "category": "Random category label from the vocabulary list.",
    "role": "Application role label (e.g. 'admin', 'user').",
    "status": "Status label (e.g. 'active', 'pending').",
    "priority": "Priority label (e.g. 'low', 'high').",
    "sku": "Stock Keeping Unit identifier.",
    "reference_code": "Generic reference / tracking code.",
    "token": "Opaque random token (URL-safe).",
    "hash": "Random hex hash digest.",
    # Temporal.
    "random_date": "Random calendar date.",
    "random_datetime": "Random datetime (date + time).",
    "random_time": "Random time of day.",
    "datetime": "Locale-aware random datetime.",
    "date_of_birth": "Plausible date of birth (date column).",
    # Special.
    "uuid4": "Random UUID v4 identifier (RFC 4122).",
    "random_choice": "Pick uniformly from the enum_values list.",
    # Structured fallbacks.
    "random_json": "Random JSON document.",
    "random_bytes": "Random binary blob (BLOB-style column).",
    "random_list": "Random list / array value.",
}


def _describe(provider: str, method: str) -> str:
    """Return a short, user-facing description for ``provider.method``.

    Looks up :data:`_METHOD_DESCRIPTIONS` first (curated copy), and falls
    back to ``"<Provider> <method spaced>"`` so plugin-supplied methods
    that haven't been classified still surface *something* on the UI.
    """
    curated = _METHOD_DESCRIPTIONS.get(method)
    if curated:
        return curated
    return f"{provider.capitalize()} {method.replace('_', ' ')}"


# ── public API ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MethodEntry:
    """One row in the generator catalogue.

    Attributes:
        provider: provider namespace (``mimesis``, ``faker``, ``builtin``,
            ``numpy``).
        method: method name within that provider.
        description: short, user-facing label for the UI picker.
        dtypes: :class:`ColumnType` values this method applies to. The
            picker uses this to filter; the server-side guard in
            :func:`dbsprout.spec.constraints.check_column_update` covers
            invariants (PK / FK uniqueness) but does **not** re-validate
            dtype, so the UI is the source of truth for this dimension.
        params: param-keys the user can set on this method.
    """

    provider: str
    method: str
    description: str
    dtypes: frozenset[ColumnType] = field(default_factory=frozenset)
    params: frozenset[str] = field(default_factory=frozenset)


def _dtypes_for_method(method: str) -> frozenset[ColumnType]:
    """Return the dtype set for ``method``, defaulting to string-shaped.

    Methods not registered in :data:`_METHOD_DTYPES` are assumed to emit
    text. This keeps the catalogue forgiving for plugin-supplied methods
    that haven't been classified yet — the picker still surfaces them on
    text columns rather than dropping them silently.
    """
    return _METHOD_DTYPES.get(method, _STRING_DTYPES)


def iter_methods() -> list[MethodEntry]:
    """Enumerate the catalogue in deterministic order.

    Order is: PATTERNS rows first (in declaration order), then any
    ``_TYPE_FALLBACKS`` entries that are not already in PATTERNS.
    Duplicates ``(provider, method)`` are de-duped — the first occurrence
    wins.
    """
    seen: set[tuple[str, str]] = set()
    entries: list[MethodEntry] = []

    def _push(provider: str, method: str) -> None:
        key = (provider, method)
        if key in seen:
            return
        seen.add(key)
        entries.append(
            MethodEntry(
                provider=provider,
                method=method,
                description=_describe(provider, method),
                dtypes=_dtypes_for_method(method),
                params=_METHOD_PARAMS.get(method, frozenset()),
            )
        )

    for pattern in PATTERNS:
        _push(pattern.provider, pattern.generator_name)
    for method, provider in _TYPE_FALLBACKS.values():
        _push(provider, method)
    return entries


def applies_to(method: str, provider: str, dtype: ColumnType) -> bool:
    """Return ``True`` when ``provider.method`` is valid for ``dtype``.

    Unknown ``(provider, method)`` pairs return ``False`` — the picker
    only renders methods present in :func:`iter_methods`, so an unknown
    method reaching this helper indicates client drift.
    """
    for entry in iter_methods():
        if entry.provider == provider and entry.method == method:
            return dtype in entry.dtypes
    return False


def param_keys_for(method: str) -> frozenset[str]:
    """Return the param-key set for ``method``, or empty if unknown."""
    return _METHOD_PARAMS.get(method, frozenset())


def providers() -> list[str]:
    """Return the sorted list of provider namespaces in the catalogue."""
    return sorted({entry.provider for entry in iter_methods()})


__all__ = [
    "MethodEntry",
    "applies_to",
    "iter_methods",
    "param_keys_for",
    "providers",
]
