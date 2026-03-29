"""Pattern registry for heuristic column-to-generator mapping.

Each pattern maps a regex (applied to the normalized column name) to a
generator name and provider. Patterns are checked in order; the first
match wins.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class PatternRule(NamedTuple):
    """A single pattern rule for column name matching."""

    regex: str
    generator_name: str
    provider: str
    base_confidence: float
    params: dict[str, Any] = {}  # noqa: RUF012


# ── Personal / Identity ──────────────────────────────────────────────────

_PERSONAL: list[PatternRule] = [
    PatternRule(r"^e?mail(_address)?$", "email", "mimesis", 0.95),
    PatternRule(r"^first_?name$", "first_name", "mimesis", 0.95),
    PatternRule(r"^last_?name$", "last_name", "mimesis", 0.95),
    PatternRule(r"^full_?name$", "full_name", "mimesis", 0.95),
    PatternRule(r"^(user_?)?name$", "username", "mimesis", 0.85),
    PatternRule(r"^username$", "username", "mimesis", 0.95),
    PatternRule(r"^password(_hash)?$", "password", "mimesis", 0.90),
    PatternRule(r"^avatar(_url)?$", "avatar_url", "mimesis", 0.90),
    PatternRule(r"^(profile_?)?image(_url)?$", "avatar_url", "mimesis", 0.85),
    PatternRule(r"^bio(graphy)?$", "text", "mimesis", 0.85),
    PatternRule(r"^gender$", "gender", "mimesis", 0.90),
    PatternRule(r"^(date_of_)?birth(day|date)?$", "date_of_birth", "mimesis", 0.90),
    PatternRule(r"^age$", "age", "builtin", 0.85),
    PatternRule(r"^ssn$", "ssn", "faker", 0.90),
    PatternRule(r"^(national_?)?id(_number)?$", "national_id", "faker", 0.80),
]

# ── Contact ──────────────────────────────────────────────────────────────

_CONTACT: list[PatternRule] = [
    PatternRule(r"^phone(_number)?$", "phone", "mimesis", 0.95),
    PatternRule(r"^(mobile|cell)(_phone)?(_number)?$", "phone", "mimesis", 0.90),
    PatternRule(r"^fax(_number)?$", "phone", "mimesis", 0.85),
    PatternRule(r"^(home|work|office)_?phone$", "phone", "mimesis", 0.85),
    PatternRule(r"^(web_?)?site(_url)?$", "url", "mimesis", 0.90),
    PatternRule(r"^website$", "url", "mimesis", 0.90),
]

# ── Address ──────────────────────────────────────────────────────────────

_ADDRESS: list[PatternRule] = [
    PatternRule(r"^address(_line)?(_[12])?$", "address", "mimesis", 0.90),
    PatternRule(r"^street(_address)?(_line)?$", "street_address", "mimesis", 0.90),
    PatternRule(r"^city$", "city", "mimesis", 0.95),
    PatternRule(r"^(state|province|region)(_code)?$", "state", "mimesis", 0.90),
    PatternRule(r"^(zip|postal)(_code)?$", "zip_code", "mimesis", 0.95),
    PatternRule(r"^country(_code)?$", "country", "mimesis", 0.90),
    PatternRule(r"^country$", "country", "mimesis", 0.95),
    PatternRule(r"^lat(itude)?$", "latitude", "mimesis", 0.90),
    PatternRule(r"^(lon|lng)(gitude)?$", "longitude", "mimesis", 0.90),
    PatternRule(r"^county$", "city", "mimesis", 0.75),
    PatternRule(r"^time_?zone$", "timezone", "mimesis", 0.90),
]

# ── Temporal ─────────────────────────────────────────────────────────────

_TEMPORAL: list[PatternRule] = [
    PatternRule(r"^created_?(at|on|date|time)?$", "datetime", "mimesis", 0.90),
    PatternRule(r"^updated_?(at|on|date|time)?$", "datetime", "mimesis", 0.90),
    PatternRule(r"^deleted_?(at|on|date)?$", "datetime", "mimesis", 0.90),
    PatternRule(r"^(start|begin)_?(date|time|at)?$", "datetime", "mimesis", 0.85),
    PatternRule(r"^(end|finish|expire[ds]?)_?(date|time|at)?$", "datetime", "mimesis", 0.85),
    PatternRule(r"^(published|posted|sent)_?(at|on|date)?$", "datetime", "mimesis", 0.85),
    PatternRule(r"^(last_)?login_?(at|date|time)?$", "datetime", "mimesis", 0.85),
    PatternRule(r"^(registered|joined|signup)_?(at|on|date)?$", "datetime", "mimesis", 0.85),
    PatternRule(r"^(due|delivery|ship)_?(date|at)?$", "datetime", "mimesis", 0.85),
    PatternRule(r"^timestamp$", "datetime", "mimesis", 0.85),
]

# ── Financial ────────────────────────────────────────────────────────────

_FINANCIAL: list[PatternRule] = [
    PatternRule(r"^price$", "price", "mimesis", 0.95),
    PatternRule(r"^(unit_)?price$", "price", "mimesis", 0.90),
    PatternRule(r"^(total_?)?amount$", "price", "mimesis", 0.85),
    PatternRule(r"^(sub_?)?total$", "price", "mimesis", 0.85),
    PatternRule(r"^cost$", "price", "mimesis", 0.90),
    PatternRule(r"^balance$", "price", "mimesis", 0.85),
    PatternRule(r"^(tax|discount|fee|charge)(_amount)?$", "price", "mimesis", 0.85),
    PatternRule(r"^currency(_code)?$", "currency_code", "mimesis", 0.90),
    PatternRule(r"^credit_card(_number)?$", "credit_card", "mimesis", 0.90),
    PatternRule(r"^(card_?)?cvv$", "cvv", "builtin", 0.85),
    PatternRule(r"^(card_?)?expir(y|ation)(_date)?$", "credit_card_expiry", "mimesis", 0.85),
]

# ── Content ──────────────────────────────────────────────────────────────

_CONTENT: list[PatternRule] = [
    PatternRule(r"^title$", "title", "mimesis", 0.90),
    PatternRule(r"^(sub_?)?title$", "title", "mimesis", 0.85),
    PatternRule(r"^description$", "text", "mimesis", 0.85),
    PatternRule(r"^(body|content|message|comment|note|summary|excerpt)$", "text", "mimesis", 0.85),
    PatternRule(r"^slug$", "slug", "mimesis", 0.90),
    PatternRule(r"^(url|link|href)$", "url", "mimesis", 0.90),
    PatternRule(r"^(image|photo|picture|thumbnail)(_url)?$", "image_url", "mimesis", 0.90),
    PatternRule(r"^(file_?)?(path|name)$", "filename", "mimesis", 0.80),
    PatternRule(r"^(mime_?)?type$", "mime_type", "builtin", 0.75),
    PatternRule(r"^(file_?)?size$", "random_int", "builtin", 0.70),
    PatternRule(r"^tags?$", "word", "mimesis", 0.75),
    PatternRule(r"^label$", "word", "mimesis", 0.75),
    PatternRule(r"^color(_code)?$", "hex_color", "mimesis", 0.85),
]

# ── Identifiers ──────────────────────────────────────────────────────────

_IDENTIFIERS: list[PatternRule] = [
    PatternRule(r"^uuid$", "uuid4", "builtin", 0.95),
    PatternRule(r"^(sku|product_code|item_code)$", "sku", "builtin", 0.85),
    PatternRule(
        r"^(ref(erence)?|tracking)(_number|_code|_id)?$", "reference_code", "builtin", 0.80
    ),
    PatternRule(r"^token$", "token", "builtin", 0.80),
    PatternRule(r"^(api_?)?key$", "token", "builtin", 0.80),
    PatternRule(r"^(hash|checksum|digest)$", "hash", "builtin", 0.80),
    PatternRule(r"^(ip_?)?address$", "ip_address", "mimesis", 0.80),
    PatternRule(r"^ip(v[46])?$", "ip_address", "mimesis", 0.85),
    PatternRule(r"^mac_?address$", "mac_address", "mimesis", 0.85),
    PatternRule(r"^(user_?)?agent$", "user_agent", "mimesis", 0.85),
]

# ── Status / Enum-like ───────────────────────────────────────────────────

_STATUS: list[PatternRule] = [
    PatternRule(r"^status$", "status", "builtin", 0.85),
    PatternRule(r"^(type|kind|category)$", "category", "builtin", 0.80),
    PatternRule(r"^role$", "role", "builtin", 0.85),
    PatternRule(r"^(priority|severity|level)$", "priority", "builtin", 0.80),
    PatternRule(
        r"^is_?(active|enabled|verified|public|admin|deleted|archived|featured)$",
        "random_bool",
        "builtin",
        0.90,
    ),
    PatternRule(r"^(has|can|should|allow)_\w+$", "random_bool", "builtin", 0.80),
    PatternRule(r"^(locale|language|lang)(_code)?$", "locale", "mimesis", 0.85),
]

# ── Quantity / Numeric ───────────────────────────────────────────────────

_QUANTITY: list[PatternRule] = [
    PatternRule(
        r"^(quantity|qty|count|num(ber)?)$", "random_int", "builtin", 0.85, {"min": 1, "max": 100}
    ),
    PatternRule(r"^(weight|height|width|length|depth|distance)$", "random_float", "builtin", 0.80),
    PatternRule(r"^(rating|score|rank)$", "random_int", "builtin", 0.85, {"min": 1, "max": 5}),
    PatternRule(
        r"^(percent|percentage|ratio)$", "random_float", "builtin", 0.80, {"min": 0.0, "max": 100.0}
    ),
    PatternRule(r"^(version|revision)$", "version", "builtin", 0.80),
    PatternRule(r"^(sort_?)?order$", "random_int", "builtin", 0.75, {"min": 0, "max": 1000}),
]


# ── Combined registry ────────────────────────────────────────────────────

PATTERNS: list[PatternRule] = [
    *_PERSONAL,
    *_CONTACT,
    *_ADDRESS,
    *_TEMPORAL,
    *_FINANCIAL,
    *_CONTENT,
    *_IDENTIFIERS,
    *_STATUS,
    *_QUANTITY,
]
