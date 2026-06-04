"""``POST /api/schema/load`` — upload a schema file and parse it (S-113).

A user uploads a schema file (SQL DDL, DBML, Mermaid, PlantUML, or Prisma) via a
multipart form; the route parses it into a :class:`~dbsprout.schema.models.DatabaseSchema`
and stores it in the per-session :class:`~dbsprout.web.workspace.Workspace`
(wired onto ``app.state.workspace`` by S-111).

Reuse, not reimplementation
---------------------------
Parsing is delegated **verbatim** to
:func:`dbsprout.schema.parsers.parse_schema_file`, which already dispatches by
file suffix, runs the plugin registry first, uses the per-format ``can_parse_*``
content detectors, and enforces a 10 MB file-size backstop. The route writes the
uploaded bytes to a temporary file with the inferred (or explicitly overridden)
suffix and hands that path to ``parse_schema_file`` — the smallest seam that
reuses every parser code path without forking the dispatch logic.

Two-tier size cap
-----------------
The :class:`~fastapi.UploadFile` body is read in chunks while summing bytes; the
route aborts with ``413`` as soon as the running total exceeds
:data:`_MAX_UPLOAD_BYTES` (a few MB). Reading incrementally (rather than
``await read()`` then ``len(...)``) bounds the in-process ``bytes`` we accumulate
and rejects the request *before* any parsing or temp-file write, so an oversize
upload cannot drive the parser. (Starlette spools the raw multipart body to a
``SpooledTemporaryFile`` during request parsing, so this cap governs how much we
read back into the app, not the transport.) The parser's own 10 MB file-size
check is the backstop; the web cap is smaller so the friendly 413 fires first.

Django / MongoDB are intentionally *not* file-content uploads:
``parse_django_models`` introspects a live Django app (app labels) and Mongo
parsing takes a connection URL — neither parses uploaded file content.

Auto-detect (S-114)
-------------------
When no ``parser`` form-field is given, suffix dispatch is the cheap first cut
(matches ``parse_schema_file``'s own ordering). For ambiguous extensions
(``.txt``, ``.schema``, no extension, …) the route additionally sniffs a small
head of the upload bytes by calling the existing ``can_parse_*`` detectors
from :mod:`dbsprout.schema.parsers` in the documented priority order
(DBML → Mermaid → PlantUML → Prisma) before falling back to ``.sql`` (DDL).
The sniffers themselves are not reimplemented — they live on the parser modules
and are reused verbatim. An explicit ``parser`` form-field **always** wins over
both filename suffix and content sniff (S-113 contract preserved).

Friendly errors (S-116)
-----------------------
Every failure path — oversize upload, empty file, unknown ``parser`` override,
parse error, and the catch-all for unexpected exceptions — flows through the
:mod:`dbsprout.web.errors` layer. Callers receive the typed envelope
``{code, message, hint?, correlation_id}`` (JSON by default, HTML fragment when
``HX-Request: true``); the route never leaks a traceback or absolute temp-file
path to the wire.

This module owns its own :class:`~fastapi.APIRouter` (``schema_load_router``),
registered by ``create_app`` inside a delimited region. It stays import-light:
stdlib helpers are imported lazily inside the handler to preserve CLI startup
time (FastAPI itself only loads under the optional ``[web]`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from dbsprout.web.errors import (
    classify_parse_error,
    raise_web_error,
    web_error_empty_file,
    web_error_file_too_large,
    web_error_internal,
    web_error_unknown_parser,
)

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace

schema_load_router = APIRouter()

#: Maximum accepted upload size (web tier). Smaller than the parser's 10 MB
#: backstop so the friendly 413 always fires first for oversize uploads.
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

#: Chunk size for the streamed size-capped read.
_CHUNK_BYTES = 64 * 1024  # 64 KB

#: Head-slice size for content sniffing. Every ``can_parse_*`` detector matches
#: on keywords near the file head (DBML ``Table``, Mermaid ``erDiagram``,
#: PlantUML ``@startuml``, Prisma ``model``/``datasource``/``generator``), so a
#: few KB is generous without copying multi-MB payloads on every upload.
_SNIFF_HEAD_BYTES = 8 * 1024  # 8 KB

#: Canonical ``parser`` override values → the file suffix understood by
#: ``parse_schema_file``. Keys are matched case-insensitively.
_PARSER_SUFFIXES: dict[str, str] = {
    "sql": ".sql",
    "ddl": ".sql",
    "dbml": ".dbml",
    "mermaid": ".mermaid",
    "mmd": ".mermaid",
    "plantuml": ".puml",
    "puml": ".puml",
    "prisma": ".prisma",
}

#: Suffixes ``parse_schema_file`` recognises (anything else → DDL fallback).
_KNOWN_SUFFIXES: frozenset[str] = frozenset(
    {".sql", ".dbml", ".mermaid", ".mmd", ".puml", ".plantuml", ".pu", ".prisma"}
)


def _workspace(request: Request) -> Workspace:
    """Typed accessor for the shared session Workspace wired in ``app.py``."""
    return cast("Workspace", request.app.state.workspace)


def _detect_suffix_from_content(head: bytes) -> str | None:
    """Sniff a head slice and return the canonical suffix, or ``None``.

    Reuses the existing ``can_parse_*`` detectors from
    :mod:`dbsprout.schema.parsers` (DBML → Mermaid → PlantUML → Prisma). Priority
    order matters: each detector matches on a *unique* keyword for its format
    (DBML ``table {``, Mermaid ``erDiagram``, PlantUML ``@startuml``,
    Prisma ``model``/``datasource``/``generator``), so overlap is rare; if it
    happens, the first match wins. DDL has no sniffer — it is the final
    fallback (matches ``parse_schema_file``'s behaviour).

    Decoding is best-effort UTF-8 with ``errors="ignore"`` for the sniff only;
    the parser sees the original bytes via the temp file, so non-UTF-8 schemas
    still work end-to-end when the user provides an explicit ``parser`` override.
    """
    from dbsprout.schema.parsers import (  # noqa: PLC0415 — lazy for startup
        can_parse_dbml,
        can_parse_mermaid,
        can_parse_plantuml,
        can_parse_prisma,
    )

    text = head.decode("utf-8", errors="ignore")
    if not text.strip():
        return None
    if can_parse_dbml(text):
        return ".dbml"
    if can_parse_mermaid(text):
        return ".mermaid"
    if can_parse_plantuml(text):
        return ".puml"
    if can_parse_prisma(text):
        return ".prisma"
    return None


def _resolve_suffix(
    filename: str | None,
    parser: str | None,
    *,
    content_head: bytes | None = None,
) -> str | None:
    """Pick the temp-file suffix: explicit ``parser`` > filename > content sniff.

    A blank/whitespace ``parser`` is treated as "not provided". With no override:
    first tries the filename suffix; if unknown/absent and ``content_head`` is
    given, sniffs the head bytes via the existing ``can_parse_*`` detectors
    (see :func:`_detect_suffix_from_content`); otherwise falls back to ``.sql``
    to match ``parse_schema_file``'s DDL behaviour.

    An unknown explicit parser value returns ``None`` — the caller raises a
    :class:`~dbsprout.web.errors.WebErrorCode.UNKNOWN_PARSER` envelope so the
    typed-error path is the *only* path back to the client.
    """
    if parser is not None and parser.strip():
        suffix = _PARSER_SUFFIXES.get(parser.strip().lower())
        if suffix is None:
            return None
        return suffix

    from pathlib import PurePosixPath  # noqa: PLC0415 — stdlib, lazy for startup

    file_suffix = PurePosixPath(filename or "").suffix.lower()
    if file_suffix in _KNOWN_SUFFIXES:
        return file_suffix
    if content_head is not None:
        sniffed = _detect_suffix_from_content(content_head)
        if sniffed is not None:
            return sniffed
    return ".sql"


async def _read_capped(upload: UploadFile, request: Request) -> bytes | Any:
    """Read the upload in chunks, surfacing 413 once the cap is exceeded.

    Reading-and-counting (rather than ``await upload.read()`` then checking
    ``len``) bounds the in-process ``bytes`` accumulated to ``_MAX_UPLOAD_BYTES``
    + one chunk and rejects an oversize upload before it is parsed or written to a
    temp file, so the route cannot be coaxed into materialising an arbitrarily
    large payload.
    """
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > _MAX_UPLOAD_BYTES:
            return raise_web_error(request, web_error_file_too_large(_MAX_UPLOAD_BYTES))
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_upload(
    request: Request,
    content: bytes,
    suffix: str,
    filename: str | None,
) -> DatabaseSchema | Any:
    """Parse uploaded bytes by writing a temp file and reusing ``parse_schema_file``.

    The temp file lives only for the duration of the parse (``with`` block) and is
    always removed, even on parse failure. ``ValueError`` / ``OSError`` from a
    real parser become a :class:`~dbsprout.web.errors.WebErrorCode.PARSE_ERROR`
    envelope; anything *unexpected* (a ``RuntimeError``, a Pydantic
    ``ValidationError`` outside the parsers' contract, etc.) is downgraded to
    :class:`~dbsprout.web.errors.WebErrorCode.INTERNAL` (500) with the real
    exception logged at ``ERROR`` and a correlation id surfaced to the user.
    """
    import tempfile  # noqa: PLC0415 — stdlib, lazy for startup
    from pathlib import Path  # noqa: PLC0415

    from dbsprout.schema.parsers import parse_schema_file  # noqa: PLC0415

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as handle:
        handle.write(content)
        handle.flush()
        path = Path(handle.name)
        try:
            return parse_schema_file(path)
        except (ValueError, OSError) as exc:
            return raise_web_error(request, classify_parse_error(exc, filename), original=exc)
        except Exception as exc:
            return raise_web_error(request, web_error_internal(), original=exc)


def _summary(schema: DatabaseSchema, source: str) -> dict[str, Any]:
    """Shape a parsed schema into a JSON-friendly load summary."""
    return {
        "source": source,
        "table_count": len(schema.tables),
        "tables": schema.table_names(),
        "dialect": schema.dialect,
    }


@schema_load_router.post("/api/schema/load", response_class=JSONResponse)
async def load_schema_upload(
    request: Request,
    file: Annotated[UploadFile, File(description="Schema file to parse.")],
    parser: Annotated[str | None, Form(description="Optional parser override.")] = None,
) -> Any:
    """Parse an uploaded schema file and store it in the session workspace.

    Returns a JSON summary (``table_count``, ``tables``, ``dialect``, ``source``)
    on success. Every failure mode flows through the S-116 error layer:

    * oversize uploads → ``413 FILE_TOO_LARGE``;
    * empty uploads → ``400 EMPTY_FILE``;
    * unknown ``parser`` override → ``400 UNKNOWN_PARSER``;
    * parser exceptions → ``400 PARSE_ERROR``;
    * anything else → ``500 INTERNAL`` (with the original exception logged).

    Bodies are JSON by default and an HTML fragment when ``HX-Request: true``;
    no traceback ever reaches the wire.
    """
    content_or_response = await _read_capped(file, request)
    # An HTMX response from the cap path short-circuits the handler.
    if not isinstance(content_or_response, bytes):
        return content_or_response
    content: bytes = content_or_response

    if not content.strip():
        return raise_web_error(request, web_error_empty_file(file.filename))

    # Resolve the suffix *after* reading so content sniffing has bytes to work
    # with when the filename suffix is unknown/absent. Explicit ``parser`` and
    # known filename suffixes never read ``content_head``.
    suffix = _resolve_suffix(file.filename, parser, content_head=content[:_SNIFF_HEAD_BYTES])
    if suffix is None:
        assert parser is not None
        return raise_web_error(request, web_error_unknown_parser(parser, sorted(_PARSER_SUFFIXES)))

    schema_or_response = _parse_upload(request, content, suffix, file.filename)
    # Same short-circuit pattern: an HTMX renderer can substitute a Response.
    if not _is_database_schema(schema_or_response):
        return schema_or_response
    schema = cast("DatabaseSchema", schema_or_response)

    source = f"upload:{file.filename}" if file.filename else "upload:<unnamed>"
    workspace = _workspace(request)
    workspace.set_schema(schema)
    workspace.set_source(source)
    # S-122: hydrate the workspace spec from the disk cache when a prior
    # session edited the spec for this exact schema. A miss leaves
    # ``workspace.spec`` ``None`` so ``GET /api/spec`` still triggers the
    # lazy heuristic build (existing S-118 contract preserved).
    workspace.hydrate_from_cache(schema.schema_hash())
    return JSONResponse(_summary(schema, source))


def _is_database_schema(obj: object) -> bool:
    """Lazy isinstance check so the module never imports ``DatabaseSchema``."""
    from dbsprout.schema.models import DatabaseSchema  # noqa: PLC0415

    return isinstance(obj, DatabaseSchema)


class PasteRequest(BaseModel):
    """Request body for ``POST /api/schema/paste`` — pasted schema text.

    Validated at the boundary: a missing, blank, or whitespace-only ``text``
    (or any unexpected field) yields FastAPI's ``422`` before the handler
    runs. Mirrors the ``ConnectRequest`` validator pattern.
    """

    model_config = ConfigDict(extra="forbid")

    text: str = Field(min_length=1, description="Pasted schema text.")
    parser: str | None = Field(
        default=None, description="Optional parser override (sql, dbml, mermaid, …)."
    )

    @field_validator("text")
    @classmethod
    def _non_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            msg = "text must not be blank"
            raise ValueError(msg)
        return stripped


@schema_load_router.post("/api/schema/paste", response_class=JSONResponse)
async def paste_schema(request: Request, body: PasteRequest) -> Any:
    """Parse pasted schema text and store it in the session workspace.

    The text is encoded to UTF-8, sniffed / suffix-resolved exactly like an
    uploaded file (reusing :func:`_resolve_suffix` and
    :func:`_detect_suffix_from_content`), then handed to :func:`_parse_upload`
    which writes a temp file and delegates to ``parse_schema_file``.  Every
    failure mode flows through the S-116 error layer:

    * blank text → ``422`` (rejected by the Pydantic validator);
    * unknown ``parser`` override → ``400 UNKNOWN_PARSER``;
    * parser exceptions → ``400 PARSE_ERROR``;
    * anything else → ``500 INTERNAL``.
    """
    content = body.text.encode("utf-8")

    suffix = _resolve_suffix(None, body.parser, content_head=content[:_SNIFF_HEAD_BYTES])
    if suffix is None:
        assert body.parser is not None
        return raise_web_error(
            request, web_error_unknown_parser(body.parser, sorted(_PARSER_SUFFIXES))
        )

    schema_or_response = _parse_upload(request, content, suffix, None)
    if not _is_database_schema(schema_or_response):
        return schema_or_response
    schema = cast("DatabaseSchema", schema_or_response)

    workspace = _workspace(request)
    workspace.set_schema(schema)
    workspace.set_source("paste")
    workspace.hydrate_from_cache(schema.schema_hash())
    return JSONResponse(_summary(schema, "paste"))
