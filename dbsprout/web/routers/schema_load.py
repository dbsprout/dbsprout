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
parsing takes a connection URL — neither parses uploaded file content. Basic
content auto-detect polish is the later S-114; here suffix/override dispatch is
sufficient.

This module owns its own :class:`~fastapi.APIRouter` (``schema_load_router``),
registered by ``create_app`` inside a delimited region. It stays import-light:
stdlib helpers are imported lazily inside the handler to preserve CLI startup
time (FastAPI itself only loads under the optional ``[web]`` extra).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any, cast

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse

if TYPE_CHECKING:
    from dbsprout.schema.models import DatabaseSchema
    from dbsprout.web.workspace import Workspace

schema_load_router = APIRouter()

#: Maximum accepted upload size (web tier). Smaller than the parser's 10 MB
#: backstop so the friendly 413 always fires first for oversize uploads.
_MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MB

#: Chunk size for the streamed size-capped read.
_CHUNK_BYTES = 64 * 1024  # 64 KB

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


def _resolve_suffix(filename: str | None, parser: str | None) -> str:
    """Pick the temp-file suffix: explicit ``parser`` override > filename suffix.

    A blank/whitespace ``parser`` is treated as "not provided". An unknown
    explicit parser value raises ``HTTPException(400)``. With no override and an
    unknown/absent filename suffix, returns ``.sql`` to match
    ``parse_schema_file``'s DDL fallback.
    """
    if parser is not None and parser.strip():
        suffix = _PARSER_SUFFIXES.get(parser.strip().lower())
        if suffix is None:
            allowed = ", ".join(sorted(_PARSER_SUFFIXES))
            raise HTTPException(
                status_code=400,
                detail=f"Unknown parser {parser!r}. Supported parsers: {allowed}.",
            )
        return suffix

    from pathlib import PurePosixPath  # noqa: PLC0415 — stdlib, lazy for startup

    file_suffix = PurePosixPath(filename or "").suffix.lower()
    if file_suffix in _KNOWN_SUFFIXES:
        return file_suffix
    return ".sql"


async def _read_capped(upload: UploadFile) -> bytes:
    """Read the upload in chunks, aborting with 413 once the cap is exceeded.

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
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Uploaded file exceeds the {_MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _parse_upload(content: bytes, suffix: str) -> DatabaseSchema:
    """Parse uploaded bytes by writing a temp file and reusing ``parse_schema_file``.

    The temp file lives only for the duration of the parse (``with`` block) and is
    always removed, even on parse failure. The parsers raise ``ValueError`` on
    malformed content (DBML wraps any underlying parse error; ``OSError`` covers
    the file IO; Pydantic ``ValidationError`` is a ``ValueError`` subclass) —
    these become a *detailed* friendly ``HTTPException(400)``. A final guard
    catches any *unexpected* exception and returns a *generic* 400 with no
    exception text, so a malfunctioning parser can never leak a traceback to the
    client (AC: "no raw traceback").
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
            raise HTTPException(
                status_code=400,
                detail=f"Could not parse the uploaded schema: {exc}",
            ) from exc
        except Exception as exc:  # defensive: never leak a traceback to the client
            raise HTTPException(
                status_code=400,
                detail="Could not parse the uploaded schema (unsupported or invalid format).",
            ) from exc


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
) -> JSONResponse:
    """Parse an uploaded schema file and store it in the session workspace.

    Returns a JSON summary (``table_count``, ``tables``, ``dialect``, ``source``)
    on success. Rejects oversize uploads with ``413`` and unknown/unparseable
    formats or empty uploads with ``400`` — always as a friendly JSON ``detail``,
    never a traceback.
    """
    suffix = _resolve_suffix(file.filename, parser)
    content = await _read_capped(file)
    if not content.strip():
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    schema = _parse_upload(content, suffix)
    source = f"upload:{file.filename}" if file.filename else "upload:<unnamed>"

    workspace = _workspace(request)
    workspace.set_schema(schema)
    workspace.set_source(source)

    return JSONResponse(_summary(schema, source))
