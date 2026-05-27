"""Live-progress + cooperative-cancel primitives for generation (S-107).

These are the low-level hooks the orchestrator emits/raises and that the
future async Job Manager (S-108) and WebSocket transport (S-109) build on. This
module deliberately contains *only* the primitives — no job runner, no
transport, no UI.

``ProgressEvent``
    A frozen value object the orchestrator hands to a ``progress_callback`` once
    per table boundary (``table_start`` before generating a table,
    ``table_done`` after). Running counts let a UI render per-table progress and
    a total.

``CancelToken`` / ``_is_cancelled``
    A *cooperative* cancel is checked at the top of the per-table loop. To suit
    both a web job runner (which holds a small cancellable object) and a unit
    test (which wants a one-line lambda), the orchestrator accepts **either** an
    object exposing ``is_cancelled() -> bool`` (described by the ``CancelToken``
    Protocol) **or** a zero-arg ``Callable[[], bool]``. ``_is_cancelled``
    normalizes both (and ``None`` → never cancelled).

``GenerationCancelled``
    Raised to unwind the generation loop on cancel. It subclasses plain
    ``Exception`` — NOT :class:`dbsprout.errors.DBSproutError`. The DBSproutError
    hierarchy is the CLI's user-facing *what/why/fix* surface; a cooperative
    cancel is an expected control signal a caller (the job runner) catches and
    maps to a ``cancelled`` job status. It carries ``tables_done`` /
    ``tables_total`` so the catcher can report how far generation got.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from collections.abc import Callable


class ProgressEvent(BaseModel):
    """An immutable progress notification emitted once per table boundary.

    ``phase`` is ``"table_start"`` (emitted before a table is generated) or
    ``"table_done"`` (after constraint enforcement). ``tables_done`` /
    ``tables_total`` track table-level progress; ``total_rows`` is the running
    cumulative row count across all tables generated so far; ``rows_in_table``
    is ``0`` on ``table_start`` and the table's row count on ``table_done``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    phase: str
    table: str | None = None
    tables_done: int = 0
    tables_total: int = 0
    rows_in_table: int = 0
    total_rows: int = 0
    message: str | None = None


@runtime_checkable
class CancelToken(Protocol):
    """A cooperative cancel signal exposing ``is_cancelled()``.

    The web Job Manager (S-108) holds an object satisfying this Protocol. The
    orchestrator also accepts a bare zero-arg ``Callable[[], bool]`` (handy in
    tests); :func:`_is_cancelled` normalizes both forms.
    """

    def is_cancelled(self) -> bool:  # pragma: no cover - structural Protocol stub
        """Return ``True`` to request cancellation, ``False`` to continue."""
        ...


class GenerationCancelled(Exception):  # noqa: N818 — control-flow cancel signal, not a user-facing error; name fixed by the S-107 acceptance criteria
    """Raised when a cooperative cancel stops generation cleanly.

    A control-flow signal (not a user-facing :class:`~dbsprout.errors.DBSproutError`).
    Carries the table-level progress reached at the moment of cancellation.
    """

    def __init__(self, *, tables_done: int, tables_total: int) -> None:
        self.tables_done = tables_done
        self.tables_total = tables_total
        super().__init__(f"Generation cancelled after {tables_done}/{tables_total} tables.")


def _is_cancelled(token: CancelToken | Callable[[], bool] | None) -> bool:
    """Normalize a cancel token to a single boolean check.

    ``None`` → never cancelled. An object with ``is_cancelled`` → call it.
    Otherwise treat *token* as a zero-arg callable and call it.
    """
    if token is None:
        return False
    if isinstance(token, CancelToken):
        return bool(token.is_cancelled())
    return bool(token())
