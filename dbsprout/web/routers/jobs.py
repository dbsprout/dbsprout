"""``POST /api/jobs/{job_id}/cancel`` — cooperative cancel for the active job (S-126).

The Studio dashboard runs the generation pipeline as a single in-flight background
job (S-108 :class:`~dbsprout.web.jobs.JobManager` + S-124 generate route). This
module owns the cancel endpoint that the Studio console's Cancel button calls:

* Looks up the :class:`~dbsprout.web.jobs.JobRecord` via the manager.
* If unknown → ``404``.
* If the job already reached a terminal state (``succeeded`` / ``failed`` /
  ``cancelled``) → ``409``. Cancelling a dead job is a no-op on the underlying
  token, but returning a friendly conflict makes the UI's race-window obvious.
* Otherwise calls :meth:`~dbsprout.web.jobs.JobManager.cancel` (which arms the
  S-107 ``cancel_token``) and returns ``{"job_id", "status"}``. The status is
  ``"cancelling"`` until the orchestrator's per-table poll observes the token
  and raises :class:`~dbsprout.generate.progress.GenerationCancelled`, after
  which :meth:`~dbsprout.web.jobs.JobManager._run` flips the record to
  ``CANCELLED`` and the WS stream emits a terminal event.

The route is **separate** from :mod:`dbsprout.web.routers.generate` on purpose:
S-125 also targets the Studio console template / generate route in parallel, so
keeping the cancel API in its own module avoids merge conflicts with that
parallel branch. Heavy imports live inside the handler to preserve the
``dbsprout serve`` lazy-import contract (mirrors S-124).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

jobs_router = APIRouter()


@jobs_router.post("/api/jobs/{job_id}/cancel")
async def cancel_job(job_id: str, request: Request) -> dict[str, Any]:
    """Arm the cooperative cancel for *job_id*; return ``{job_id, status}``.

    Returns ``404`` if the id is unknown and ``409`` if the job already reached
    a terminal state. On success, status is ``"cancelling"`` (or the already-
    ``"cancelled"`` terminal value if the worker reacted between the lookup and
    the response — both are AC-valid).
    """
    from dbsprout.web.jobs import JobError, JobStatus  # noqa: PLC0415

    manager = request.app.state.job_manager
    try:
        record = manager.get(job_id)
    except JobError as exc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown job {job_id!r}.",
        ) from exc

    terminal = {JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED}
    if record.status in terminal:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=(
                f"Job {job_id!r} already reached terminal state "
                f"{record.status.value!r}; cannot cancel."
            ),
        )

    manager.cancel(job_id)
    # The status may already have flipped to CANCELLED if the worker observed
    # the token before this line; otherwise it stays RUNNING here and the UI
    # shows "cancelling" until the WS terminal frame.
    reported = "cancelling" if record.status is JobStatus.RUNNING else record.status.value
    return {"job_id": job_id, "status": reported}
