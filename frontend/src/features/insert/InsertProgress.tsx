import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { ApiError } from "../../api/client";
import { cancelJob, getJob, queryKeys } from "../../api/endpoints";
import { isTerminal } from "../generate/isTerminal";

interface InsertProgressProps {
  jobId: string;
  /** Poll interval in ms while the job is still running. */
  pollMs?: number;
  /** Fired once when the job first reaches the succeeded state. */
  onSucceeded?: () => void;
}

/**
 * Poll GET /api/jobs/{jobId} on an interval and render the live insert status.
 * Polling stops as soon as the job reaches a terminal state (succeeded / failed
 * / cancelled). A Cancel button arms the cooperative cancel via
 * POST /api/jobs/{id}/cancel; the next poll observes the cancelled state and the
 * poll settles. On failure the (already credential-scrubbed) server error is
 * shown; a transport failure of the poll itself surfaces the typed ApiError.
 *
 * Modelled on the P1b-3 generate ProgressConsole — polling only (per-table
 * progress streams over a WebSocket on the backend, out of scope for this slice).
 */
export function InsertProgress({ jobId, pollMs = 500, onSucceeded }: InsertProgressProps) {
  const job = useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: () => getJob(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => (isTerminal(query.state.data?.status) ? false : pollMs),
  });

  const cancel = useMutation({ mutationFn: () => cancelJob(jobId) });

  const status = job.data?.status;
  const succeeded = status === "succeeded";

  useEffect(() => {
    if (succeeded) onSucceeded?.();
  }, [succeeded, onSucceeded]);

  if (job.isError) {
    return <p role="alert" className="db-notice-alert">{(job.error as ApiError).message}</p>;
  }

  if (!job.data) {
    return <p className="db-notice-muted">Starting insert…</p>;
  }

  const terminal = isTerminal(status);
  const { error } = job.data;

  return (
    <div className="db-subsection flex flex-col gap-2">
      <p className="text-sm text-slate-700">
        Status: <strong className="text-slate-900">{status}</strong>
      </p>
      {!terminal && (
        <button type="button" className="db-btn-danger self-start" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
          Cancel
        </button>
      )}
      {status === "failed" && error && <p role="alert" className="db-notice-alert">{error}</p>}
      {status === "cancelled" && <p className="db-notice-muted">Insert cancelled.</p>}
    </div>
  );
}
