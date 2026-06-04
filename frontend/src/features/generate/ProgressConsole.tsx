import { useMutation, useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { ApiError } from "../../api/client";
import { cancelJob, getJob, queryKeys } from "../../api/endpoints";
import { isTerminal } from "./isTerminal";
import { useJobSocket } from "./useJobSocket";

interface ProgressConsoleProps {
  jobId: string;
  /** Poll interval in ms while the job is still running. */
  pollMs?: number;
  /** Fired once when the job first reaches the succeeded state. */
  onSucceeded?: () => void;
}

/**
 * Render a generate job's live progress. The per-table progress line streams
 * over the `/ws/jobs/{jobId}` WebSocket (`useJobSocket`); a `GET /api/jobs/{id}`
 * poll runs alongside as the disconnect fallback so the run still reaches a
 * terminal status if the socket drops or never opens.
 *
 * Polling stops as soon as EITHER signal reports terminal — the WS terminal
 * frame or a terminal poll. On failure the (credential-scrubbed) server error is
 * shown; a transport/HTTP failure of the poll itself surfaces the typed
 * ApiError message. `onSucceeded` fires exactly once on the first succeeded
 * signal from either source.
 *
 * While the run is non-terminal a Cancel button arms the cooperative cancel via
 * POST /api/jobs/{id}/cancel (the existing `cancelJob`, modelled on the P1c-2
 * insert-cancel pattern); the next poll observes the `cancelled` state, the poll
 * settles, and the "Run cancelled." line renders.
 */
export function ProgressConsole({ jobId, pollMs = 500, onSucceeded }: ProgressConsoleProps) {
  const { progress, terminal: wsTerminal } = useJobSocket(jobId);

  const job = useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: () => getJob(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      // Stop polling once either the poll OR the WebSocket reports terminal.
      if (wsTerminal || isTerminal(query.state.data?.status)) return false;
      return pollMs;
    },
  });

  const cancel = useMutation({ mutationFn: () => cancelJob(jobId) });

  // The WS terminal status (when present) wins; otherwise fall back to the poll.
  const status = progress?.status ?? job.data?.status;
  const succeeded = status === "succeeded";

  useEffect(() => {
    if (succeeded) onSucceeded?.();
  }, [succeeded, onSucceeded]);

  if (job.isError) {
    return <p role="alert" className="db-notice-alert">{(job.error as ApiError).message}</p>;
  }

  if (!job.data && !progress) {
    return <p className="db-notice-muted">Starting job…</p>;
  }

  const engine = job.data?.engine;
  const seed = job.data?.seed;
  // Prefer the WS terminal error when the socket reported the failure; else poll.
  const error = progress?.status === "failed" ? progress.error : job.data?.error;

  return (
    <div className="db-subsection flex flex-col gap-2">
      <p className="text-sm text-slate-700">
        Status: <strong className="text-slate-900">{status}</strong>
      </p>
      <p className="text-sm text-slate-500">
        Engine: {engine ?? "—"} · Seed: {seed ?? "—"}
      </p>
      {progress && progress.table && (
        <p className="font-mono text-sm text-slate-600">
          Table: <strong>{progress.table}</strong> · {progress.tablesDone} / {progress.tablesTotal}{" "}
          tables · {progress.totalRows} rows
        </p>
      )}
      {!isTerminal(status) && (
        <button type="button" className="db-btn-danger self-start" disabled={cancel.isPending} onClick={() => cancel.mutate()}>
          Cancel
        </button>
      )}
      {status === "failed" && error && <p role="alert" className="db-notice-alert">{error}</p>}
      {status === "cancelled" && <p className="db-notice-muted">Run cancelled.</p>}
    </div>
  );
}
