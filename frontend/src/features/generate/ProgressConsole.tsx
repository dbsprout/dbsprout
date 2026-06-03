import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { ApiError } from "../../api/client";
import { getJob, queryKeys } from "../../api/endpoints";
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

  // The WS terminal status (when present) wins; otherwise fall back to the poll.
  const status = progress?.status ?? job.data?.status;
  const succeeded = status === "succeeded";

  useEffect(() => {
    if (succeeded) onSucceeded?.();
  }, [succeeded, onSucceeded]);

  if (job.isError) {
    return <p role="alert">{(job.error as ApiError).message}</p>;
  }

  if (!job.data && !progress) {
    return <p>Starting job…</p>;
  }

  const engine = job.data?.engine;
  const seed = job.data?.seed;
  // Prefer the WS terminal error when the socket reported the failure; else poll.
  const error = progress?.status === "failed" ? progress.error : job.data?.error;

  return (
    <div>
      <p>
        Status: <strong>{status}</strong>
      </p>
      <p>
        Engine: {engine ?? "—"} · Seed: {seed ?? "—"}
      </p>
      {progress && progress.table && (
        <p>
          Table: <strong>{progress.table}</strong> · {progress.tablesDone} / {progress.tablesTotal}{" "}
          tables · {progress.totalRows} rows
        </p>
      )}
      {status === "failed" && error && <p role="alert">{error}</p>}
      {status === "cancelled" && <p>Run cancelled.</p>}
    </div>
  );
}
