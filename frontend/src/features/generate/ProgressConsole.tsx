import { useQuery } from "@tanstack/react-query";
import { useEffect } from "react";
import { ApiError } from "../../api/client";
import { getJob, queryKeys } from "../../api/endpoints";
import { isTerminal } from "./isTerminal";

interface ProgressConsoleProps {
  jobId: string;
  /** Poll interval in ms while the job is still running. */
  pollMs?: number;
  /** Fired once when the job first reaches the succeeded state. */
  onSucceeded?: () => void;
}

/**
 * Poll GET /api/jobs/{jobId} on an interval and render the live status. Polling
 * stops as soon as the job reaches a terminal state (succeeded / failed /
 * cancelled). On failure the (already credential-scrubbed) server error is shown;
 * a transport/HTTP failure of the poll itself surfaces the typed ApiError message.
 *
 * Per-table progress events stream over a WebSocket on the backend; this slice
 * intentionally uses polling only, so the console reports status transitions
 * rather than per-row progress.
 */
export function ProgressConsole({ jobId, pollMs = 500, onSucceeded }: ProgressConsoleProps) {
  const job = useQuery({
    queryKey: queryKeys.job(jobId),
    queryFn: () => getJob(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => (isTerminal(query.state.data?.status) ? false : pollMs),
  });

  const status = job.data?.status;
  const succeeded = status === "succeeded";

  useEffect(() => {
    if (succeeded) onSucceeded?.();
  }, [succeeded, onSucceeded]);

  if (job.isError) {
    return <p role="alert">{(job.error as ApiError).message}</p>;
  }

  if (!job.data) {
    return <p>Starting job…</p>;
  }

  const { engine, seed, error } = job.data;

  return (
    <div>
      <p>
        Status: <strong>{status}</strong>
      </p>
      <p>
        Engine: {engine ?? "—"} · Seed: {seed ?? "—"}
      </p>
      {status === "failed" && error && <p role="alert">{error}</p>}
      {status === "cancelled" && <p>Run cancelled.</p>}
    </div>
  );
}
