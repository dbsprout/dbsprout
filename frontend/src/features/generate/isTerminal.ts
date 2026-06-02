import type { JobStatus } from "../../api/types";

const TERMINAL: ReadonlySet<JobStatus> = new Set<JobStatus>([
  "succeeded",
  "failed",
  "cancelled",
]);

/** True once a job has reached a terminal state — used to stop polling. */
export function isTerminal(status: JobStatus | undefined): boolean {
  return status !== undefined && TERMINAL.has(status);
}
