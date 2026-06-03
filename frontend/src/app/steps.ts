import type { QueryClient } from "@tanstack/react-query";
import { queryKeys } from "../api/endpoints";

export interface Step {
  /** Stable identifier (also the App section it targets). */
  id: string;
  /** Human-readable step title shown in the stepper. */
  title: string;
  /**
   * Precondition for advancing PAST this step. Reads the TanStack Query cache;
   * pure and synchronous so it is unit-testable as a truth table.
   */
  gate: (qc: QueryClient) => boolean;
}

/** A loaded schema has at least one table in the schema cache. */
export function hasSchema(qc: QueryClient): boolean {
  const data = qc.getQueryData(queryKeys.schema) as
    | { tables?: unknown[] }
    | undefined;
  return !!data?.tables && data.tables.length > 0;
}

/** A spec exists with at least one table-spec in the spec cache. */
export function hasSpec(qc: QueryClient): boolean {
  const data = qc.getQueryData(queryKeys.spec) as
    | { tables?: unknown[] }
    | undefined;
  return !!data?.tables && data.tables.length > 0;
}

/**
 * Any cached job has succeeded. The job query key is parameterized
 * (`["job", jobId]`), so scan every matching cache entry rather than a single
 * fixed-key lookup.
 */
export function hasSucceededJob(qc: QueryClient): boolean {
  return qc
    .getQueryCache()
    .findAll({ queryKey: ["job"] })
    .some(
      (q) =>
        (q.state.data as { status?: string } | undefined)?.status === "succeeded",
    );
}

const always = (): boolean => true;

/**
 * The 7-step guided sequence, mapped to the existing App sections. Each gate is
 * the precondition for advancing PAST that step (Back is always allowed). Steps
 * with no hard precondition in this slice gate `true`; P3-2 may tighten them.
 */
export const STEPS: readonly Step[] = [
  { id: "start", title: "Start", gate: hasSchema },
  { id: "schema", title: "Schema", gate: hasSchema },
  { id: "configure", title: "Configure", gate: hasSpec },
  { id: "generate", title: "Generate", gate: hasSucceededJob },
  { id: "validate", title: "Validate", gate: always },
  { id: "output", title: "Output", gate: always },
  { id: "runs", title: "Runs & Quality", gate: always },
] as const;
