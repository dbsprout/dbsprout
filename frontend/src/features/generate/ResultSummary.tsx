import { useQuery } from "@tanstack/react-query";
import { getJobResult, getSpec, queryKeys } from "../../api/endpoints";
import type { JobTableResult, TableSpec } from "../../api/types";

interface ResultSummaryProps {
  /**
   * The completed generation job to summarise. When supplied (P4-4), the real
   * per-table generated counts + durations are read from
   * GET /api/jobs/{id}/result. When omitted, the component falls back to the
   * /api/spec approximation (spec counts equal generated counts because
   * generation honours the spec) — used by callers not yet wired to a jobId.
   */
  jobId?: string;
}

/** Render the real per-table result envelope (P4-4). */
function RealSummary({ jobId }: { jobId: string }) {
  const result = useQuery({
    queryKey: queryKeys.jobResult(jobId),
    queryFn: () => getJobResult(jobId),
  });

  if (result.isLoading) {
    return <p>Summarising…</p>;
  }
  if (result.isError || !result.data || !Array.isArray(result.data.tables)) {
    return <p>No summary available.</p>;
  }

  const { tables, total_rows, total_tables, total_duration_ms } = result.data;

  return (
    <div>
      <h3>Result summary</h3>
      <ul>
        {tables.map((t: JobTableResult) => (
          <li key={t.table_name}>
            {t.table_name} — {t.row_count.toLocaleString()} rows ·{" "}
            {t.duration_ms.toLocaleString()} ms
          </li>
        ))}
      </ul>
      <p>
        Total: <strong>{total_rows.toLocaleString()}</strong> rows across {total_tables} tables in{" "}
        {total_duration_ms.toLocaleString()} ms
      </p>
    </div>
  );
}

/** Render the /api/spec-derived approximation (no jobId available). */
function SpecSummary() {
  const spec = useQuery({ queryKey: queryKeys.spec, queryFn: getSpec });

  if (spec.isLoading) {
    return <p>Summarising…</p>;
  }
  if (spec.isError || !spec.data) {
    return <p>No summary available.</p>;
  }

  const tables = spec.data.tables;
  const total = tables.reduce((sum, t) => sum + t.row_count, 0);

  return (
    <div>
      <h3>Result summary</h3>
      <ul>
        {tables.map((t: TableSpec) => (
          <li key={t.table_name}>
            {t.table_name} — {t.row_count.toLocaleString()} rows
          </li>
        ))}
      </ul>
      <p>
        Total: <strong>{total.toLocaleString()}</strong> rows across {tables.length} tables
      </p>
    </div>
  );
}

/**
 * Per-table generated row-count summary shown after a successful run.
 *
 * With a `jobId` (P4-4) the counts + per-table / total durations are the *real*
 * generated values read from GET /api/jobs/{id}/result. Without one, the counts
 * are read from the loaded spec (GET /api/spec): generation honours the spec's
 * per-table row_count, so spec counts equal generated counts — the legacy
 * approximation kept for callers not yet wired to a jobId.
 */
export function ResultSummary({ jobId }: ResultSummaryProps) {
  return jobId ? <RealSummary jobId={jobId} /> : <SpecSummary />;
}
