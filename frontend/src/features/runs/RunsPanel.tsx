import { useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { listRuns, queryKeys } from "../../api/endpoints";
import type { RunRow } from "../../api/types";

interface RunsPanelProps {
  /** Called with a run id when a history row is clicked (drives QualityPanel). */
  onSelectRun: (runId: number) => void;
}

/**
 * Paginated run-history table over GET /api/runs. Holds the current page and
 * exposes Prev/Next (gated on has_prev/has_next). Clicking a row lifts the run
 * id up so the sibling QualityPanel can show that run's metrics. Honest
 * empty-state when the CLI has never run; an error line if the request fails.
 */
export function RunsPanel({ onSelectRun }: RunsPanelProps) {
  const [page, setPage] = useState(1);
  const runs = useQuery({
    queryKey: queryKeys.runs(page),
    queryFn: () => listRuns(page),
  });

  if (runs.isLoading) {
    return <p className="db-notice-muted">Loading runs…</p>;
  }
  if (runs.isError || !runs.data) {
    return (
      <p role="alert" className="db-notice-alert">
        Could not load runs.
      </p>
    );
  }

  const { rows, has_prev, has_next, page: current, total_pages, total_runs } = runs.data;

  if (total_runs === 0) {
    return (
      <p className="db-notice-muted">
        No runs yet — your generation history will appear here after you generate.
      </p>
    );
  }

  return (
    <div>
      <h3 className="db-subsection-title">Run history</h3>
      <div className="overflow-x-auto rounded-md border border-slate-200">
      <table className="db-table">
        <thead>
          <tr>
            <th>Started</th>
            <th>Engine</th>
            <th>Provider</th>
            <th>Rows</th>
            <th>Tables</th>
            <th>Cost (USD)</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((r: RunRow) => (
            <tr
              key={r.id ?? r.started_at}
              onClick={() => r.id != null && onSelectRun(r.id)}
              style={{ cursor: r.id != null ? "pointer" : "default" }}
              className="hover:bg-accent-50"
            >
              <td>{r.started_at}</td>
              <td>{r.engine}</td>
              <td>{r.provider ?? "—"}</td>
              <td>{r.total_rows.toLocaleString()}</td>
              <td>{r.total_tables}</td>
              <td>{r.cost.toFixed(4)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
      <div className="mt-3 flex items-center gap-3">
        <button
          type="button"
          className="db-btn-secondary"
          disabled={!has_prev}
          onClick={() => setPage((p) => Math.max(1, p - 1))}
        >
          Prev
        </button>
        <span className="text-sm text-slate-500">
          Page {current} of {total_pages}
        </span>
        <button
          type="button"
          className="db-btn-secondary"
          disabled={!has_next}
          onClick={() => setPage((p) => p + 1)}
        >
          Next
        </button>
      </div>
    </div>
  );
}
