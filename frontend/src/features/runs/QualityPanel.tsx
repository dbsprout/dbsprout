import { useQuery } from "@tanstack/react-query";
import { getQuality, queryKeys } from "../../api/endpoints";
import type { QualityRow } from "../../api/types";

/** Map a quality status to its pill colour class (pass/warn/fail). */
function statusPill(status: string): string {
  if (status === "pass") return "db-pill-pass";
  if (status === "warn") return "db-pill-warn";
  return "db-pill-fail";
}

interface QualityPanelProps {
  /** Run to show metrics for; the latest run when undefined. */
  runId: number | undefined;
}

/**
 * Pass/fail/warn quality table over GET /api/quality for the selected (or
 * latest) run. Each metric carries a classified status badge. Honest
 * empty-state when no run exists or the run id is unknown (found=false).
 */
export function QualityPanel({ runId }: QualityPanelProps) {
  const quality = useQuery({
    queryKey: queryKeys.quality(runId),
    queryFn: () => getQuality(runId),
  });

  if (quality.isLoading) {
    return <p className="db-notice-muted">Loading quality…</p>;
  }
  if (quality.isError || !quality.data) {
    return (
      <p role="alert" className="db-notice-alert">
        Could not load quality.
      </p>
    );
  }
  if (!quality.data.found || quality.data.rows.length === 0) {
    return (
      <p className="db-notice-muted">
        No quality data yet. Quality metrics are recorded when a generation run completes — run{" "}
        <strong>Generate</strong> to populate them, or use the <strong>Validate</strong> step for a
        live integrity check of your current data.
      </p>
    );
  }

  return (
    <div className="mt-4">
      <h3 className="db-subsection-title">Quality metrics</h3>
      <div className="overflow-x-auto rounded-md border border-slate-200">
      <table className="db-table">
        <thead>
          <tr>
            <th>Type</th>
            <th>Metric</th>
            <th>Score</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {quality.data.rows.map((r: QualityRow) => (
            <tr key={`${r.metric_type}:${r.metric_name}`} data-status={r.status}>
              <td>{r.metric_type}</td>
              <td>{r.metric_name}</td>
              <td>{r.score.toFixed(4)}</td>
              <td>
                <span className={statusPill(r.status)}>{r.status}</span>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  );
}
