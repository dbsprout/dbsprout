import { useQuery } from "@tanstack/react-query";
import { getQuality, queryKeys } from "../../api/endpoints";
import type { QualityRow } from "../../api/types";

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
    return <p>Loading quality…</p>;
  }
  if (quality.isError || !quality.data) {
    return <p role="alert">Could not load quality.</p>;
  }
  if (!quality.data.found || quality.data.rows.length === 0) {
    return <p>No quality data for this run.</p>;
  }

  return (
    <div>
      <h3>Quality metrics</h3>
      <table>
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
              <td>{r.status}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
