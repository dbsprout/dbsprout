import { useQuery } from "@tanstack/react-query";
import { getCosts, queryKeys } from "../../api/endpoints";
import type { ProviderCost } from "../../api/types";

/**
 * LLM cost summary over GET /api/costs: headline totals plus a per-provider
 * breakdown table. A chart is optional; the table satisfies the AC. Honest
 * empty-state for offline/heuristic runs with no LLM calls (all zeros).
 */
export function CostsPanel() {
  const costs = useQuery({ queryKey: queryKeys.costs, queryFn: getCosts });

  if (costs.isLoading) {
    return <p className="db-notice-muted">Loading costs…</p>;
  }
  if (costs.isError || !costs.data) {
    return (
      <p role="alert" className="db-notice-alert">
        Could not load costs.
      </p>
    );
  }

  const { total_cost, total_tokens, total_calls, avg_cost_per_run, per_provider } = costs.data;

  if (total_calls === 0) {
    return (
      <p className="db-notice-muted">No LLM calls recorded (offline / heuristic runs).</p>
    );
  }

  return (
    <div className="mt-4">
      <h3 className="db-subsection-title">LLM costs</h3>
      <ul className="mb-3 grid grid-cols-2 gap-2 text-sm text-slate-700 sm:grid-cols-4">
        <li>
          Total cost: <strong>${total_cost.toFixed(4)}</strong>
        </li>
        <li>Total tokens: {total_tokens.toLocaleString()}</li>
        <li>Total calls: {total_calls.toLocaleString()}</li>
        <li>Avg cost / run: ${avg_cost_per_run.toFixed(4)}</li>
      </ul>
      <div className="overflow-x-auto rounded-md border border-slate-200">
      <table className="db-table">
        <thead>
          <tr>
            <th>Provider</th>
            <th>Cost (USD)</th>
            <th>Tokens</th>
            <th>Calls</th>
          </tr>
        </thead>
        <tbody>
          {per_provider.map((p: ProviderCost) => (
            <tr key={p.provider}>
              <td>{p.provider}</td>
              <td>{p.cost.toFixed(4)}</td>
              <td>{p.tokens.toLocaleString()}</td>
              <td>{p.calls.toLocaleString()}</td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  );
}
