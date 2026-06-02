import { useQuery } from "@tanstack/react-query";
import { getSpec, queryKeys } from "../../api/endpoints";
import type { TableSpec } from "../../api/types";

/**
 * Per-table generated row-count summary shown after a successful run.
 *
 * The GET /api/jobs/{id} envelope does not carry the GenerateResult, so the
 * counts are read from the loaded spec (GET /api/spec): generation honours the
 * spec's per-table row_count, so spec counts equal generated counts. A richer
 * job-result envelope is a planned follow-up.
 */
export function ResultSummary() {
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
