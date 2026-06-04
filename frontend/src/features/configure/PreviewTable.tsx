import { useQuery } from "@tanstack/react-query";
import { getPreview, queryKeys } from "../../api/endpoints";

/** Live sample rows for one table; refreshed by invalidating its preview query. */
export function PreviewTable({ table }: { table: string }) {
  const { data, isLoading, isError } = useQuery({
    queryKey: queryKeys.preview(table),
    queryFn: () => getPreview(table),
  });

  if (isLoading) {
    return <p className="db-notice-muted">Loading preview…</p>;
  }
  if (isError || !data || data.rows.length === 0) {
    return (
      <p className="db-notice-muted">No preview yet — run generate to see sample rows.</p>
    );
  }

  const cols = Object.keys(data.rows[0]);
  return (
    <div className="db-subsection overflow-x-auto">
    <table aria-label={`preview of ${table}`} className="db-table">
      <thead>
        <tr>
          {cols.map((c) => (
            <th key={c}>{c}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {data.rows.map((row, i) => (
          <tr key={i}>
            {cols.map((c) => (
              <td key={c}>{String(row[c] ?? "")}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
    </div>
  );
}
