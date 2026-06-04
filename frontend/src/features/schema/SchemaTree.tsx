import { useQuery } from "@tanstack/react-query";
import { getSchema, queryKeys } from "../../api/endpoints";

export function SchemaTree() {
  const { data, isLoading, isError } = useQuery({
    queryKey: queryKeys.schema,
    queryFn: getSchema,
  });

  if (isLoading) {
    return <p className="db-notice-muted">Loading schema…</p>;
  }

  if (isError || !data) {
    return <p className="db-notice-muted">No schema loaded yet — pick a source to begin.</p>;
  }

  return (
    <ul className="flex flex-col gap-3">
      {data.tables.map((table) => {
        const fkColumns = new Set(table.foreign_keys.flatMap((fk) => fk.columns));
        return (
          <li key={table.name}>
            <strong className="font-mono text-sm font-semibold text-slate-900">{table.name}</strong>
            <ul className="mt-1 flex flex-col gap-0.5 border-l border-slate-200 pl-3">
              {table.columns.map((col) => {
                const isPk = table.primary_key.includes(col.name);
                const isFk = fkColumns.has(col.name);
                return (
                  <li key={col.name} className="flex items-center gap-2 text-sm">
                    {isPk && (
                      <span className="db-badge-pk" title="Primary key" aria-hidden="true">
                        🔑 PK
                      </span>
                    )}
                    {isFk && (
                      <span className="db-badge-fk" title="Foreign key" aria-hidden="true">
                        ↗ FK
                      </span>
                    )}
                    <span className="font-mono text-slate-700">
                      {col.name}: <span className="text-slate-400">{col.type}</span>
                    </span>
                  </li>
                );
              })}
            </ul>
          </li>
        );
      })}
    </ul>
  );
}
