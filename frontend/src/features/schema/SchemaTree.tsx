import { useQuery } from "@tanstack/react-query";
import { getSchema, queryKeys } from "../../api/endpoints";

export function SchemaTree() {
  const { data, isLoading, isError } = useQuery({
    queryKey: queryKeys.schema,
    queryFn: getSchema,
  });

  if (isLoading) {
    return <p>Loading schema…</p>;
  }

  if (isError || !data) {
    return <p>No schema loaded yet — pick a source to begin.</p>;
  }

  return (
    <ul>
      {data.tables.map((table) => {
        const fkColumns = new Set(table.foreign_keys.flatMap((fk) => fk.columns));
        return (
          <li key={table.name}>
            <strong>{table.name}</strong>
            <ul>
              {table.columns.map((col) => {
                const isPk = table.primary_key.includes(col.name);
                const isFk = fkColumns.has(col.name);
                return (
                  <li key={col.name}>
                    {isPk && "🔑 "}
                    {isFk && "↗ "}
                    {col.name}: {col.type}
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
