import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { exportData, getSchema, getSpec, queryKeys } from "../../api/endpoints";
import type { ExportFormat } from "../../api/types";

const FORMATS: readonly ExportFormat[] = ["sql", "csv", "json", "parquet"];

/**
 * Output → Export panel (P1c-1).
 *
 * Pick a file format (SQL / CSV / JSON / Parquet) and an optional subset of tables,
 * then POST /api/export and let the browser save the streamed file. The Export
 * button stays disabled until a spec exists (GET /api/spec succeeds) — that is the
 * "a generation has run / spec exists" signal. An empty table selection means
 * "every table", so the request omits the `tables` key. A non-OK response surfaces
 * a typed {@link ApiError} message (the backend has already scrubbed internals).
 */
export function ExportPanel() {
  const [format, setFormat] = useState<ExportFormat>("sql");
  const [selected, setSelected] = useState<string[]>([]);

  const spec = useQuery({ queryKey: queryKeys.spec, queryFn: getSpec, retry: false });
  const schema = useQuery({ queryKey: queryKeys.schema, queryFn: getSchema, retry: false });

  const tableNames = schema.data?.tables.map((t) => t.name) ?? [];

  const mutation = useMutation({
    mutationFn: () => exportData(format, selected.length > 0 ? selected : undefined),
  });

  function handleTablesChange(e: React.ChangeEvent<HTMLSelectElement>) {
    setSelected(Array.from(e.target.selectedOptions, (o) => o.value));
  }

  const ready = spec.isSuccess;

  return (
    <div>
      <label>
        format
        <select
          aria-label="export format"
          value={format}
          onChange={(e) => setFormat(e.target.value as ExportFormat)}
        >
          {FORMATS.map((f) => (
            <option key={f} value={f}>
              {f}
            </option>
          ))}
        </select>
      </label>
      <label>
        tables (optional — default all)
        <select
          aria-label="export tables"
          multiple
          value={selected}
          onChange={handleTablesChange}
        >
          {tableNames.map((name) => (
            <option key={name} value={name}>
              {name}
            </option>
          ))}
        </select>
      </label>
      <button
        type="button"
        disabled={!ready || mutation.isPending}
        onClick={() => mutation.mutate()}
      >
        Export
      </button>
      {mutation.isError && <p role="alert">{(mutation.error as ApiError).message}</p>}
    </div>
  );
}
