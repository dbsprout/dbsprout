import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type KeyboardEvent, useRef } from "react";
import { ApiError } from "../../api/client";
import { getSpec, putTableRowCount, queryKeys } from "../../api/endpoints";
import type { TableSpec } from "../../api/types";

export function SpecPanel() {
  const qc = useQueryClient();
  const { data, isLoading, isError } = useQuery({
    queryKey: queryKeys.spec,
    queryFn: getSpec,
  });

  const mutation = useMutation({
    mutationFn: (v: { table: string; rowCount: number }) =>
      putTableRowCount(v.table, v.rowCount),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.spec }),
  });

  if (isLoading) {
    return <p className="db-notice-muted">Loading spec…</p>;
  }

  if (isError) {
    return <p className="db-notice-muted">No schema loaded — pick a source first.</p>;
  }

  return (
    <div className="db-subsection">
      <p className="db-subsection-title">Rows per table</p>
      {mutation.isError && (
        <p role="alert" className="db-notice-alert mb-2">{(mutation.error as ApiError).message}</p>
      )}
      <ul className="flex flex-col gap-1">
        {data?.tables.map((t: TableSpec) => (
          <TableRow key={t.table_name} spec={t} onCommit={(rowCount) => mutation.mutate({ table: t.table_name, rowCount })} />
        ))}
      </ul>
    </div>
  );
}

function TableRow({ spec, onCommit }: { spec: TableSpec; onCommit: (rowCount: number) => void }) {
  const inputRef = useRef<HTMLInputElement>(null);

  function commit() {
    const raw = inputRef.current?.value ?? "";
    const n = parseInt(raw, 10);
    if (Number.isInteger(n) && n >= 1 && n !== spec.row_count) {
      onCommit(n);
    }
  }

  function handleKeyDown(e: KeyboardEvent<HTMLInputElement>) {
    if (e.key === "Enter") {
      commit();
    }
  }

  return (
    <li className="flex items-center justify-between gap-3 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm">
      <span className="font-mono text-slate-700">{spec.table_name}</span>
      <input
        ref={inputRef}
        type="number"
        aria-label={`rows for ${spec.table_name}`}
        defaultValue={spec.row_count}
        min={1}
        onBlur={commit}
        onKeyDown={handleKeyDown}
        className="w-28 rounded-md border border-slate-300 px-2 py-1 text-right text-sm outline-none focus:border-accent-500 focus:ring-2 focus:ring-accent-500/40"
      />
    </li>
  );
}
