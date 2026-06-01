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
    return <p>Loading spec…</p>;
  }

  if (isError) {
    return <p>No schema loaded — pick a source first.</p>;
  }

  return (
    <div>
      {mutation.isError && (
        <p role="alert">{(mutation.error as ApiError).message}</p>
      )}
      <ul>
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
    <li>
      <span>{spec.table_name}</span>
      <input
        ref={inputRef}
        type="number"
        aria-label={`rows for ${spec.table_name}`}
        defaultValue={spec.row_count}
        min={1}
        onBlur={commit}
        onKeyDown={handleKeyDown}
      />
    </li>
  );
}
