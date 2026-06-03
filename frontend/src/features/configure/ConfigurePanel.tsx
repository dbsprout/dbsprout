import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import {
  getPreview,
  getSpec,
  listGenerators,
  putColumnSpec,
  // ─── P2b-2 ───
  putTableAdvanced,
  // ─── end P2b-2 ───
  queryKeys,
} from "../../api/endpoints";
import type {
  // ─── P2b-2 ───
  CorrelationRule,
  DerivedColumn,
  // ─── end P2b-2 ───
  GeneratorConfig,
  TableSpec,
} from "../../api/types";
import { ColumnGrid } from "./ColumnGrid";
import { ColumnInspector } from "./ColumnInspector";
// ─── P2b-2 ───
import { CorrelationsEditor } from "./CorrelationsEditor";
import { DerivedColumns } from "./DerivedColumns";
// ─── end P2b-2 ───
import { PreviewTable } from "./PreviewTable";
import { SpecAssist } from "./SpecAssist";

/**
 * Configure surface: a table picker drives a per-column generator grid, a Column
 * Inspector for the focused column, and a live preview. Generator/inspector edits
 * persist via `putColumnSpec` and invalidate both the spec and the table preview.
 */
export function ConfigurePanel() {
  const qc = useQueryClient();
  const [selectedTable, setSelectedTable] = useState<string | null>(null);
  const [selectedColumn, setSelectedColumn] = useState<string | null>(null);

  const spec = useQuery({ queryKey: queryKeys.spec, queryFn: getSpec });
  const generators = useQuery({
    queryKey: queryKeys.generators,
    queryFn: () => listGenerators(),
  });

  const tables = spec.data?.tables ?? [];
  const table = tables.find((t) => t.table_name === selectedTable) ?? tables[0];

  const preview = useQuery({
    queryKey: table ? queryKeys.preview(table.table_name) : ["preview", "__none__"],
    queryFn: () => getPreview(table!.table_name),
    enabled: !!table,
  });

  const mutation = useMutation({
    mutationFn: (v: { column: string; cfg: GeneratorConfig }) =>
      putColumnSpec(table!.table_name, v.column, v.cfg),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.spec });
      if (table) qc.invalidateQueries({ queryKey: queryKeys.preview(table.table_name) });
    },
  });

  // ─── P2b-2 ─── advanced packs (correlations + derived) persistence.
  const advancedMutation = useMutation({
    mutationFn: (body: { correlations?: CorrelationRule[]; derived?: DerivedColumn[] }) =>
      putTableAdvanced(table!.table_name, body),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.spec }),
  });
  // ─── end P2b-2 ───

  if (spec.isLoading) {
    return <p>Loading spec…</p>;
  }
  if (spec.isError || !table) {
    return <p>No schema loaded — pick a source first.</p>;
  }

  const activeTable: TableSpec = table;
  const focusedCfg = selectedColumn ? activeTable.columns[selectedColumn] : null;

  return (
    <div>
      {/* ═══ P2b-3 ═══ */}
      {/* AI spec-assist: an LLM proposes a full DataSpec for the loaded schema;
          on success the spec query is invalidated so this grid repaints. */}
      <SpecAssist />
      {/* ═══ end P2b-3 ═══ */}
      <label>
        table
        <select
          aria-label="configure table"
          value={activeTable.table_name}
          onChange={(e) => {
            setSelectedTable(e.target.value);
            setSelectedColumn(null);
          }}
        >
          {tables.map((t) => (
            <option key={t.table_name} value={t.table_name}>
              {t.table_name}
            </option>
          ))}
        </select>
      </label>

      {mutation.isError && <p role="alert">Failed to save column.</p>}

      <ColumnGrid
        spec={activeTable}
        methods={generators.data?.methods ?? []}
        previewRow={preview.data?.rows[0] ?? null}
        onSelect={(c) => setSelectedColumn(c)}
        onGeneratorChange={(column, cfg) => mutation.mutate({ column, cfg })}
      />

      {selectedColumn && focusedCfg && (
        <ColumnInspector
          column={selectedColumn}
          cfg={focusedCfg}
          onSave={(cfg) => mutation.mutate({ column: selectedColumn, cfg })}
        />
      )}

      {/* ═══ P2b-2 ═══ advanced packs: correlations + derived columns */}
      {advancedMutation.isError && <p role="alert">Failed to save advanced packs.</p>}
      <CorrelationsEditor
        table={activeTable.table_name}
        columns={Object.keys(activeTable.columns)}
        tables={tables.map((t) => t.table_name)}
        rules={activeTable.correlations}
        onSave={(correlations) => advancedMutation.mutate({ correlations })}
      />
      <DerivedColumns
        table={activeTable.table_name}
        columns={Object.keys(activeTable.columns)}
        derived={activeTable.derived}
        onSave={(derived) => advancedMutation.mutate({ derived })}
      />
      {/* ═══ end P2b-2 ═══ */}

      <PreviewTable table={activeTable.table_name} />
    </div>
  );
}
