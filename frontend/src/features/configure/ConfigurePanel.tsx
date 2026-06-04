import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useState } from "react";
import { useSelection } from "../../app/SelectionProvider";
import {
  getPreview,
  getSchema,
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

  // ─── P5-7 ─── snapshot of the cross-panel "why" for the drill notice. The P4-7
  // effect nulls the global selection immediately, so we must capture the reason
  // into local state here rather than read it from the (now-cleared) selection.
  const [drillNote, setDrillNote] = useState<{
    table: string;
    column: string | null;
    reason: string;
  } | null>(null);

  // ─── P4-7 ─── cross-panel drill: a Validate violation (or any caller) can focus
  // this grid on a specific table+column. Apply the target to the table-picker
  // state, then clear it so a later manual table switch isn't overridden.
  const { selection, clearSelection } = useSelection();
  useEffect(() => {
    if (!selection) return;
    setSelectedTable(selection.table);
    setSelectedColumn(selection.column);
    // ─── P5-7 ─── capture (or clear) the drill notice from the same target.
    setDrillNote(
      selection.reason
        ? { table: selection.table, column: selection.column, reason: selection.reason }
        : null,
    );
    clearSelection();
  }, [selection, clearSelection]);
  // ─── end P4-7 ───

  const spec = useQuery({ queryKey: queryKeys.spec, queryFn: getSpec });
  const generators = useQuery({
    queryKey: queryKeys.generators,
    queryFn: () => listGenerators(),
  });
  // ─── P4-12 ─── the SQL type that drives the dtype filter lives in the schema
  // tree (`ColumnNode.type`), not in the spec; load it to feed the dormant filter.
  const schema = useQuery({ queryKey: queryKeys.schema, queryFn: getSchema });

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

  // ─── P4-12 ─── column name → raw SQL type for the active table, read from the
  // schema tree. Feeds the dtype filter (P4-1) in ColumnGrid/ColumnInspector;
  // columns absent here fall back to "unfiltered" (the child's safe default).
  const columnTypes = useMemo<Record<string, string>>(() => {
    const node = schema.data?.tables?.find((t) => t.name === table?.table_name);
    if (!node) {
      return {};
    }
    return Object.fromEntries(node.columns.map((c) => [c.name, c.type]));
  }, [schema.data, table?.table_name]);

  if (spec.isLoading) {
    return <p className="db-notice-muted">Loading spec…</p>;
  }
  if (spec.isError || !table) {
    return <p className="db-notice-muted">No schema loaded — pick a source first.</p>;
  }

  const activeTable: TableSpec = table;
  const focusedCfg = selectedColumn ? activeTable.columns[selectedColumn] : null;

  return (
    <div className="flex flex-col gap-4">
      {/* ═══ P2b-3 ═══ */}
      {/* AI spec-assist: an LLM proposes a full DataSpec for the loaded schema;
          on success the spec query is invalidated so this grid repaints. */}
      <SpecAssist />
      {/* ═══ end P2b-3 ═══ */}
      <label className="db-field">
        <span className="db-label">table</span>
        <select
          aria-label="configure table"
          className="db-input"
          value={activeTable.table_name}
          onChange={(e) => {
            setSelectedTable(e.target.value);
            setSelectedColumn(null);
            // ─── P5-7 ─── a manual table switch ends the cross-panel context.
            setDrillNote(null);
          }}
        >
          {tables.map((t) => (
            <option key={t.table_name} value={t.table_name}>
              {t.table_name}
            </option>
          ))}
        </select>
      </label>

      {/* ─── P5-7 ─── explain WHY a cell was focused after a Validate drill. */}
      {drillNote && (
        <div role="status" className="db-notice-status flex items-start justify-between gap-2">
          <span>
            Focused{" "}
            <span className="font-mono">
              {drillNote.table}
              {drillNote.column ? `.${drillNote.column}` : ""}
            </span>{" "}
            — flagged by Validate: {drillNote.reason}
          </span>
          <button
            type="button"
            aria-label="dismiss notice"
            className="db-btn-secondary"
            onClick={() => setDrillNote(null)}
          >
            ×
          </button>
        </div>
      )}

      {mutation.isError && <p role="alert" className="db-notice-alert">Failed to save column.</p>}

      <ColumnGrid
        spec={activeTable}
        methods={generators.data?.methods ?? []}
        previewRow={preview.data?.rows[0] ?? null}
        // ─── P4-12 ─── per-column SQL types activate the dtype-filtered dropdowns.
        columnTypes={columnTypes}
        // ─── P4-7 ─── scroll/highlight the drilled-into column (no-op if absent).
        focusColumn={selectedColumn}
        onSelect={(c) => {
          setSelectedColumn(c);
          // ─── P5-7 ─── picking another column ends the cross-panel context.
          setDrillNote(null);
        }}
        onGeneratorChange={(column, cfg) => mutation.mutate({ column, cfg })}
      />

      {selectedColumn && focusedCfg && (
        <ColumnInspector
          column={selectedColumn}
          cfg={focusedCfg}
          // ─── P4-12 ─── feed the inspector's dtype-filtered generator picker.
          methods={generators.data?.methods ?? []}
          columnType={columnTypes[selectedColumn]}
          onSave={(cfg) => mutation.mutate({ column: selectedColumn, cfg })}
        />
      )}

      {/* ═══ P2b-2 ═══ advanced packs: correlations + derived columns */}
      {advancedMutation.isError && <p role="alert" className="db-notice-alert">Failed to save advanced packs.</p>}
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
