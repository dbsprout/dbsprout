import type { IntegrityByTable } from "../../api/types";

/** A single integrity check rolled up across every table. */
export interface IntegrityCheckRow {
  check: "fk" | "unique" | "not_null" | "check";
  label: string;
  count: number;
  passed: boolean;
}

const CHECKS: readonly { check: IntegrityCheckRow["check"]; label: string; field: keyof IntegrityByTable }[] = [
  { check: "fk", label: "Foreign keys", field: "fk_violations" },
  { check: "unique", label: "Uniqueness", field: "unique_violations" },
  { check: "not_null", label: "NOT NULL", field: "not_null_violations" },
  { check: "check", label: "CHECK", field: "check_violations" },
];

/**
 * Roll the per-table `by_table` buckets up into a fixed four-row per-check
 * summary. A check "passes" when its total violation count across all tables is
 * zero. The four checks are always present (stable order) so the panel renders a
 * consistent grid even for a clean or empty run.
 */
export function integritySummary(byTable: IntegrityByTable[]): IntegrityCheckRow[] {
  return CHECKS.map(({ check, label, field }) => {
    const count = byTable.reduce((sum, row) => sum + (row[field] as number), 0);
    return { check, label, count, passed: count === 0 };
  });
}
