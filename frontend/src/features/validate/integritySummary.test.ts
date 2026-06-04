import { expect, test } from "vitest";
import type { IntegrityByTable } from "../../api/types";
import { integritySummary } from "./integritySummary";

const CLEAN: IntegrityByTable[] = [
  { table: "users", fk_violations: 0, unique_violations: 0, not_null_violations: 0, check_violations: 0 },
  { table: "orders", fk_violations: 0, unique_violations: 0, not_null_violations: 0, check_violations: 0 },
];

const MIXED: IntegrityByTable[] = [
  { table: "users", fk_violations: 0, unique_violations: 2, not_null_violations: 0, check_violations: 0 },
  { table: "orders", fk_violations: 3, unique_violations: 0, not_null_violations: 1, check_violations: 0 },
];

test("a clean by_table yields every check passed with zero counts", () => {
  const summary = integritySummary(CLEAN);

  expect(summary).toEqual([
    { check: "fk", label: "Foreign keys", count: 0, passed: true },
    { check: "unique", label: "Uniqueness", count: 0, passed: true },
    { check: "not_null", label: "NOT NULL", count: 0, passed: true },
    { check: "check", label: "CHECK", count: 0, passed: true },
  ]);
});

test("aggregates violation counts per check across tables and flags failures", () => {
  const summary = integritySummary(MIXED);
  const byCheck = Object.fromEntries(summary.map((s) => [s.check, s]));

  expect(byCheck.fk).toMatchObject({ count: 3, passed: false });
  expect(byCheck.unique).toMatchObject({ count: 2, passed: false });
  expect(byCheck.not_null).toMatchObject({ count: 1, passed: false });
  expect(byCheck.check).toMatchObject({ count: 0, passed: true });
});

test("an empty by_table still reports all four checks as passed", () => {
  const summary = integritySummary([]);
  expect(summary).toHaveLength(4);
  expect(summary.every((s) => s.passed && s.count === 0)).toBe(true);
});
