import { describe, expect, test } from "vitest";
import { drillReason, violationHelp } from "./violationHelp";

describe("violationHelp", () => {
  test("pk_uniqueness maps to the duplicate-key remedy", () => {
    const h = violationHelp("pk_uniqueness");
    expect(h.what).toMatch(/share the same/i);
    expect(h.fix).toMatch(/re-generate/i);
    expect(h.fix).toMatch(/composite primary keys/i);
  });

  test("a bare 'unique' check also maps to the duplicate-key remedy", () => {
    const h = violationHelp("unique");
    expect(h.what).toMatch(/unique/i);
    expect(h.fix).toMatch(/re-generate|row count/i);
  });

  test("fk_satisfaction maps to the orphaned-foreign-key remedy", () => {
    const h = violationHelp("fk_satisfaction");
    expect(h.what).toMatch(/parent row|orphan/i);
    expect(h.fix).toMatch(/foreign key|parent table/i);
  });

  test("not_null maps to the NOT NULL remedy", () => {
    const h = violationHelp("not_null");
    expect(h.what).toMatch(/null/i);
    expect(h.fix).toMatch(/nullable rate/i);
  });

  test("check maps to the CHECK-constraint remedy", () => {
    const h = violationHelp("check");
    expect(h.what).toMatch(/check constraint/i);
    expect(h.fix).toMatch(/min\/max|allowed values/i);
  });

  test("classification is case-insensitive", () => {
    expect(violationHelp("FK_SATISFACTION")).toEqual(violationHelp("fk_satisfaction"));
    expect(violationHelp("Not_Null")).toEqual(violationHelp("not_null"));
  });

  test("an unknown check falls back to generic guidance", () => {
    const h = violationHelp("mystery_check");
    expect(h.what).toMatch(/failed/i);
    expect(h.fix).toMatch(/re-generate/i);
  });
});

describe("drillReason", () => {
  test("returns a short non-empty reason per known class", () => {
    for (const check of ["pk_uniqueness", "unique", "fk_satisfaction", "not_null", "check"]) {
      const r = drillReason(check);
      expect(r.length).toBeGreaterThan(0);
      expect(r.length).toBeLessThan(120);
    }
  });

  test("duplicate-key reason mentions re-generating", () => {
    expect(drillReason("pk_uniqueness")).toMatch(/re-generate|seed/i);
  });

  test("fk reason mentions the foreign key", () => {
    expect(drillReason("fk_satisfaction")).toMatch(/foreign key|parent/i);
  });

  test("unknown check still returns a usable fallback reason", () => {
    expect(drillReason("mystery_check")).toMatch(/integrity|re-generate/i);
  });
});
