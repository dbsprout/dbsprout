import { describe, expect, test } from "vitest";
import type { GeneratorMethod } from "../../api/types";
import { filterMethodsByDtype, sqlTypeToDtype } from "./columnDtype";

describe("sqlTypeToDtype", () => {
  test.each([
    ["VARCHAR(255)", "VARCHAR"],
    ["varchar", "VARCHAR"],
    ["character varying(64)", "VARCHAR"],
    ["TEXT", "TEXT"],
    ["int", "INTEGER"],
    ["int4", "INTEGER"],
    ["integer", "INTEGER"],
    ["serial", "INTEGER"],
    ["int8", "BIGINT"],
    ["bigint", "BIGINT"],
    ["bigserial", "BIGINT"],
    ["smallint", "SMALLINT"],
    ["int2", "SMALLINT"],
    ["float8", "FLOAT"],
    ["double precision", "FLOAT"],
    ["real", "FLOAT"],
    ["numeric(10,2)", "DECIMAL"],
    ["decimal", "DECIMAL"],
    ["bool", "BOOLEAN"],
    ["boolean", "BOOLEAN"],
    ["date", "DATE"],
    ["datetime", "DATETIME"],
    ["timestamp", "TIMESTAMP"],
    ["timestamptz", "TIMESTAMP"],
    ["timestamp with time zone", "TIMESTAMP"],
    ["time", "TIME"],
    ["uuid", "UUID"],
    ["json", "JSON"],
    ["jsonb", "JSON"],
    ["bytea", "BINARY"],
    ["blob", "BINARY"],
    ["binary", "BINARY"],
  ])("maps %s -> %s", (raw, expected) => {
    expect(sqlTypeToDtype(raw)).toBe(expected);
  });

  test("normalizes surrounding whitespace and case", () => {
    expect(sqlTypeToDtype("  VarChar(12) ")).toBe("VARCHAR");
  });

  test.each([["", null], ["   ", null], ["geography", null], ["weird_custom_type", null]])(
    "returns null for unmappable %s",
    (raw, expected) => {
      expect(sqlTypeToDtype(raw)).toBe(expected);
    },
  );
});

const M = (provider: string, method: string, dtypes: string[]): GeneratorMethod => ({
  provider,
  method,
  description: "",
  example: "",
  dtypes,
  params: [],
});

const methods: GeneratorMethod[] = [
  M("mimesis", "email", ["VARCHAR", "TEXT"]),
  M("mimesis", "name", ["VARCHAR", "TEXT"]),
  M("builtin", "increment", ["INTEGER", "BIGINT"]),
  M("numpy", "random_int", ["INTEGER"]),
];

describe("filterMethodsByDtype", () => {
  test("keeps only methods whose dtypes include the column dtype", () => {
    const out = filterMethodsByDtype(methods, "INTEGER", false);
    expect(out.map((m) => m.method)).toEqual(["increment", "random_int"]);
  });

  test("showAll bypasses the filter entirely", () => {
    const out = filterMethodsByDtype(methods, "INTEGER", true);
    expect(out).toHaveLength(methods.length);
  });

  test("a null dtype disables filtering (shows all)", () => {
    const out = filterMethodsByDtype(methods, null, false);
    expect(out).toHaveLength(methods.length);
  });

  test("a dtype no method supports yields an empty list (escape hatch needed)", () => {
    const out = filterMethodsByDtype(methods, "UUID", false);
    expect(out).toEqual([]);
  });
});
