import type { GeneratorMethod } from "../../api/types";
import { genLabel } from "./genLabel";

/** A `<select>` option: a "provider/method" value and its display label. */
export interface GeneratorOption {
  value: string;
  label: string;
}

/**
 * Normalize a raw SQL column type string (e.g. `VARCHAR(255)`, `int4`,
 * `timestamptz`) to a `ColumnType.name` — the UPPERCASE token the API uses in
 * `GeneratorMethod.dtypes` and the `?dtype=` filter (`dbsprout.schema.ColumnType`).
 *
 * Mirrors the server's `_resolve_dtype` intent: case-insensitive, but here we
 * additionally fold common dialect spellings (Postgres `int4`/`serial`,
 * `timestamptz`, `jsonb`, …) onto the canonical name. Anything unrecognized
 * returns `null`, which callers treat as "don't filter" (show all) — a safe,
 * non-blocking default rather than hiding every generator for an exotic type.
 */
export function sqlTypeToDtype(raw: string): string | null {
  // Strip parameters (`(255)`, `(10,2)`) and surrounding whitespace, then lowercase.
  const base = raw
    .replace(/\(.*$/, "")
    .trim()
    .toLowerCase();
  if (base === "") {
    return null;
  }
  return DTYPE_BY_SQL[base] ?? null;
}

/**
 * Filter generator methods to those compatible with `dtype`. A `null` dtype or
 * `showAll` bypasses the filter (returns the list unchanged), so the picker
 * always has an escape hatch and never strands a column with an exotic type.
 */
export function filterMethodsByDtype(
  methods: GeneratorMethod[],
  dtype: string | null,
  showAll: boolean,
): GeneratorMethod[] {
  if (showAll || dtype === null) {
    return methods;
  }
  return methods.filter((m) => m.dtypes.includes(dtype));
}

/**
 * Build the generator `<select>` options for a column: dtype-filtered method
 * labels plus the column's `current` value (always selectable, even when
 * filtered out or unknown), marked "(incompatible)" when it is hidden by the
 * active filter. Shared by `ColumnGrid` and `ColumnInspector` so both pickers
 * behave identically. `columnType` is the raw SQL type (may be `undefined`).
 */
export function generatorOptions(
  methods: GeneratorMethod[],
  columnType: string | undefined,
  showAll: boolean,
  current: string,
): GeneratorOption[] {
  const dtype = columnType ? sqlTypeToDtype(columnType) : null;
  const compatible = filterMethodsByDtype(methods, dtype, showAll).map((m) =>
    genLabel(m.provider, m.method),
  );
  const opts: GeneratorOption[] = compatible.map((value) => ({ value, label: value }));
  if (!compatible.includes(current)) {
    const incompatible = dtype !== null && !showAll;
    opts.unshift({ value: current, label: incompatible ? `${current} (incompatible)` : current });
  }
  return opts;
}

/**
 * Raw-SQL spelling → canonical `ColumnType.name`. Keys are lowercase and
 * parameter-free (the helper normalizes input the same way). Covers the common
 * Postgres / MySQL / SQLite / MSSQL aliases; the canonical names themselves are
 * included so a server-normalized value round-trips.
 */
const DTYPE_BY_SQL: Readonly<Record<string, string>> = {
  // INTEGER
  int: "INTEGER",
  int4: "INTEGER",
  integer: "INTEGER",
  serial: "INTEGER",
  serial4: "INTEGER",
  mediumint: "INTEGER",
  // BIGINT
  bigint: "BIGINT",
  int8: "BIGINT",
  bigserial: "BIGINT",
  serial8: "BIGINT",
  // SMALLINT
  smallint: "SMALLINT",
  int2: "SMALLINT",
  smallserial: "SMALLINT",
  tinyint: "SMALLINT",
  // FLOAT
  float: "FLOAT",
  float4: "FLOAT",
  float8: "FLOAT",
  real: "FLOAT",
  "double precision": "FLOAT",
  double: "FLOAT",
  // DECIMAL
  decimal: "DECIMAL",
  numeric: "DECIMAL",
  money: "DECIMAL",
  // BOOLEAN
  bool: "BOOLEAN",
  boolean: "BOOLEAN",
  bit: "BOOLEAN",
  // VARCHAR
  varchar: "VARCHAR",
  "character varying": "VARCHAR",
  "char varying": "VARCHAR",
  char: "VARCHAR",
  character: "VARCHAR",
  nvarchar: "VARCHAR",
  nchar: "VARCHAR",
  string: "VARCHAR",
  // TEXT
  text: "TEXT",
  longtext: "TEXT",
  mediumtext: "TEXT",
  ntext: "TEXT",
  clob: "TEXT",
  // DATE
  date: "DATE",
  // DATETIME
  datetime: "DATETIME",
  datetime2: "DATETIME",
  smalldatetime: "DATETIME",
  // TIMESTAMP
  timestamp: "TIMESTAMP",
  timestamptz: "TIMESTAMP",
  "timestamp with time zone": "TIMESTAMP",
  "timestamp without time zone": "TIMESTAMP",
  // TIME
  time: "TIME",
  timetz: "TIME",
  // UUID
  uuid: "UUID",
  uniqueidentifier: "UUID",
  // JSON
  json: "JSON",
  jsonb: "JSON",
  // BINARY
  binary: "BINARY",
  varbinary: "BINARY",
  bytea: "BINARY",
  blob: "BINARY",
  // ENUM
  enum: "ENUM",
  // ARRAY
  array: "ARRAY",
};
