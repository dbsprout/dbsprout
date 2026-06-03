import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { ColumnGrid } from "./ColumnGrid";
import type { GeneratorConfig, GeneratorMethod, TableSpec } from "../../api/types";

const emailCfg: GeneratorConfig = {
  provider: "mimesis",
  method: "email",
  params: {},
  distribution: null,
  distribution_params: {},
  min_value: null,
  max_value: null,
  enum_values: null,
  format_pattern: null,
  unique: false,
  nullable_rate: 0,
  vectorized: false,
};

const spec: TableSpec = {
  table_name: "users",
  row_count: 10,
  derived: [],
  correlations: [],
  cardinality: null,
  columns: {
    id: { ...emailCfg, method: "increment", unique: true },
    email: { ...emailCfg, params: { domain: "x.com" } },
  },
};

const methods: GeneratorMethod[] = [
  { provider: "mimesis", method: "email", description: "", example: "", dtypes: [], params: [] },
  { provider: "mimesis", method: "name", description: "", example: "", dtypes: [], params: [] },
  { provider: "mimesis", method: "increment", description: "", example: "", dtypes: [], params: [] },
];

test("renders a row per column with the current generator selected", () => {
  render(
    <ColumnGrid
      spec={spec}
      methods={methods}
      previewRow={{ id: 1, email: "a@b.c" }}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  expect(screen.getByText("email")).toBeInTheDocument();
  expect(screen.getByLabelText(/generator for email/i)).toHaveValue("mimesis/email");
  expect(screen.getByText("a@b.c")).toBeInTheDocument();
});

test("shows the params summary and a dash when empty", () => {
  render(
    <ColumnGrid
      spec={spec}
      methods={methods}
      previewRow={null}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  expect(screen.getByText(/"domain":"x.com"/)).toBeInTheDocument();
});

test("changing a generator calls onGeneratorChange with the spread config", () => {
  const onChange = vi.fn();
  render(
    <ColumnGrid
      spec={spec}
      methods={methods}
      previewRow={null}
      onSelect={() => undefined}
      onGeneratorChange={onChange}
    />,
  );
  fireEvent.change(screen.getByLabelText(/generator for email/i), {
    target: { value: "mimesis/name" },
  });
  expect(onChange).toHaveBeenCalledWith(
    "email",
    expect.objectContaining({ provider: "mimesis", method: "name", params: { domain: "x.com" } }),
  );
});

test("clicking a column row calls onSelect", () => {
  const onSelect = vi.fn();
  render(
    <ColumnGrid
      spec={spec}
      methods={methods}
      previewRow={null}
      onSelect={onSelect}
      onGeneratorChange={() => undefined}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /inspect email/i }));
  expect(onSelect).toHaveBeenCalledWith("email");
});

test("renders a unique badge for unique columns", () => {
  render(
    <ColumnGrid
      spec={spec}
      methods={methods}
      previewRow={null}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  expect(screen.getByText("unique")).toBeInTheDocument();
});

// ─── P4-1: dtype-filtered generator dropdown ───
const typedMethods: GeneratorMethod[] = [
  { provider: "mimesis", method: "email", description: "", example: "", dtypes: ["VARCHAR", "TEXT"], params: [] },
  { provider: "mimesis", method: "name", description: "", example: "", dtypes: ["VARCHAR", "TEXT"], params: [] },
  { provider: "builtin", method: "increment", description: "", example: "", dtypes: ["INTEGER", "BIGINT"], params: [] },
];

function optionValues(select: HTMLSelectElement): string[] {
  return Array.from(select.options).map((o) => o.value);
}

test("filters the generator options to the column's dtype when columnTypes is given", () => {
  render(
    <ColumnGrid
      spec={spec}
      methods={typedMethods}
      previewRow={null}
      columnTypes={{ id: "INTEGER", email: "VARCHAR(255)" }}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  const emailSelect = screen.getByLabelText(/generator for email/i) as HTMLSelectElement;
  // VARCHAR column: only the two string generators (plus its own current value).
  expect(optionValues(emailSelect)).toEqual(
    expect.arrayContaining(["mimesis/email", "mimesis/name"]),
  );
  expect(optionValues(emailSelect)).not.toContain("builtin/increment");
});

test("always keeps the current generator selectable even when it is incompatible", () => {
  // email column persisted with an INTEGER-only generator; VARCHAR dtype would hide it.
  const oddSpec: TableSpec = {
    ...spec,
    columns: { email: { ...emailCfg, provider: "builtin", method: "increment" } },
  };
  render(
    <ColumnGrid
      spec={oddSpec}
      methods={typedMethods}
      previewRow={null}
      columnTypes={{ email: "VARCHAR(255)" }}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  const emailSelect = screen.getByLabelText(/generator for email/i) as HTMLSelectElement;
  expect(emailSelect.value).toBe("builtin/increment");
  expect(optionValues(emailSelect)).toContain("builtin/increment");
});

test("the show-all toggle reveals incompatible generators (escape hatch)", () => {
  render(
    <ColumnGrid
      spec={spec}
      methods={typedMethods}
      previewRow={null}
      columnTypes={{ id: "INTEGER", email: "VARCHAR(255)" }}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  const emailSelect = screen.getByLabelText(/generator for email/i) as HTMLSelectElement;
  expect(optionValues(emailSelect)).not.toContain("builtin/increment");
  fireEvent.click(screen.getByLabelText(/show all generators/i));
  expect(optionValues(emailSelect)).toContain("builtin/increment");
});

test("without columnTypes the dropdown is unfiltered (backwards compatible)", () => {
  render(
    <ColumnGrid
      spec={spec}
      methods={typedMethods}
      previewRow={null}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  const emailSelect = screen.getByLabelText(/generator for email/i) as HTMLSelectElement;
  expect(optionValues(emailSelect)).toEqual(
    expect.arrayContaining(["mimesis/email", "mimesis/name", "builtin/increment"]),
  );
});

test("an unmappable column type disables filtering for that column", () => {
  render(
    <ColumnGrid
      spec={spec}
      methods={typedMethods}
      previewRow={null}
      columnTypes={{ email: "geography" }}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  const emailSelect = screen.getByLabelText(/generator for email/i) as HTMLSelectElement;
  expect(optionValues(emailSelect)).toContain("builtin/increment");
});
// ─── end P4-1 ───

// ─── P4-2: virtualization ───
afterEach(() => {
  vi.restoreAllMocks();
});

function wideSpec(columnCount: number): TableSpec {
  const columns: Record<string, GeneratorConfig> = {};
  for (let i = 0; i < columnCount; i += 1) {
    columns[`col_${i}`] = { ...emailCfg };
  }
  return { table_name: "wide", row_count: 10, derived: [], correlations: [], cardinality: null, columns };
}

test("virtualizes large column counts — only a windowed subset of rows is mounted", () => {
  render(
    <ColumnGrid
      spec={wideSpec(200)}
      methods={methods}
      previewRow={null}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  const rendered = screen.getAllByRole("button", { name: /^inspect col_/ });
  // A 480px viewport at ~40px/row windows to ~12 rows plus overscan — far fewer than 200.
  expect(rendered.length).toBeGreaterThan(0);
  expect(rendered.length).toBeLessThan(40);
  // The first column is within the initial window.
  expect(screen.getByRole("button", { name: "inspect col_0" })).toBeInTheDocument();
});

test("scrolling the virtualized body mounts later rows and unmounts earlier ones", () => {
  render(
    <ColumnGrid
      spec={wideSpec(200)}
      methods={methods}
      previewRow={null}
      onSelect={() => undefined}
      onGeneratorChange={() => undefined}
    />,
  );
  // The scroll container is the only `overflow:auto` ancestor of the rows.
  const grid = screen.getByLabelText("columns of wide");
  const scroller = grid.querySelector("tbody") as HTMLElement;
  expect(scroller).toBeTruthy();
  // Far-down rows are not mounted initially.
  expect(screen.queryByRole("button", { name: "inspect col_150" })).not.toBeInTheDocument();
  // Drive a scroll: offset is read from scrollTop, so set it and dispatch the event.
  Object.defineProperty(scroller, "scrollTop", { value: 150 * 40, configurable: true });
  fireEvent.scroll(scroller);
  expect(screen.getByRole("button", { name: "inspect col_150" })).toBeInTheDocument();
  // The very first row has scrolled out of the window and unmounted.
  expect(screen.queryByRole("button", { name: "inspect col_0" })).not.toBeInTheDocument();
});
// ─── end P4-2 ───
