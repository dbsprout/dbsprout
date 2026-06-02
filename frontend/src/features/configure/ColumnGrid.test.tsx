import { fireEvent, render, screen } from "@testing-library/react";
import { expect, test, vi } from "vitest";
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
