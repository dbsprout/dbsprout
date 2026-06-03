import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import { ColumnInspector } from "./ColumnInspector";
import type { GeneratorConfig } from "../../api/types";

const cfg: GeneratorConfig = {
  provider: "mimesis",
  method: "email",
  params: { domain: "x.com" },
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

/** A numeric cfg pre-populated with the advanced fields, for round-trip tests. */
const numericCfg: GeneratorConfig = {
  ...cfg,
  method: "random_int",
  params: {},
  distribution: "normal",
  distribution_params: { mean: 50, std: 10 },
  min_value: 0,
  max_value: 100,
  enum_values: ["a", "b"],
};

function save() {
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
}

// ── existing behaviour (regression) ──────────────────────────────────────────

test("shows the focused column's provider/method", () => {
  render(<ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />);
  expect(screen.getByText("mimesis/email")).toBeInTheDocument();
});

test("saving an edited null % persists the new nullable_rate", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/null %/i), { target: { value: "0.25" } });
  save();
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ nullable_rate: 0.25 }));
});

test("toggling unique persists it", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.click(screen.getByLabelText(/^unique$/i));
  save();
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ unique: true }));
});

test("editing params JSON persists the parsed object", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/^params$/i), {
    target: { value: '{"domain":"y.io"}' },
  });
  save();
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ params: { domain: "y.io" } }));
});

test("invalid params JSON blocks save and shows an error", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/^params$/i), { target: { value: "{ not json" } });
  save();
  expect(onSave).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent(/invalid json/i);
});

test("re-roll calls onSave with the current cfg", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.click(screen.getByRole("button", { name: /re-roll/i }));
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ params: { domain: "x.com" } }));
});

test("re-keys local state when the focused column changes", () => {
  const onSave = vi.fn();
  const { rerender } = render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/null %/i), { target: { value: "0.9" } });
  const next: GeneratorConfig = { ...cfg, method: "name", nullable_rate: 0.1, params: {} };
  rerender(<ColumnInspector column="age" cfg={next} onSave={onSave} />);
  expect(screen.getByLabelText(/null %/i)).toHaveValue(0.1);
});

// ── distribution select ──────────────────────────────────────────────────────

test("distribution select defaults to the cfg's distribution", () => {
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={() => undefined} />);
  expect(screen.getByLabelText(/distribution/i)).toHaveValue("normal");
});

test("distribution select defaults to none when null", () => {
  render(<ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />);
  expect(screen.getByLabelText(/distribution/i)).toHaveValue("");
});

test("offers the curated distribution options", () => {
  render(<ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />);
  const select = screen.getByLabelText(/distribution/i);
  for (const name of ["uniform", "normal", "exponential", "zipf", "poisson", "lognormal"]) {
    expect(within(select).getByRole("option", { name })).toBeInTheDocument();
  }
});

test("selecting a distribution persists it", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/distribution/i), { target: { value: "zipf" } });
  save();
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ distribution: "zipf" }));
});

test("choosing the none distribution option persists null", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/distribution/i), { target: { value: "" } });
  save();
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ distribution: null }));
});

// ── distribution params (numeric kv) ─────────────────────────────────────────

test("renders existing distribution params as editable rows", () => {
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={() => undefined} />);
  expect(screen.getByDisplayValue("mean")).toBeInTheDocument();
  expect(screen.getByDisplayValue("std")).toBeInTheDocument();
});

test("editing a distribution param persists the parsed numeric record", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  const valueInput = screen.getByLabelText(/value for mean/i);
  fireEvent.change(valueInput, { target: { value: "42" } });
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ distribution_params: { mean: 42, std: 10 } }),
  );
});

test("adding a new distribution param row persists it", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.click(screen.getByRole("button", { name: /add param/i }));
  fireEvent.change(screen.getByLabelText(/param name 0/i), { target: { value: "s" } });
  fireEvent.change(screen.getByLabelText(/value for s/i), { target: { value: "1.5" } });
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ distribution_params: { s: 1.5 } }),
  );
});

test("blank-key distribution param rows are dropped on save", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.click(screen.getByRole("button", { name: /add param/i }));
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ distribution_params: {} }),
  );
});

test("a non-numeric distribution param value blocks save with an error", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/value for mean/i), { target: { value: "abc" } });
  save();
  expect(onSave).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent(/numeric/i);
});

test("removing a distribution param row drops it on save", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  fireEvent.click(screen.getByRole("button", { name: /remove param std/i }));
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ distribution_params: { mean: 50 } }),
  );
});

// ── min / max ────────────────────────────────────────────────────────────────

test("min/max default from cfg and persist as numbers", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  expect(screen.getByLabelText(/^min$/i)).toHaveValue("0");
  expect(screen.getByLabelText(/^max$/i)).toHaveValue("100");
  fireEvent.change(screen.getByLabelText(/^max$/i), { target: { value: "200" } });
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ min_value: 0, max_value: 200 }),
  );
});

test("blank min/max persist as null", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/^min$/i), { target: { value: "" } });
  fireEvent.change(screen.getByLabelText(/^max$/i), { target: { value: "" } });
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ min_value: null, max_value: null }),
  );
});

test("min greater than max blocks save with an inline error", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/^min$/i), { target: { value: "300" } });
  save();
  expect(onSave).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent(/min.*max/i);
});

test("a non-numeric min blocks save with an error", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/^min$/i), { target: { value: "x" } });
  save();
  expect(onSave).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent(/numeric/i);
});

// ── enum values list editor ──────────────────────────────────────────────────

test("renders existing enum values and persists edits", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="status" cfg={numericCfg} onSave={onSave} />);
  const enumInput = screen.getByLabelText(/enum values/i);
  expect(enumInput).toHaveValue("a\nb");
  fireEvent.change(enumInput, { target: { value: "a\nb\nc" } });
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ enum_values: ["a", "b", "c"] }),
  );
});

test("an all-blank enum editor persists null", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="status" cfg={numericCfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/enum values/i), { target: { value: "  \n  " } });
  save();
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ enum_values: null }));
});

test("enum editor accepts comma-separated entries and trims them", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="status" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/enum values/i), { target: { value: " red , green ,blue" } });
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ enum_values: ["red", "green", "blue"] }),
  );
});

// ── combined round-trip ──────────────────────────────────────────────────────

test("save preserves untouched advanced fields", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({
      distribution: "normal",
      distribution_params: { mean: 50, std: 10 },
      min_value: 0,
      max_value: 100,
      enum_values: ["a", "b"],
    }),
  );
});
