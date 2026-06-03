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

// ── P4-3: dual-edit refresh (grid edits the SAME focused column) ──────────────

test("an external cfg change re-seeds a clean draft without reselection", () => {
  const { rerender } = render(
    <ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />,
  );
  expect(screen.getByLabelText(/null %/i)).toHaveValue(0);
  // Grid edits the SAME column → ConfigurePanel feeds a new cfg identity, same column key.
  const external: GeneratorConfig = { ...cfg, nullable_rate: 0.4, distribution: "zipf" };
  rerender(<ColumnInspector column="email" cfg={external} onSave={() => undefined} />);
  expect(screen.getByLabelText(/null %/i)).toHaveValue(0.4);
  expect(screen.getByLabelText(/distribution/i)).toHaveValue("zipf");
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("after a clean re-seed, save persists the latest cfg, not the stale one", () => {
  const onSave = vi.fn();
  const { rerender } = render(
    <ColumnInspector column="email" cfg={cfg} onSave={onSave} />,
  );
  const external: GeneratorConfig = { ...cfg, nullable_rate: 0.4 };
  rerender(<ColumnInspector column="email" cfg={external} onSave={onSave} />);
  save();
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ nullable_rate: 0.4 }));
});

test("an external cfg change does not clobber an in-progress unsaved edit", () => {
  const { rerender } = render(
    <ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />,
  );
  // User starts editing locally (draft is now dirty).
  fireEvent.change(screen.getByLabelText(/null %/i), { target: { value: "0.9" } });
  const external: GeneratorConfig = { ...cfg, nullable_rate: 0.4 };
  rerender(<ColumnInspector column="email" cfg={external} onSave={() => undefined} />);
  // The user's edit survives; a non-destructive notice appears.
  expect(screen.getByLabelText(/null %/i)).toHaveValue(0.9);
  expect(screen.getByRole("status")).toHaveTextContent(/chang/i);
});

test("no external change shows no notice and no refresh affordance", () => {
  render(<ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /refresh/i })).not.toBeInTheDocument();
});

test("the refresh affordance pulls the latest cfg into a dirty draft and dismisses the notice", () => {
  const { rerender } = render(
    <ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />,
  );
  fireEvent.change(screen.getByLabelText(/null %/i), { target: { value: "0.9" } });
  const external: GeneratorConfig = { ...cfg, nullable_rate: 0.4 };
  rerender(<ColumnInspector column="email" cfg={external} onSave={() => undefined} />);
  fireEvent.click(screen.getByRole("button", { name: /refresh/i }));
  expect(screen.getByLabelText(/null %/i)).toHaveValue(0.4);
  expect(screen.queryByRole("status")).not.toBeInTheDocument();
});

test("refresh re-seeds every advanced field from the latest cfg", () => {
  const onSave = vi.fn();
  const { rerender } = render(
    <ColumnInspector column="age" cfg={cfg} onSave={onSave} />,
  );
  fireEvent.change(screen.getByLabelText(/null %/i), { target: { value: "0.9" } });
  rerender(<ColumnInspector column="age" cfg={numericCfg} onSave={onSave} />);
  fireEvent.click(screen.getByRole("button", { name: /refresh/i }));
  expect(screen.getByLabelText(/^min$/i)).toHaveValue("0");
  expect(screen.getByLabelText(/^max$/i)).toHaveValue("100");
  expect(screen.getByLabelText(/enum values/i)).toHaveValue("a\nb");
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({
      nullable_rate: 0,
      distribution: "normal",
      distribution_params: { mean: 50, std: 10 },
      min_value: 0,
      max_value: 100,
      enum_values: ["a", "b"],
    }),
  );
});

// ─── P4-1: dtype-filtered generator picker ───
import type { GeneratorMethod } from "../../api/types";

const inspectorMethods: GeneratorMethod[] = [
  { provider: "mimesis", method: "email", description: "", example: "", dtypes: ["VARCHAR", "TEXT"], params: [] },
  { provider: "mimesis", method: "name", description: "", example: "", dtypes: ["VARCHAR", "TEXT"], params: [] },
  { provider: "builtin", method: "increment", description: "", example: "", dtypes: ["INTEGER"], params: [] },
];

function genOptionValues(): string[] {
  const select = screen.getByLabelText(/generator method/i) as HTMLSelectElement;
  return Array.from(select.options).map((o) => o.value);
}

test("no generator picker renders when methods are not supplied (backwards compatible)", () => {
  render(<ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />);
  expect(screen.queryByLabelText(/generator method/i)).not.toBeInTheDocument();
});

test("the generator picker filters to the column's dtype", () => {
  render(
    <ColumnInspector
      column="email"
      cfg={cfg}
      methods={inspectorMethods}
      columnType="VARCHAR(255)"
      onSave={() => undefined}
    />,
  );
  expect(genOptionValues()).toEqual(
    expect.arrayContaining(["mimesis/email", "mimesis/name"]),
  );
  expect(genOptionValues()).not.toContain("builtin/increment");
});

test("the show-all toggle exposes incompatible generators in the inspector", () => {
  render(
    <ColumnInspector
      column="email"
      cfg={cfg}
      methods={inspectorMethods}
      columnType="VARCHAR(255)"
      onSave={() => undefined}
    />,
  );
  expect(genOptionValues()).not.toContain("builtin/increment");
  fireEvent.click(screen.getByLabelText(/show all generators/i));
  expect(genOptionValues()).toContain("builtin/increment");
});

test("changing the generator persists provider/method via onSave", () => {
  const onSave = vi.fn();
  render(
    <ColumnInspector
      column="email"
      cfg={cfg}
      methods={inspectorMethods}
      columnType="VARCHAR(255)"
      onSave={onSave}
    />,
  );
  fireEvent.change(screen.getByLabelText(/generator method/i), {
    target: { value: "mimesis/name" },
  });
  save();
  expect(onSave).toHaveBeenCalledWith(
    expect.objectContaining({ provider: "mimesis", method: "name" }),
  );
});
// ─── end P4-1 ───
