import { fireEvent, render, screen } from "@testing-library/react";
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

test("shows the focused column's provider/method", () => {
  render(<ColumnInspector column="email" cfg={cfg} onSave={() => undefined} />);
  expect(screen.getByText("mimesis/email")).toBeInTheDocument();
});

test("saving an edited null % persists the new nullable_rate", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/null %/i), { target: { value: "0.25" } });
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ nullable_rate: 0.25 }));
});

test("toggling unique persists it", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.click(screen.getByLabelText(/unique/i));
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ unique: true }));
});

test("editing params JSON persists the parsed object", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/params/i), {
    target: { value: '{"domain":"y.io"}' },
  });
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
  expect(onSave).toHaveBeenCalledWith(expect.objectContaining({ params: { domain: "y.io" } }));
});

test("invalid params JSON blocks save and shows an error", () => {
  const onSave = vi.fn();
  render(<ColumnInspector column="email" cfg={cfg} onSave={onSave} />);
  fireEvent.change(screen.getByLabelText(/params/i), { target: { value: "{ not json" } });
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
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
  const { rerender } = render(
    <ColumnInspector column="email" cfg={cfg} onSave={onSave} />,
  );
  fireEvent.change(screen.getByLabelText(/null %/i), { target: { value: "0.9" } });
  const next: GeneratorConfig = { ...cfg, method: "name", nullable_rate: 0.1, params: {} };
  rerender(<ColumnInspector column="age" cfg={next} onSave={onSave} />);
  expect(screen.getByLabelText(/null %/i)).toHaveValue(0.1);
});
