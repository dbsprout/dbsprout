import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import type { DerivedColumn } from "../../api/types";
import { DerivedColumns } from "./DerivedColumns";

const COLUMNS = ["qty", "price", "total"];

test("lists existing derived columns", () => {
  const derived: DerivedColumn[] = [
    { column: "total", expression: "qty * price", depends_on: ["qty", "price"] },
  ];
  render(<DerivedColumns table="orders" columns={COLUMNS} derived={derived} onSave={vi.fn()} />);
  const list = screen.getByLabelText(/derived columns for orders/i);
  expect(within(list).getByText(/qty \* price/i)).toBeInTheDocument();
});

test("adds a derived column and calls onSave", () => {
  const onSave = vi.fn();
  render(<DerivedColumns table="orders" columns={COLUMNS} derived={[]} onSave={onSave} />);

  fireEvent.change(screen.getByLabelText(/derived column name/i), {
    target: { value: "total" },
  });
  fireEvent.change(screen.getByLabelText(/derived expression/i), {
    target: { value: "qty * price" },
  });
  const deps = screen.getByLabelText(/depends on/i) as HTMLSelectElement;
  for (const opt of Array.from(deps.options)) {
    if (opt.value === "qty" || opt.value === "price") opt.selected = true;
  }
  fireEvent.change(deps);
  fireEvent.click(screen.getByRole("button", { name: /add derived/i }));

  expect(onSave).toHaveBeenCalledTimes(1);
  const saved = onSave.mock.calls[0][0] as DerivedColumn[];
  expect(saved).toEqual([
    { column: "total", expression: "qty * price", depends_on: ["qty", "price"] },
  ]);
});

test("rejects an empty column name or expression", () => {
  const onSave = vi.fn();
  render(<DerivedColumns table="orders" columns={COLUMNS} derived={[]} onSave={onSave} />);
  fireEvent.click(screen.getByRole("button", { name: /add derived/i }));
  expect(onSave).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent(/name and expression/i);
});

test("removes a derived column", () => {
  const onSave = vi.fn();
  const derived: DerivedColumn[] = [
    { column: "total", expression: "qty * price", depends_on: ["qty"] },
    { column: "tax", expression: "total * 0.1", depends_on: ["total"] },
  ];
  render(<DerivedColumns table="orders" columns={COLUMNS} derived={derived} onSave={onSave} />);
  fireEvent.click(screen.getByRole("button", { name: /remove derived total/i }));
  const saved = onSave.mock.calls[0][0] as DerivedColumn[];
  expect(saved).toHaveLength(1);
  expect(saved[0].column).toBe("tax");
});
