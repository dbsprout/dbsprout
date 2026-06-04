import { fireEvent, render, screen, within } from "@testing-library/react";
import { expect, test, vi } from "vitest";
import type { CorrelationRule } from "../../api/types";
import { CorrelationsEditor } from "./CorrelationsEditor";

const COLUMNS = ["city", "state", "zip"];
const TABLES = ["users", "orders"];

test("lists the existing correlation rules", () => {
  const rules: CorrelationRule[] = [
    { columns: ["city", "state"], lookup_table: null, strategy: "lookup" },
  ];
  render(
    <CorrelationsEditor
      table="users"
      columns={COLUMNS}
      tables={TABLES}
      rules={rules}
      onSave={vi.fn()}
    />,
  );
  const list = screen.getByLabelText(/correlation rules for users/i);
  expect(within(list).getByText(/city, state/i)).toBeInTheDocument();
});

test("adds a rule and calls onSave with the new list", () => {
  const onSave = vi.fn();
  render(
    <CorrelationsEditor
      table="users"
      columns={COLUMNS}
      tables={TABLES}
      rules={[]}
      onSave={onSave}
    />,
  );
  // Select two columns in the multi-select.
  const select = screen.getByLabelText(/columns for new correlation/i) as HTMLSelectElement;
  for (const opt of Array.from(select.options)) {
    if (opt.value === "city" || opt.value === "state") opt.selected = true;
  }
  fireEvent.change(select);
  fireEvent.click(screen.getByRole("button", { name: /add correlation/i }));

  expect(onSave).toHaveBeenCalledTimes(1);
  const saved = onSave.mock.calls[0][0] as CorrelationRule[];
  expect(saved).toHaveLength(1);
  expect(saved[0].columns).toEqual(["city", "state"]);
  expect(saved[0].strategy).toBe("lookup");
});

test("includes the chosen lookup table when set", () => {
  const onSave = vi.fn();
  render(
    <CorrelationsEditor
      table="users"
      columns={COLUMNS}
      tables={TABLES}
      rules={[]}
      onSave={onSave}
    />,
  );
  const select = screen.getByLabelText(/columns for new correlation/i) as HTMLSelectElement;
  select.options[0].selected = true; // city
  fireEvent.change(select);
  fireEvent.change(screen.getByLabelText(/lookup table/i), { target: { value: "orders" } });
  fireEvent.click(screen.getByRole("button", { name: /add correlation/i }));

  const saved = onSave.mock.calls[0][0] as CorrelationRule[];
  expect(saved[0].lookup_table).toBe("orders");
});

test("does not add a rule when no columns are selected", () => {
  const onSave = vi.fn();
  render(
    <CorrelationsEditor
      table="users"
      columns={COLUMNS}
      tables={TABLES}
      rules={[]}
      onSave={onSave}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /add correlation/i }));
  expect(onSave).not.toHaveBeenCalled();
  expect(screen.getByRole("alert")).toHaveTextContent(/select at least one column/i);
});

test("removes a rule and calls onSave with the shorter list", () => {
  const onSave = vi.fn();
  const rules: CorrelationRule[] = [
    { columns: ["city", "state"], lookup_table: null, strategy: "lookup" },
    { columns: ["zip"], lookup_table: null, strategy: "lookup" },
  ];
  render(
    <CorrelationsEditor
      table="users"
      columns={COLUMNS}
      tables={TABLES}
      rules={rules}
      onSave={onSave}
    />,
  );
  fireEvent.click(screen.getByRole("button", { name: /remove correlation 1/i }));
  const saved = onSave.mock.calls[0][0] as CorrelationRule[];
  expect(saved).toHaveLength(1);
  expect(saved[0].columns).toEqual(["zip"]);
});
