import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { expect, test } from "vitest";
import { SelectionProvider, useSelection } from "./SelectionProvider";

function wrapper({ children }: { children: ReactNode }) {
  return <SelectionProvider>{children}</SelectionProvider>;
}

test("defaults to a null selection", () => {
  const { result } = renderHook(() => useSelection(), { wrapper });
  expect(result.current.selection).toBeNull();
});

test("setSelection stores the focus target", () => {
  const { result } = renderHook(() => useSelection(), { wrapper });
  act(() => result.current.setSelection({ table: "users", column: "email" }));
  expect(result.current.selection).toEqual({ table: "users", column: "email" });
});

test("setSelection tolerates a null column (table-level violation)", () => {
  const { result } = renderHook(() => useSelection(), { wrapper });
  act(() => result.current.setSelection({ table: "orders", column: null }));
  expect(result.current.selection).toEqual({ table: "orders", column: null });
});

// ─── P5-7: an optional cross-panel reason rides along with the target ───
test("setSelection round-trips an optional reason", () => {
  const { result } = renderHook(() => useSelection(), { wrapper });
  act(() =>
    result.current.setSelection({
      table: "users",
      column: "id",
      reason: "duplicate key value — re-generate",
    }),
  );
  expect(result.current.selection).toEqual({
    table: "users",
    column: "id",
    reason: "duplicate key value — re-generate",
  });
});

test("clearSelection resets to null", () => {
  const { result } = renderHook(() => useSelection(), { wrapper });
  act(() => result.current.setSelection({ table: "users", column: "email" }));
  act(() => result.current.clearSelection());
  expect(result.current.selection).toBeNull();
});

test("useSelection throws outside a provider", () => {
  expect(() => renderHook(() => useSelection())).toThrow(/SelectionProvider/);
});
