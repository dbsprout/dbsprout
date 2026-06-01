import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { SpecPanel } from "./SpecPanel";

afterEach(() => vi.unstubAllGlobals());

const SPEC = {
  version: "1.0",
  tables: [{ table_name: "users", row_count: 100, columns: {}, derived: [], correlations: [], cardinality: null }],
  global_seed: 42, schema_hash: "", model_used: null, created_at: null,
};

test("lists tables with their row counts", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify(SPEC), { status: 200, headers: { "Content-Type": "application/json" } })));
  renderWithClient(<SpecPanel />);
  await waitFor(() => expect(screen.getByText("users")).toBeInTheDocument());
  expect(screen.getByLabelText(/rows for users/i)).toHaveValue(100);
});

test("editing a row count PUTs the new value", async () => {
  const m = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    if (String(input) === "/api/spec" && (!init || init.method === undefined)) {
      return new Response(JSON.stringify(SPEC), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response(JSON.stringify({ table_name: "users", row_count: 500 }), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", m);
  renderWithClient(<SpecPanel />);
  const input = await screen.findByLabelText(/rows for users/i);
  fireEvent.change(input, { target: { value: "500" } });
  fireEvent.blur(input);
  await waitFor(() =>
    expect(m.mock.calls.some(([u, i]) => String(u) === "/api/spec/tables/users" && (i as RequestInit | undefined)?.method === "PUT")).toBe(true),
  );
});

test("empty state when no schema (409)", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ detail: { code: "NO_SCHEMA", message: "no schema" } }), { status: 409, headers: { "Content-Type": "application/json" } })));
  renderWithClient(<SpecPanel />);
  await waitFor(() => expect(screen.getByText(/no schema/i)).toBeInTheDocument());
});
