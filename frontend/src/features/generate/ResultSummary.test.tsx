import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { ResultSummary } from "./ResultSummary";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SPEC = {
  version: "1.0",
  global_seed: 42,
  schema_hash: "",
  model_used: null,
  created_at: null,
  tables: [
    { table_name: "users", row_count: 100, columns: {}, derived: [], correlations: [], cardinality: null },
    { table_name: "orders", row_count: 250, columns: {}, derived: [], correlations: [], cardinality: null },
  ],
};

test("lists each table with its generated row count", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(SPEC)));

  renderWithClient(<ResultSummary />);

  await waitFor(() => expect(screen.getByText(/users/)).toBeInTheDocument());
  expect(screen.getByText(/users/).textContent).toMatch(/100/);
  expect(screen.getByText(/orders/).textContent).toMatch(/250/);
});

test("shows the total row count across tables", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(SPEC)));

  renderWithClient(<ResultSummary />);

  await waitFor(() => expect(screen.getByText(/350/)).toBeInTheDocument());
});

test("shows an empty-state line when the spec cannot be read", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({ detail: { code: "NO_SCHEMA", message: "no spec" } }, 409),
    ),
  );

  renderWithClient(<ResultSummary />);

  await waitFor(() => expect(screen.getByText(/no summary available/i)).toBeInTheDocument());
});
