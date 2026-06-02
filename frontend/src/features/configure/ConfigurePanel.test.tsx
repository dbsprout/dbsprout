import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { ConfigurePanel } from "./ConfigurePanel";

const SPEC = {
  version: "1.0",
  global_seed: 42,
  schema_hash: "",
  model_used: null,
  created_at: null,
  tables: [
    {
      table_name: "users",
      row_count: 10,
      derived: [],
      correlations: [],
      cardinality: null,
      columns: {
        email: {
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
        },
      },
    },
    {
      table_name: "orders",
      row_count: 5,
      derived: [],
      correlations: [],
      cardinality: null,
      columns: {
        total: {
          provider: "numpy",
          method: "uniform",
          params: {},
          distribution: null,
          distribution_params: {},
          min_value: null,
          max_value: null,
          enum_values: null,
          format_pattern: null,
          unique: false,
          nullable_rate: 0,
          vectorized: true,
        },
      },
    },
  ],
};

const GENERATORS = {
  providers: ["mimesis", "numpy"],
  methods: [
    { provider: "mimesis", method: "email", description: "", example: "", dtypes: [], params: [] },
    { provider: "mimesis", method: "name", description: "", example: "", dtypes: [], params: [] },
    { provider: "numpy", method: "uniform", description: "", example: "", dtypes: [], params: [] },
  ],
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function router() {
  return vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/spec") return json(SPEC);
    if (url === "/api/generators") return json(GENERATORS);
    if (url.startsWith("/api/preview/users"))
      return json({ table: "users", limit: 100, total: 1, rows: [{ email: "a@b.c" }] });
    if (url.startsWith("/api/preview/orders"))
      return json({ table: "orders", limit: 100, total: 1, rows: [{ total: 9.5 }] });
    if (url.startsWith("/api/spec/tables/"))
      return json({ provider: "mimesis", method: "name", params: {} });
    return json({}, 404);
  });
}

afterEach(() => vi.unstubAllGlobals());

test("renders the grid for the first table with its preview sample", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(<ConfigurePanel />);
  await waitFor(() =>
    expect(screen.getByLabelText(/generator for email/i)).toBeInTheDocument(),
  );
  expect(await screen.findAllByText("a@b.c")).not.toHaveLength(0);
});

test("grid shows the live sample value from preview", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(<ConfigurePanel />);
  const grid = await screen.findByLabelText(/columns of users/i);
  await waitFor(() => expect(within(grid).getByText("a@b.c")).toBeInTheDocument());
});

test("changing the table picker drives the grid", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(<ConfigurePanel />);
  const picker = await screen.findByLabelText(/configure table/i);
  fireEvent.change(picker, { target: { value: "orders" } });
  await waitFor(() =>
    expect(screen.getByLabelText(/generator for total/i)).toBeInTheDocument(),
  );
});

test("changing a generator PUTs the column and refreshes the preview", async () => {
  const m = router();
  vi.stubGlobal("fetch", m);
  renderWithClient(<ConfigurePanel />);
  const select = await screen.findByLabelText(/generator for email/i);
  fireEvent.change(select, { target: { value: "mimesis/name" } });
  await waitFor(() =>
    expect(
      m.mock.calls.some(
        ([u, i]) =>
          String(u) === "/api/spec/tables/users/columns/email" &&
          (i as RequestInit | undefined)?.method === "PUT",
      ),
    ).toBe(true),
  );
  await waitFor(() =>
    expect(
      m.mock.calls.filter(([u]) => String(u).startsWith("/api/preview/users")).length,
    ).toBeGreaterThan(1),
  );
});

test("inspector opens on column click and saves edits", async () => {
  const m = router();
  vi.stubGlobal("fetch", m);
  renderWithClient(<ConfigurePanel />);
  fireEvent.click(await screen.findByRole("button", { name: /inspect email/i }));
  expect(await screen.findByLabelText(/inspector for email/i)).toBeInTheDocument();
  fireEvent.click(screen.getByRole("button", { name: /^save$/i }));
  await waitFor(() =>
    expect(
      m.mock.calls.some(
        ([u, i]) =>
          String(u) === "/api/spec/tables/users/columns/email" &&
          (i as RequestInit | undefined)?.method === "PUT",
      ),
    ).toBe(true),
  );
});

test("empty state when no schema loaded (409)", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      String(input) === "/api/spec"
        ? json({ detail: { code: "NO_SCHEMA", message: "no schema" } }, 409)
        : json({}, 200),
    ),
  );
  renderWithClient(<ConfigurePanel />);
  await waitFor(() => expect(screen.getByText(/no schema loaded/i)).toBeInTheDocument());
});
