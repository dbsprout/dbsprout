import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { useSelection, type SelectionTarget } from "../../app/SelectionProvider";
import { ConfigurePanel } from "./ConfigurePanel";

/** A sibling that drills a cross-panel selection, sharing the test's SelectionProvider. */
function Driller({ target }: { target: SelectionTarget }) {
  const { setSelection } = useSelection();
  return (
    <button type="button" onClick={() => setSelection(target)}>
      drill
    </button>
  );
}

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
    {
      provider: "mimesis",
      method: "email",
      description: "",
      example: "",
      dtypes: ["VARCHAR"],
      params: [],
    },
    {
      provider: "mimesis",
      method: "name",
      description: "",
      example: "",
      dtypes: ["VARCHAR"],
      params: [],
    },
    {
      provider: "numpy",
      method: "uniform",
      description: "",
      example: "",
      dtypes: ["INTEGER", "FLOAT"],
      params: [],
    },
  ],
};

// ─── P4-12: schema-tree carries the per-column SQL types that drive the dtype
// filter. `users.email` is VARCHAR → only VARCHAR generators are compatible;
// `orders.total` is FLOAT → only numeric generators are compatible. `users`
// also has a column with no schema entry to exercise the unfiltered fallback.
const SCHEMA = {
  table_count: 2,
  dialect: "sqlite",
  source: "test",
  tables: [
    {
      name: "users",
      primary_key: ["email"],
      foreign_keys: [],
      columns: [
        {
          name: "email",
          type: "VARCHAR(255)",
          nullable: false,
          unique: true,
          autoincrement: false,
          default: null,
          max_length: 255,
        },
      ],
    },
    {
      name: "orders",
      primary_key: [],
      foreign_keys: [],
      columns: [
        {
          name: "total",
          type: "FLOAT",
          nullable: false,
          unique: false,
          autoincrement: false,
          default: null,
          max_length: null,
        },
      ],
    },
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
    if (url === "/api/schema") return json(SCHEMA);
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

// ─── P4-7: cross-panel drill focuses the grid ───
test("a drilled selection switches the table and focuses + opens the inspector on the column", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(
    <>
      <Driller target={{ table: "orders", column: "total" }} />
      <ConfigurePanel />
    </>,
  );
  // Starts on the first table (users).
  const picker = await screen.findByLabelText(/configure table/i);
  expect(picker).toHaveValue("users");
  // Drill into orders.total.
  fireEvent.click(screen.getByRole("button", { name: "drill" }));
  await waitFor(() => expect(picker).toHaveValue("orders"));
  // The grid now shows the orders column and the inspector opens on it.
  expect(await screen.findByLabelText(/generator for total/i)).toBeInTheDocument();
  expect(await screen.findByLabelText(/inspector for total/i)).toBeInTheDocument();
  // The drilled row is highlighted.
  const row = screen.getByRole("button", { name: "inspect total" }).closest("[data-focused]");
  expect(row).toHaveAttribute("data-focused", "true");
});

test("a drill to a table with a null column selects the table without opening the inspector", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(
    <>
      <Driller target={{ table: "orders", column: null }} />
      <ConfigurePanel />
    </>,
  );
  const picker = await screen.findByLabelText(/configure table/i);
  fireEvent.click(screen.getByRole("button", { name: "drill" }));
  await waitFor(() => expect(picker).toHaveValue("orders"));
  // No column focused → no inspector and no highlighted row.
  expect(screen.queryByLabelText(/inspector for/i)).not.toBeInTheDocument();
  expect(document.querySelectorAll('[data-focused="true"]').length).toBe(0);
});

test("a drill to a column that no longer exists selects the table but is otherwise a no-op", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(
    <>
      <Driller target={{ table: "orders", column: "ghost_column" }} />
      <ConfigurePanel />
    </>,
  );
  const picker = await screen.findByLabelText(/configure table/i);
  fireEvent.click(screen.getByRole("button", { name: "drill" }));
  await waitFor(() => expect(picker).toHaveValue("orders"));
  // Grid still renders; no crash, no inspector, no highlight for the missing column.
  expect(await screen.findByLabelText(/generator for total/i)).toBeInTheDocument();
  expect(screen.queryByLabelText(/inspector for ghost_column/i)).not.toBeInTheDocument();
  expect(document.querySelectorAll('[data-focused="true"]').length).toBe(0);
});
// ─── end P4-7 ───

// ─── P4-12: ConfigurePanel feeds per-column SQL types into the dormant filter ───

/** Option labels of a `<select>`, in DOM order. */
function optionValues(select: HTMLElement): string[] {
  return Array.from(select.querySelectorAll("option")).map((o) => o.value);
}

test("grid generator dropdown filters to dtype-compatible generators for the column's SQL type", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(<ConfigurePanel />);
  // users.email is VARCHAR → only the VARCHAR generators are offered; the FLOAT
  // generator (numpy/uniform) is filtered out.
  const select = await screen.findByLabelText(/generator for email/i);
  await waitFor(() => expect(optionValues(select)).toContain("mimesis/name"));
  const values = optionValues(select);
  expect(values).toContain("mimesis/email");
  expect(values).toContain("mimesis/name");
  expect(values).not.toContain("numpy/uniform");
});

test("grid 'show all generators' escape hatch restores the filtered-out generators", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(<ConfigurePanel />);
  const select = await screen.findByLabelText(/generator for email/i);
  await waitFor(() => expect(optionValues(select)).toContain("mimesis/name"));
  // Ticking "show all generators" in the grid un-filters the dropdown.
  fireEvent.click(screen.getByLabelText(/show all generators/i));
  await waitFor(() => expect(optionValues(select)).toContain("numpy/uniform"));
});

test("inspector generator picker filters by the focused column's SQL type and 'show all' restores", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(<ConfigurePanel />);
  fireEvent.click(await screen.findByRole("button", { name: /inspect email/i }));
  const inspector = await screen.findByLabelText(/inspector for email/i);
  // The inspector's method picker is present (driven by methods + columnType) and
  // filtered to VARCHAR generators.
  const picker = await within(inspector).findByRole("combobox", { name: /generator method/i });
  await waitFor(() => expect(optionValues(picker)).toContain("mimesis/name"));
  expect(optionValues(picker)).not.toContain("numpy/uniform");
  // The inspector's own "show all generators" restores the filtered-out generator.
  fireEvent.click(within(inspector).getByLabelText(/show all generators/i));
  await waitFor(() => expect(optionValues(picker)).toContain("numpy/uniform"));
});

test("a column with no schema-tree entry stays unfiltered (show-all fallback)", async () => {
  // Schema omits orders.total's type entirely → columnTypes has no entry for it,
  // so the grid dropdown is unfiltered and offers every generator.
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/spec") return json(SPEC);
      if (url === "/api/generators") return json(GENERATORS);
      if (url === "/api/schema")
        return json({
          table_count: 1,
          dialect: "sqlite",
          source: "test",
          tables: [{ name: "orders", primary_key: [], foreign_keys: [], columns: [] }],
        });
      if (url.startsWith("/api/preview/")) return json({ table: "x", limit: 100, total: 0, rows: [] });
      return json({}, 404);
    }),
  );
  renderWithClient(<ConfigurePanel />);
  const picker = await screen.findByLabelText(/configure table/i);
  fireEvent.change(picker, { target: { value: "orders" } });
  const select = await screen.findByLabelText(/generator for total/i);
  // Unknown type → every generator stays selectable.
  await waitFor(() => expect(optionValues(select)).toContain("numpy/uniform"));
  expect(optionValues(select)).toContain("mimesis/email");
  expect(optionValues(select)).toContain("mimesis/name");
});
// ─── end P4-12 ───
