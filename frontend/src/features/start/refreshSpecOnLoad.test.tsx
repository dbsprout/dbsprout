import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { SelectionProvider } from "../../app/SelectionProvider";
import { ConfigurePanel } from "../configure/ConfigurePanel";
import { SamplePicker } from "./SamplePicker";

// ─── P5-12 ───
// End-to-end reproduction of the stuck-session bug: an app that opens with NO
// schema leaves the `spec` query in its 409 error state. Loading a sample must
// invalidate `spec` (and preview) — not just `schema` — so Configure refetches
// and renders the grid instead of staying on "No schema loaded".

afterEach(() => vi.unstubAllGlobals());

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
  ],
};

const GENERATORS = {
  providers: ["mimesis"],
  methods: [
    { provider: "mimesis", method: "email", description: "", example: "", dtypes: ["VARCHAR"], params: [] },
  ],
};

const SCHEMA = {
  table_count: 1,
  dialect: "sqlite",
  source: "sample:ecommerce",
  tables: [
    {
      name: "users",
      primary_key: ["email"],
      foreign_keys: [],
      columns: [
        { name: "email", type: "VARCHAR(255)", nullable: false, unique: true, autoincrement: false, default: null, max_length: 255 },
      ],
    },
  ],
};

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } });
}

/** Render both panels under ONE QueryClient so the loader's invalidation reaches Configure. */
function renderShared() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <SelectionProvider>
        <ConfigurePanel />
        <SamplePicker onLoaded={() => undefined} />
      </SelectionProvider>
    </QueryClientProvider>,
  );
}

test("loading a sample un-sticks Configure from its initial 'No schema loaded' 409 state", async () => {
  // Stateful spec endpoint: 409 until a sample is loaded, 200 after.
  let schemaLoaded = false;
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/spec") {
      return schemaLoaded ? json(SPEC) : json({ detail: { code: "NO_SCHEMA", message: "no schema" } }, 409);
    }
    if (url === "/api/schema") {
      return schemaLoaded ? json(SCHEMA) : json({ detail: { code: "NO_SCHEMA", message: "no schema" } }, 409);
    }
    if (url === "/api/generators") return json(GENERATORS);
    if (url === "/api/samples") {
      return json({ samples: [{ name: "ecommerce", title: "E-commerce", description: "demo", dialect: "sqlite", table_count: 1 }] });
    }
    if (url === "/api/schema/sample") {
      schemaLoaded = true;
      return json({ source: "sample:ecommerce", table_count: 1, tables: ["users"], dialect: "sqlite" });
    }
    if (url.startsWith("/api/preview/")) return json({ table: "users", limit: 100, total: 1, rows: [{ email: "a@b.c" }] });
    return json({}, 404);
  });
  vi.stubGlobal("fetch", fetchMock);

  renderShared();

  // Initially stuck: spec 409 → Configure shows the empty state.
  await waitFor(() => expect(screen.getByText(/no schema loaded/i)).toBeInTheDocument());

  // Load the sample (same session, no page refresh).
  fireEvent.click(await screen.findByRole("button", { name: /E-commerce/i }));

  // The fix: spec is invalidated → refetched (now 200) → Configure renders the grid.
  await waitFor(() => expect(screen.getByLabelText(/generator for email/i)).toBeInTheDocument());
  expect(screen.queryByText(/no schema loaded/i)).not.toBeInTheDocument();
});
