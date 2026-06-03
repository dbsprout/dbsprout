// ─── P2b-2 ─── ConfigurePanel integration for the advanced-packs editors.
import { fireEvent, screen, waitFor } from "@testing-library/react";
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
        city: makeCfg("mimesis", "city"),
        state: makeCfg("mimesis", "state"),
      },
    },
  ],
};

function makeCfg(provider: string, method: string) {
  return {
    provider,
    method,
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
  };
}

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

function router() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/spec") return json(SPEC);
    if (url === "/api/generators") return json({ providers: [], methods: [] });
    if (url.startsWith("/api/preview/"))
      return json({ table: "users", limit: 100, total: 0, rows: [] });
    if (url.endsWith("/advanced"))
      return json({ table_name: "users", correlations: [], derived: [] });
    if (url.startsWith("/api/spec/tables/")) return json(makeCfg("mimesis", "city"));
    return json({}, 404);
  });
}

afterEach(() => vi.unstubAllGlobals());

test("renders the correlations + derived editors for the active table", async () => {
  vi.stubGlobal("fetch", router());
  renderWithClient(<ConfigurePanel />);
  expect(await screen.findByLabelText(/correlations for users/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/derived for users/i)).toBeInTheDocument();
});

test("saving a correlation PUTs /advanced and invalidates the spec query", async () => {
  const m = router();
  vi.stubGlobal("fetch", m);
  renderWithClient(<ConfigurePanel />);

  const select = (await screen.findByLabelText(
    /columns for new correlation/i,
  )) as HTMLSelectElement;
  select.options[0].selected = true; // city
  fireEvent.change(select);
  fireEvent.click(screen.getByRole("button", { name: /add correlation/i }));

  await waitFor(() =>
    expect(
      m.mock.calls.some(
        ([u, i]) =>
          String(u) === "/api/spec/tables/users/advanced" &&
          (i as RequestInit | undefined)?.method === "PUT",
      ),
    ).toBe(true),
  );
  // The spec query is refetched (a second GET /api/spec after the PUT).
  await waitFor(() =>
    expect(m.mock.calls.filter(([u]) => String(u) === "/api/spec").length).toBeGreaterThan(1),
  );
});

test("saving a derived column PUTs /advanced", async () => {
  const m = router();
  vi.stubGlobal("fetch", m);
  renderWithClient(<ConfigurePanel />);

  fireEvent.change(await screen.findByLabelText(/derived column name/i), {
    target: { value: "label" },
  });
  fireEvent.change(screen.getByLabelText(/derived expression/i), {
    target: { value: "city" },
  });
  fireEvent.click(screen.getByRole("button", { name: /add derived/i }));

  await waitFor(() =>
    expect(
      m.mock.calls.some(
        ([u, i]) =>
          String(u) === "/api/spec/tables/users/advanced" &&
          (i as RequestInit | undefined)?.method === "PUT",
      ),
    ).toBe(true),
  );
});
