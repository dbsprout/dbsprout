import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { StartPanel } from "./StartPanel";

afterEach(() => vi.unstubAllGlobals());

// ═══ P5-3 ═══ — the panel now reads the loaded schema to decide collapse; the
// default empty schema keeps the picker (and the existing tab / SavedConnections
// tests) visible, while a non-empty `schema` drives the collapsed-summary tests.
const EMPTY_SCHEMA = { table_count: 0, dialect: null, source: null, tables: [] };

function stubFetch(
  connections: { name: string; url: string }[] = [],
  schema: unknown = EMPTY_SCHEMA,
) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      let body: unknown = { samples: [] };
      if (url.endsWith("/api/connections")) body = { connections };
      else if (url.endsWith("/api/schema")) body = schema;
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}

function loadedSchema(tableCount: number) {
  return {
    table_count: tableCount,
    dialect: "sqlite",
    source: "sample:ecommerce",
    tables: Array.from({ length: tableCount }, (_, i) => ({
      name: `t${i}`,
      primary_key: [],
      columns: [],
      foreign_keys: [],
    })),
  };
}

test("renders source tabs and switches to Paste", () => {
  stubFetch();
  renderWithClient(<StartPanel onLoaded={() => undefined} />);
  for (const name of [/live database/i, /upload/i, /paste/i, /sample/i]) {
    expect(screen.getByRole("tab", { name })).toBeInTheDocument();
  }
  fireEvent.click(screen.getByRole("tab", { name: /paste/i }));
  expect(screen.getByLabelText(/paste schema/i)).toBeInTheDocument();
});

test("mounts SavedConnections on the Live database tab (P2a-2)", async () => {
  stubFetch([]);
  renderWithClient(<StartPanel onLoaded={() => undefined} />);
  // The connect (Live database) tab is the default — SavedConnections is here.
  const region = await screen.findByRole("region", { name: /saved connections/i });
  await waitFor(() =>
    expect(within(region).getByText(/no saved connections/i)).toBeInTheDocument(),
  );
});

test("hides SavedConnections on non-Live tabs (P5-2)", async () => {
  stubFetch([]);
  renderWithClient(<StartPanel onLoaded={() => undefined} />);
  // Present on the default connect tab…
  await screen.findByRole("region", { name: /saved connections/i });
  // …and gone once another source tab is active.
  fireEvent.click(screen.getByRole("tab", { name: /paste/i }));
  expect(
    screen.queryByRole("region", { name: /saved connections/i }),
  ).not.toBeInTheDocument();
});

test("loading a saved connection surfaces the URL on the Live tab", async () => {
  stubFetch([{ name: "prod", url: "postgresql://u:@db/app" }]);
  renderWithClient(<StartPanel onLoaded={() => undefined} />);

  // SavedConnections lives on the default Live database tab (P5-2 scoping).
  const region = await screen.findByRole("region", { name: /saved connections/i });
  await waitFor(() => expect(within(region).getByText("prod")).toBeInTheDocument());
  fireEvent.click(within(region).getByRole("button", { name: /^load$/i }));

  expect(screen.getByRole("status")).toHaveTextContent("postgresql://u:@db/app");
  expect(screen.getByRole("tab", { name: /live database/i })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

// ═══ P5-3 ═══ — once a schema is loaded the panel collapses to a compact
// summary instead of leaving the full picker open forever.
test("collapses to a compact summary when a schema is loaded (P5-3)", async () => {
  stubFetch([], loadedSchema(3));
  renderWithClient(<StartPanel onLoaded={() => undefined} />);

  const status = await screen.findByRole("status");
  expect(status).toHaveTextContent(/schema loaded/i);
  expect(status).toHaveTextContent("3");
  // The source tabs are gone in the collapsed state.
  expect(screen.queryByRole("tab", { name: /paste/i })).not.toBeInTheDocument();
});

test('"Change source" re-opens the picker (P5-3)', async () => {
  stubFetch([], loadedSchema(3));
  renderWithClient(<StartPanel onLoaded={() => undefined} />);

  const changeBtn = await screen.findByRole("button", { name: /change source/i });
  fireEvent.click(changeBtn);

  // The source tabs are back once the user chooses to change the source.
  expect(screen.getByRole("tab", { name: /paste/i })).toBeInTheDocument();
  expect(screen.getByRole("tab", { name: /live database/i })).toBeInTheDocument();
});
