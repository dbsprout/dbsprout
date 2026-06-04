import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import * as endpoints from "../../api/endpoints";
import { ApiError } from "../../api/client";
import { ExportPanel } from "./ExportPanel";

afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SPEC = { version: "1", tables: [{ table_name: "users", row_count: 10 }] };
const SCHEMA = {
  table_count: 2,
  dialect: "sqlite",
  source: "sample:ecommerce",
  tables: [
    { name: "users", primary_key: [], columns: [], foreign_keys: [] },
    { name: "orders", primary_key: [], columns: [], foreign_keys: [] },
  ],
};

/** Stub fetch so GET /api/spec + GET /api/schema both resolve (panel is ready). */
function stubReadyApi() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/spec")) return jsonResponse(SPEC);
      if (url.endsWith("/api/schema")) return jsonResponse(SCHEMA);
      return jsonResponse({});
    }),
  );
}

test("renders a format select with SQL/CSV/JSON/Parquet and a table multi-select", async () => {
  stubReadyApi();
  renderWithClient(<ExportPanel />);

  const fmt = screen.getByLabelText(/export format/i);
  expect(fmt).toBeInTheDocument();
  for (const f of ["sql", "csv", "json", "parquet"]) {
    expect(screen.getByRole("option", { name: f })).toBeInTheDocument();
  }
  await waitFor(() => expect(screen.getByLabelText(/export tables/i)).toBeInTheDocument());
  await waitFor(() =>
    expect(screen.getByRole("option", { name: "orders" })).toBeInTheDocument(),
  );
});

test("the Export button is disabled until a spec is available", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.endsWith("/api/spec")) {
        return jsonResponse({ detail: { code: "NO_SPEC", message: "no spec" } }, 404);
      }
      return jsonResponse(SCHEMA);
    }),
  );
  renderWithClient(<ExportPanel />);
  await waitFor(() =>
    expect(screen.getByRole("button", { name: /export/i })).toBeDisabled(),
  );
});

test("changing the format and clicking Export calls exportData with format + selected tables", async () => {
  stubReadyApi();
  const spy = vi.spyOn(endpoints, "exportData").mockResolvedValue(undefined);
  renderWithClient(<ExportPanel />);

  await waitFor(() =>
    expect(screen.getByRole("button", { name: /export/i })).toBeEnabled(),
  );

  fireEvent.change(screen.getByLabelText(/export format/i), { target: { value: "csv" } });
  const tableSelect = screen.getByLabelText(/export tables/i) as HTMLSelectElement;
  // Select just "users".
  for (const opt of Array.from(tableSelect.options)) {
    opt.selected = opt.value === "users";
  }
  fireEvent.change(tableSelect);

  fireEvent.click(screen.getByRole("button", { name: /export/i }));

  await waitFor(() => expect(spy).toHaveBeenCalledWith("csv", ["users"]));
});

test("with no table selected it exports all tables (undefined subset)", async () => {
  stubReadyApi();
  const spy = vi.spyOn(endpoints, "exportData").mockResolvedValue(undefined);
  renderWithClient(<ExportPanel />);

  await waitFor(() =>
    expect(screen.getByRole("button", { name: /export/i })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: /export/i }));

  await waitFor(() => expect(spy).toHaveBeenCalledWith("sql", undefined));
});

test("a failed export surfaces a typed error message", async () => {
  stubReadyApi();
  vi.spyOn(endpoints, "exportData").mockRejectedValue(
    new ApiError(409, { code: "NO_RUN", message: "no run on the workspace" }),
  );
  renderWithClient(<ExportPanel />);

  await waitFor(() =>
    expect(screen.getByRole("button", { name: /export/i })).toBeEnabled(),
  );
  fireEvent.click(screen.getByRole("button", { name: /export/i }));

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/no run on the workspace/i),
  );
});
