import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { SchemaTree } from "./SchemaTree";

afterEach(() => vi.unstubAllGlobals());

function stub(body: unknown, status = 200) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })),
  );
}

test("renders tables and columns from GET /api/schema", async () => {
  stub({
    table_count: 1,
    dialect: "sqlite",
    source: "sample:ecommerce",
    tables: [
      {
        name: "users",
        primary_key: ["id"],
        columns: [{ name: "id", type: "INTEGER", nullable: false, unique: true, autoincrement: true, default: null, max_length: null }],
        foreign_keys: [],
      },
    ],
  });
  renderWithClient(<SchemaTree />);
  await waitFor(() => expect(screen.getByText("users")).toBeInTheDocument());
  expect(screen.getByText(/id/)).toBeInTheDocument();
});

test("shows an empty state when no schema is loaded (404)", async () => {
  stub({ detail: { code: "NO_SCHEMA", message: "No schema loaded" } }, 404);
  renderWithClient(<SchemaTree />);
  await waitFor(() => expect(screen.getByText(/no schema/i)).toBeInTheDocument());
});
