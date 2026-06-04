import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { PreviewTable } from "./PreviewTable";

afterEach(() => vi.unstubAllGlobals());

function stub(status: number, body: unknown) {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify(body), {
          status,
          headers: { "Content-Type": "application/json" },
        }),
    ),
  );
}

test("renders sample rows for the active table", async () => {
  stub(200, { table: "users", limit: 100, total: 1, rows: [{ id: 1, email: "a@b.c" }] });
  renderWithClient(<PreviewTable table="users" />);
  await waitFor(() => expect(screen.getByText("a@b.c")).toBeInTheDocument());
  expect(screen.getByLabelText(/preview of users/i)).toBeInTheDocument();
});

test("shows an empty hint when no generation has run (404)", async () => {
  stub(404, { detail: "No generation result available." });
  renderWithClient(<PreviewTable table="users" />);
  await waitFor(() => expect(screen.getByText(/no preview yet/i)).toBeInTheDocument());
});

test("shows an empty hint when the table has zero rows", async () => {
  stub(200, { table: "users", limit: 100, total: 0, rows: [] });
  renderWithClient(<PreviewTable table="users" />);
  await waitFor(() => expect(screen.getByText(/no preview yet/i)).toBeInTheDocument());
});
