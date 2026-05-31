import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { ConnectForm } from "./ConnectForm";

afterEach(() => vi.unstubAllGlobals());

function stubOk() {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/api/connect/test")) {
      return new Response(JSON.stringify({ ok: true, dialect: "sqlite", server_version: "3.47", table_count: 2, latency_ms: 4 }), { status: 200, headers: { "Content-Type": "application/json" } });
    }
    return new Response(JSON.stringify({ source: "db: sqlite", table_count: 2, tables: ["a", "b"], dialect: "sqlite" }), { status: 200, headers: { "Content-Type": "application/json" } });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

test("Test Connection shows probe result", async () => {
  stubOk();
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/connection url/i), { target: { value: "sqlite:////tmp/x.db" } });
  fireEvent.click(screen.getByRole("button", { name: /test connection/i }));
  await waitFor(() => expect(screen.getByText(/connected/i)).toBeInTheDocument());
  expect(screen.getByText(/2 tables/i)).toBeInTheDocument();
});

test("Connect introspects and calls onLoaded", async () => {
  const fetchMock = stubOk();
  const onLoaded = vi.fn();
  renderWithClient(<ConnectForm onLoaded={onLoaded} />);
  fireEvent.change(screen.getByLabelText(/connection url/i), { target: { value: "sqlite:////tmp/x.db" } });
  fireEvent.click(screen.getByRole("button", { name: /^connect/i }));
  await waitFor(() => expect(onLoaded).toHaveBeenCalled());
  expect(fetchMock.mock.calls.some(([u]) => String(u).endsWith("/api/connect"))).toBe(true);
});
