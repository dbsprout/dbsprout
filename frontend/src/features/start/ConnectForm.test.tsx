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

function urlInput(): HTMLInputElement {
  return screen.getByLabelText(/connection url/i) as HTMLInputElement;
}

test("selecting an SSL mode folds sslmode into the URL preview", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/ssl mode/i), { target: { value: "require" } });
  expect(urlInput().value).toContain("sslmode=require");
});

test("entering a schema folds search_path into the URL preview", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/^schema/i), { target: { value: "analytics" } });
  expect(urlInput().value).toContain("options=-csearch_path%3Danalytics");
});

test("entering a connect timeout folds connect_timeout into the URL preview", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/connect timeout/i), { target: { value: "12" } });
  expect(urlInput().value).toContain("connect_timeout=12");
});

test("free-form params textarea folds key=value pairs into the URL preview", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/extra parameters/i), {
    target: { value: "application_name=dbsprout\nkeepalives=1" },
  });
  const value = urlInput().value;
  expect(value).toContain("application_name=dbsprout");
  expect(value).toContain("keepalives=1");
});

test("advanced fields are not shown for sqlite", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/database type/i), { target: { value: "sqlite" } });
  expect(screen.queryByLabelText(/ssl mode/i)).toBeNull();
});
