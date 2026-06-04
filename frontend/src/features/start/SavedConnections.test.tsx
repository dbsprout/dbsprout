import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { SavedConnections } from "./SavedConnections";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("lists saved connections and emits the stored URL on Load", async () => {
  const fetchMock = vi.fn(async () =>
    jsonResponse({
      connections: [{ name: "prod", url: "postgresql://u:@db/app" }],
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  const onLoad = vi.fn();
  renderWithClient(<SavedConnections onLoad={onLoad} />);

  await waitFor(() => expect(screen.getByText("prod")).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /load/i }));
  expect(onLoad).toHaveBeenCalledWith("postgresql://u:@db/app");
});

test("shows an empty state when there are no saved connections", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ connections: [] })));
  renderWithClient(<SavedConnections onLoad={vi.fn()} />);
  await waitFor(() =>
    expect(screen.getByText(/no saved connections/i)).toBeInTheDocument(),
  );
});

test("saves the current name + URL via POST", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url.endsWith("/api/connections") && init?.method === "POST") {
      return jsonResponse({ name: "stage", url: "mysql://u:@db/app" });
    }
    return jsonResponse({ connections: [] });
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWithClient(<SavedConnections onLoad={vi.fn()} />);

  await waitFor(() =>
    expect(screen.getByText(/no saved connections/i)).toBeInTheDocument(),
  );
  fireEvent.change(screen.getByLabelText(/name/i), { target: { value: "stage" } });
  fireEvent.change(screen.getByLabelText(/connection url/i), {
    target: { value: "mysql://u:secret@db/app" },
  });
  fireEvent.click(screen.getByRole("button", { name: /save connection/i }));

  await waitFor(() =>
    expect(
      fetchMock.mock.calls.some(
        ([u, i]) => String(u).endsWith("/api/connections") && i?.method === "POST",
      ),
    ).toBe(true),
  );
});

test("deletes a saved connection via DELETE", async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "DELETE") {
      return jsonResponse({ deleted: true });
    }
    return jsonResponse({ connections: [{ name: "prod", url: "postgresql://u:@db/app" }] });
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWithClient(<SavedConnections onLoad={vi.fn()} />);

  await waitFor(() => expect(screen.getByText("prod")).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /delete/i }));

  await waitFor(() =>
    expect(
      fetchMock.mock.calls.some(
        ([u, i]) =>
          String(u).endsWith("/api/connections/prod") && i?.method === "DELETE",
      ),
    ).toBe(true),
  );
});

test("surfaces a typed-envelope error on delete failure", async () => {
  const fetchMock = vi.fn(async (_input: RequestInfo | URL, init?: RequestInit) => {
    if (init?.method === "DELETE") {
      return jsonResponse(
        { detail: { code: "NOT_FOUND", message: "No saved connection named 'prod'." } },
        404,
      );
    }
    return jsonResponse({ connections: [{ name: "prod", url: "postgresql://u:@db/app" }] });
  });
  vi.stubGlobal("fetch", fetchMock);
  renderWithClient(<SavedConnections onLoad={vi.fn()} />);

  await waitFor(() => expect(screen.getByText("prod")).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /delete/i }));
  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/no saved connection/i),
  );
});
