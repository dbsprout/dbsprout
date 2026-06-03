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

// ─── P4-8: URL-paste → field auto-fill ───

function input(label: RegExp): HTMLInputElement {
  return screen.getByLabelText(label) as HTMLInputElement;
}

test("pasting a postgres URL fills the structured fields", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(urlInput(), {
    target: { value: "postgresql://admin:secret@db.host:6543/shop" },
  });
  expect(input(/^host/i).value).toBe("db.host");
  expect(input(/^port/i).value).toBe("6543");
  expect(input(/^user/i).value).toBe("admin");
  expect((screen.getByLabelText("Database") as HTMLInputElement).value).toBe("shop");
  expect((screen.getByLabelText(/database type/i) as HTMLSelectElement).value).toBe("postgresql");
});

test("pasting a URL with advanced params fills the Advanced section", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(urlInput(), {
    target: {
      value:
        "postgresql://admin:secret@db.host:5432/shop" +
        "?sslmode=require&options=-csearch_path%3Danalytics&connect_timeout=12" +
        "&application_name=dbsprout",
    },
  });
  expect((screen.getByLabelText(/ssl mode/i) as HTMLSelectElement).value).toBe("require");
  expect(input(/^schema/i).value).toBe("analytics");
  expect(input(/connect timeout/i).value).toBe("12");
  expect((screen.getByLabelText(/extra parameters/i) as HTMLTextAreaElement).value).toContain(
    "application_name=dbsprout",
  );
});

test("pasting a sqlite URL switches db-type and fills the file path", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(urlInput(), { target: { value: "sqlite:////tmp/seed.db" } });
  expect((screen.getByLabelText(/database type/i) as HTMLSelectElement).value).toBe("sqlite");
  expect(input(/file path/i).value).toBe("/tmp/seed.db");
});

test("the pasted URL stays verbatim in the URL field (round-trip safe)", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  const pasted = "postgresql://admin:secret@db.host:5432/shop?sslmode=require";
  fireEvent.change(urlInput(), { target: { value: pasted } });
  expect(urlInput().value).toBe(pasted);
});

// ─── end P4-8 ───

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

// ─── P2a-3 ───

function bodyOf(call: unknown): Record<string, unknown> {
  const init = (call as [string, RequestInit])[1];
  return JSON.parse(String(init.body)) as Record<string, unknown>;
}

test("filling the SSH bastion sends an ssh block on Connect", async () => {
  const fetchMock = stubOk();
  const onLoaded = vi.fn();
  renderWithClient(<ConnectForm onLoaded={onLoaded} />);
  fireEvent.change(screen.getByLabelText(/connection url/i), {
    target: { value: "postgresql://u:p@db.internal:5432/app" },
  });
  fireEvent.change(screen.getByLabelText(/ssh bastion host/i), {
    target: { value: "bastion.example.com" },
  });
  fireEvent.change(screen.getByLabelText(/ssh user/i), { target: { value: "deploy" } });
  fireEvent.change(screen.getByLabelText(/ssh key path/i), { target: { value: "/home/me/.ssh/id" } });
  fireEvent.click(screen.getByRole("button", { name: /^connect/i }));
  await waitFor(() => expect(onLoaded).toHaveBeenCalled());
  const connectCall = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/api/connect"));
  expect(connectCall).toBeDefined();
  expect(bodyOf(connectCall)).toMatchObject({
    ssh: { host: "bastion.example.com", user: "deploy", key_path: "/home/me/.ssh/id" },
  });
});

test("no ssh block is sent when the bastion host is left blank", async () => {
  const fetchMock = stubOk();
  const onLoaded = vi.fn();
  renderWithClient(<ConnectForm onLoaded={onLoaded} />);
  fireEvent.change(screen.getByLabelText(/connection url/i), {
    target: { value: "postgresql://u:p@db.internal:5432/app" },
  });
  fireEvent.click(screen.getByRole("button", { name: /^connect/i }));
  await waitFor(() => expect(onLoaded).toHaveBeenCalled());
  const connectCall = fetchMock.mock.calls.find(([u]) => String(u).endsWith("/api/connect"));
  expect(bodyOf(connectCall).ssh).toBeUndefined();
});

test("SSH fields are not shown for sqlite", () => {
  renderWithClient(<ConnectForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/database type/i), { target: { value: "sqlite" } });
  expect(screen.queryByLabelText(/ssh bastion host/i)).toBeNull();
});
