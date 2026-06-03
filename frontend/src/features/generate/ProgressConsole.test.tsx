import { act, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { ProgressConsole } from "./ProgressConsole";

// ─── P4-5 ─── A stand-in WebSocket the tests drive synchronously. The newest
// instance is captured on `.last`. WS tests install it; pure-poll tests leave
// `WebSocket` undefined so the console runs poll-only (its disconnect fallback).
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static get last(): MockWebSocket {
    return MockWebSocket.instances[MockWebSocket.instances.length - 1];
  }
  url: string;
  closed = false;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }
  close(): void {
    this.closed = true;
  }
  open(): void {
    act(() => this.onopen?.());
  }
  send(frame: unknown): void {
    act(() => this.onmessage?.({ data: JSON.stringify(frame) }));
  }
  drop(): void {
    act(() => this.onclose?.());
  }
}

beforeEach(() => {
  MockWebSocket.instances = [];
  // Inert by default: the socket is created but the test must drive it. Poll-only
  // tests leave it un-driven so the hook stays connected=false → poll is the
  // source of truth (its disconnect fallback). WS tests call open()/send().
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
});

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const base = {
  id: "j1",
  kind: "generate",
  engine: "heuristic",
  seed: 42,
  started_at: "2026-06-02T00:00:00+00:00",
  finished_at: null,
  error: null,
};

// A fetch stub that yields `running` for the first `runningCalls` calls, then `final`.
function staged(final: Record<string, unknown>, runningCalls: number) {
  let n = 0;
  return vi.fn(async () => {
    n += 1;
    if (n <= runningCalls) return jsonResponse({ ...base, status: "running" });
    return jsonResponse({ ...base, ...final });
  });
}

test("polls the job and renders the running status with engine + seed", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse({ ...base, status: "running" }));
  vi.stubGlobal("fetch", m);

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);

  await waitFor(() => expect(screen.getByText(/running/i)).toBeInTheDocument());
  expect(screen.getByText(/heuristic/)).toBeInTheDocument();
  expect(screen.getByText(/42/)).toBeInTheDocument();
  expect(String(m.mock.calls[0][0])).toBe("/api/jobs/j1");
});

test("stops polling once the job succeeds", async () => {
  const m = staged({ status: "succeeded", finished_at: "2026-06-02T00:00:05+00:00" }, 1);
  vi.stubGlobal("fetch", m);

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);

  await waitFor(() => expect(screen.getByText(/succeeded/i)).toBeInTheDocument());

  const settled = m.mock.calls.length;
  await new Promise((r) => setTimeout(r, 80));
  // No further /api/jobs polls after the terminal frame.
  expect(m.mock.calls.length).toBe(settled);
});

test("renders the typed error and stops polling when the job fails", async () => {
  const m = staged({ status: "failed", error: "engine 'spec' needs a LoRA adapter" }, 1);
  vi.stubGlobal("fetch", m);

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/needs a LoRA adapter/i),
  );

  const settled = m.mock.calls.length;
  await new Promise((r) => setTimeout(r, 80));
  expect(m.mock.calls.length).toBe(settled);
});

test("surfaces a transport error when the poll itself fails", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({ detail: { code: "NOT_FOUND", message: "unknown job 'j1'" } }, 404),
    ),
  );

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/unknown job/i),
  );
});

test("invokes onSucceeded when the job reaches succeeded", async () => {
  const m = staged({ status: "succeeded" }, 1);
  vi.stubGlobal("fetch", m);
  const onSucceeded = vi.fn();

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} onSucceeded={onSucceeded} />);

  await waitFor(() => expect(onSucceeded).toHaveBeenCalledTimes(1));
});

// ─── P4-5: live WebSocket progress ───

const wsEvent = (over: Record<string, unknown> = {}) => ({
  phase: "table_done",
  table: "orders",
  tables_done: 2,
  tables_total: 5,
  rows_in_table: 250,
  total_rows: 600,
  message: null,
  ...over,
});

test("renders live per-table progress from the WebSocket", async () => {
  // Poll yields a bare `running` status; the live per-table line comes from WS.
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ...base, status: "running" })));

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);
  await waitFor(() => expect(screen.getByText(/running/i)).toBeInTheDocument());

  MockWebSocket.last.open();
  MockWebSocket.last.send(wsEvent());

  await waitFor(() => expect(screen.getByText(/orders/)).toBeInTheDocument());
  // Per-table counters surfaced from the live frame.
  expect(screen.getByText(/2\s*\/\s*5/)).toBeInTheDocument();
  expect(screen.getByText(/600/)).toBeInTheDocument();
});

test("stops cleanly and fires onSucceeded on the WS terminal frame", async () => {
  // Poll never returns terminal on its own; the WS terminal frame ends the run.
  const m = vi.fn(async () => jsonResponse({ ...base, status: "running" }));
  vi.stubGlobal("fetch", m);
  const onSucceeded = vi.fn();

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} onSucceeded={onSucceeded} />);
  await waitFor(() => expect(screen.getByText(/running/i)).toBeInTheDocument());

  MockWebSocket.last.open();
  MockWebSocket.last.send(wsEvent());
  MockWebSocket.last.send({ phase: "terminal", status: "succeeded", error: null });

  await waitFor(() => expect(onSucceeded).toHaveBeenCalledTimes(1));
  // The terminal frame closes the socket and stops polling.
  expect(MockWebSocket.last.closed).toBe(true);
  const settled = m.mock.calls.length;
  await new Promise((r) => setTimeout(r, 80));
  expect(m.mock.calls.length).toBe(settled);
});

test("renders the WS terminal error when the socket reports a failure", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ...base, status: "running" })));

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);
  await waitFor(() => expect(screen.getByText(/running/i)).toBeInTheDocument());

  MockWebSocket.last.open();
  MockWebSocket.last.send({ phase: "terminal", status: "failed", error: "boom: bad spec" });

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/boom: bad spec/i),
  );
});

test("falls back to polling when the socket drops before terminal", async () => {
  // After the socket drops mid-run, polling carries the job to its terminal state.
  const m = staged({ status: "succeeded", finished_at: "2026-06-02T00:00:05+00:00" }, 1);
  vi.stubGlobal("fetch", m);

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);
  await waitFor(() => expect(screen.getByText(/running/i)).toBeInTheDocument());

  MockWebSocket.last.open();
  MockWebSocket.last.drop(); // disconnect before any terminal frame

  // Poll fallback still drives the console to a terminal status.
  await waitFor(() => expect(screen.getByText(/succeeded/i)).toBeInTheDocument());
});
