import { act, fireEvent, screen, waitFor } from "@testing-library/react";
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

// ─── P4-6: cancel affordance ───
// A router fetch stub: /api/jobs/{id}/cancel records the call + flips the job to
// `cancelled`; bare /api/jobs/{id} polls return `running` until cancel, then
// `cancelled`. The WS is left inert so the poll is the source of truth.
function cancelFetch() {
  const cancelCalls: string[] = [];
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/cancel")) {
      cancelCalls.push(url);
      return jsonResponse({ job_id: "j1", status: "cancelled" });
    }
    if (url.startsWith("/api/jobs/")) {
      const status = cancelCalls.length > 0 ? "cancelled" : "running";
      return jsonResponse({ ...base, status });
    }
    return jsonResponse({}, 500);
  });
  return { fetchMock, cancelCalls };
}

test("shows a Cancel button while the job runs and clicking it calls cancelJob", async () => {
  const { fetchMock, cancelCalls } = cancelFetch();
  vi.stubGlobal("fetch", fetchMock);

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);

  const cancelBtn = await screen.findByRole("button", { name: /cancel/i });
  fireEvent.click(cancelBtn);

  await waitFor(() => expect(cancelCalls.length).toBeGreaterThan(0));
  expect(cancelCalls[0]).toBe("/api/jobs/j1/cancel");
});

test("the poll settles on cancelled; the console shows it and stops polling", async () => {
  const { fetchMock } = cancelFetch();
  vi.stubGlobal("fetch", fetchMock);

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);

  fireEvent.click(await screen.findByRole("button", { name: /cancel/i }));

  await waitFor(() => expect(screen.getByText(/run cancelled\./i)).toBeInTheDocument());
  // The Cancel button is gone once the run is terminal.
  expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();

  // No further /api/jobs polls after the terminal (cancelled) state.
  const pollCalls = () =>
    fetchMock.mock.calls.filter(
      (c) => String(c[0]).startsWith("/api/jobs/") && !String(c[0]).endsWith("/cancel"),
    ).length;
  const settled = pollCalls();
  await new Promise((r) => setTimeout(r, 80));
  expect(pollCalls()).toBe(settled);
});

test("no Cancel button once the job has already succeeded", async () => {
  const m = staged({ status: "succeeded", finished_at: "2026-06-02T00:00:05+00:00" }, 1);
  vi.stubGlobal("fetch", m);

  renderWithClient(<ProgressConsole jobId="j1" pollMs={20} />);

  await waitFor(() => expect(screen.getByText(/succeeded/i)).toBeInTheDocument());
  expect(screen.queryByRole("button", { name: /cancel/i })).not.toBeInTheDocument();
});
