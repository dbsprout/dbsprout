import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { useJobSocket } from "./useJobSocket";

/**
 * Minimal stand-in for the browser WebSocket. The newest instance is captured
 * on `MockWebSocket.last` so a test can drive the lifecycle synchronously
 * (`emitOpen` / `emitMessage` / `emitClose` / `emitError`).
 */
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

  emitOpen(): void {
    act(() => this.onopen?.());
  }
  emitMessage(frame: unknown): void {
    act(() => this.onmessage?.({ data: JSON.stringify(frame) }));
  }
  emitClose(): void {
    act(() => this.onclose?.());
  }
  emitError(): void {
    act(() => this.onerror?.());
  }
}

beforeEach(() => {
  MockWebSocket.instances = [];
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
});

afterEach(() => vi.unstubAllGlobals());

const eventFrame = (over: Record<string, unknown> = {}) => ({
  phase: "table_done",
  table: "users",
  tables_done: 1,
  tables_total: 3,
  rows_in_table: 100,
  total_rows: 100,
  message: null,
  ...over,
});

test("opens the socket for the job and surfaces per-table progress", () => {
  const { result } = renderHook(() => useJobSocket("j1"));

  expect(MockWebSocket.instances).toHaveLength(1);
  expect(MockWebSocket.last.url).toMatch(/\/ws\/jobs\/j1$/);

  MockWebSocket.last.emitOpen();
  MockWebSocket.last.emitMessage(eventFrame());

  expect(result.current.connected).toBe(true);
  expect(result.current.terminal).toBe(false);
  expect(result.current.progress).toMatchObject({
    table: "users",
    tablesDone: 1,
    tablesTotal: 3,
    totalRows: 100,
    status: null,
  });
});

test("marks terminal + status on the terminal frame and closes the socket", () => {
  const { result } = renderHook(() => useJobSocket("j1"));

  MockWebSocket.last.emitOpen();
  MockWebSocket.last.emitMessage(eventFrame());
  MockWebSocket.last.emitMessage({ phase: "terminal", status: "succeeded", error: null });

  expect(result.current.terminal).toBe(true);
  expect(result.current.progress?.status).toBe("succeeded");
  expect(result.current.progress?.error).toBeNull();
  expect(MockWebSocket.last.closed).toBe(true);
});

test("carries the failure message from a failed terminal frame", () => {
  const { result } = renderHook(() => useJobSocket("j1"));

  MockWebSocket.last.emitOpen();
  MockWebSocket.last.emitMessage({ phase: "terminal", status: "failed", error: "boom" });

  expect(result.current.progress?.status).toBe("failed");
  expect(result.current.progress?.error).toBe("boom");
});

test("surfaces a pre-terminal disconnect as connected=false / terminal=false (fallback signal)", () => {
  const { result } = renderHook(() => useJobSocket("j1"));

  MockWebSocket.last.emitOpen();
  expect(result.current.connected).toBe(true);

  MockWebSocket.last.emitClose();

  expect(result.current.connected).toBe(false);
  expect(result.current.terminal).toBe(false);
});

test("treats an error before the terminal frame as a disconnect", () => {
  const { result } = renderHook(() => useJobSocket("j1"));

  MockWebSocket.last.emitOpen();
  MockWebSocket.last.emitError();

  expect(result.current.connected).toBe(false);
  expect(result.current.terminal).toBe(false);
});

test("a close AFTER the terminal frame stays terminal (clean shutdown)", () => {
  const { result } = renderHook(() => useJobSocket("j1"));

  MockWebSocket.last.emitOpen();
  MockWebSocket.last.emitMessage({ phase: "terminal", status: "succeeded", error: null });
  MockWebSocket.last.emitClose();

  expect(result.current.terminal).toBe(true);
  expect(result.current.progress?.status).toBe("succeeded");
});

test("closes the socket on unmount (no leak)", () => {
  const { unmount } = renderHook(() => useJobSocket("j1"));
  const sock = MockWebSocket.last;

  unmount();

  expect(sock.closed).toBe(true);
});

test("opens a fresh socket when the jobId changes", () => {
  const { rerender } = renderHook(({ id }) => useJobSocket(id), {
    initialProps: { id: "j1" },
  });
  const first = MockWebSocket.last;

  rerender({ id: "j2" });

  expect(first.closed).toBe(true);
  expect(MockWebSocket.instances).toHaveLength(2);
  expect(MockWebSocket.last.url).toMatch(/\/ws\/jobs\/j2$/);
});

test("does not open a socket for an empty jobId", () => {
  renderHook(() => useJobSocket(""));
  expect(MockWebSocket.instances).toHaveLength(0);
});

test("degrades to poll-only when WebSocket is unavailable", () => {
  vi.stubGlobal("WebSocket", undefined);
  const { result } = renderHook(() => useJobSocket("j1"));

  expect(result.current.connected).toBe(false);
  expect(result.current.terminal).toBe(false);
  expect(result.current.progress).toBeNull();
});

test("ignores a malformed (non-JSON) frame without crashing", () => {
  const { result } = renderHook(() => useJobSocket("j1"));
  MockWebSocket.last.emitOpen();

  act(() => MockWebSocket.last.onmessage?.({ data: "not json{" }));

  expect(result.current.connected).toBe(true);
  expect(result.current.progress).toBeNull();
});
