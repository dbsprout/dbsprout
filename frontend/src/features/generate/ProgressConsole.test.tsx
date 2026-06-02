import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { ProgressConsole } from "./ProgressConsole";

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
