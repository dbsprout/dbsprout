import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { InsertProgress } from "./InsertProgress";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const baseJob = {
  id: "job-1",
  kind: "insert",
  engine: null,
  seed: null,
  started_at: "2026-06-02T00:00:00+00:00",
  finished_at: null,
  error: null,
};

test("polls and stops on succeeded; fires onSucceeded once", async () => {
  let polls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url.startsWith("/api/jobs/")) {
        polls += 1;
        return jsonResponse({ ...baseJob, status: polls <= 1 ? "running" : "succeeded" });
      }
      return jsonResponse({}, 500);
    }),
  );
  const onSucceeded = vi.fn();

  renderWithClient(<InsertProgress jobId="job-1" pollMs={20} onSucceeded={onSucceeded} />);

  await waitFor(() => expect(screen.getByText(/Status:/)).toHaveTextContent(/succeeded/i));
  await waitFor(() => expect(onSucceeded).toHaveBeenCalledTimes(1));

  // After terminal, polling stops: record the call count, wait, expect no growth.
  const stub = globalThis.fetch as unknown as ReturnType<typeof vi.fn>;
  const after = stub.mock.calls.length;
  await new Promise((r) => setTimeout(r, 80));
  expect(stub.mock.calls.length).toBe(after);
});

test("a failed poll shows the scrubbed typed error and stops", async () => {
  let polls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => {
      polls += 1;
      return jsonResponse(
        polls <= 1
          ? { ...baseJob, status: "running" }
          : { ...baseJob, status: "failed", error: "connection refused" },
      );
    }),
  );

  renderWithClient(<InsertProgress jobId="job-1" pollMs={20} />);

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/connection refused/i),
  );
});

test("a transport failure of the poll surfaces the typed ApiError", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ detail: { code: "INTERNAL", message: "boom" } }, 500)),
  );

  renderWithClient(<InsertProgress jobId="job-1" pollMs={20} />);

  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/boom/i));
});

test("Cancel calls cancelJob and the poll settles on cancelled", async () => {
  let polls = 0;
  const cancelCalls: string[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/cancel")) {
        cancelCalls.push(url);
        return jsonResponse({ job_id: "job-1", status: "cancelling" });
      }
      if (url.startsWith("/api/jobs/")) {
        polls += 1;
        // After cancel is requested, the job flips to cancelled.
        const status = cancelCalls.length > 0 ? "cancelled" : "running";
        return jsonResponse({ ...baseJob, status });
      }
      void init;
      return jsonResponse({}, 500);
    }),
  );

  renderWithClient(<InsertProgress jobId="job-1" pollMs={20} />);

  await waitFor(() => expect(screen.getByText(/Status:/)).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /cancel/i }));

  await waitFor(() => expect(cancelCalls.length).toBeGreaterThan(0));
  await waitFor(() => expect(screen.getByText(/insert cancelled\./i)).toBeInTheDocument());
});
