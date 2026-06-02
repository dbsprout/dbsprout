import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { GenerateSurface } from "./GenerateSurface";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const SPEC = {
  version: "1.0",
  global_seed: 42,
  schema_hash: "",
  model_used: null,
  created_at: null,
  tables: [
    { table_name: "users", row_count: 100, columns: {}, derived: [], correlations: [], cardinality: null },
  ],
};

const job = {
  id: "job-1",
  kind: "generate",
  engine: "heuristic",
  seed: 42,
  started_at: "2026-06-02T00:00:00+00:00",
  finished_at: null,
  error: null,
};

/**
 * One router fetch stub: /api/generate → job id, /api/jobs/* → running then
 * succeeded, /api/spec → SPEC. Models the full slice end-to-end.
 */
function makeFetch() {
  let jobPolls = 0;
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/generate") return jsonResponse({ job_id: "job-1", seed: 42 });
    if (url.startsWith("/api/jobs/")) {
      jobPolls += 1;
      const status = jobPolls <= 1 ? "running" : "succeeded";
      return jsonResponse({ ...job, status });
    }
    if (url === "/api/spec") return jsonResponse(SPEC);
    return jsonResponse({ detail: { code: "X", message: "unexpected" } }, 500);
  });
}

test("start → job id → polling → success summary (happy path)", async () => {
  vi.stubGlobal("fetch", makeFetch());

  renderWithClient(<GenerateSurface pollMs={20} />);

  // Before a run, no console / summary is shown.
  expect(screen.queryByText(/Status:/)).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: /generate/i }));

  // Console appears and polls.
  await waitFor(() => expect(screen.getByText(/Status:/)).toBeInTheDocument());

  // On success the per-table summary renders.
  await waitFor(() => expect(screen.getByText(/Result summary/i)).toBeInTheDocument());
  expect(screen.getByText(/users/).textContent).toMatch(/100/);
});

test("error path: a failed run shows the typed error and no summary", async () => {
  let jobPolls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/generate") return jsonResponse({ job_id: "job-1", seed: 42 });
      if (url.startsWith("/api/jobs/")) {
        jobPolls += 1;
        return jobPolls <= 1
          ? jsonResponse({ ...job, status: "running" })
          : jsonResponse({ ...job, status: "failed", error: "no schema loaded" });
      }
      return jsonResponse(SPEC);
    }),
  );

  renderWithClient(<GenerateSurface pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /generate/i }));

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/no schema loaded/i),
  );
  expect(screen.queryByText(/Result summary/i)).not.toBeInTheDocument();
});
