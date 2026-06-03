import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { ResultSummary } from "./ResultSummary";

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
    { table_name: "orders", row_count: 250, columns: {}, derived: [], correlations: [], cardinality: null },
  ],
};

// ─── P4-4 ─── the real result envelope returned by GET /api/jobs/{id}/result.
const JOB_RESULT = {
  job_id: "job-1",
  total_rows: 312,
  total_tables: 2,
  total_duration_ms: 1234,
  tables: [
    { table_name: "users", row_count: 92, duration_ms: 410 },
    { table_name: "orders", row_count: 220, duration_ms: 824 },
  ],
};

// ── spec-fallback (no jobId): unchanged behaviour ──

test("lists each table with its generated row count (spec fallback)", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(SPEC)));

  renderWithClient(<ResultSummary />);

  await waitFor(() => expect(screen.getByText(/users/)).toBeInTheDocument());
  expect(screen.getByText(/users/).textContent).toMatch(/100/);
  expect(screen.getByText(/orders/).textContent).toMatch(/250/);
});

test("shows the total row count across tables (spec fallback)", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(SPEC)));

  renderWithClient(<ResultSummary />);

  await waitFor(() => expect(screen.getByText(/350/)).toBeInTheDocument());
});

test("shows an empty-state line when the spec cannot be read", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({ detail: { code: "NO_SCHEMA", message: "no spec" } }, 409),
    ),
  );

  renderWithClient(<ResultSummary />);

  await waitFor(() => expect(screen.getByText(/no summary available/i)).toBeInTheDocument());
});

// ─── P4-4: with a jobId, read the REAL per-table counts + durations ───

test("with a jobId, renders actual generated per-table counts + duration", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url === "/api/jobs/job-1/result") return jsonResponse(JOB_RESULT);
    // /api/spec must NOT be the source when a jobId is supplied.
    return jsonResponse(SPEC);
  });
  vi.stubGlobal("fetch", fetchMock);

  renderWithClient(<ResultSummary jobId="job-1" />);

  // Real counts (92 / 220), not the spec approximation (100 / 250).
  await waitFor(() => expect(screen.getByText(/users/).textContent).toMatch(/92/));
  expect(screen.getByText(/orders/).textContent).toMatch(/220/);
  // Per-table duration is shown.
  expect(screen.getByText(/users/).textContent).toMatch(/410/);
  expect(screen.getByText(/orders/).textContent).toMatch(/824/);

  // Real total rows + total duration.
  await waitFor(() => expect(screen.getByText(/312/)).toBeInTheDocument());
  expect(screen.getByText(/1234|1,234/).textContent).toBeTruthy();

  // The result endpoint was hit; the spec endpoint was not consulted.
  const calledUrls = fetchMock.mock.calls.map((c) => String(c[0]));
  expect(calledUrls).toContain("/api/jobs/job-1/result");
  expect(calledUrls).not.toContain("/api/spec");
});

test("with a jobId, shows an empty-state line when the result cannot be read", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({ detail: { code: "NO_RESULT", message: "no result yet" } }, 409),
    ),
  );

  renderWithClient(<ResultSummary jobId="job-1" />);

  await waitFor(() => expect(screen.getByText(/no summary available/i)).toBeInTheDocument());
});
