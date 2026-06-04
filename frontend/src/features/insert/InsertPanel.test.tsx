import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { InsertPanel } from "./InsertPanel";

afterEach(() => vi.unstubAllGlobals());

const TOKEN = "PAYLOAD-b64.SIGNATURE-b64";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const PREVIEW = {
  target: "postgresql://***@localhost/app",
  dialect: "postgresql",
  scope: [
    { table: "users", row_count: 100 },
    { table: "orders", row_count: 250 },
  ],
  total_rows: 350,
  confirmation_token: TOKEN,
  warnings: ["table 'orders' references 'users'; ensure parents exist."],
};

const baseJob = {
  id: "job-1",
  kind: "insert",
  engine: null,
  seed: null,
  started_at: "2026-06-02T00:00:00+00:00",
  finished_at: null,
  error: null,
};

/** Router stub: preview → insert → job polls running then succeeded. */
function makeFetch(opts: { jobOutcome?: "succeeded" | "failed" | "cancelled" } = {}) {
  const outcome = opts.jobOutcome ?? "succeeded";
  let jobPolls = 0;
  const insertBodies: unknown[] = [];
  const fn = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input);
    if (url === "/api/insert/preview") return jsonResponse(PREVIEW);
    if (url === "/api/insert") {
      insertBodies.push(JSON.parse(init?.body as string));
      return jsonResponse({
        job_id: "job-1",
        scope: PREVIEW.scope,
        total_rows: PREVIEW.total_rows,
        writer: "PgCopyWriter",
        method: "auto",
        scope_warnings: [],
      });
    }
    if (url.endsWith("/cancel")) return jsonResponse({ job_id: "job-1", status: "cancelling" });
    if (url.startsWith("/api/jobs/")) {
      jobPolls += 1;
      return jsonResponse({ ...baseJob, status: jobPolls <= 1 ? "running" : outcome });
    }
    return jsonResponse({ detail: { code: "X", message: "unexpected" } }, 500);
  });
  return { fn, insertBodies };
}

test("preview → confirm → insert (happy path): scope, warnings, success summary", async () => {
  const { fn, insertBodies } = makeFetch();
  vi.stubGlobal("fetch", fn);

  renderWithClient(<InsertPanel pollMs={20} />);

  // No confirm gate before previewing.
  expect(screen.queryByRole("button", { name: /confirm & insert/i })).not.toBeInTheDocument();

  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));

  // Scope + per-table row counts + warnings render.
  await waitFor(() => expect(screen.getByText(/users — 100 rows/)).toBeInTheDocument());
  expect(screen.getByText(/orders — 250 rows/)).toBeInTheDocument();
  expect(screen.getByText(/350/)).toBeInTheDocument();
  expect(screen.getByText(/ensure parents exist/i)).toBeInTheDocument();

  // The confirm gate is now present.
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));

  // Job polls then a success summary renders.
  await waitFor(() => expect(screen.getByText(/Status:/)).toHaveTextContent(/succeeded/i));
  await waitFor(() => expect(screen.getByText(/Inserted/i)).toBeInTheDocument());
  expect(screen.getByText(/PgCopyWriter/)).toBeInTheDocument();

  // The token round-tripped into the insert request body.
  expect((insertBodies[0] as { confirmation_token: string }).confirmation_token).toBe(TOKEN);
});

test("the confirmation token is never rendered nor logged", async () => {
  const { fn } = makeFetch();
  vi.stubGlobal("fetch", fn);
  const logSpy = vi.spyOn(console, "log").mockImplementation(() => undefined);
  const infoSpy = vi.spyOn(console, "info").mockImplementation(() => undefined);
  const warnSpy = vi.spyOn(console, "warn").mockImplementation(() => undefined);
  const errSpy = vi.spyOn(console, "error").mockImplementation(() => undefined);

  const { container } = renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /confirm & insert/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));
  await waitFor(() => expect(screen.getByText(/Inserted/i)).toBeInTheDocument());

  // Token absent from the rendered DOM.
  expect(container.innerHTML).not.toContain(TOKEN);
  // Token absent from every console call.
  for (const spy of [logSpy, infoSpy, warnSpy, errSpy]) {
    for (const call of spy.mock.calls) {
      expect(JSON.stringify(call)).not.toContain(TOKEN);
    }
  }
  logSpy.mockRestore();
  infoSpy.mockRestore();
  warnSpy.mockRestore();
  errSpy.mockRestore();
});

test("polling stops on terminal success (no further job polls)", async () => {
  const { fn } = makeFetch();
  vi.stubGlobal("fetch", fn);

  renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /confirm & insert/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));
  await waitFor(() => expect(screen.getByText(/Inserted/i)).toBeInTheDocument());

  const jobCallsAfter = fn.mock.calls.filter((c) => String(c[0]).startsWith("/api/jobs/")).length;
  await new Promise((r) => setTimeout(r, 80));
  const jobCallsLater = fn.mock.calls.filter((c) => String(c[0]).startsWith("/api/jobs/")).length;
  expect(jobCallsLater).toBe(jobCallsAfter);
});

test("cancel: Cancel calls the endpoint and the run reports cancelled", async () => {
  const { fn } = makeFetch({ jobOutcome: "cancelled" });
  vi.stubGlobal("fetch", fn);

  renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /confirm & insert/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));

  await waitFor(() => expect(screen.getByRole("button", { name: /^cancel$/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /^cancel$/i }));

  await waitFor(() =>
    expect(fn.mock.calls.some((c) => String(c[0]).endsWith("/cancel"))).toBe(true),
  );
  await waitFor(() => expect(screen.getByText(/insert cancelled\./i)).toBeInTheDocument());
  // No success summary on a cancelled run.
  expect(screen.queryByText(/Inserted/i)).not.toBeInTheDocument();
});

test("error path: a failed job shows the scrubbed error and no summary", async () => {
  let jobPolls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/insert/preview") return jsonResponse(PREVIEW);
      if (url === "/api/insert")
        return jsonResponse({
          job_id: "job-1",
          scope: PREVIEW.scope,
          total_rows: PREVIEW.total_rows,
          writer: "SaBatchWriter",
          method: "auto",
          scope_warnings: [],
        });
      jobPolls += 1;
      return jobPolls <= 1
        ? jsonResponse({ ...baseJob, status: "running" })
        : jsonResponse({ ...baseJob, status: "failed", error: "connection refused" });
    }),
  );

  renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /confirm & insert/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/connection refused/i),
  );
  expect(screen.queryByText(/Inserted/i)).not.toBeInTheDocument();
});

test("rejected token (403) surfaces a typed error and returns to the preview step", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/insert/preview") return jsonResponse(PREVIEW);
      if (url === "/api/insert")
        return jsonResponse(
          { detail: { code: "WRITE_GUARD_REJECTED", message: "confirmation token rejected" } },
          403,
        );
      return jsonResponse({ ...baseJob, status: "running" });
    }),
  );

  renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /confirm & insert/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/confirmation token rejected/i),
  );
  // No job started, so no status console.
  expect(screen.queryByText(/Status:/)).not.toBeInTheDocument();
});

test("a failed preview (e.g. 409 NO_RUN) shows the typed error and no confirm gate", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({ detail: { code: "NO_RUN", message: "no generation result" } }, 409),
    ),
  );

  renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/no generation result/i),
  );
  expect(screen.queryByRole("button", { name: /confirm & insert/i })).not.toBeInTheDocument();
});

test("success summary renders scope_warnings when the insert returns them", async () => {
  let jobPolls = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      if (url === "/api/insert/preview") return jsonResponse(PREVIEW);
      if (url === "/api/insert")
        return jsonResponse({
          job_id: "job-1",
          scope: PREVIEW.scope,
          total_rows: PREVIEW.total_rows,
          writer: "SaBatchWriter",
          method: "batch",
          scope_warnings: ["partial scope: 'orders' parents may be missing"],
        });
      jobPolls += 1;
      return jsonResponse({ ...baseJob, status: jobPolls <= 1 ? "running" : "succeeded" });
    }),
  );

  renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /confirm & insert/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));

  await waitFor(() => expect(screen.getByText(/Inserted/i)).toBeInTheDocument());
  expect(screen.getByText(/parents may be missing/i)).toBeInTheDocument();
});

test("selecting a method sends it on the insert request", async () => {
  const { fn, insertBodies } = makeFetch();
  vi.stubGlobal("fetch", fn);

  renderWithClient(<InsertPanel pollMs={20} />);
  fireEvent.click(screen.getByRole("button", { name: /preview insert/i }));
  await waitFor(() => expect(screen.getByRole("button", { name: /confirm & insert/i })).toBeInTheDocument());

  fireEvent.change(screen.getByLabelText(/insert method/i), { target: { value: "batch" } });
  fireEvent.click(screen.getByRole("button", { name: /confirm & insert/i }));

  await waitFor(() => expect(insertBodies.length).toBeGreaterThan(0));
  expect((insertBodies[0] as { method: string }).method).toBe("batch");
});
