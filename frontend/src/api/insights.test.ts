import { afterEach, expect, test, vi } from "vitest";
import { ApiError } from "./client";
import { getCosts, getQuality, listRuns, queryKeys } from "./endpoints";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("listRuns GETs /api/runs with no page param by default", async () => {
  const payload = {
    rows: [],
    page: 1,
    total_pages: 1,
    total_runs: 0,
    has_prev: false,
    has_next: false,
  };
  const m = vi.fn<typeof fetch>(async () => jsonResponse(payload));
  vi.stubGlobal("fetch", m);

  const out = await listRuns();

  expect(out).toEqual(payload);
  expect(String(m.mock.calls[0][0])).toBe("/api/runs");
});

test("listRuns appends ?page= when a page is given", async () => {
  const m = vi.fn<typeof fetch>(async () =>
    jsonResponse({
      rows: [],
      page: 3,
      total_pages: 5,
      total_runs: 42,
      has_prev: true,
      has_next: true,
    }),
  );
  vi.stubGlobal("fetch", m);

  const out = await listRuns(3);

  expect(out.page).toBe(3);
  expect(String(m.mock.calls[0][0])).toBe("/api/runs?page=3");
});

test("getQuality GETs /api/quality without run_id by default", async () => {
  const m = vi.fn<typeof fetch>(async () =>
    jsonResponse({ found: false, run_id: null, rows: [] }),
  );
  vi.stubGlobal("fetch", m);

  const out = await getQuality();

  expect(out.found).toBe(false);
  expect(String(m.mock.calls[0][0])).toBe("/api/quality");
});

test("getQuality appends ?run_id= when given", async () => {
  const m = vi.fn<typeof fetch>(async () =>
    jsonResponse({ found: true, run_id: 7, rows: [] }),
  );
  vi.stubGlobal("fetch", m);

  await getQuality(7);

  expect(String(m.mock.calls[0][0])).toBe("/api/quality?run_id=7");
});

test("getCosts GETs /api/costs and returns the summary", async () => {
  const payload = {
    total_cost: 0.08,
    total_tokens: 3000,
    total_calls: 2,
    avg_cost_per_run: 0.08,
    per_provider: [{ provider: "openai", cost: 0.05, tokens: 2000, calls: 1 }],
  };
  const m = vi.fn<typeof fetch>(async () => jsonResponse(payload));
  vi.stubGlobal("fetch", m);

  const out = await getCosts();

  expect(out).toEqual(payload);
  expect(String(m.mock.calls[0][0])).toBe("/api/costs");
});

test("a non-OK insights response throws a typed ApiError", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ detail: { code: "BOOM", message: "nope" } }, 500)),
  );

  await expect(listRuns()).rejects.toBeInstanceOf(ApiError);
});

test("queryKeys namespace runs by page, quality by run id, and costs", () => {
  expect(queryKeys.runs(2)).toEqual(["runs", 2]);
  expect(queryKeys.runs()).toEqual(["runs", undefined]);
  expect(queryKeys.quality(7)).toEqual(["quality", 7]);
  expect(queryKeys.quality()).toEqual(["quality", undefined]);
  expect(queryKeys.costs).toEqual(["costs"]);
});
