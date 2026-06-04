import { afterEach, expect, test, vi } from "vitest";
import { ApiError } from "./client";
import { generate, getJob, queryKeys } from "./endpoints";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("generate POSTs engine + seed and returns { job_id, seed }", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse({ job_id: "j1", seed: 42 }));
  vi.stubGlobal("fetch", m);

  const out = await generate({ engine: "heuristic", seed: 42 });

  expect(out).toEqual({ job_id: "j1", seed: 42 });
  const [url, init] = m.mock.calls[0];
  expect(String(url)).toBe("/api/generate");
  expect(init?.method).toBe("POST");
  expect(JSON.parse(init?.body as string)).toEqual({
    engine: "heuristic",
    seed: 42,
  });
});

test("generate sends seed: null so the server materialises one", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse({ job_id: "j2", seed: 999 }));
  vi.stubGlobal("fetch", m);

  const out = await generate({ engine: "spec", seed: null });

  expect(out.seed).toBe(999);
  const [, init] = m.mock.calls[0];
  expect(JSON.parse(init?.body as string)).toEqual({
    engine: "spec",
    seed: null,
  });
});

test("getJob GETs /api/jobs/{id} and returns the record", async () => {
  const record = {
    id: "j1",
    kind: "generate",
    status: "running",
    engine: "heuristic",
    seed: 42,
    started_at: "2026-06-02T00:00:00+00:00",
    finished_at: null,
    error: null,
  };
  const m = vi.fn<typeof fetch>(async () => jsonResponse(record));
  vi.stubGlobal("fetch", m);

  const out = await getJob("j1");

  expect(out).toEqual(record);
  expect(String(m.mock.calls[0][0])).toBe("/api/jobs/j1");
});

test("getJob url-encodes the job id", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse({ id: "a/b" }));
  vi.stubGlobal("fetch", m);

  await getJob("a/b");

  expect(String(m.mock.calls[0][0])).toBe("/api/jobs/a%2Fb");
});

test("a non-OK generate response throws a typed ApiError", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({ detail: { code: "NO_SCHEMA", message: "no schema loaded" } }, 400),
    ),
  );

  await expect(generate({ engine: "heuristic", seed: null })).rejects.toBeInstanceOf(ApiError);
});

test("queryKeys.job namespaces by job id", () => {
  expect(queryKeys.job("j1")).toEqual(["job", "j1"]);
});
