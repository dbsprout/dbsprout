import { afterEach, expect, test, vi } from "vitest";
import { ApiError } from "./client";
import { assistSpec } from "./endpoints";

afterEach(() => vi.unstubAllGlobals());

function stub(body: unknown, status = 200) {
  const m = vi.fn(
    async () =>
      new Response(JSON.stringify(body), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", m);
  return m;
}

const OK = {
  provider: "embedded",
  model_used: "qwen",
  schema_hash: "abc123",
  tables: 2,
  total_columns: 7,
};

test("assistSpec POSTs /api/spec/assist with an empty body by default", async () => {
  const m = stub(OK);
  const out = await assistSpec();
  expect(out.tables).toBe(2);
  const [url, init] = m.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toBe("/api/spec/assist");
  expect(init.method).toBe("POST");
  expect(JSON.parse(String(init.body))).toEqual({});
});

test("assistSpec forwards an explicit provider in the body", async () => {
  const m = stub({ ...OK, provider: "cloud" });
  const out = await assistSpec("cloud");
  expect(out.provider).toBe("cloud");
  const [, init] = m.mock.calls[0] as unknown as [string, RequestInit];
  expect(JSON.parse(String(init.body))).toEqual({ provider: "cloud" });
});

test("assistSpec throws a typed ApiError on a 503 envelope", async () => {
  stub({ detail: { code: "LLM_UNAVAILABLE", message: "provider missing" } }, 503);
  await expect(assistSpec("embedded")).rejects.toMatchObject({
    name: "ApiError",
    status: 503,
    code: "LLM_UNAVAILABLE",
  });
  // Sanity: it is an ApiError instance, so components can read `.message`.
  await expect(assistSpec("embedded")).rejects.toBeInstanceOf(ApiError);
});
