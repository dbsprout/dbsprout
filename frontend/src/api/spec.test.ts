import { afterEach, expect, test, vi } from "vitest";
import { getSpec, listGenerators, putColumnSpec, putTableRowCount } from "./endpoints";
import type { GeneratorConfig } from "./types";

afterEach(() => vi.unstubAllGlobals());

function stub(body: unknown, status = 200) {
  const m = vi.fn(async () =>
    new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } }),
  );
  vi.stubGlobal("fetch", m);
  return m;
}

test("getSpec GETs /api/spec", async () => {
  const m = stub({ version: "1.0", tables: [], global_seed: 42, schema_hash: "", model_used: null, created_at: null });
  const spec = await getSpec();
  expect(spec.version).toBe("1.0");
  expect(String((m.mock.calls as unknown as [string, RequestInit?][][])[0][0])).toBe("/api/spec");
});

test("listGenerators passes dtype query", async () => {
  const m = stub({ providers: [], methods: [] });
  await listGenerators("VARCHAR");
  expect(String((m.mock.calls as unknown as [string, RequestInit?][][])[0][0])).toBe("/api/generators?dtype=VARCHAR");
});

test("putTableRowCount PUTs row_count", async () => {
  const m = stub({ table_name: "users", row_count: 500 });
  const out = await putTableRowCount("users", 500);
  expect(out.row_count).toBe(500);
  const [url, init] = m.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toBe("/api/spec/tables/users");
  expect(init.method).toBe("PUT");
  expect(JSON.parse(String(init.body))).toEqual({ row_count: 500 });
});

test("putColumnSpec PUTs the GeneratorConfig", async () => {
  const cfg: GeneratorConfig = {
    provider: "mimesis", method: "email", params: {}, distribution: null, distribution_params: {},
    min_value: null, max_value: null, enum_values: null, format_pattern: null, unique: false,
    nullable_rate: 0, vectorized: false,
  };
  const m = stub(cfg);
  await putColumnSpec("users", "email", cfg);
  expect(String((m.mock.calls as unknown as [string, RequestInit?][][])[0][0])).toBe("/api/spec/tables/users/columns/email");
});
