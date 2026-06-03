// ─── P2b-2 ─── client tests for putTableAdvanced (advanced packs).
import { afterEach, expect, test, vi } from "vitest";
import { putTableAdvanced } from "./endpoints";
import type { CorrelationRule, DerivedColumn } from "./types";

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

test("putTableAdvanced PUTs correlations + derived to the advanced route", async () => {
  const correlations: CorrelationRule[] = [
    { columns: ["city", "state"], lookup_table: null, strategy: "lookup" },
  ];
  const derived: DerivedColumn[] = [
    { column: "price", expression: "qty * 2", depends_on: ["qty"] },
  ];
  const m = stub({ table_name: "users", correlations, derived });

  const out = await putTableAdvanced("users", { correlations, derived });

  expect(out.table_name).toBe("users");
  expect(out.correlations[0].columns).toEqual(["city", "state"]);
  const [url, init] = m.mock.calls[0] as unknown as [string, RequestInit];
  expect(url).toBe("/api/spec/tables/users/advanced");
  expect(init.method).toBe("PUT");
  expect(JSON.parse(String(init.body))).toEqual({ correlations, derived });
});

test("putTableAdvanced encodes the table name", async () => {
  const m = stub({ table_name: "order items", correlations: [], derived: [] });
  await putTableAdvanced("order items", { correlations: [] });
  const [url] = m.mock.calls[0] as unknown as [string];
  expect(url).toBe("/api/spec/tables/order%20items/advanced");
});

test("putTableAdvanced surfaces a typed 422 error", async () => {
  stub({ detail: { code: "UNKNOWN_COLUMN_REF", message: "no such column" } }, 422);
  await expect(
    putTableAdvanced("users", { correlations: [{ columns: ["ghost"], lookup_table: null, strategy: "lookup" }] }),
  ).rejects.toMatchObject({ status: 422, code: "UNKNOWN_COLUMN_REF" });
});
