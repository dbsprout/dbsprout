import { afterEach, expect, test, vi } from "vitest";
import { getPreview, queryKeys } from "./endpoints";

afterEach(() => vi.unstubAllGlobals());

test("getPreview GETs /api/preview/{table} (URL-encoded)", async () => {
  const body = { table: "users", limit: 100, total: 2, rows: [{ id: 1 }, { id: 2 }] };
  const m = vi.fn(
    async () =>
      new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", m);
  const out = await getPreview("users");
  expect(out.rows).toHaveLength(2);
  expect(String((m.mock.calls as unknown as [string][])[0][0])).toBe("/api/preview/users");
});

test("getPreview URL-encodes the table name", async () => {
  const m = vi.fn(
    async () =>
      new Response(JSON.stringify({ table: "a b", limit: 100, total: 0, rows: [] }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", m);
  await getPreview("a b");
  expect(String((m.mock.calls as unknown as [string][])[0][0])).toBe("/api/preview/a%20b");
});

test("preview query key factory is table-scoped", () => {
  expect(queryKeys.preview("orders")).toEqual(["preview", "orders"]);
});
