import { QueryClient } from "@tanstack/react-query";
import { expect, test, vi } from "vitest";
import { invalidateSchemaQueries } from "./invalidateSchema";

// ─── P5-12 ───
// Loading a schema must refresh every schema-DERIVED query, not just the schema
// tree — otherwise a `spec` query stuck in its 409 error state never refetches
// and Configure stays on "No schema loaded". This guards that the helper fans
// the invalidation out to the schema, spec, AND preview query families.

/** The query key passed to a given invalidateQueries call, if any. */
function invalidatedKeys(spy: ReturnType<typeof vi.fn>): readonly unknown[][] {
  return spy.mock.calls.map(([arg]) => (arg as { queryKey: unknown[] }).queryKey);
}

test("invalidates the schema, spec, and preview query families", () => {
  const qc = new QueryClient();
  const spy = vi.spyOn(qc, "invalidateQueries");

  invalidateSchemaQueries(qc);

  const keys = invalidatedKeys(spy as unknown as ReturnType<typeof vi.fn>);
  expect(keys).toContainEqual(["schema"]);
  expect(keys).toContainEqual(["spec"]);
  expect(keys).toContainEqual(["preview"]);
});

test("does not invalidate schema alone — the spec query must be refreshed too", () => {
  const qc = new QueryClient();
  const spy = vi.spyOn(qc, "invalidateQueries");

  invalidateSchemaQueries(qc);

  const keys = invalidatedKeys(spy as unknown as ReturnType<typeof vi.fn>);
  // The whole point of the fix: spec is invalidated alongside schema.
  expect(keys.some((k) => k[0] === "spec")).toBe(true);
});
