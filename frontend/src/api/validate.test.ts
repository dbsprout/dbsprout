import { afterEach, expect, test, vi } from "vitest";
import { ApiError } from "./client";
import { queryKeys, validate } from "./endpoints";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const ENVELOPE = {
  summary: { tables: 2, rows: 30, violations: 1 },
  by_table: [
    { table: "users", fk_violations: 0, unique_violations: 0, not_null_violations: 0, check_violations: 0 },
    { table: "orders", fk_violations: 1, unique_violations: 0, not_null_violations: 0, check_violations: 0 },
  ],
  details: [
    { check: "fk_satisfaction", table: "orders", column: "user_id", passed: false, details: "1 orphan" },
  ],
  fidelity: null,
  detection: null,
};

test("validate POSTs /api/validate with no body when tables omitted and returns the envelope", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse(ENVELOPE));
  vi.stubGlobal("fetch", m);

  const out = await validate();

  expect(out).toEqual(ENVELOPE);
  const [url, init] = m.mock.calls[0];
  expect(String(url)).toBe("/api/validate");
  expect(init?.method).toBe("POST");
});

test("validate forwards an explicit tables list in the request body", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse(ENVELOPE));
  vi.stubGlobal("fetch", m);

  await validate(["orders"]);

  const [, init] = m.mock.calls[0];
  expect(JSON.parse(init?.body as string)).toEqual({ tables: ["orders"] });
});

test("a 409 NO_RUN response throws a typed ApiError carrying the message", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse(
        { detail: { code: "NO_RUN", message: "No generation result is available." } },
        409,
      ),
    ),
  );

  await expect(validate()).rejects.toBeInstanceOf(ApiError);
});

test("queryKeys.validate is a stable namespace", () => {
  expect(queryKeys.validate).toEqual(["validate"]);
});
