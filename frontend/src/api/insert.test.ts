import { afterEach, expect, test, vi } from "vitest";
import { ApiError } from "./client";
import { cancelJob, insertData, insertPreview } from "./endpoints";

afterEach(() => vi.unstubAllGlobals());

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
  confirmation_token: "payload.sig",
};

test("insertPreview POSTs { tables } and returns the preview envelope", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse(PREVIEW));
  vi.stubGlobal("fetch", m);

  const out = await insertPreview(["users", "orders"]);

  expect(out).toEqual(PREVIEW);
  const [url, init] = m.mock.calls[0];
  expect(String(url)).toBe("/api/insert/preview");
  expect(init?.method).toBe("POST");
  expect(JSON.parse(init?.body as string)).toEqual({ tables: ["users", "orders"] });
});

test("insertPreview with no tables sends tables: null (whole-DB scope)", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse(PREVIEW));
  vi.stubGlobal("fetch", m);

  await insertPreview();

  const [, init] = m.mock.calls[0];
  expect(JSON.parse(init?.body as string)).toEqual({ tables: null });
});

test("insertData POSTs token + method and returns { job_id, ... }", async () => {
  const res = {
    job_id: "job-9",
    scope: [{ table: "users", row_count: 100 }],
    total_rows: 100,
    writer: "PgCopyWriter",
    method: "auto",
    scope_warnings: [],
  };
  const m = vi.fn<typeof fetch>(async () => jsonResponse(res));
  vi.stubGlobal("fetch", m);

  const out = await insertData({
    tables: ["users"],
    confirmation_token: "payload.sig",
    method: "auto",
  });

  expect(out).toEqual(res);
  const [url, init] = m.mock.calls[0];
  expect(String(url)).toBe("/api/insert");
  expect(init?.method).toBe("POST");
  expect(JSON.parse(init?.body as string)).toEqual({
    tables: ["users"],
    confirmation_token: "payload.sig",
    method: "auto",
  });
});

test("cancelJob POSTs /api/jobs/{id}/cancel and returns { job_id, status }", async () => {
  const m = vi.fn<typeof fetch>(async () =>
    jsonResponse({ job_id: "job-9", status: "cancelling" }),
  );
  vi.stubGlobal("fetch", m);

  const out = await cancelJob("job-9");

  expect(out).toEqual({ job_id: "job-9", status: "cancelling" });
  const [url, init] = m.mock.calls[0];
  expect(String(url)).toBe("/api/jobs/job-9/cancel");
  expect(init?.method).toBe("POST");
});

test("cancelJob url-encodes the job id", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse({ job_id: "a/b", status: "cancelling" }));
  vi.stubGlobal("fetch", m);

  await cancelJob("a/b");

  expect(String(m.mock.calls[0][0])).toBe("/api/jobs/a%2Fb/cancel");
});

test("a rejected token (403) surfaces a typed ApiError", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse(
        { detail: { code: "WRITE_GUARD_REJECTED", message: "token rejected" } },
        403,
      ),
    ),
  );

  await expect(
    insertData({ tables: null, confirmation_token: "bad", method: "auto" }),
  ).rejects.toBeInstanceOf(ApiError);
});
