import { afterEach, describe, expect, test, vi } from "vitest";
import { ApiError, apiGet, apiPost } from "./client";

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  test("apiGet returns parsed JSON on 200", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => jsonResponse({ ok: true })));
    await expect(apiGet<{ ok: boolean }>("/api/x")).resolves.toEqual({ ok: true });
  });

  test("apiPost sends JSON body and returns parsed JSON", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () => jsonResponse({ source: "sample:e" }));
    vi.stubGlobal("fetch", fetchMock);
    const out = await apiPost("/api/schema/sample", { name: "e" });
    expect(out).toEqual({ source: "sample:e" });
    const [, init] = fetchMock.mock.calls[0];
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ name: "e" });
    expect(new Headers(init?.headers).get("content-type")).toContain("application/json");
  });

  test("throws ApiError carrying the envelope code on 4xx", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse({ detail: { code: "NOT_FOUND", message: "Unknown sample" } }, 404),
      ),
    );
    await expect(apiPost("/api/schema/sample", { name: "z" })).rejects.toMatchObject({
      name: "ApiError",
      status: 404,
      code: "NOT_FOUND",
      message: "Unknown sample",
    });
  });

  test("throws ApiError on non-JSON / network-ish failure", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("boom", { status: 500 })));
    await expect(apiGet("/api/x")).rejects.toBeInstanceOf(ApiError);
  });
});
