import { afterEach, describe, expect, test, vi } from "vitest";
import { ApiError, apiDownload, apiGet, apiPost } from "./client";

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

describe("apiDownload", () => {
  function blobResponse(body: string, filename?: string, status = 200): Response {
    const headers: Record<string, string> = { "Content-Type": "application/sql" };
    if (filename) headers["Content-Disposition"] = `attachment; filename="${filename}"`;
    return new Response(body, { status, headers });
  }

  function stubDownloadDom() {
    const click = vi.fn();
    const anchor = { href: "", download: "", click } as unknown as HTMLAnchorElement;
    const create = vi
      .spyOn(document, "createElement")
      .mockImplementation((tag: string) =>
        tag === "a" ? anchor : ({} as HTMLElement),
      );
    vi.spyOn(document.body, "appendChild").mockImplementation((n) => n);
    vi.spyOn(document.body, "removeChild").mockImplementation((n) => n);
    const createObjectURL = vi.fn(() => "blob:mock-url");
    const revokeObjectURL = vi.fn();
    vi.stubGlobal("URL", { ...URL, createObjectURL, revokeObjectURL });
    return { anchor, click, create, createObjectURL, revokeObjectURL };
  }

  test("downloads the blob with the Content-Disposition filename and revokes the url", async () => {
    const fetchMock = vi.fn<typeof fetch>(async () =>
      blobResponse("INSERT INTO users ...", "users.sql"),
    );
    vi.stubGlobal("fetch", fetchMock);
    const dom = stubDownloadDom();

    await apiDownload("/api/export", { format: "sql" }, "fallback.bin");

    const [url, init] = fetchMock.mock.calls[0];
    expect(String(url)).toBe("/api/export");
    expect(init?.method).toBe("POST");
    expect(JSON.parse(String(init?.body))).toEqual({ format: "sql" });
    expect(dom.anchor.download).toBe("users.sql");
    expect(dom.anchor.href).toBe("blob:mock-url");
    expect(dom.click).toHaveBeenCalledOnce();
    expect(dom.revokeObjectURL).toHaveBeenCalledWith("blob:mock-url");
  });

  test("falls back to the provided name when there is no Content-Disposition header", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => blobResponse("a,b,c", undefined)));
    const dom = stubDownloadDom();

    await apiDownload("/api/export", { format: "csv" }, "dbsprout-export.csv");

    expect(dom.anchor.download).toBe("dbsprout-export.csv");
    expect(dom.click).toHaveBeenCalledOnce();
  });

  test("throws ApiError from the JSON error envelope and triggers no download", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        jsonResponse({ detail: { code: "NO_RUN", message: "no run on the workspace" } }, 409),
      ),
    );
    const dom = stubDownloadDom();

    await expect(
      apiDownload("/api/export", { format: "sql" }, "fallback.sql"),
    ).rejects.toMatchObject({ name: "ApiError", status: 409, code: "NO_RUN" });
    expect(dom.click).not.toHaveBeenCalled();
  });
});
