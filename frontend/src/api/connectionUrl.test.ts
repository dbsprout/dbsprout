import { afterEach, describe, expect, test, vi } from "vitest";
import { buildConnectionUrl, defaultPort, type ConnectionFields } from "./connectionUrl";
import { uploadSchema } from "./endpoints";

const base: ConnectionFields = {
  type: "postgresql",
  host: "localhost",
  port: "5432",
  user: "admin",
  password: "p@ss word",
  database: "shop",
  filePath: "",
};

describe("buildConnectionUrl", () => {
  test("postgres url with url-encoded credentials", () => {
    expect(buildConnectionUrl(base)).toBe("postgresql://admin:p%40ss%20word@localhost:5432/shop");
  });
  test("sqlite uses file path", () => {
    expect(buildConnectionUrl({ ...base, type: "sqlite", filePath: "/tmp/x.db" })).toBe("sqlite:////tmp/x.db");
  });
  test("omits empty port and credentials gracefully", () => {
    expect(buildConnectionUrl({ ...base, port: "", user: "", password: "" })).toBe("postgresql://localhost/shop");
  });
  test("defaultPort per dialect", () => {
    expect(defaultPort("postgresql")).toBe("5432");
    expect(defaultPort("mysql")).toBe("3306");
    expect(defaultPort("mssql")).toBe("1433");
    expect(defaultPort("sqlite")).toBe("");
  });
});

afterEach(() => vi.unstubAllGlobals());

test("uploadSchema posts multipart FormData without a JSON content-type", async () => {
  const fetchMock = vi.fn(async () =>
    new Response(JSON.stringify({ source: "upload:x.sql", table_count: 1, tables: ["t"], dialect: "sqlite" }), {
      status: 200, headers: { "Content-Type": "application/json" },
    }),
  );
  vi.stubGlobal("fetch", fetchMock);
  const file = new File(["CREATE TABLE t (id INTEGER);"], "x.sql", { type: "text/plain" });
  const out = await uploadSchema(file);
  expect(out.tables).toEqual(["t"]);
  const call = fetchMock.mock.calls[0] as unknown as Parameters<typeof fetch>;
  const init = call[1];
  expect(init?.method).toBe("POST");
  expect(init?.body).toBeInstanceOf(FormData);
  expect(new Headers(init?.headers).get("content-type")).toBeNull();
});
