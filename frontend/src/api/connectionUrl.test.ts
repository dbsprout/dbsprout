import { afterEach, describe, expect, test, vi } from "vitest";
import { buildConnectionUrl, defaultPort, SSL_MODES, type ConnectionFields } from "./connectionUrl";
import { uploadSchema } from "./endpoints";

const base: ConnectionFields = {
  type: "postgresql",
  host: "localhost",
  port: "5432",
  user: "admin",
  password: "p@ss word",
  database: "shop",
  filePath: "",
  sslMode: "",
  sslCa: "",
  sslCert: "",
  sslKey: "",
  schema: "",
  connectTimeout: "",
  params: [],
};

describe("buildConnectionUrl — legacy behaviour (blank advanced)", () => {
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

describe("buildConnectionUrl — ssl mode + connect timeout", () => {
  test("appends sslmode for postgres", () => {
    expect(buildConnectionUrl({ ...base, sslMode: "require" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop?sslmode=require",
    );
  });
  test("accepts every valid ssl mode", () => {
    for (const mode of SSL_MODES) {
      expect(buildConnectionUrl({ ...base, sslMode: mode })).toContain(`sslmode=${mode}`);
    }
  });
  test("skips an invalid ssl mode", () => {
    expect(buildConnectionUrl({ ...base, sslMode: "bogus" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop",
    );
  });
  test("mysql does not emit sslmode", () => {
    expect(buildConnectionUrl({ ...base, type: "mysql", port: "3306", sslMode: "require" })).toBe(
      "mysql://admin:p%40ss%20word@localhost:3306/shop",
    );
  });
  test("appends connect_timeout when a non-negative integer", () => {
    expect(buildConnectionUrl({ ...base, connectTimeout: "10" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop?connect_timeout=10",
    );
  });
  test("skips a non-integer connect timeout", () => {
    expect(buildConnectionUrl({ ...base, connectTimeout: "ten" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop",
    );
    expect(buildConnectionUrl({ ...base, connectTimeout: "-3" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop",
    );
  });
  test("sqlite ignores all advanced fields", () => {
    expect(
      buildConnectionUrl({
        ...base,
        type: "sqlite",
        filePath: "/tmp/x.db",
        sslMode: "require",
        connectTimeout: "10",
        schema: "analytics",
        params: [{ key: "a", value: "b" }],
      }),
    ).toBe("sqlite:////tmp/x.db");
  });
});

describe("buildConnectionUrl — ssl certificates (dialect-aware)", () => {
  test("postgres maps ca/cert/key to sslrootcert/sslcert/sslkey", () => {
    expect(
      buildConnectionUrl({ ...base, sslCa: "/c/ca.pem", sslCert: "/c/cl.pem", sslKey: "/c/cl.key" }),
    ).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop" +
        "?sslrootcert=%2Fc%2Fca.pem&sslcert=%2Fc%2Fcl.pem&sslkey=%2Fc%2Fcl.key",
    );
  });
  test("mysql maps ca/cert/key to ssl_ca/ssl_cert/ssl_key", () => {
    expect(
      buildConnectionUrl({
        ...base,
        type: "mysql",
        port: "3306",
        sslCa: "/c/ca.pem",
        sslCert: "/c/cl.pem",
        sslKey: "/c/cl.key",
      }),
    ).toBe(
      "mysql://admin:p%40ss%20word@localhost:3306/shop" +
        "?ssl_ca=%2Fc%2Fca.pem&ssl_cert=%2Fc%2Fcl.pem&ssl_key=%2Fc%2Fcl.key",
    );
  });
  test("encodes cert paths containing spaces", () => {
    expect(buildConnectionUrl({ ...base, sslCa: "/my certs/ca.pem" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop?sslrootcert=%2Fmy%20certs%2Fca.pem",
    );
  });
});

describe("buildConnectionUrl — schema / search_path", () => {
  test("postgres folds schema into options=-csearch_path", () => {
    expect(buildConnectionUrl({ ...base, schema: "analytics" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop?options=-csearch_path%3Danalytics",
    );
  });
  test("postgres url-encodes the schema name", () => {
    expect(buildConnectionUrl({ ...base, schema: "app schema" })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop?options=-csearch_path%3Dapp%20schema",
    );
  });
  test("mysql ignores schema (schema == database)", () => {
    expect(buildConnectionUrl({ ...base, type: "mysql", port: "3306", schema: "analytics" })).toBe(
      "mysql://admin:p%40ss%20word@localhost:3306/shop",
    );
  });
});

describe("buildConnectionUrl — free-form params", () => {
  test("appends params verbatim in insertion order", () => {
    expect(
      buildConnectionUrl({
        ...base,
        params: [
          { key: "application_name", value: "dbsprout" },
          { key: "target_session_attrs", value: "read-write" },
        ],
      }),
    ).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop" +
        "?application_name=dbsprout&target_session_attrs=read-write",
    );
  });
  test("skips params with an empty key", () => {
    expect(buildConnectionUrl({ ...base, params: [{ key: "", value: "x" }, { key: "a", value: "1" }] })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop?a=1",
    );
  });
  test("url-encodes param keys and values", () => {
    expect(buildConnectionUrl({ ...base, params: [{ key: "k v", value: "a&b" }] })).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop?k%20v=a%26b",
    );
  });
  test("full permutation: sslmode + cert + schema + timeout + 2 params, ordered", () => {
    expect(
      buildConnectionUrl({
        ...base,
        sslMode: "verify-full",
        sslCa: "/c/ca.pem",
        schema: "analytics",
        connectTimeout: "5",
        params: [
          { key: "application_name", value: "dbsprout" },
          { key: "keepalives", value: "1" },
        ],
      }),
    ).toBe(
      "postgresql://admin:p%40ss%20word@localhost:5432/shop" +
        "?sslmode=verify-full" +
        "&sslrootcert=%2Fc%2Fca.pem" +
        "&options=-csearch_path%3Danalytics" +
        "&connect_timeout=5" +
        "&application_name=dbsprout" +
        "&keepalives=1",
    );
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
