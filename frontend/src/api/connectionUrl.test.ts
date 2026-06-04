import { afterEach, describe, expect, test, vi } from "vitest";
import {
  buildConnectionUrl,
  defaultPort,
  parseConnectionUrl,
  SSL_MODES,
  type ConnectionFields,
} from "./connectionUrl";
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

describe("parseConnectionUrl — basic authority", () => {
  test("parses postgres user/password/host/port/database", () => {
    const f = parseConnectionUrl("postgresql://admin:secret@db.host:5432/shop");
    expect(f.type).toBe("postgresql");
    expect(f.user).toBe("admin");
    expect(f.password).toBe("secret");
    expect(f.host).toBe("db.host");
    expect(f.port).toBe("5432");
    expect(f.database).toBe("shop");
  });

  test("decodes percent-encoded credentials", () => {
    const f = parseConnectionUrl("postgresql://admin:p%40ss%20word@localhost:5432/shop");
    expect(f.user).toBe("admin");
    expect(f.password).toBe("p@ss word");
  });

  test("handles omitted port, credentials and database", () => {
    const f = parseConnectionUrl("postgresql://localhost/shop");
    expect(f.user).toBe("");
    expect(f.password).toBe("");
    expect(f.port).toBe("");
    expect(f.host).toBe("localhost");
    expect(f.database).toBe("shop");
  });

  test("handles a user with no password", () => {
    const f = parseConnectionUrl("postgresql://admin@localhost:5432/shop");
    expect(f.user).toBe("admin");
    expect(f.password).toBe("");
  });

  test("parses mysql with its default port", () => {
    const f = parseConnectionUrl("mysql://admin:secret@localhost:3306/shop");
    expect(f.type).toBe("mysql");
    expect(f.port).toBe("3306");
  });

  test("tolerates a SQLAlchemy +driver scheme suffix", () => {
    const f = parseConnectionUrl("postgresql+psycopg2://admin:secret@localhost:5432/shop");
    expect(f.type).toBe("postgresql");
    expect(f.host).toBe("localhost");
  });

  test("maps the postgres scheme alias", () => {
    expect(parseConnectionUrl("postgres://localhost/shop").type).toBe("postgresql");
  });

  test("returns all ConnectionFields keys populated (drop-in form state)", () => {
    const f = parseConnectionUrl("postgresql://localhost/shop");
    expect(Object.keys(f).sort()).toEqual(
      [
        "connectTimeout",
        "database",
        "filePath",
        "host",
        "params",
        "password",
        "port",
        "schema",
        "sslCa",
        "sslCert",
        "sslKey",
        "sslMode",
        "type",
        "user",
      ].sort(),
    );
  });

  test("never throws on a malformed url; falls back to postgresql", () => {
    const f = parseConnectionUrl("not-a-url");
    expect(f.type).toBe("postgresql");
  });
});

describe("parseConnectionUrl — sqlite", () => {
  test("extracts the file path as the exact inverse of the builder", () => {
    const f = parseConnectionUrl("sqlite:////tmp/x.db");
    expect(f.type).toBe("sqlite");
    expect(f.filePath).toBe("/tmp/x.db");
  });

  test("relative sqlite path", () => {
    expect(parseConnectionUrl("sqlite:///data.db").filePath).toBe("data.db");
  });
});

describe("parseConnectionUrl — advanced query (inverse of buildAdvancedQuery)", () => {
  test("parses sslmode for postgres", () => {
    expect(parseConnectionUrl("postgresql://h/db?sslmode=require").sslMode).toBe("require");
  });

  test("keeps an unrecognised sslmode value as a free param (no data loss)", () => {
    const f = parseConnectionUrl("postgresql://h/db?sslmode=bogus");
    expect(f.sslMode).toBe("");
    expect(f.params).toEqual([{ key: "sslmode", value: "bogus" }]);
  });

  test("parses postgres cert keys into ca/cert/key", () => {
    const f = parseConnectionUrl(
      "postgresql://h/db?sslrootcert=%2Fc%2Fca.pem&sslcert=%2Fc%2Fcl.pem&sslkey=%2Fc%2Fcl.key",
    );
    expect(f.sslCa).toBe("/c/ca.pem");
    expect(f.sslCert).toBe("/c/cl.pem");
    expect(f.sslKey).toBe("/c/cl.key");
  });

  test("parses mysql cert keys into ca/cert/key", () => {
    const f = parseConnectionUrl(
      "mysql://h:3306/db?ssl_ca=%2Fc%2Fca.pem&ssl_cert=%2Fc%2Fcl.pem&ssl_key=%2Fc%2Fcl.key",
    );
    expect(f.sslCa).toBe("/c/ca.pem");
    expect(f.sslCert).toBe("/c/cl.pem");
    expect(f.sslKey).toBe("/c/cl.key");
  });

  test("parses schema out of options=-csearch_path", () => {
    expect(parseConnectionUrl("postgresql://h/db?options=-csearch_path%3Danalytics").schema).toBe(
      "analytics",
    );
  });

  test("decodes a schema name containing a space", () => {
    expect(
      parseConnectionUrl("postgresql://h/db?options=-csearch_path%3Dapp%20schema").schema,
    ).toBe("app schema");
  });

  test("parses connect_timeout", () => {
    expect(parseConnectionUrl("postgresql://h/db?connect_timeout=10").connectTimeout).toBe("10");
  });

  test("collects unrecognised pairs as free params in order", () => {
    const f = parseConnectionUrl(
      "postgresql://h/db?application_name=dbsprout&target_session_attrs=read-write",
    );
    expect(f.params).toEqual([
      { key: "application_name", value: "dbsprout" },
      { key: "target_session_attrs", value: "read-write" },
    ]);
  });

  test("decodes percent-encoded param keys and values", () => {
    const f = parseConnectionUrl("postgresql://h/db?k%20v=a%26b");
    expect(f.params).toEqual([{ key: "k v", value: "a&b" }]);
  });

  test("an unknown options= value (not search_path) stays a free param", () => {
    const f = parseConnectionUrl("postgresql://h/db?options=-cfoo%3Dbar");
    expect(f.schema).toBe("");
    expect(f.params).toEqual([{ key: "options", value: "-cfoo=bar" }]);
  });
});

describe("parseConnectionUrl ∘ buildConnectionUrl — round trip", () => {
  const fixtures: ConnectionFields[] = [
    base,
    { ...base, port: "", user: "", password: "" },
    { ...base, type: "sqlite", filePath: "/tmp/x.db" },
    { ...base, type: "mysql", port: "3306" },
    { ...base, sslMode: "require" },
    { ...base, sslCa: "/c/ca.pem", sslCert: "/c/cl.pem", sslKey: "/c/cl.key" },
    { ...base, type: "mysql", port: "3306", sslCa: "/c/ca.pem" },
    { ...base, schema: "analytics" },
    { ...base, connectTimeout: "5" },
    { ...base, params: [{ key: "application_name", value: "dbsprout" }, { key: "keepalives", value: "1" }] },
    {
      ...base,
      sslMode: "verify-full",
      sslCa: "/c/ca.pem",
      schema: "analytics",
      connectTimeout: "5",
      params: [
        { key: "application_name", value: "dbsprout" },
        { key: "keepalives", value: "1" },
      ],
    },
  ];

  test.each(fixtures.map((f, i) => [i, f] as const))(
    "build(parse(build(f))) === build(f) — fixture %i",
    (_i, f) => {
      const url = buildConnectionUrl(f);
      expect(buildConnectionUrl(parseConnectionUrl(url))).toBe(url);
    },
  );
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
