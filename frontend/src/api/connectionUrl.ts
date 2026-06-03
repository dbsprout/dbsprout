export type DbType = "postgresql" | "mysql" | "sqlite" | "mssql";

/** Valid libpq `sslmode` values (Postgres). Used to validate the form select. */
export const SSL_MODES = [
  "disable",
  "allow",
  "prefer",
  "require",
  "verify-ca",
  "verify-full",
] as const;

export type SslMode = (typeof SSL_MODES)[number];

/** A single free-form connection query parameter (`key=value`). */
export interface ConnectionParam {
  key: string;
  value: string;
}

export interface ConnectionFields {
  type: DbType;
  host: string;
  port: string;
  user: string;
  password: string;
  database: string;
  filePath: string;
  /** libpq `sslmode` (Postgres only); empty = unset. */
  sslMode: string;
  /** Path to the CA certificate. */
  sslCa: string;
  /** Path to the client certificate. */
  sslCert: string;
  /** Path to the client private key. */
  sslKey: string;
  /** Schema / search_path (Postgres only). */
  schema: string;
  /** Connection timeout in seconds (non-negative integer). */
  connectTimeout: string;
  /** Arbitrary extra query parameters appended verbatim. */
  params: ConnectionParam[];
}

const DEFAULT_PORTS: Record<DbType, string> = {
  postgresql: "5432",
  mysql: "3306",
  mssql: "1433",
  sqlite: "",
};

export function defaultPort(type: DbType): string {
  return DEFAULT_PORTS[type];
}

/** Per-dialect SSL certificate query-key mapping. `null` means "not supported as a query key". */
const SSL_KEYS: Record<DbType, { ca: string; cert: string; key: string } | null> = {
  postgresql: { ca: "sslrootcert", cert: "sslcert", key: "sslkey" },
  mysql: { ca: "ssl_ca", cert: "ssl_cert", key: "ssl_key" },
  mssql: null,
  sqlite: null,
};

function isValidSslMode(value: string): value is SslMode {
  return (SSL_MODES as readonly string[]).includes(value);
}

function isNonNegativeInteger(value: string): boolean {
  return /^\d+$/.test(value);
}

/**
 * Build the ordered list of advanced query pairs (`key=value`, already URL-encoded).
 * Dialect-aware; returns an empty list for SQLite. Pure + immutable.
 */
function buildAdvancedQuery(f: ConnectionFields): string[] {
  if (f.type === "sqlite") {
    return [];
  }

  const pairs: string[] = [];
  const push = (key: string, value: string): void => {
    pairs.push(`${encodeURIComponent(key)}=${encodeURIComponent(value)}`);
  };

  // 1. sslmode (Postgres only; must be a recognised libpq value).
  if (f.type === "postgresql" && f.sslMode.length > 0 && isValidSslMode(f.sslMode)) {
    push("sslmode", f.sslMode);
  }

  // 2. SSL certificate paths (dialect-aware keys).
  const sslKeys = SSL_KEYS[f.type];
  if (sslKeys) {
    if (f.sslCa.length > 0) push(sslKeys.ca, f.sslCa);
    if (f.sslCert.length > 0) push(sslKeys.cert, f.sslCert);
    if (f.sslKey.length > 0) push(sslKeys.key, f.sslKey);
  }

  // 3. schema / search_path (Postgres `options=-csearch_path%3D<schema>`).
  if (f.type === "postgresql" && f.schema.length > 0) {
    // The `=` inside the option value must stay percent-encoded as %3D, so build it
    // literally and only encode the schema name.
    push("options", `-csearch_path=${f.schema}`);
  }

  // 4. connect timeout (non-negative integer only).
  if (f.connectTimeout.length > 0 && isNonNegativeInteger(f.connectTimeout)) {
    push("connect_timeout", f.connectTimeout);
  }

  // 5. free-form params, insertion order, empty keys skipped.
  for (const p of f.params) {
    if (p.key.length > 0) {
      push(p.key, p.value);
    }
  }

  return pairs;
}

/** Empty ConnectionFields — every key present so a parse result drops into form state. */
const EMPTY_FIELDS: ConnectionFields = {
  type: "postgresql",
  host: "",
  port: "",
  user: "",
  password: "",
  database: "",
  filePath: "",
  sslMode: "",
  sslCa: "",
  sslCert: "",
  sslKey: "",
  schema: "",
  connectTimeout: "",
  params: [],
};

/** Map a URL scheme (sans `+driver` suffix) to a DbType. Unknown → postgresql. */
function schemeToDbType(scheme: string): DbType {
  const head = scheme.split("+")[0]?.toLowerCase() ?? "";
  switch (head) {
    case "postgresql":
    case "postgres":
      return "postgresql";
    case "mysql":
      return "mysql";
    case "mssql":
    case "sqlserver":
      return "mssql";
    case "sqlite":
      return "sqlite";
    default:
      return "postgresql";
  }
}

function safeDecode(value: string): string {
  try {
    return decodeURIComponent(value);
  } catch {
    return value;
  }
}

/** Reverse of `SSL_KEYS`: query-key → which cert slot it fills, for a given dialect. */
function certSlotFor(type: DbType, key: string): "ca" | "cert" | "key" | null {
  const keys = SSL_KEYS[type];
  if (!keys) {
    return null;
  }
  if (key === keys.ca) return "ca";
  if (key === keys.cert) return "cert";
  if (key === keys.key) return "key";
  return null;
}

/**
 * Fold a single decoded query pair into the accumulating fields. Recognised keys are
 * consumed into structured slots; everything else accumulates as a free-form param.
 * Returns a new ConnectionFields (immutable).
 */
function applyQueryPair(f: ConnectionFields, key: string, value: string): ConnectionFields {
  // sslmode (postgres, recognised value only).
  if (f.type === "postgresql" && key === "sslmode" && isValidSslMode(value)) {
    return { ...f, sslMode: value };
  }

  // dialect-aware SSL cert keys.
  const slot = certSlotFor(f.type, key);
  if (slot === "ca") return { ...f, sslCa: value };
  if (slot === "cert") return { ...f, sslCert: value };
  if (slot === "key") return { ...f, sslKey: value };

  // schema folded into options=-csearch_path=<schema> (postgres).
  if (f.type === "postgresql" && key === "options") {
    const prefix = "-csearch_path=";
    if (value.startsWith(prefix)) {
      return { ...f, schema: value.slice(prefix.length) };
    }
  }

  // connect timeout.
  if (key === "connect_timeout") {
    return { ...f, connectTimeout: value };
  }

  // anything else → free-form param, in encounter order.
  return { ...f, params: [...f.params, { key, value }] };
}

/** Parse the `?a=b&c=d` query string into structured + free-form fields. */
function parseQuery(f: ConnectionFields, query: string): ConnectionFields {
  if (query.length === 0) {
    return f;
  }
  let out = f;
  for (const pair of query.split("&")) {
    if (pair.length === 0) {
      continue;
    }
    const eq = pair.indexOf("=");
    const rawKey = eq === -1 ? pair : pair.slice(0, eq);
    const rawValue = eq === -1 ? "" : pair.slice(eq + 1);
    out = applyQueryPair(out, safeDecode(rawKey), safeDecode(rawValue));
  }
  return out;
}

/**
 * Inverse of {@link buildConnectionUrl}: parse a SQLAlchemy-style connection URL back into
 * structured {@link ConnectionFields}. Dialect-aware (sslmode / cert keys / search_path /
 * connect_timeout / free params). Pure, immutable, and never throws — a malformed URL
 * yields best-effort fields defaulting to the postgresql dialect.
 *
 * Engineered as the left-inverse of the builder for canonical builder output, so that
 * `buildConnectionUrl(parseConnectionUrl(url)) === url` for any URL the builder can emit.
 */
export function parseConnectionUrl(url: string): ConnectionFields {
  const schemeSplit = url.indexOf("://");
  if (schemeSplit === -1) {
    return { ...EMPTY_FIELDS, host: url };
  }

  const scheme = url.slice(0, schemeSplit);
  const type = schemeToDbType(scheme);
  const rest = url.slice(schemeSplit + 3);

  // SQLite: everything after `sqlite:///` is the file path (exact inverse of the builder).
  if (type === "sqlite") {
    return { ...EMPTY_FIELDS, type, filePath: rest.replace(/^\//, "") };
  }

  // Split off the query string first.
  const queryIdx = rest.indexOf("?");
  const authority = queryIdx === -1 ? rest : rest.slice(0, queryIdx);
  const query = queryIdx === -1 ? "" : rest.slice(queryIdx + 1);

  // authority = [user[:password]@]host[:port][/database]
  const atIdx = authority.indexOf("@");
  const credPart = atIdx === -1 ? "" : authority.slice(0, atIdx);
  const hostPart = atIdx === -1 ? authority : authority.slice(atIdx + 1);

  let user = "";
  let password = "";
  if (credPart.length > 0) {
    const colon = credPart.indexOf(":");
    if (colon === -1) {
      user = safeDecode(credPart);
    } else {
      user = safeDecode(credPart.slice(0, colon));
      password = safeDecode(credPart.slice(colon + 1));
    }
  }

  // hostPart = host[:port][/database]
  const slash = hostPart.indexOf("/");
  const hostPort = slash === -1 ? hostPart : hostPart.slice(0, slash);
  const database = slash === -1 ? "" : hostPart.slice(slash + 1);

  const portColon = hostPort.lastIndexOf(":");
  const host = portColon === -1 ? hostPort : hostPort.slice(0, portColon);
  const port = portColon === -1 ? "" : hostPort.slice(portColon + 1);

  const fields: ConnectionFields = {
    ...EMPTY_FIELDS,
    type,
    host,
    port,
    user,
    password,
    database,
  };

  return parseQuery(fields, query);
}

export function buildConnectionUrl(f: ConnectionFields): string {
  if (f.type === "sqlite") {
    return `sqlite:///${f.filePath}`; // "/tmp/x.db" -> "sqlite:////tmp/x.db"
  }
  const cred =
    f.user.length > 0
      ? `${encodeURIComponent(f.user)}${f.password ? `:${encodeURIComponent(f.password)}` : ""}@`
      : "";
  const port = f.port.length > 0 ? `:${f.port}` : "";
  const db = f.database.length > 0 ? `/${f.database}` : "";
  const base = `${f.type}://${cred}${f.host}${port}${db}`;
  const query = buildAdvancedQuery(f);
  return query.length > 0 ? `${base}?${query.join("&")}` : base;
}
