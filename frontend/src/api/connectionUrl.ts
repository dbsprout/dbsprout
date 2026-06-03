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
