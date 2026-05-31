export type DbType = "postgresql" | "mysql" | "sqlite" | "mssql";

export interface ConnectionFields {
  type: DbType;
  host: string;
  port: string;
  user: string;
  password: string;
  database: string;
  filePath: string;
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
  return `${f.type}://${cred}${f.host}${port}${db}`;
}
