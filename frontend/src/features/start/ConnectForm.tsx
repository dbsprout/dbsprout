import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import {
  buildConnectionUrl,
  defaultPort,
  type ConnectionFields,
  type DbType,
} from "../../api/connectionUrl";
import { connect, connectTest, queryKeys } from "../../api/endpoints";
import type { ConnectionProbe } from "../../api/types";

interface ConnectFormProps {
  onLoaded: () => void;
}

const DB_TYPES: DbType[] = ["postgresql", "mysql", "sqlite", "mssql"];

const DEFAULT_FIELDS: ConnectionFields = {
  type: "postgresql",
  host: "localhost",
  port: "5432",
  user: "",
  password: "",
  database: "",
  filePath: "",
};

export function ConnectForm({ onLoaded }: ConnectFormProps) {
  const [fields, setFields] = useState<ConnectionFields>(DEFAULT_FIELDS);
  const [url, setUrl] = useState<string>(buildConnectionUrl(DEFAULT_FIELDS));

  const qc = useQueryClient();

  const testM = useMutation({
    mutationFn: () => connectTest(url),
  });

  const connectM = useMutation({
    mutationFn: () => connect(url),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.schema });
      onLoaded();
    },
  });

  const isPending = testM.isPending || connectM.isPending;

  function updateFields(next: ConnectionFields) {
    setFields(next);
    setUrl(buildConnectionUrl(next));
  }

  function handleTypeChange(type: DbType) {
    const next: ConnectionFields = {
      ...fields,
      type,
      port: defaultPort(type),
    };
    updateFields(next);
  }

  function handleFieldChange(key: keyof ConnectionFields, value: string) {
    const next: ConnectionFields = { ...fields, [key]: value };
    updateFields(next);
  }

  function renderProbeResult(data: ConnectionProbe) {
    return (
      <p>
        {`Connected · ${data.dialect} ${data.server_version} · ${data.table_count} tables · ${data.latency_ms} ms`}
      </p>
    );
  }

  return (
    <div>
      <div>
        <label htmlFor="db-type">Database type</label>
        <select
          id="db-type"
          value={fields.type}
          onChange={(e) => handleTypeChange(e.target.value as DbType)}
        >
          {DB_TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
      </div>

      {fields.type === "sqlite" ? (
        <div>
          <label htmlFor="file-path">File path</label>
          <input
            id="file-path"
            type="text"
            value={fields.filePath}
            onChange={(e) => handleFieldChange("filePath", e.target.value)}
          />
        </div>
      ) : (
        <>
          <div>
            <label htmlFor="host">Host</label>
            <input
              id="host"
              type="text"
              value={fields.host}
              onChange={(e) => handleFieldChange("host", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="port">Port</label>
            <input
              id="port"
              type="text"
              value={fields.port}
              onChange={(e) => handleFieldChange("port", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="user">User</label>
            <input
              id="user"
              type="text"
              value={fields.user}
              onChange={(e) => handleFieldChange("user", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="password">Password</label>
            <input
              id="password"
              type="password"
              value={fields.password}
              onChange={(e) => handleFieldChange("password", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="database">Database</label>
            <input
              id="database"
              type="text"
              value={fields.database}
              onChange={(e) => handleFieldChange("database", e.target.value)}
            />
          </div>
        </>
      )}

      <div>
        <label htmlFor="connection-url">Connection URL</label>
        <input
          id="connection-url"
          type="text"
          aria-label="Connection URL"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
      </div>

      {testM.data && renderProbeResult(testM.data)}
      {testM.isError && (
        <p role="alert">
          {(testM.error as ApiError).message}
          {(testM.error as ApiError).hint && (
            <span> — {(testM.error as ApiError).hint}</span>
          )}
        </p>
      )}
      {connectM.isError && (
        <p role="alert">
          {(connectM.error as ApiError).message}
          {(connectM.error as ApiError).hint && (
            <span> — {(connectM.error as ApiError).hint}</span>
          )}
        </p>
      )}

      <div>
        <button
          type="button"
          onClick={() => testM.mutate()}
          disabled={isPending}
        >
          Test Connection
        </button>
        <button
          type="button"
          onClick={() => connectM.mutate()}
          disabled={isPending}
        >
          Connect &amp; continue
        </button>
      </div>
    </div>
  );
}
