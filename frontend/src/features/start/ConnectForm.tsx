import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import {
  buildConnectionUrl,
  defaultPort,
  parseConnectionUrl,
  SSL_MODES,
  type ConnectionFields,
  type ConnectionParam,
  type DbType,
} from "../../api/connectionUrl";
import { connect, connectTest, queryKeys } from "../../api/endpoints";
import type { ConnectionProbe, SshTunnelInput } from "../../api/types";

// ─── P2a-3 ───
// SSH-tunnel bastion fields, held as component state (the SSH block is NOT part
// of the connection URL — it travels alongside it). The key is referenced by
// path only; nothing is uploaded. An ssh block is sent only when a bastion host
// is filled in.
interface SshFields {
  host: string;
  port: string;
  user: string;
  keyPath: string;
}

const DEFAULT_SSH: SshFields = { host: "", port: "", user: "", keyPath: "" };

/** Build the optional SSH block; `undefined` (no tunnel) when no bastion host. */
function buildSsh(f: SshFields): SshTunnelInput | undefined {
  if (f.host.trim().length === 0) {
    return undefined;
  }
  const ssh: SshTunnelInput = { host: f.host.trim(), user: f.user.trim(), key_path: f.keyPath.trim() };
  const port = Number.parseInt(f.port, 10);
  if (Number.isFinite(port) && port > 0) {
    return { ...ssh, port };
  }
  return ssh;
}
// ─── end P2a-3 ───

// ─── P4-10: SSH incomplete-block inline validation ───
// Validate the SSH block at the boundary: a fully-empty block means "no tunnel"
// and is allowed, but the moment ANY field is set the block must be complete —
// host + user + key path are all required (port is optional, defaults to 22
// server-side). This catches a partial block (which the backend would 422) with
// inline errors before submit. Pure function → unit-testable, no side effects.

/** Per-field SSH error messages; only present for required fields left blank. */
interface SshErrors {
  host?: string;
  user?: string;
  keyPath?: string;
}

const SSH_FIELD_KEYS: (keyof SshFields)[] = ["host", "port", "user", "keyPath"];

/** True when at least one SSH field has a non-blank value. */
function isSshTouched(f: SshFields): boolean {
  return SSH_FIELD_KEYS.some((k) => f[k].trim().length > 0);
}

/**
 * Required-field errors for a touched SSH block. A fully-empty block yields no
 * errors (no tunnel); otherwise host/user/key_path must each be non-blank.
 */
function sshErrors(f: SshFields): SshErrors {
  if (!isSshTouched(f)) {
    return {};
  }
  const errors: SshErrors = {};
  if (f.host.trim().length === 0) {
    errors.host = "SSH bastion host is required when configuring a tunnel";
  }
  if (f.user.trim().length === 0) {
    errors.user = "SSH user is required when configuring a tunnel";
  }
  if (f.keyPath.trim().length === 0) {
    errors.keyPath = "SSH key path is required when configuring a tunnel";
  }
  return errors;
}
// ─── end P4-10 ───

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
  sslMode: "",
  sslCa: "",
  sslCert: "",
  sslKey: "",
  schema: "",
  connectTimeout: "",
  params: [],
};

/** Parse a `key=value` per-line textarea into params. Blank lines skipped; first `=` splits. */
function parseParams(text: string): ConnectionParam[] {
  const out: ConnectionParam[] = [];
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (line.length === 0) {
      continue;
    }
    const eq = line.indexOf("=");
    if (eq === -1) {
      out.push({ key: line, value: "" });
    } else {
      out.push({ key: line.slice(0, eq).trim(), value: line.slice(eq + 1).trim() });
    }
  }
  return out;
}

/** Serialize parsed params back into the textarea's `key=value` per-line form. */
function paramsToText(params: ConnectionParam[]): string {
  return params.map((p) => (p.value.length > 0 ? `${p.key}=${p.value}` : p.key)).join("\n");
}

/** String-valued keys of ConnectionFields (everything except `type` and `params`). */
type StringFieldKey = Exclude<keyof ConnectionFields, "type" | "params">;

export function ConnectForm({ onLoaded }: ConnectFormProps) {
  const [fields, setFields] = useState<ConnectionFields>(DEFAULT_FIELDS);
  const [url, setUrl] = useState<string>(buildConnectionUrl(DEFAULT_FIELDS));
  const [paramsText, setParamsText] = useState<string>("");
  // ─── P2a-3 ───
  const [ssh, setSsh] = useState<SshFields>(DEFAULT_SSH);

  function handleSshChange(key: keyof SshFields, value: string) {
    setSsh((prev) => ({ ...prev, [key]: value }));
  }
  // ─── end P2a-3 ───

  const qc = useQueryClient();

  const testM = useMutation({
    mutationFn: () => connectTest(url, buildSsh(ssh)),
  });

  const connectM = useMutation({
    mutationFn: () => connect(url, buildSsh(ssh)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.schema });
      onLoaded();
    },
  });

  // ─── P4-10 ─── Inline SSH validation: block submit while the block is partial.
  const sshFieldErrors = sshErrors(ssh);
  const sshIncomplete = Object.keys(sshFieldErrors).length > 0;
  // ─── end P4-10 ───

  const isPending = testM.isPending || connectM.isPending;
  const submitBlocked = isPending || sshIncomplete;

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

  function handleFieldChange(key: StringFieldKey, value: string) {
    const next: ConnectionFields = { ...fields, [key]: value };
    updateFields(next);
  }

  function handleParamsChange(text: string) {
    setParamsText(text);
    updateFields({ ...fields, params: parseParams(text) });
  }

  // ─── P4-8 ───
  // Pasting / editing the URL reverse-populates the structured form (incl. the
  // Advanced section). The raw URL is kept verbatim in state — so what you paste is
  // exactly what is sent on Test/Connect — while `fields` and the params textarea are
  // derived from it via `parseConnectionUrl` (the inverse of `buildConnectionUrl`).
  function handleUrlChange(raw: string) {
    setUrl(raw);
    const parsed = parseConnectionUrl(raw);
    setFields(parsed);
    setParamsText(paramsToText(parsed.params));
  }
  // ─── end P4-8 ───

  function renderProbeResult(data: ConnectionProbe) {
    return (
      <p className="db-notice-status">
        {`Connected · ${data.dialect} ${data.server_version} · ${data.table_count} tables · ${data.latency_ms} ms`}
      </p>
    );
  }

  return (
    <div className="flex flex-col gap-3">
      <div className="db-field">
        <label htmlFor="db-type" className="db-label">Database type</label>
        <select
          id="db-type"
          className="db-input"
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
          <label htmlFor="file-path" className="db-label">File path</label>
          <input
            className="db-input"
            id="file-path"
            type="text"
            value={fields.filePath}
            onChange={(e) => handleFieldChange("filePath", e.target.value)}
          />
        </div>
      ) : (
        <>
          <div>
            <label htmlFor="host" className="db-label">Host</label>
            <input
              className="db-input"
              id="host"
              type="text"
              value={fields.host}
              onChange={(e) => handleFieldChange("host", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="port" className="db-label">Port</label>
            <input
              className="db-input"
              id="port"
              type="text"
              value={fields.port}
              onChange={(e) => handleFieldChange("port", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="user" className="db-label">User</label>
            <input
              className="db-input"
              id="user"
              type="text"
              value={fields.user}
              onChange={(e) => handleFieldChange("user", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="password" className="db-label">Password</label>
            <input
              className="db-input"
              id="password"
              type="password"
              value={fields.password}
              onChange={(e) => handleFieldChange("password", e.target.value)}
            />
          </div>
          <div>
            <label htmlFor="database" className="db-label">Database</label>
            <input
              className="db-input"
              id="database"
              type="text"
              value={fields.database}
              onChange={(e) => handleFieldChange("database", e.target.value)}
            />
          </div>

          <details className="rounded-md border border-slate-200 p-3">
            <summary className="cursor-pointer text-sm font-medium text-slate-700">Advanced</summary>
            <div className="mt-3 flex flex-col gap-3">
              <label htmlFor="ssl-mode" className="db-label">SSL mode</label>
              <select
                className="db-input"
                id="ssl-mode"
                value={fields.sslMode}
                onChange={(e) => handleFieldChange("sslMode", e.target.value)}
              >
                <option value="">(default)</option>
                {SSL_MODES.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label htmlFor="ssl-ca" className="db-label">CA certificate path</label>
              <input
                className="db-input"
                id="ssl-ca"
                type="text"
                value={fields.sslCa}
                onChange={(e) => handleFieldChange("sslCa", e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="ssl-cert" className="db-label">Client certificate path</label>
              <input
                className="db-input"
                id="ssl-cert"
                type="text"
                value={fields.sslCert}
                onChange={(e) => handleFieldChange("sslCert", e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="ssl-key" className="db-label">Client key path</label>
              <input
                className="db-input"
                id="ssl-key"
                type="text"
                value={fields.sslKey}
                onChange={(e) => handleFieldChange("sslKey", e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="schema" className="db-label">Schema / search_path</label>
              <input
                className="db-input"
                id="schema"
                type="text"
                value={fields.schema}
                onChange={(e) => handleFieldChange("schema", e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="connect-timeout" className="db-label">Connect timeout (s)</label>
              <input
                className="db-input"
                id="connect-timeout"
                type="text"
                inputMode="numeric"
                value={fields.connectTimeout}
                onChange={(e) => handleFieldChange("connectTimeout", e.target.value)}
              />
            </div>
            <div>
              <label htmlFor="extra-params" className="db-label">Extra parameters (one key=value per line)</label>
              <textarea
                className="db-input"
                id="extra-params"
                value={paramsText}
                onChange={(e) => handleParamsChange(e.target.value)}
              />
            </div>
            {/* ═══ P2a-3 ═══ SSH tunnel (bastion). Sent only when a host is set. */}
            <fieldset className="db-fieldset flex flex-col gap-3">
              <legend className="db-legend">SSH tunnel (optional)</legend>
              <div>
                <label htmlFor="ssh-host" className="db-label">SSH bastion host</label>
                <input
                  className="db-input"
                  id="ssh-host"
                  type="text"
                  value={ssh.host}
                  aria-invalid={sshFieldErrors.host ? true : undefined}
                  aria-describedby={sshFieldErrors.host ? "ssh-host-error" : undefined}
                  onChange={(e) => handleSshChange("host", e.target.value)}
                />
                {sshFieldErrors.host && (
                  <p id="ssh-host-error" role="alert" className="mt-1 text-xs text-red-600">
                    {sshFieldErrors.host}
                  </p>
                )}
              </div>
              <div>
                <label htmlFor="ssh-port" className="db-label">SSH bastion port</label>
                <input
                  className="db-input"
                  id="ssh-port"
                  type="text"
                  inputMode="numeric"
                  placeholder="22"
                  value={ssh.port}
                  onChange={(e) => handleSshChange("port", e.target.value)}
                />
              </div>
              <div>
                <label htmlFor="ssh-user" className="db-label">SSH user</label>
                <input
                  className="db-input"
                  id="ssh-user"
                  type="text"
                  value={ssh.user}
                  aria-invalid={sshFieldErrors.user ? true : undefined}
                  aria-describedby={sshFieldErrors.user ? "ssh-user-error" : undefined}
                  onChange={(e) => handleSshChange("user", e.target.value)}
                />
                {sshFieldErrors.user && (
                  <p id="ssh-user-error" role="alert" className="mt-1 text-xs text-red-600">
                    {sshFieldErrors.user}
                  </p>
                )}
              </div>
              <div>
                <label htmlFor="ssh-key-path" className="db-label">SSH key path</label>
                <input
                  className="db-input"
                  id="ssh-key-path"
                  type="text"
                  value={ssh.keyPath}
                  aria-invalid={sshFieldErrors.keyPath ? true : undefined}
                  aria-describedby={sshFieldErrors.keyPath ? "ssh-key-path-error" : undefined}
                  onChange={(e) => handleSshChange("keyPath", e.target.value)}
                />
                {sshFieldErrors.keyPath && (
                  <p id="ssh-key-path-error" role="alert" className="mt-1 text-xs text-red-600">
                    {sshFieldErrors.keyPath}
                  </p>
                )}
              </div>
            </fieldset>
            {/* ═══ end P2a-3 ═══ */}
          </details>
        </>
      )}

      <div className="db-field">
        <label htmlFor="connection-url" className="db-label">Connection URL</label>
        <input
          className="db-input font-mono"
          id="connection-url"
          type="text"
          aria-label="Connection URL"
          value={url}
          onChange={(e) => handleUrlChange(e.target.value)}
        />
      </div>

      {testM.data && renderProbeResult(testM.data)}
      {testM.isError && (
        <p role="alert" className="db-notice-alert">
          {(testM.error as ApiError).message}
          {(testM.error as ApiError).hint && (
            <span> — {(testM.error as ApiError).hint}</span>
          )}
        </p>
      )}
      {connectM.isError && (
        <p role="alert" className="db-notice-alert">
          {(connectM.error as ApiError).message}
          {(connectM.error as ApiError).hint && (
            <span> — {(connectM.error as ApiError).hint}</span>
          )}
        </p>
      )}

      <div className="flex gap-2">
        <button
          type="button"
          className="db-btn-secondary"
          onClick={() => testM.mutate()}
          disabled={submitBlocked}
        >
          Test Connection
        </button>
        <button
          type="button"
          className="db-btn-primary"
          onClick={() => connectM.mutate()}
          disabled={submitBlocked}
        >
          Connect &amp; continue
        </button>
      </div>
    </div>
  );
}
