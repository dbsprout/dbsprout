import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import {
  deleteConnection,
  getConnections,
  queryKeys,
  saveConnection,
} from "../../api/endpoints";

interface SavedConnectionsProps {
  /** Emitted with the stored (password-stripped / ${ENV_VAR}) URL on load. */
  onLoad: (url: string) => void;
}

/**
 * P2a-2: list / save / delete reusable connections persisted to
 * `.dbsprout/connections.toml`. Loading a saved connection emits its stored URL
 * via `onLoad` so the parent (StartPanel) can hand it to the connect flow.
 * Passwords are never stored — the server strips them before persisting.
 */
export function SavedConnections({ onLoad }: SavedConnectionsProps) {
  const [name, setName] = useState("");
  const [url, setUrl] = useState("");

  const qc = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: queryKeys.connections,
    queryFn: getConnections,
  });

  const invalidate = () =>
    qc.invalidateQueries({ queryKey: queryKeys.connections });

  const saveM = useMutation({
    mutationFn: () => saveConnection(name.trim(), url.trim()),
    onSuccess: () => {
      setName("");
      setUrl("");
      invalidate();
    },
  });

  const deleteM = useMutation({
    mutationFn: (target: string) => deleteConnection(target),
    onSuccess: invalidate,
  });

  const isPending = saveM.isPending || deleteM.isPending;
  const canSave = name.trim().length > 0 && url.trim().length > 0 && !isPending;

  const error = (saveM.error ?? deleteM.error) as ApiError | null;

  return (
    <section aria-label="Saved connections" className="db-subsection">
      <h3 className="db-subsection-title">Saved connections</h3>

      {error && <p role="alert" className="db-notice-alert mb-3">{error.message}</p>}

      {isLoading ? (
        <p className="db-notice-muted">Loading saved connections…</p>
      ) : data && data.connections.length > 0 ? (
        <ul className="mb-3 flex flex-col gap-2">
          {data.connections.map((c) => (
            <li
              key={c.name}
              className="flex flex-wrap items-center gap-2 rounded-md border border-slate-200 bg-white px-3 py-2 text-sm"
            >
              <span className="font-medium text-slate-900">{c.name}</span>{" "}
              <code className="font-mono text-xs text-slate-500">{c.url}</code>{" "}
              <button
                type="button"
                className="db-btn-secondary ml-auto"
                onClick={() => onLoad(c.url)}
                disabled={isPending}
              >
                Load
              </button>{" "}
              <button
                type="button"
                className="db-btn-danger"
                onClick={() => deleteM.mutate(c.name)}
                disabled={isPending}
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="db-notice-muted mb-3">No saved connections yet.</p>
      )}

      <div className="flex flex-wrap items-end gap-2">
        <div className="db-field mb-0">
          <label htmlFor="saved-name" className="db-label">Name</label>
          <input
            id="saved-name"
            type="text"
            className="db-input"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </div>
        <div className="db-field mb-0 flex-1">
          <label htmlFor="saved-url" className="db-label">Connection URL</label>
          <input
            id="saved-url"
            type="text"
            className="db-input font-mono"
            value={url}
            onChange={(e) => setUrl(e.target.value)}
          />
        </div>
        <button type="button" className="db-btn-primary" onClick={() => saveM.mutate()} disabled={!canSave}>
          Save connection
        </button>
      </div>
    </section>
  );
}
