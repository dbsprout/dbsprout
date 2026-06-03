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
    <section aria-label="Saved connections">
      <h3>Saved connections</h3>

      {error && <p role="alert">{error.message}</p>}

      {isLoading ? (
        <p>Loading saved connections…</p>
      ) : data && data.connections.length > 0 ? (
        <ul>
          {data.connections.map((c) => (
            <li key={c.name}>
              <span>{c.name}</span>{" "}
              <code>{c.url}</code>{" "}
              <button
                type="button"
                onClick={() => onLoad(c.url)}
                disabled={isPending}
              >
                Load
              </button>{" "}
              <button
                type="button"
                onClick={() => deleteM.mutate(c.name)}
                disabled={isPending}
              >
                Delete
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p>No saved connections yet.</p>
      )}

      <div>
        <label htmlFor="saved-name">Name</label>
        <input
          id="saved-name"
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        <label htmlFor="saved-url">Connection URL</label>
        <input
          id="saved-url"
          type="text"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
        />
        <button type="button" onClick={() => saveM.mutate()} disabled={!canSave}>
          Save connection
        </button>
      </div>
    </section>
  );
}
