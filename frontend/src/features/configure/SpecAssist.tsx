import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { assistSpec, queryKeys } from "../../api/endpoints";
import type { SpecProvider } from "../../api/types";

/**
 * AI spec-assist (P2b-3): a provider picker + an "AI assist" button that asks an
 * LLM to propose a full DataSpec for the loaded schema. The default provider is
 * the offline "embedded" path; "cloud" is opt-in. On success the server stores
 * the proposal on the workspace, so we invalidate the `spec` query to repaint
 * the configure grid. Failures (provider unavailable, no schema) surface as a
 * typed alert message rather than a crash.
 *
 * P4-11 — cloud key-entry UX. Selecting "cloud" reveals a small sub-panel that
 * lets the user pick a *model* and name the *environment variable* that holds
 * the provider API key. The key VALUE is never collected, sent, or stored: only
 * the env-var name + model (both non-secret) ride on the request, and the server
 * reads the actual key from its own process environment. A missing key degrades
 * to the typed `LLM_UNAVAILABLE` 503, which we render as an actionable panel
 * (message + server `hint`) rather than a bare error.
 */
// P4-11 defaults — the common OpenAI case needs no further typing.
const DEFAULT_CLOUD_MODEL = "gpt-4o-mini";
const DEFAULT_API_KEY_ENV = "OPENAI_API_KEY";

export function SpecAssist() {
  const qc = useQueryClient();
  const [provider, setProvider] = useState<SpecProvider>("embedded");
  // ─── P4-11 ─── session-only, in-component state. NEVER persisted to
  // localStorage / the query cache, and only ever holds an env-var *name* (not
  // a key value).
  const [model, setModel] = useState(DEFAULT_CLOUD_MODEL);
  const [apiKeyEnv, setApiKeyEnv] = useState(DEFAULT_API_KEY_ENV);
  const isCloud = provider === "cloud";

  const mutation = useMutation({
    mutationFn: () =>
      assistSpec(
        provider,
        isCloud ? { model: model.trim(), api_key_env: apiKeyEnv.trim() } : undefined,
      ),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.spec }),
  });

  const error = mutation.isError ? (mutation.error as ApiError) : null;

  return (
    <section aria-label="AI spec assist">
      <label>
        provider
        <select
          aria-label="spec-assist provider"
          value={provider}
          onChange={(e) => setProvider(e.target.value as SpecProvider)}
          disabled={mutation.isPending}
        >
          <option value="embedded">Embedded (offline)</option>
          <option value="cloud">Cloud (opt-in)</option>
        </select>
      </label>

      {/* ─── P4-11 ─── cloud key-entry sub-panel (revealed only for cloud). */}
      {isCloud && (
        <fieldset aria-label="cloud provider settings">
          <label>
            cloud model
            <input
              aria-label="cloud model"
              type="text"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={mutation.isPending}
            />
          </label>
          <label>
            API key env var
            <input
              aria-label="api key env var"
              type="text"
              value={apiKeyEnv}
              onChange={(e) => setApiKeyEnv(e.target.value)}
              disabled={mutation.isPending}
            />
          </label>
          <p>
            The API key is read from this environment variable on the dbsprout
            server. Its value is never sent from the browser or stored.
          </p>
        </fieldset>
      )}

      <button
        type="button"
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending}
      >
        AI assist
      </button>
      {mutation.isPending && <span>Proposing…</span>}
      {error && (
        <div role="alert">
          <p>{error.message}</p>
          {error.hint && <p>{error.hint}</p>}
        </div>
      )}
    </section>
  );
}
