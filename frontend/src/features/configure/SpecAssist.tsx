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
 */
export function SpecAssist() {
  const qc = useQueryClient();
  const [provider, setProvider] = useState<SpecProvider>("embedded");

  const mutation = useMutation({
    mutationFn: () => assistSpec(provider),
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.spec }),
  });

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
      <button
        type="button"
        onClick={() => mutation.mutate()}
        disabled={mutation.isPending}
      >
        AI assist
      </button>
      {mutation.isPending && <span>Proposing…</span>}
      {mutation.isError && (
        <p role="alert">{(mutation.error as ApiError).message}</p>
      )}
    </section>
  );
}
