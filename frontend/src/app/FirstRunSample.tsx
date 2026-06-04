import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient, type QueryClient } from "@tanstack/react-query";
import { ApiError } from "../api/client";
import { listSamples, loadSample, queryKeys } from "../api/endpoints";
import { useMode } from "./ModeProvider";

const SEEN_KEY = "dbsprout.guided.seen";

/** localStorage read, quota/private-mode/SSR safe (mirrors ModeProvider). */
function readSeen(): boolean {
  try {
    return localStorage.getItem(SEEN_KEY) === "true";
  } catch {
    return false;
  }
}

/** Record that the user has entered guided mode at least once. */
function markSeen(): void {
  try {
    localStorage.setItem(SEEN_KEY, "true");
  } catch {
    // Storage unavailable (private mode / quota) — in-memory only.
  }
}

function hasSchemaLoaded(qc: QueryClient): boolean {
  const data = qc.getQueryData(queryKeys.schema) as { tables?: unknown[] } | undefined;
  return !!data?.tables && data.tables.length > 0;
}

/**
 * First-guided-entry Sample offer. Shown ONLY the first time the user enters
 * guided mode (localStorage `dbsprout.guided.seen` unset) AND when no schema is
 * loaded. Non-destructive: it never auto-overwrites a loaded schema — the seed is
 * a single user click, reusing the same `listSamples`/`loadSample` plumbing as
 * the Start tab's SamplePicker. The seen flag is set on the first guided entry,
 * so a reload or a second entry never re-prompts.
 */
export function FirstRunSample() {
  const { mode } = useMode();
  const qc = useQueryClient();
  // Latches true the first time guided mode is entered while the "seen" flag is
  // unset — whether the app starts in guided OR the user toggles advanced→guided.
  // Once latched it stays true for the rest of this mount, so the offer does not
  // vanish mid-interaction; `markSeen()` ensures a future mount/reload never
  // re-triggers it.
  const [firstEntry, setFirstEntry] = useState(() => mode === "guided" && !readSeen());

  useEffect(() => {
    if (mode === "guided" && !readSeen()) {
      setFirstEntry(true);
      markSeen();
    }
  }, [mode]);

  const { data } = useQuery({
    queryKey: queryKeys.samples,
    queryFn: listSamples,
    enabled: firstEntry,
  });

  const mutation = useMutation({
    mutationFn: loadSample,
    onSuccess: () => qc.invalidateQueries({ queryKey: queryKeys.schema }),
  });

  if (mode !== "guided" || !firstEntry) return null;
  if (hasSchemaLoaded(qc)) return null;

  const samples = data?.samples ?? [];
  if (samples.length === 0) return null;
  const first = samples[0];

  return (
    <aside aria-label="Get started with a sample">
      <p>New here? Load a sample database to walk through the steps.</p>
      {mutation.isError && <p role="alert">{(mutation.error as ApiError).message}</p>}
      <button
        type="button"
        onClick={() => mutation.mutate(first.name)}
        disabled={mutation.isPending}
      >
        Load sample: {first.title}
      </button>
    </aside>
  );
}
