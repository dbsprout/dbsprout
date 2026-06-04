import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { listSamples, loadSample, queryKeys } from "../../api/endpoints";

interface SamplePickerProps {
  onLoaded: () => void;
}

export function SamplePicker({ onLoaded }: SamplePickerProps) {
  const { data, isLoading } = useQuery({
    queryKey: queryKeys.samples,
    queryFn: listSamples,
  });

  // ═══ P5-2 ═══ — title of the most recently loaded sample, used to confirm the
  // load (a visible status) instead of changing the schema silently.
  const [loadedTitle, setLoadedTitle] = useState<string | null>(null);

  const qc = useQueryClient();
  const mutation = useMutation({
    mutationFn: loadSample,
    onSuccess: (_result, name) => {
      const title = data?.samples.find((s) => s.name === name)?.title ?? name;
      setLoadedTitle(title);
      qc.invalidateQueries({ queryKey: queryKeys.schema });
      onLoaded();
    },
  });

  if (isLoading) {
    return <p className="db-notice-muted">Loading samples…</p>;
  }

  return (
    <div>
      {/* ═══ P5-2 ═══ — explain what a sample is for, so the action is not a leap. */}
      <p className="db-notice-muted mb-3">
        A sample loads an example schema so you can try Configure → Generate with no real database.
      </p>
      {mutation.isError && (
        <p role="alert" className="db-notice-alert mb-3">{(mutation.error as ApiError).message}</p>
      )}
      <ul className="flex flex-col gap-2">
        {data?.samples.map((s) => (
          <li key={s.name}>
            <button
              onClick={() => mutation.mutate(s.name)}
              disabled={mutation.isPending}
              className="w-full cursor-pointer rounded-md border border-slate-200 bg-white px-3 py-2 text-left text-sm transition hover:border-accent-400 hover:bg-accent-50 disabled:opacity-50"
            >
              <span className="font-medium text-slate-900">{s.title}</span>
              {" — "}
              <span>{s.description}</span>
              {" ("}
              <span>{s.table_count}</span>
              {" tables, "}
              <span>{s.dialect}</span>
              {")"}
            </button>
          </li>
        ))}
      </ul>
      {/* ═══ P5-2 ═══ — confirm the load explicitly and point the way forward. */}
      {loadedTitle && (
        <p role="status" className="db-notice-status mt-3">
          Loaded the {loadedTitle} sample — head to Configure &amp; Generate.
        </p>
      )}
    </div>
  );
}
