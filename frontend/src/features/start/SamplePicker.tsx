import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
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

  const qc = useQueryClient();
  const mutation = useMutation({
    mutationFn: loadSample,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.schema });
      onLoaded();
    },
  });

  if (isLoading) {
    return <p className="db-notice-muted">Loading samples…</p>;
  }

  return (
    <div>
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
    </div>
  );
}
