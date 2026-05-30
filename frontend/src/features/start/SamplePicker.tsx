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
    return <p>Loading samples…</p>;
  }

  return (
    <div>
      {mutation.isError && (
        <p role="alert">{(mutation.error as ApiError).message}</p>
      )}
      <ul>
        {data?.samples.map((s) => (
          <li key={s.name}>
            <button
              onClick={() => mutation.mutate(s.name)}
              disabled={mutation.isPending}
            >
              <span>{s.title}</span>
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
