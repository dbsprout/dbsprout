import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { queryKeys, uploadSchema } from "../../api/endpoints";

interface UploadFormProps {
  onLoaded: () => void;
}

export function UploadForm({ onLoaded }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const qc = useQueryClient();

  const mutation = useMutation({
    mutationFn: (f: File) => uploadSchema(f),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.schema });
      onLoaded();
    },
  });

  return (
    <div>
      <div>
        <label htmlFor="schema-file">Schema file</label>
        <input
          id="schema-file"
          type="file"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>
      {mutation.isError && (
        <p role="alert">{(mutation.error as ApiError).message}</p>
      )}
      <button
        type="button"
        onClick={() => {
          if (file) mutation.mutate(file);
        }}
        disabled={mutation.isPending || file === null}
      >
        Load file
      </button>
    </div>
  );
}
