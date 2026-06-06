import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { uploadSchema } from "../../api/endpoints";
import { invalidateSchemaQueries } from "../../api/invalidateSchema";

interface UploadFormProps {
  onLoaded: () => void;
}

export function UploadForm({ onLoaded }: UploadFormProps) {
  const [file, setFile] = useState<File | null>(null);
  const qc = useQueryClient();

  const mutation = useMutation({
    mutationFn: (f: File) => uploadSchema(f),
    // ═══ P5-12 ═══ — refresh spec + preview alongside schema.
    onSuccess: () => {
      invalidateSchemaQueries(qc);
      onLoaded();
    },
  });

  return (
    <div className="flex flex-col gap-3">
      <div className="db-field">
        <label htmlFor="schema-file" className="db-label">Schema file</label>
        <input
          id="schema-file"
          type="file"
          className="text-sm text-slate-600 file:mr-3 file:rounded-md file:border-0 file:bg-accent-600 file:px-3 file:py-1.5 file:text-sm file:font-medium file:text-white hover:file:bg-accent-700"
          onChange={(e) => setFile(e.target.files?.[0] ?? null)}
        />
      </div>
      {mutation.isError && (
        <p role="alert" className="db-notice-alert">{(mutation.error as ApiError).message}</p>
      )}
      <button
        type="button"
        className="db-btn-primary self-start"
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
