import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { pasteSchema, queryKeys } from "../../api/endpoints";

interface PasteFormProps {
  onLoaded: () => void;
}

export function PasteForm({ onLoaded }: PasteFormProps) {
  const [text, setText] = useState<string>("");
  const qc = useQueryClient();

  const mutation = useMutation({
    mutationFn: (t: string) => pasteSchema(t),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.schema });
      onLoaded();
    },
  });

  return (
    <div className="flex flex-col gap-3">
      <label htmlFor="paste-schema" className="db-label">Paste schema</label>
      <textarea
        id="paste-schema"
        className="db-input font-mono"
        value={text}
        onChange={(e) => setText(e.target.value)}
        rows={10}
      />
      {mutation.isError && (
        <p role="alert" className="db-notice-alert">{(mutation.error as ApiError).message}</p>
      )}
      <button
        type="button"
        className="db-btn-primary self-start"
        onClick={() => mutation.mutate(text)}
        disabled={mutation.isPending || text.trim().length === 0}
      >
        Load pasted schema
      </button>
    </div>
  );
}
