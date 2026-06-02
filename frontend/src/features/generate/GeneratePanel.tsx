import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { generate } from "../../api/endpoints";
import type { Engine } from "../../api/types";
import { parseSeed } from "./parseSeed";

const ENGINES: readonly Engine[] = ["heuristic", "spec", "statistical", "finetuned"];

interface GeneratePanelProps {
  /** Called with the job id once a generation job has been accepted. */
  onStarted: (jobId: string) => void;
}

/**
 * Start a generation run: pick an engine + optional seed, then POST /api/generate.
 * On success the returned job id is lifted up so the parent can mount the progress
 * console. A failed start (e.g. 409 single-active, 400 no schema) renders a typed
 * message; the backend has already scrubbed any credentials from it.
 */
export function GeneratePanel({ onStarted }: GeneratePanelProps) {
  const [engine, setEngine] = useState<Engine>("heuristic");
  const [seed, setSeed] = useState("");

  const mutation = useMutation({
    mutationFn: () => generate({ engine, seed: parseSeed(seed) }),
    onSuccess: (res) => onStarted(res.job_id),
  });

  return (
    <div>
      <label>
        engine
        <select
          aria-label="generate engine"
          value={engine}
          onChange={(e) => setEngine(e.target.value as Engine)}
        >
          {ENGINES.map((e) => (
            <option key={e} value={e}>
              {e}
            </option>
          ))}
        </select>
      </label>
      <label>
        seed
        <input
          aria-label="generate seed"
          type="text"
          inputMode="numeric"
          placeholder="auto"
          value={seed}
          onChange={(e) => setSeed(e.target.value)}
        />
      </label>
      <button type="button" disabled={mutation.isPending} onClick={() => mutation.mutate()}>
        Generate
      </button>
      {mutation.isError && (
        <p role="alert">{(mutation.error as ApiError).message}</p>
      )}
    </div>
  );
}
