import { useState } from "react";
import { GeneratePanel } from "./GeneratePanel";
import { ProgressConsole } from "./ProgressConsole";
import { ResultSummary } from "./ResultSummary";

interface GenerateSurfaceProps {
  /** Poll interval forwarded to the console (kept short in tests). */
  pollMs?: number;
}

/**
 * The Generate surface: pick engine + seed and start a run (GeneratePanel),
 * watch it poll to a terminal state (ProgressConsole), and — once it succeeds —
 * show the per-table row-count summary (ResultSummary). The console and summary
 * are only mounted after a job has started / succeeded so nothing polls early.
 */
export function GenerateSurface({ pollMs }: GenerateSurfaceProps) {
  const [jobId, setJobId] = useState<string | null>(null);
  const [succeeded, setSucceeded] = useState(false);

  function handleStarted(id: string) {
    setSucceeded(false);
    setJobId(id);
  }

  return (
    <div className="flex flex-col gap-4">
      <GeneratePanel onStarted={handleStarted} />
      {jobId && (
        <ProgressConsole
          jobId={jobId}
          pollMs={pollMs}
          onSucceeded={() => setSucceeded(true)}
        />
      )}
      {succeeded && jobId && <ResultSummary jobId={jobId} />}
    </div>
  );
}
