import { useState } from "react";
import { CostsPanel } from "./CostsPanel";
import { QualityPanel } from "./QualityPanel";
import { RunsPanel } from "./RunsPanel";

/**
 * The "Runs & Quality" surface: the paginated run history (RunsPanel), the
 * pass/fail/warn quality table for the selected (or latest) run (QualityPanel),
 * and the LLM cost summary (CostsPanel). Selecting a row in the history focuses
 * the quality table on that run; with nothing selected it shows the latest run.
 */
export function RunsSurface() {
  const [selectedRun, setSelectedRun] = useState<number | undefined>(undefined);

  return (
    <div>
      <RunsPanel onSelectRun={setSelectedRun} />
      <QualityPanel runId={selectedRun} />
      <CostsPanel />
    </div>
  );
}
