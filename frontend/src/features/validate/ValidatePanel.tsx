import { useMutation } from "@tanstack/react-query";
import { ApiError } from "../../api/client";
import { validate } from "../../api/endpoints";
import type {
  DetectionReport,
  FidelityReport,
  IntegrityDetail,
  ValidateResponse,
} from "../../api/types";
import { integritySummary } from "./integritySummary";

/** Where a violation points; the seam the Configure grid can later focus on. */
export interface DrillTarget {
  table: string;
  column: string | null;
}

interface ValidatePanelProps {
  /**
   * Invoked when the user drills into a violation row. Thin seam: the SPA has no
   * shared cross-panel selection store yet, so App.tsx wires a no-op stub. A
   * future story can route this to focus the Configure ColumnGrid on the cell.
   */
  onDrill?: (target: DrillTarget) => void;
}

/**
 * The Validate surface: run POST /api/validate against the last generation run
 * and render an integrity summary (per-check pass/fail), per-violation rows with
 * a drill affordance, and — only when present — fidelity + detection sections.
 * A typed error (e.g. 409 NO_RUN when nothing has been generated) surfaces as a
 * scrubbed alert message.
 */
export function ValidatePanel({ onDrill }: ValidatePanelProps) {
  const mutation = useMutation<ValidateResponse>({ mutationFn: () => validate() });
  const report = mutation.data;

  return (
    <div>
      <button type="button" disabled={mutation.isPending} onClick={() => mutation.mutate()}>
        Validate
      </button>

      {mutation.isError && <p role="alert">{(mutation.error as ApiError).message}</p>}

      {report && (
        <div>
          <IntegrityBlock report={report} />
          <ViolationRows details={report.details} onDrill={onDrill} />
          {report.fidelity && <FidelityBlock report={report.fidelity} />}
          {report.detection && <DetectionBlock report={report.detection} />}
        </div>
      )}
    </div>
  );
}

function IntegrityBlock({ report }: { report: ValidateResponse }) {
  const summary = integritySummary(report.by_table);
  return (
    <section>
      <h3>Integrity</h3>
      <p>
        {report.summary.violations === 0
          ? "No violations — every check passed."
          : `${report.summary.violations} violation${report.summary.violations === 1 ? "" : "s"} across ${report.summary.tables} tables.`}
      </p>
      <ul>
        {summary.map((row) => (
          <li key={row.check}>
            {row.label}: {row.passed ? "pass" : `fail (${row.count})`}
          </li>
        ))}
      </ul>
    </section>
  );
}

function ViolationRows({
  details,
  onDrill,
}: {
  details: IntegrityDetail[];
  onDrill?: (target: DrillTarget) => void;
}) {
  if (details.length === 0) return null;
  return (
    <section>
      <h3>Violations</h3>
      <ul>
        {details.map((d, i) => (
          <li key={`${d.table}.${d.column}.${d.check}.${i}`}>
            {d.check} · {d.table}
            {d.column ? `.${d.column}` : ""} — {d.details}
            <button
              type="button"
              onClick={() => onDrill?.({ table: d.table, column: d.column })}
            >
              Drill to cell
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

function FidelityBlock({ report }: { report: FidelityReport }) {
  return (
    <section>
      <h3>Fidelity</h3>
      <p>
        Score: {report.overall_score} — {report.passed ? "pass" : "fail"}
      </p>
      <ul>
        {report.metrics.map((m, i) => (
          <li key={`${m.metric}.${m.table}.${m.column}.${i}`}>
            {m.metric} · {m.table}
            {m.column ? `.${m.column}` : ""}: {m.score}
          </li>
        ))}
      </ul>
    </section>
  );
}

function DetectionBlock({ report }: { report: DetectionReport }) {
  return (
    <section>
      <h3>Detection</h3>
      <p>
        Score: {report.overall_score} — {report.passed ? "pass" : "fail"}
      </p>
      <ul>
        {report.metrics.map((m, i) => (
          <li key={`${m.metric}.${m.table}.${i}`}>
            {m.metric} · {m.table}: {m.accuracy}
          </li>
        ))}
      </ul>
    </section>
  );
}
