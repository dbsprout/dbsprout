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
import { drillReason, violationHelp } from "./violationHelp";

/** Where a violation points; the seam the Configure grid can later focus on. */
export interface DrillTarget {
  table: string;
  column: string | null;
  /** P5-7: a short cross-panel reason shown in the Configure drill notice. */
  reason?: string;
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
    <div className="flex flex-col gap-3">
      <button type="button" className="db-btn-primary self-start" disabled={mutation.isPending} onClick={() => mutation.mutate()}>
        Validate
      </button>

      {mutation.isError && <p role="alert" className="db-notice-alert">{(mutation.error as ApiError).message}</p>}

      {report && (
        <div className="flex flex-col gap-4">
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
    <section className="db-subsection">
      <h3 className="db-subsection-title">Integrity</h3>
      <p className="mb-2 text-sm text-slate-600">
        {report.summary.violations === 0
          ? "No violations — every check passed."
          : `${report.summary.violations} violation${report.summary.violations === 1 ? "" : "s"} across ${report.summary.tables} tables.`}
      </p>
      <ul className="flex flex-col gap-1">
        {summary.map((row) => (
          <li
            key={row.check}
            className="flex flex-wrap items-center justify-between gap-x-2 gap-y-1 rounded-md border border-slate-200 bg-white px-3 py-1.5 text-sm"
          >
            <span className="text-slate-700">{row.label}</span>
            <span className={row.passed ? "db-pill-pass" : "db-pill-fail"}>
              {row.passed ? "pass" : `fail (${row.count})`}
            </span>
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
    <section className="db-subsection">
      <h3 className="db-subsection-title">Violations</h3>
      {/* ─── P5-7 ─── cheap actionable hint: most violations clear on a re-seed. */}
      <p className="mb-2 text-xs text-slate-500">
        Tip: most violations clear by re-generating on the Generate step with a different seed.
      </p>
      <ul className="flex flex-col gap-1">
        {details.map((d, i) => {
          const help = violationHelp(d.check);
          return (
            <li
              key={`${d.table}.${d.column}.${d.check}.${i}`}
              className="flex flex-wrap items-center gap-2 rounded-md border border-red-200 bg-red-50 px-3 py-1.5 text-sm text-red-800"
            >
              <span className="font-mono">
                {d.check} · {d.table}
                {d.column ? `.${d.column}` : ""} — {d.details}
              </span>
              <button
                type="button"
                className="db-btn-secondary ml-auto"
                onClick={() => onDrill?.({ table: d.table, column: d.column, reason: drillReason(d.check) })}
              >
                Drill to cell
              </button>
              {/* ─── P5-7 ─── plain-language "what it means" + "how to fix it". */}
              <p className="basis-full text-xs text-red-700/90">
                {help.what} <span className="font-semibold">Fix:</span> {help.fix}
              </p>
            </li>
          );
        })}
      </ul>
    </section>
  );
}

function FidelityBlock({ report }: { report: FidelityReport }) {
  return (
    <section className="db-subsection">
      <h3 className="db-subsection-title">Fidelity</h3>
      <p className="mb-2 flex items-center gap-2 text-sm text-slate-600">
        Score: {report.overall_score}
        <span className={report.passed ? "db-pill-pass" : "db-pill-fail"}>
          {report.passed ? "pass" : "fail"}
        </span>
      </p>
      <ul className="flex flex-col gap-1 font-mono text-sm text-slate-700">
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
    <section className="db-subsection">
      <h3 className="db-subsection-title">Detection</h3>
      <p className="mb-2 flex items-center gap-2 text-sm text-slate-600">
        Score: {report.overall_score}
        <span className={report.passed ? "db-pill-pass" : "db-pill-fail"}>
          {report.passed ? "pass" : "fail"}
        </span>
      </p>
      <ul className="flex flex-col gap-1 font-mono text-sm text-slate-700">
        {report.metrics.map((m, i) => (
          <li key={`${m.metric}.${m.table}.${i}`}>
            {m.metric} · {m.table}: {m.accuracy}
          </li>
        ))}
      </ul>
    </section>
  );
}
