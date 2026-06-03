import type { StepId } from "./steps";

export interface CoachEntry {
  /** Short, action-oriented heading for the step. */
  title: string;
  /** One sentence of guidance on what to do in this step. */
  body: string;
}

/**
 * One short, action-oriented blurb per guided step. Keyed by `StepId` so the map
 * can never drift from the step model (enforced by `coachCopy.test.ts`).
 */
export const COACH_COPY: Record<StepId, CoachEntry> = {
  start: {
    title: "Pick a data source",
    body: "Connect a live database, upload or paste a schema, or load a bundled sample to begin.",
  },
  schema: {
    title: "Review the schema",
    body: "Check the tables, columns, and foreign keys DBSprout found before you configure generation.",
  },
  configure: {
    title: "Configure generation",
    body: "Set row counts and per-column generators, then preview a few rows to confirm the data looks right.",
  },
  generate: {
    title: "Generate the data",
    body: "Choose an engine and a seed, run generation, and watch the live progress.",
  },
  validate: {
    title: "Validate integrity",
    body: "Run the quality checks — foreign keys, uniqueness, NOT NULL, and CHECK constraints.",
  },
  output: {
    title: "Export or insert",
    body: "Download the data in your format of choice, or insert it straight into a target database.",
  },
  runs: {
    title: "Review runs & quality",
    body: "Inspect past runs, quality metrics, and LLM costs for this workspace.",
  },
};
