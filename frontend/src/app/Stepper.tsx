import { useQueryClient } from "@tanstack/react-query";
import { useRef, useSyncExternalStore } from "react";
import { useMode } from "./ModeProvider";
import { STEPS } from "./steps";

/**
 * Guided-mode stepper: shows the current step and Next/Back. Next is gated by
 * the current step's precondition, read from the TanStack Query cache; the
 * component subscribes to cache mutations via useSyncExternalStore so the gate
 * re-evaluates when a schema/spec/job lands. Renders nothing in advanced mode.
 *
 * `currentStep` lives in the mode context (P3-2 will consume it for the focus
 * overlay). This slice renders no dimming — every App section still shows.
 */
export function Stepper() {
  const { mode, currentStep, setStep } = useMode();
  const qc = useQueryClient();

  // Subscribe to the whole query cache so the gate recomputes on any mutation
  // (schema load, spec creation, job success). The snapshot is a monotonic
  // version bumped on every notification — using a count or hash would miss
  // IN-PLACE data updates (e.g. a polled job flipping running→succeeded on the
  // SAME jobId key), since the cache-entry count is unchanged. useSyncExternalStore
  // bails out of re-rendering when the snapshot is `===` to the previous, so the
  // version must strictly increase whenever any cached data changes.
  const versionRef = useRef(0);
  useSyncExternalStore(
    (onChange) =>
      qc.getQueryCache().subscribe(() => {
        versionRef.current += 1;
        onChange();
      }),
    () => versionRef.current,
    () => versionRef.current,
  );

  if (mode !== "guided") return null;

  const step = STEPS[currentStep];
  const isFirst = currentStep === 0;
  const isLast = currentStep === STEPS.length - 1;
  const canAdvance = !isLast && step.gate(qc);

  return (
    <nav aria-label="Guided steps">
      <p>
        Step {currentStep + 1} of {STEPS.length}: <strong>{step.title}</strong>
      </p>
      <button
        type="button"
        onClick={() => setStep(currentStep - 1)}
        disabled={isFirst}
      >
        Back
      </button>
      <button
        type="button"
        onClick={() => setStep(currentStep + 1)}
        disabled={!canAdvance}
      >
        Next
      </button>
    </nav>
  );
}
