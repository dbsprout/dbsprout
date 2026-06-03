import { useQueryClient } from "@tanstack/react-query";
import { useSyncExternalStore } from "react";
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
  // (schema load, spec creation, job success). The snapshot is the cache's
  // change-count; the subscribe callback also fires on in-place setQueryData,
  // forcing a re-render where the gate is recomputed.
  useSyncExternalStore(
    (onChange) => qc.getQueryCache().subscribe(onChange),
    () => qc.getQueryCache().getAll().length,
    () => 0,
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
