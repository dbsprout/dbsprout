import { useMode } from "./ModeProvider";
import { STEPS } from "./steps";
import { COACH_COPY } from "./coachCopy";

/**
 * Renders the coach copy for the current guided step. Returns null in advanced
 * mode. Keyed off the step model (`STEPS[currentStep].id`) so the copy can never
 * drift from the stepper.
 */
export function CoachPanel() {
  const { mode, currentStep } = useMode();
  if (mode !== "guided") return null;

  const step = STEPS[currentStep];
  const copy = step ? COACH_COPY[step.id] : undefined;
  if (!copy) return null;

  return (
    <aside
      aria-label="Step guidance"
      className="mt-3 rounded-lg border border-accent-200 bg-accent-50 px-4 py-3"
    >
      <h3 className="text-sm font-semibold text-accent-900">{copy.title}</h3>
      <p className="mt-1 text-sm text-accent-800">{copy.body}</p>
    </aside>
  );
}
