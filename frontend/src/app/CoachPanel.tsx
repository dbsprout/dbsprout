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
    <aside aria-label="Step guidance">
      <h3>{copy.title}</h3>
      <p>{copy.body}</p>
    </aside>
  );
}
