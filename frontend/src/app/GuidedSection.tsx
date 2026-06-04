import type { ReactNode } from "react";
import { useMode } from "./ModeProvider";
import { STEPS, type StepId } from "./steps";

// React 18's HTMLAttributes does not yet type the `inert` attribute (it lives in
// React's experimental types). Declare it locally so the focus overlay can mark a
// non-active section non-tabbable / non-interactive without pulling in the
// experimental types. React 18.3 forwards the lowercase attribute to the DOM.
declare module "react" {
  interface HTMLAttributes<T> {
    inert?: string;
  }
}

interface GuidedSectionProps {
  /** The step id this section belongs to (the Output step owns Export + Insert). */
  stepId: StepId;
  children: ReactNode;
}

/**
 * Wraps an App `<section>` with the guided wizard overlay. In advanced mode it is
 * a plain `<section>` (byte-identical — no overlay attributes), so the advanced
 * regression stays green. In guided mode this is a true one-step-at-a-time wizard:
 * only the active step's section is visible; every other section is HIDDEN via the
 * `hidden` (display:none) utility — removed from layout, not merely dimmed — while
 * staying `aria-hidden` and `inert` (out of the accessibility tree, non-tabbable).
 * Hidden sections are ALWAYS MOUNTED (children still rendered), so each panel's
 * query/UI state survives step changes — hide via display:none, never unmount.
 */
/** The card frame shared by every workbench section. */
const CARD = "db-card";

export function GuidedSection({ stepId, children }: GuidedSectionProps) {
  const { mode, currentStep } = useMode();

  if (mode !== "guided") {
    return <section className={CARD}>{children}</section>;
  }

  const active = STEPS[currentStep]?.id === stepId;
  if (active) {
    // The active guided step gets an accent ring/emphasis.
    return <section className={`${CARD} ring-2 ring-accent-500`}>{children}</section>;
  }

  // Non-active guided section: hidden (display:none) but mounted, so its panel
  // state survives step changes. aria-hidden/inert stay (harmless + keep it out
  // of the accessibility tree).
  return (
    <section aria-hidden="true" inert="" className={`${CARD} hidden`}>
      {children}
    </section>
  );
}
