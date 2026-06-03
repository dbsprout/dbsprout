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
 * Wraps an App `<section>` with the guided focus overlay. In advanced mode it is
 * a plain `<section>` (byte-identical — no overlay attributes), so the advanced
 * regression stays green. In guided mode the active step's section stays
 * interactive; every other section is dimmed, `aria-hidden`, and `inert`
 * (non-tabbable / non-interactive) — but ALWAYS MOUNTED, so panel/query state
 * survives step changes (dim via attributes/style, never unmount).
 */
export function GuidedSection({ stepId, children }: GuidedSectionProps) {
  const { mode, currentStep } = useMode();

  if (mode !== "guided") {
    return <section>{children}</section>;
  }

  const active = STEPS[currentStep]?.id === stepId;
  if (active) {
    return <section>{children}</section>;
  }

  return (
    <section aria-hidden="true" inert="" style={{ opacity: 0.4 }}>
      {children}
    </section>
  );
}
