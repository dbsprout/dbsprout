import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, test } from "vitest";
import { ModeProvider, useMode } from "./ModeProvider";
import { STEP_IDS } from "./steps";
import { COACH_COPY } from "./coachCopy";
import { CoachPanel } from "./CoachPanel";

beforeEach(() => localStorage.clear());
afterEach(() => localStorage.clear());

function StepSetter({ step }: { step: number }) {
  const { setStep } = useMode();
  setStep(step);
  return null;
}

test("renders nothing in advanced mode", () => {
  const { container } = render(
    <ModeProvider>
      <CoachPanel />
    </ModeProvider>,
  );
  expect(container).toBeEmptyDOMElement();
});

test("guided mode shows the coach copy for each of the 7 steps", () => {
  STEP_IDS.forEach((id, i) => {
    localStorage.setItem("dbsprout.mode", "guided");
    const { unmount } = render(
      <ModeProvider>
        <StepSetter step={i} />
        <CoachPanel />
      </ModeProvider>,
    );
    expect(screen.getByText(COACH_COPY[id].title)).toBeInTheDocument();
    expect(screen.getByText(COACH_COPY[id].body)).toBeInTheDocument();
    unmount();
  });
});
