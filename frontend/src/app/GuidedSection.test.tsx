import { render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, test } from "vitest";
import { ModeProvider } from "./ModeProvider";
import type { StepId } from "./steps";
import { GuidedSection } from "./GuidedSection";

beforeEach(() => localStorage.clear());
afterEach(() => localStorage.clear());

function renderSection(mode: "advanced" | "guided", stepId: StepId) {
  localStorage.setItem("dbsprout.mode", mode);
  return render(
    <ModeProvider>
      <GuidedSection stepId={stepId}>
        <button>inner</button>
      </GuidedSection>
    </ModeProvider>,
  );
}

test("advanced mode renders a plain section with no overlay attrs", () => {
  const { container } = renderSection("advanced", "schema");
  const section = container.querySelector("section")!;
  expect(section).not.toHaveAttribute("aria-hidden");
  expect(section).not.toHaveAttribute("inert");
  expect(section.getAttribute("style")).toBeNull();
  expect(screen.getByRole("button", { name: "inner" })).toBeInTheDocument();
});

test("guided mode: the active step (currentStep=0 -> start) is interactive", () => {
  const { container } = renderSection("guided", "start");
  const section = container.querySelector("section")!;
  expect(section).not.toHaveAttribute("aria-hidden");
  expect(section).not.toHaveAttribute("inert");
  expect(screen.getByRole("button", { name: "inner" })).toBeInTheDocument();
});

test("guided mode: a non-active section is hidden (display:none) but mounted", () => {
  const { container } = renderSection("guided", "schema"); // currentStep=0=start
  const section = container.querySelector("section")!;
  // True wizard: the non-active section is removed from layout via the `hidden`
  // (display:none) utility — NOT dimmed (no opacity). aria-hidden/inert remain
  // (harmless on a display:none node, and keep it out of the accessibility tree).
  expect(section).toHaveClass("hidden");
  expect(section.style.opacity).toBe("");
  expect(section).toHaveAttribute("aria-hidden", "true");
  expect(section).toHaveAttribute("inert");
  // Still mounted — children present (state preserved across step changes). The
  // button is removed from the accessibility tree by aria-hidden, so query it as
  // hidden / by text to prove it is in the DOM (not unmounted). `getByText` finds
  // display:none nodes because jsdom does not apply layout.
  expect(screen.getByText("inner")).toBeInTheDocument();
  expect(
    screen.getByRole("button", { name: "inner", hidden: true }),
  ).toBeInTheDocument();
});
