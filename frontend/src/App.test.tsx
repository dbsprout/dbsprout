import { screen } from "@testing-library/react";
import type { ReactElement } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ModeProvider } from "./app/ModeProvider";
import { renderWithClient } from "./test/renderWithClient";
import { App } from "./App";

// AppShell now reads the mode context; render the App inside a provider. Clear
// persisted mode so every test starts in the default advanced mode.
function renderApp(ui: ReactElement = <App />) {
  return renderWithClient(<ModeProvider>{ui}</ModeProvider>);
}

function stubSamplesFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ samples: [] }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    ),
  );
}

beforeEach(() => localStorage.clear());
afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

test("renders the Workbench shell", () => {
  stubSamplesFetch();
  renderApp();
  expect(screen.getByText("DBSprout Workbench")).toBeInTheDocument();
});

test("mounts the Configure surface", () => {
  stubSamplesFetch();
  renderApp();
  expect(screen.getByRole("heading", { name: "Configure" })).toBeInTheDocument();
});

test("mounts the Generate surface", () => {
  stubSamplesFetch();
  renderApp();
  expect(screen.getByRole("heading", { name: "Generate" })).toBeInTheDocument();
});

test("mounts the Insert surface", () => {
  stubSamplesFetch();
  renderApp();
  expect(screen.getByRole("heading", { name: "Insert" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /preview insert/i })).toBeInTheDocument();
});

test("mounts the Validate surface", () => {
  stubSamplesFetch();
  renderApp();
  expect(screen.getByRole("heading", { name: "Validate" })).toBeInTheDocument();
});

test("mounts the Runs & Quality surface (P1c-4)", () => {
  stubSamplesFetch();
  renderApp();
  expect(screen.getByRole("heading", { name: "Runs & Quality" })).toBeInTheDocument();
});

// ─── P3-1: advanced-mode regression guard ───
// Advanced is the default and must stay byte-identical: all 7 sections render
// and no guided stepper is present. The mode toggle is the only visible addition.
test("advanced mode renders all 7 sections and no stepper (regression)", () => {
  stubSamplesFetch();
  renderApp();
  for (const name of [
    "Start",
    "Configure",
    "Generate",
    "Output",
    "Insert",
    "Validate",
    "Runs & Quality",
  ]) {
    expect(screen.getByRole("heading", { name })).toBeInTheDocument();
  }
  expect(screen.queryByText(/Step 1 of 7/)).not.toBeInTheDocument();
  expect(screen.getByRole("button", { name: /guided/i })).toBeInTheDocument();
});

// ─── P3-2: guided focus overlay + coach copy ───
// Advanced sections carry NO overlay attributes (byte-identical regression guard).
test("advanced mode: no section is inert or aria-hidden", () => {
  stubSamplesFetch();
  const { container } = renderApp();
  const sections = Array.from(container.querySelectorAll("main > section"));
  expect(sections.length).toBeGreaterThan(0);
  for (const s of sections) {
    expect(s).not.toHaveAttribute("inert");
    expect(s).not.toHaveAttribute("aria-hidden");
  }
});

test("guided mode: only the active step's section is interactive, others inert", () => {
  stubSamplesFetch();
  localStorage.setItem("dbsprout.mode", "guided");
  const { container } = renderApp();
  const sections = Array.from(container.querySelectorAll("main > section"));
  // currentStep defaults to 0 (Start) → at least one non-active section is inert.
  const inertCount = sections.filter((s) => s.hasAttribute("inert")).length;
  expect(inertCount).toBeGreaterThan(0);
  // The Start section (active) is NOT inert.
  const startHeading = screen.getByRole("heading", { name: "Start" });
  const startSection = startHeading.closest("section")!;
  expect(startSection).not.toHaveAttribute("inert");
});

test("guided mode: coach copy for the active (Start) step is shown", () => {
  stubSamplesFetch();
  localStorage.setItem("dbsprout.mode", "guided");
  renderApp();
  expect(screen.getByText("Pick a data source")).toBeInTheDocument();
});

// ─── P4-7: cross-panel selection store wiring ───
// The full App now reads useSelection() and wires ValidatePanel.onDrill into the
// store (App.tsx). The store → ConfigurePanel → ColumnGrid focus path is exercised
// end-to-end in ConfigurePanel.test.tsx against the real store; here we only guard
// that mounting the whole shell inside the new provider stays crash-free and that
// the Validate drill affordance is present (a real, non-noop callback is attached).
test("mounts the full shell inside the SelectionProvider without regressions", () => {
  stubSamplesFetch();
  renderApp();
  // All seven sections still render (advanced default) — the new provider is inert
  // until a drill dispatches a selection.
  for (const name of ["Start", "Configure", "Generate", "Output", "Insert", "Validate", "Runs & Quality"]) {
    expect(screen.getByRole("heading", { name })).toBeInTheDocument();
  }
  // The Validate run affordance is wired (its onDrill now routes into the store).
  expect(screen.getByRole("button", { name: /^validate$/i })).toBeInTheDocument();
});
// ─── end P4-7 ───
