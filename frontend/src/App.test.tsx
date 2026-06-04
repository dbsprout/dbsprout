import { fireEvent, screen, waitFor } from "@testing-library/react";
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

// ─── P5-5: a fetch stub that lets the real ValidatePanel render one violation ───
// Routes by URL/method: POST /api/validate returns a single not-null violation on
// orders.total; every other request gets a benign empty payload so the rest of the
// shell mounts crash-free. Lets a real "Drill to cell" click flow through
// ValidatePanel → App.onDrill → Mode/Selection without a backend.
const VALIDATE_VIOLATION = {
  summary: { tables: 1, rows: 10, violations: 1 },
  by_table: [
    {
      table: "orders",
      fk_violations: 0,
      unique_violations: 0,
      not_null_violations: 1,
      check_violations: 0,
    },
  ],
  details: [
    {
      check: "not_null",
      table: "orders",
      column: "total",
      passed: false,
      details: "1 null row",
    },
  ],
  fidelity: null,
  detection: null,
};

function stubValidateFetch() {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      const method = (init?.method ?? "GET").toUpperCase();
      const json = (body: unknown) =>
        new Response(JSON.stringify(body), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      if (url.includes("/api/validate") && method === "POST") {
        return json(VALIDATE_VIOLATION);
      }
      if (url.includes("/api/samples")) return json({ samples: [] });
      if (url.includes("/api/costs")) {
        // Zero-shaped so CostsPanel takes its "no LLM calls" branch (total_calls
        // === 0) instead of calling .toFixed on an undefined total.
        return json({
          total_cost: 0,
          total_tokens: 0,
          total_calls: 0,
          avg_cost_per_run: 0,
          per_provider: [],
        });
      }
      // schema / spec / generators / preview / connections / runs etc. — a superset
      // of benign empties so whichever field a panel reads is an empty array/object.
      return json({ tables: [], methods: [], rows: [], connections: [], samples: [], runs: [] });
    }),
  );
}

/**
 * The `<section>` wrapping a heading, for asserting guided hidden/active state.
 * `hidden: true` so a heading inside a `hidden`/`aria-hidden` (guided) section is
 * still matched — otherwise role queries drop inaccessible nodes by default.
 */
function sectionFor(headingName: string): HTMLElement {
  return screen
    .getByRole("heading", { name: headingName, hidden: true })
    .closest("section")!;
}

async function drillFirstViolation() {
  // `hidden: true` — in guided mode the Validate section is hidden (display:none)
  // but still mounted, so its buttons exist; default role queries would skip them.
  fireEvent.click(screen.getByRole("button", { name: /^validate$/i, hidden: true }));
  await waitFor(() =>
    expect(
      screen.getByRole("button", { name: /drill to cell/i, hidden: true }),
    ).toBeInTheDocument(),
  );
  fireEvent.click(screen.getByRole("button", { name: /drill to cell/i, hidden: true }));
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

// ─── P5-5: Validate drill navigates to Configure in guided mode ───
// In the guided wizard only the active step is visible; Configure (step 3) is
// hidden while you're on Validate (step 5). Drilling a violation must navigate the
// wizard to Configure so the now-visible ConfigurePanel shows the focused cell.
test("guided mode: drilling a violation navigates the wizard to Configure", async () => {
  stubValidateFetch();
  localStorage.setItem("dbsprout.mode", "guided");
  renderApp();

  // Configure starts hidden (we're on the Start step, idx 0).
  expect(sectionFor("Configure")).toHaveClass("hidden");
  expect(sectionFor("Configure")).toHaveAttribute("aria-hidden");
  expect(screen.getByText(/Step 1 of 7/)).toBeInTheDocument();

  // To reach the Validate panel's run button it must be mounted — guided keeps all
  // steps mounted, so the Validate button exists even though its section is hidden.
  await drillFirstViolation();

  // The wizard has navigated to the Configure step (idx 2 → "Step 3 of 7").
  await waitFor(() => expect(screen.getByText(/Step 3 of 7/)).toBeInTheDocument());
  // Configure is now the active step: no longer hidden / aria-hidden.
  expect(sectionFor("Configure")).not.toHaveClass("hidden");
  expect(sectionFor("Configure")).not.toHaveAttribute("aria-hidden");
});

test("advanced mode: drilling a violation sets selection without wizard navigation", async () => {
  stubValidateFetch();
  // Default advanced mode — no stepper at all.
  renderApp();
  expect(screen.queryByText(/Step \d+ of 7/)).not.toBeInTheDocument();

  await drillFirstViolation();

  // Advanced shows every section; no wizard navigation occurs (still no stepper).
  expect(screen.queryByText(/Step \d+ of 7/)).not.toBeInTheDocument();
  expect(screen.getByRole("heading", { name: "Configure" })).toBeInTheDocument();
  // Configure was already visible in advanced mode — no hidden class to clear.
  expect(sectionFor("Configure")).not.toHaveClass("hidden");
});
// ─── end P5-5 ───
