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
