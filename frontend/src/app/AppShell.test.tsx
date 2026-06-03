import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen } from "@testing-library/react";
import { afterEach, beforeEach, expect, test } from "vitest";
import { ModeProvider } from "./ModeProvider";
import { AppShell } from "./AppShell";

// Mode is persisted to localStorage; clear it so each test starts in advanced.
beforeEach(() => localStorage.clear());
afterEach(() => localStorage.clear());

function renderShell() {
  const client = new QueryClient();
  return render(
    <QueryClientProvider client={client}>
      <ModeProvider>
        <AppShell>
          <p>child-content</p>
        </AppShell>
      </ModeProvider>
    </QueryClientProvider>,
  );
}

test("advanced mode shows no stepper and renders children", () => {
  renderShell();
  expect(screen.getByText("child-content")).toBeInTheDocument();
  expect(screen.queryByText(/Step 1 of 7/)).not.toBeInTheDocument();
});

test("toggle switches to guided and reveals the stepper", () => {
  renderShell();
  fireEvent.click(screen.getByRole("button", { name: /guided/i }));
  expect(screen.getByText(/Step 1 of 7/)).toBeInTheDocument();
  // Children still present — no remount.
  expect(screen.getByText("child-content")).toBeInTheDocument();
});

test("toggle reflects the active mode via aria-pressed", () => {
  renderShell();
  const guided = screen.getByRole("button", { name: /guided/i });
  const advanced = screen.getByRole("button", { name: /advanced/i });
  expect(guided).toHaveAttribute("aria-pressed", "false");
  expect(advanced).toHaveAttribute("aria-pressed", "true");
  fireEvent.click(guided);
  expect(guided).toHaveAttribute("aria-pressed", "true");
  expect(advanced).toHaveAttribute("aria-pressed", "false");
});
