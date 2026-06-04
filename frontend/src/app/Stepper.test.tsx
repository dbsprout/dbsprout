import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { act, fireEvent, render, screen } from "@testing-library/react";
import { useEffect } from "react";
import { expect, test } from "vitest";
import { queryKeys } from "../api/endpoints";
import { ModeProvider, useMode } from "./ModeProvider";
import { Stepper } from "./Stepper";

/** Flips the provider to guided mode once on mount (no render loop). */
function ForceGuided() {
  const { setMode } = useMode();
  useEffect(() => setMode("guided"), [setMode]);
  return null;
}

function GuidedHarness({ client }: { client: QueryClient }) {
  return (
    <QueryClientProvider client={client}>
      <ModeProvider>
        <ForceGuided />
        <Stepper />
      </ModeProvider>
    </QueryClientProvider>
  );
}

test("renders nothing in advanced mode", () => {
  const client = new QueryClient();
  const { container } = render(
    <QueryClientProvider client={client}>
      <ModeProvider>
        <Stepper />
      </ModeProvider>
    </QueryClientProvider>,
  );
  expect(container).toBeEmptyDOMElement();
});

test("shows the first step and gated Next/Back in guided mode", () => {
  const client = new QueryClient();
  render(<GuidedHarness client={client} />);
  expect(screen.getByText(/Step 1 of 7/)).toBeInTheDocument();
  expect(screen.getByText("Start")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /back/i })).toBeDisabled();
  // No schema yet → Start gate fails → Next disabled.
  expect(screen.getByRole("button", { name: /next/i })).toBeDisabled();
});

test("Next enables reactively once a schema lands, then advances", () => {
  const client = new QueryClient();
  render(<GuidedHarness client={client} />);
  const next = () => screen.getByRole("button", { name: /next/i });
  expect(next()).toBeDisabled();
  act(() => {
    client.setQueryData(queryKeys.schema, { tables: [{ name: "users" }] });
  });
  expect(next()).toBeEnabled();
  fireEvent.click(next());
  expect(screen.getByText(/Step 2 of 7/)).toBeInTheDocument();
  expect(screen.getByText("Schema")).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /back/i })).toBeEnabled();
});

/** Jumps the provider to the Generate step (index 3) once on mount. */
function ForceGenerateStep() {
  const { setMode, setStep } = useMode();
  useEffect(() => {
    setMode("guided");
    setStep(3);
  }, [setMode, setStep]);
  return null;
}

// Regression: a polled job flips running→succeeded IN PLACE on the SAME jobId
// key, so the cache-entry count is unchanged. The stepper's snapshot must still
// advance (monotonic version) or Next would never re-enable at Generate.
test("Generate's Next re-enables on an in-place job running→succeeded flip", () => {
  const client = new QueryClient();
  client.setQueryData(queryKeys.job("x"), { status: "running" });
  render(
    <QueryClientProvider client={client}>
      <ModeProvider>
        <ForceGenerateStep />
        <Stepper />
      </ModeProvider>
    </QueryClientProvider>,
  );
  const next = () => screen.getByRole("button", { name: /next/i });
  expect(screen.getByText("Generate")).toBeInTheDocument();
  expect(next()).toBeDisabled();
  act(() => {
    client.setQueryData(queryKeys.job("x"), { status: "succeeded" });
  });
  expect(next()).toBeEnabled();
});
