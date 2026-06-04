import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { CostsPanel } from "./CostsPanel";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const COSTS = {
  total_cost: 0.08,
  total_tokens: 3000,
  total_calls: 2,
  avg_cost_per_run: 0.08,
  per_provider: [
    { provider: "openai", cost: 0.05, tokens: 2000, calls: 1 },
    { provider: "anthropic", cost: 0.03, tokens: 1000, calls: 1 },
  ],
};

test("renders totals and a per-provider breakdown", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(COSTS)));

  renderWithClient(<CostsPanel />);

  await waitFor(() => expect(screen.getByText("openai")).toBeInTheDocument());
  expect(screen.getByText("anthropic")).toBeInTheDocument();
  expect(screen.getByText(/3,000/)).toBeInTheDocument(); // total tokens
});

test("shows an empty-state when there were no LLM calls", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({
        total_cost: 0,
        total_tokens: 0,
        total_calls: 0,
        avg_cost_per_run: 0,
        per_provider: [],
      }),
    ),
  );

  renderWithClient(<CostsPanel />);

  await waitFor(() => expect(screen.getByText(/no llm costs yet/i)).toBeInTheDocument());
  expect(screen.getByText(/cloud\/embedded llm/i)).toBeInTheDocument();
});

test("shows an error state when the request fails", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ detail: { code: "BOOM", message: "nope" } }, 500)),
  );

  renderWithClient(<CostsPanel />);

  await waitFor(() => expect(screen.getByText(/could not load costs/i)).toBeInTheDocument());
});
