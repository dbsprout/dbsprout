import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { RunsSurface } from "./RunsSurface";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status: 200,
    headers: { "Content-Type": "application/json" },
  });
}

function route(url: string | URL): Response {
  const u = String(url);
  if (u.startsWith("/api/runs")) {
    return jsonResponse({
      rows: [],
      page: 1,
      total_pages: 1,
      total_runs: 0,
      has_prev: false,
      has_next: false,
    });
  }
  if (u.startsWith("/api/quality")) {
    return jsonResponse({ found: false, run_id: null, rows: [] });
  }
  if (u.startsWith("/api/costs")) {
    return jsonResponse({
      total_cost: 0,
      total_tokens: 0,
      total_calls: 0,
      avg_cost_per_run: 0,
      per_provider: [],
    });
  }
  return jsonResponse({});
}

test("mounts the runs, quality and costs panels", async () => {
  vi.stubGlobal("fetch", vi.fn(async (url: string | URL) => route(url)));

  renderWithClient(<RunsSurface />);

  await waitFor(() => expect(screen.getByText(/no runs yet/i)).toBeInTheDocument());
  expect(screen.getByText(/no quality data/i)).toBeInTheDocument();
  expect(screen.getByText(/no llm calls/i)).toBeInTheDocument();
});
