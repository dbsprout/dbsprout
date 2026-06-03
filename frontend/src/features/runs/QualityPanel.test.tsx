import { screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { QualityPanel } from "./QualityPanel";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const QUALITY = {
  found: true,
  run_id: 7,
  rows: [
    { metric_type: "integrity", metric_name: "fk_valid", score: 1, passed: true, status: "pass", details_json: null },
    { metric_type: "fidelity", metric_name: "distribution", score: 0.62, passed: true, status: "warn", details_json: null },
    { metric_type: "detection", metric_name: "classifier", score: 0.4, passed: false, status: "fail", details_json: null },
  ],
};

test("renders pass / warn / fail metric rows", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(QUALITY)));

  renderWithClient(<QualityPanel runId={undefined} />);

  await waitFor(() => expect(screen.getByText("fk_valid")).toBeInTheDocument());
  expect(screen.getByText("distribution")).toBeInTheDocument();
  expect(screen.getByText("classifier")).toBeInTheDocument();
  expect(screen.getByText("pass")).toBeInTheDocument();
  expect(screen.getByText("warn")).toBeInTheDocument();
  expect(screen.getByText("fail")).toBeInTheDocument();
});

test("requests the given run id", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse(QUALITY));
  vi.stubGlobal("fetch", m);

  renderWithClient(<QualityPanel runId={7} />);

  await waitFor(() => expect(screen.getByText("fk_valid")).toBeInTheDocument());
  expect(String(m.mock.calls[0][0])).toBe("/api/quality?run_id=7");
});

test("shows an empty-state when no run was found", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ found: false, run_id: null, rows: [] })),
  );

  renderWithClient(<QualityPanel runId={undefined} />);

  await waitFor(() => expect(screen.getByText(/no quality data/i)).toBeInTheDocument());
});

test("shows an error state when the request fails", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ detail: { code: "BOOM", message: "nope" } }, 500)),
  );

  renderWithClient(<QualityPanel runId={undefined} />);

  await waitFor(() => expect(screen.getByText(/could not load quality/i)).toBeInTheDocument());
});
