import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { GeneratePanel } from "./GeneratePanel";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

test("renders engine select with the four engines, a seed input and a Generate button", () => {
  renderWithClient(<GeneratePanel onStarted={() => undefined} />);

  const select = screen.getByLabelText(/generate engine/i);
  expect(select).toBeInTheDocument();
  for (const engine of ["heuristic", "spec", "statistical", "finetuned"]) {
    expect(screen.getByRole("option", { name: engine })).toBeInTheDocument();
  }
  expect(screen.getByLabelText(/generate seed/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /generate/i })).toBeInTheDocument();
});

test("clicking Generate POSTs the chosen engine + parsed seed and reports the job id", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse({ job_id: "job-7", seed: 5 }));
  vi.stubGlobal("fetch", m);
  const onStarted = vi.fn();

  renderWithClient(<GeneratePanel onStarted={onStarted} />);

  fireEvent.change(screen.getByLabelText(/generate engine/i), { target: { value: "spec" } });
  fireEvent.change(screen.getByLabelText(/generate seed/i), { target: { value: "5" } });
  fireEvent.click(screen.getByRole("button", { name: /generate/i }));

  await waitFor(() => expect(onStarted).toHaveBeenCalledWith("job-7"));
  const [url, init] = m.mock.calls[0];
  expect(String(url)).toBe("/api/generate");
  expect(JSON.parse(init?.body as string)).toEqual({ engine: "spec", seed: 5 });
});

test("a blank seed box sends seed: null (server materialises one)", async () => {
  const m = vi.fn<typeof fetch>(async () => jsonResponse({ job_id: "job-8", seed: 123 }));
  vi.stubGlobal("fetch", m);

  renderWithClient(<GeneratePanel onStarted={() => undefined} />);
  fireEvent.click(screen.getByRole("button", { name: /generate/i }));

  await waitFor(() => expect(m).toHaveBeenCalled());
  expect(JSON.parse(m.mock.calls[0][1]?.body as string)).toEqual({
    engine: "heuristic",
    seed: null,
  });
});

test("a failed start surfaces a typed error and does not report a job id", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({ detail: { code: "JOB_ACTIVE", message: "a job is already running" } }, 409),
    ),
  );
  const onStarted = vi.fn();

  renderWithClient(<GeneratePanel onStarted={onStarted} />);
  fireEvent.click(screen.getByRole("button", { name: /generate/i }));

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/already running/i),
  );
  expect(onStarted).not.toHaveBeenCalled();
});
