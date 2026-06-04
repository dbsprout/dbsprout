import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { RunsPanel } from "./RunsPanel";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const PAGE1 = {
  rows: [
    {
      id: 2,
      started_at: "2026-05-20T12:05:00+00:00",
      engine: "spec",
      provider: "openai",
      total_rows: 4242,
      total_tables: 2,
      duration_ms: 5000,
      cost: 0.08,
    },
    {
      id: 1,
      started_at: "2026-05-20T12:00:00+00:00",
      engine: "heuristic",
      provider: null,
      total_rows: 100,
      total_tables: 1,
      duration_ms: 12,
      cost: 0,
    },
  ],
  page: 1,
  total_pages: 2,
  total_runs: 12,
  has_prev: false,
  has_next: true,
};

test("lists run history rows with engine and totals", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAGE1)));

  renderWithClient(<RunsPanel onSelectRun={() => undefined} />);

  await waitFor(() => expect(screen.getByText("spec")).toBeInTheDocument());
  expect(screen.getByText("heuristic")).toBeInTheDocument();
  expect(screen.getByText(/4,242/)).toBeInTheDocument();
});

test("renders a dash for a run with no provider", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAGE1)));

  renderWithClient(<RunsPanel onSelectRun={() => undefined} />);

  await waitFor(() => expect(screen.getByText("heuristic")).toBeInTheDocument());
  expect(screen.getAllByText("—").length).toBeGreaterThan(0);
});

test("Next advances the page and disables Prev on page 1", async () => {
  const fetchMock = vi.fn(async (url: string | URL) => {
    const u = String(url);
    if (u.includes("page=2")) {
      return jsonResponse({ ...PAGE1, page: 2, has_prev: true, has_next: false, rows: [] });
    }
    return jsonResponse(PAGE1);
  });
  vi.stubGlobal("fetch", fetchMock);

  renderWithClient(<RunsPanel onSelectRun={() => undefined} />);

  await waitFor(() => expect(screen.getByText("spec")).toBeInTheDocument());
  const prev = screen.getByRole("button", { name: /prev/i });
  expect(prev).toBeDisabled();

  fireEvent.click(screen.getByRole("button", { name: /next/i }));

  await waitFor(() =>
    expect(fetchMock.mock.calls.some(([u]) => String(u).includes("page=2"))).toBe(true),
  );
});

test("clicking a row reports the selected run id", async () => {
  const onSelect = vi.fn();
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(PAGE1)));

  renderWithClient(<RunsPanel onSelectRun={onSelect} />);

  await waitFor(() => expect(screen.getByText("spec")).toBeInTheDocument());
  fireEvent.click(screen.getByText("spec"));
  expect(onSelect).toHaveBeenCalledWith(2);
});

test("shows an empty-state when there are no runs", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse({
        rows: [],
        page: 1,
        total_pages: 1,
        total_runs: 0,
        has_prev: false,
        has_next: false,
      }),
    ),
  );

  renderWithClient(<RunsPanel onSelectRun={() => undefined} />);

  await waitFor(() =>
    expect(
      screen.getByText(/no runs yet — your generation history will appear here/i),
    ).toBeInTheDocument(),
  );
});

test("shows an error state when the request fails", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ detail: { code: "BOOM", message: "nope" } }, 500)),
  );

  renderWithClient(<RunsPanel onSelectRun={() => undefined} />);

  await waitFor(() => expect(screen.getByText(/could not load runs/i)).toBeInTheDocument());
});
