import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { ValidatePanel } from "./ValidatePanel";

afterEach(() => vi.unstubAllGlobals());

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const BASE = {
  summary: { tables: 2, rows: 30, violations: 1 },
  by_table: [
    { table: "users", fk_violations: 0, unique_violations: 0, not_null_violations: 0, check_violations: 0 },
    { table: "orders", fk_violations: 1, unique_violations: 0, not_null_violations: 0, check_violations: 0 },
  ],
  details: [
    { check: "fk_satisfaction", table: "orders", column: "user_id", passed: false, details: "1 orphan row" },
  ],
  fidelity: null,
  detection: null,
};

const FIDELITY = {
  overall_score: 0.92,
  passed: true,
  metrics: [{ metric: "ks", table: "users", column: "age", score: 0.95, details: "ok" }],
};

const DETECTION = {
  overall_score: 0.51,
  passed: true,
  metrics: [{ metric: "c2st", table: "users", accuracy: 0.51, details: "indistinguishable" }],
};

async function clickRun() {
  fireEvent.click(screen.getByRole("button", { name: /validate/i }));
}

test("runs validation and renders the integrity summary plus a violation row", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(BASE)));

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() => expect(screen.getByText(/foreign keys/i)).toBeInTheDocument());
  expect(screen.getByText(/1 orphan row/)).toBeInTheDocument();
  expect(screen.getByText(/orders/)).toBeInTheDocument();
});

test("renders fidelity and detection sections only when present", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => jsonResponse({ ...BASE, fidelity: FIDELITY, detection: DETECTION })),
  );

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() => expect(screen.getByText(/fidelity/i)).toBeInTheDocument());
  expect(screen.getByText(/detection/i)).toBeInTheDocument();
  expect(screen.getByText(/0\.92/)).toBeInTheDocument();
});

test("omits fidelity and detection sections when both are null", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(BASE)));

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() => expect(screen.getByText(/foreign keys/i)).toBeInTheDocument());
  expect(screen.queryByText(/fidelity/i)).not.toBeInTheDocument();
  expect(screen.queryByText(/detection/i)).not.toBeInTheDocument();
});

test("a clean run reports no violations", async () => {
  const clean = {
    ...BASE,
    summary: { tables: 2, rows: 30, violations: 0 },
    by_table: BASE.by_table.map((t) => ({ ...t, fk_violations: 0 })),
    details: [],
  };
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(clean)));

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() => expect(screen.getByText(/no violations/i)).toBeInTheDocument());
});

test("a violation row drill button invokes onDrill with the offending table and column", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(BASE)));
  const onDrill = vi.fn();

  renderWithClient(<ValidatePanel onDrill={onDrill} />);
  await clickRun();

  await waitFor(() => expect(screen.getByRole("button", { name: /drill/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /drill/i }));

  expect(onDrill).toHaveBeenCalledWith({ table: "orders", column: "user_id" });
});

test("a typed error surfaces as an alert without leaking internals", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () =>
      jsonResponse(
        { detail: { code: "NO_RUN", message: "No generation result is available." } },
        409,
      ),
    ),
  );

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/no generation result/i),
  );
});
