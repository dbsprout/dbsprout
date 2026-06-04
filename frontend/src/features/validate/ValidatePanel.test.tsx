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

test("a violation row drill button invokes onDrill with the offending table, column, and a reason", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(BASE)));
  const onDrill = vi.fn();

  renderWithClient(<ValidatePanel onDrill={onDrill} />);
  await clickRun();

  await waitFor(() => expect(screen.getByRole("button", { name: /drill/i })).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /drill/i }));

  // ─── P5-7: the drill now carries a short cross-panel reason for Configure ───
  expect(onDrill).toHaveBeenCalledWith(
    expect.objectContaining({
      table: "orders",
      column: "user_id",
      reason: expect.stringMatching(/foreign key|parent|orphan/i),
    }),
  );
});

// ─── P5-7: each violation explains what it means + how to fix it ───
test("an fk violation row shows a plain-language explanation and a fix", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(BASE)));

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() => expect(screen.getByText(/1 orphan row/)).toBeInTheDocument());
  // What it means.
  expect(screen.getByText(/references a parent row that does not exist/i)).toBeInTheDocument();
  // How to fix it.
  expect(screen.getByText(/sampled from real parent rows/i)).toBeInTheDocument();
});

test("a duplicate-key (unique/pk) violation shows the re-generate remedy", async () => {
  const dup = {
    ...BASE,
    summary: { tables: 1, rows: 10, violations: 1 },
    by_table: [
      { table: "users", fk_violations: 0, unique_violations: 1, not_null_violations: 0, check_violations: 0 },
    ],
    details: [
      { check: "pk_uniqueness", table: "users", column: "id", passed: false, details: "2 duplicate keys" },
    ],
  };
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(dup)));

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() => expect(screen.getByText(/2 duplicate keys/)).toBeInTheDocument());
  expect(screen.getByText(/share the same value/i)).toBeInTheDocument();
  // The duplicate-key remedy mentions re-generating with a different seed.
  expect(screen.getAllByText(/re-generate/i).length).toBeGreaterThan(0);
});

test("the Violations section shows a re-seed tip", async () => {
  vi.stubGlobal("fetch", vi.fn(async () => jsonResponse(BASE)));

  renderWithClient(<ValidatePanel />);
  await clickRun();

  await waitFor(() => expect(screen.getByText(/1 orphan row/)).toBeInTheDocument());
  expect(screen.getByText(/re-generating on the Generate step/i)).toBeInTheDocument();
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
