import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { queryKeys } from "../../api/endpoints";
import { SpecAssist } from "./SpecAssist";

afterEach(() => vi.unstubAllGlobals());

function json(body: unknown, status = 200) {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  });
}

const OK = {
  provider: "embedded",
  model_used: "qwen",
  schema_hash: "h",
  tables: 2,
  total_columns: 7,
};

function renderWithClient(ui: React.ReactElement) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const utils = render(<QueryClientProvider client={client}>{ui}</QueryClientProvider>);
  return { client, ...utils };
}

test("clicking AI assist POSTs /api/spec/assist with the default embedded provider", async () => {
  const m = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    if (String(input) === "/api/spec/assist") return json(OK);
    return json({}, 404);
  });
  vi.stubGlobal("fetch", m);

  renderWithClient(<SpecAssist />);
  fireEvent.click(screen.getByRole("button", { name: /ai assist/i }));

  await waitFor(() =>
    expect(
      m.mock.calls.some(
        ([u, i]) =>
          String(u) === "/api/spec/assist" &&
          (i as RequestInit | undefined)?.method === "POST",
      ),
    ).toBe(true),
  );
  const [, init] = m.mock.calls.find(([u]) => String(u) === "/api/spec/assist") as [
    string,
    RequestInit,
  ];
  // The picker defaults to the offline "embedded" provider, sent explicitly.
  expect(JSON.parse(String(init.body))).toEqual({ provider: "embedded" });
});

test("selecting Cloud forwards provider:cloud", async () => {
  const m = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) => {
    if (String(input) === "/api/spec/assist") return json({ ...OK, provider: "cloud" });
    return json({}, 404);
  });
  vi.stubGlobal("fetch", m);

  renderWithClient(<SpecAssist />);
  fireEvent.change(screen.getByLabelText(/spec-assist provider/i), {
    target: { value: "cloud" },
  });
  fireEvent.click(screen.getByRole("button", { name: /ai assist/i }));

  await waitFor(() => {
    const call = m.mock.calls.find(([u]) => String(u) === "/api/spec/assist");
    expect(call).toBeDefined();
    expect(JSON.parse(String((call as [string, RequestInit])[1].body))).toEqual({
      provider: "cloud",
    });
  });
});

test("success invalidates the spec query so the grid repaints", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      String(input) === "/api/spec/assist" ? json(OK) : json({}, 404),
    ),
  );
  const { client } = renderWithClient(<SpecAssist />);
  const spy = vi.spyOn(client, "invalidateQueries");

  fireEvent.click(screen.getByRole("button", { name: /ai assist/i }));

  await waitFor(() =>
    expect(spy).toHaveBeenCalledWith({ queryKey: queryKeys.spec }),
  );
});

test("a typed error surfaces as an alert message", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      String(input) === "/api/spec/assist"
        ? json({ detail: { code: "LLM_UNAVAILABLE", message: "provider missing" } }, 503)
        : json({}, 404),
    ),
  );
  renderWithClient(<SpecAssist />);
  fireEvent.click(screen.getByRole("button", { name: /ai assist/i }));

  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/provider missing/i),
  );
});

test("shows progress while the proposal is in flight", async () => {
  let resolve: ((r: Response) => void) | undefined;
  vi.stubGlobal(
    "fetch",
    vi.fn(
      (input: RequestInfo | URL) =>
        new Promise<Response>((res) => {
          if (String(input) === "/api/spec/assist") resolve = res;
        }),
    ),
  );
  renderWithClient(<SpecAssist />);
  fireEvent.click(screen.getByRole("button", { name: /ai assist/i }));

  await waitFor(() => expect(screen.getByText(/proposing/i)).toBeInTheDocument());
  resolve?.(json(OK));
});
