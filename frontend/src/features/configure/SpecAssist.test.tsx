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
    // P4-11: cloud now also carries its non-secret model + api-key-env defaults.
    expect(JSON.parse(String((call as [string, RequestInit])[1].body))).toEqual({
      provider: "cloud",
      model: "gpt-4o-mini",
      api_key_env: "OPENAI_API_KEY",
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

// ─── P4-11: cloud key-entry UX ───────────────────────────────────────────────

test("the cloud key sub-panel is hidden while embedded is selected", () => {
  renderWithClient(<SpecAssist />);
  // Embedded is the default, so the cloud-only model / env-var inputs are absent.
  expect(screen.queryByLabelText(/cloud model/i)).not.toBeInTheDocument();
  expect(screen.queryByLabelText(/api key env/i)).not.toBeInTheDocument();
});

test("selecting Cloud reveals the model + API-key-env inputs with safe defaults", () => {
  renderWithClient(<SpecAssist />);
  fireEvent.change(screen.getByLabelText(/spec-assist provider/i), {
    target: { value: "cloud" },
  });

  const model = screen.getByLabelText(/cloud model/i) as HTMLInputElement;
  const keyEnv = screen.getByLabelText(/api key env/i) as HTMLInputElement;
  expect(model).toBeInTheDocument();
  expect(keyEnv).toBeInTheDocument();
  // Sensible defaults that need no further typing for the common case.
  expect(model.value).toBe("gpt-4o-mini");
  expect(keyEnv.value).toBe("OPENAI_API_KEY");
});

test("cloud assist forwards provider, model and api_key_env (never a raw key)", async () => {
  const m = vi.fn(async (input: RequestInfo | URL, _init?: RequestInit) =>
    String(input) === "/api/spec/assist" ? json({ ...OK, provider: "cloud" }) : json({}, 404),
  );
  vi.stubGlobal("fetch", m);

  renderWithClient(<SpecAssist />);
  fireEvent.change(screen.getByLabelText(/spec-assist provider/i), {
    target: { value: "cloud" },
  });
  fireEvent.change(screen.getByLabelText(/cloud model/i), {
    target: { value: "gpt-4o" },
  });
  fireEvent.change(screen.getByLabelText(/api key env/i), {
    target: { value: "MY_OPENAI_KEY" },
  });
  fireEvent.click(screen.getByRole("button", { name: /ai assist/i }));

  await waitFor(() => {
    const call = m.mock.calls.find(([u]) => String(u) === "/api/spec/assist");
    expect(call).toBeDefined();
    const body = JSON.parse(String((call as [string, RequestInit])[1].body));
    expect(body).toEqual({
      provider: "cloud",
      model: "gpt-4o",
      api_key_env: "MY_OPENAI_KEY",
    });
    // Security: the body must carry only an env-var *name*, never a key value.
    expect(JSON.stringify(body)).not.toMatch(/sk-/);
    expect(Object.keys(body)).not.toContain("api_key");
    expect(Object.keys(body)).not.toContain("key");
  });
});

test("a missing-key 503 renders an actionable panel with the server hint", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) =>
      String(input) === "/api/spec/assist"
        ? json(
            {
              detail: {
                code: "LLM_UNAVAILABLE",
                message:
                  "cloud provider key not found — set the OPENAI_API_KEY environment variable",
                hint: "Install an LLM provider extra or use the heuristic spec.",
              },
            },
            503,
          )
        : json({}, 404),
    ),
  );
  renderWithClient(<SpecAssist />);
  fireEvent.change(screen.getByLabelText(/spec-assist provider/i), {
    target: { value: "cloud" },
  });
  fireEvent.click(screen.getByRole("button", { name: /ai assist/i }));

  // Both the typed message AND the actionable hint are shown (not a bare error).
  await waitFor(() =>
    expect(screen.getByRole("alert")).toHaveTextContent(/OPENAI_API_KEY/),
  );
  expect(screen.getByRole("alert")).toHaveTextContent(/heuristic spec/i);
});
