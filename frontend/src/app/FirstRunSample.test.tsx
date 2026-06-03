import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { queryKeys } from "../api/endpoints";
import { ModeProvider } from "./ModeProvider";
import { FirstRunSample } from "./FirstRunSample";

const SEEN = "dbsprout.guided.seen";

function sampleFetch() {
  return vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/api/samples")) {
      return new Response(
        JSON.stringify({
          samples: [
            {
              name: "ecommerce",
              title: "E-commerce",
              description: "demo",
              dialect: "sqlite",
              table_count: 7,
            },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(
      JSON.stringify({ source: "sample:ecommerce", table_count: 7, tables: [], dialect: "sqlite" }),
      { status: 200, headers: { "Content-Type": "application/json" } },
    );
  });
}

function renderFirstRun(client = new QueryClient()) {
  localStorage.setItem("dbsprout.mode", "guided");
  return render(
    <QueryClientProvider client={client}>
      <ModeProvider>
        <FirstRunSample />
      </ModeProvider>
    </QueryClientProvider>,
  );
}

beforeEach(() => localStorage.clear());
afterEach(() => {
  vi.unstubAllGlobals();
  localStorage.clear();
});

test("advanced mode shows nothing and does not set the seen flag", () => {
  vi.stubGlobal("fetch", sampleFetch());
  localStorage.setItem("dbsprout.mode", "advanced");
  const { container } = render(
    <QueryClientProvider client={new QueryClient()}>
      <ModeProvider>
        <FirstRunSample />
      </ModeProvider>
    </QueryClientProvider>,
  );
  expect(container).toBeEmptyDOMElement();
  expect(localStorage.getItem(SEEN)).toBeNull();
});

test("first guided entry with no schema offers a sample and seeds on click", async () => {
  const fetchMock = sampleFetch();
  vi.stubGlobal("fetch", fetchMock);
  renderFirstRun();
  await waitFor(() => expect(screen.getByText(/E-commerce/i)).toBeInTheDocument());
  // The seen flag is recorded on first guided entry so a second entry never re-prompts.
  expect(localStorage.getItem(SEEN)).toBe("true");
  fireEvent.click(screen.getByRole("button", { name: /E-commerce/i }));
  await waitFor(() =>
    expect(
      fetchMock.mock.calls.some(([u]) => String(u).endsWith("/api/schema/sample")),
    ).toBe(true),
  );
});

test("does not offer when a schema is already loaded (no overwrite)", () => {
  vi.stubGlobal("fetch", sampleFetch());
  const client = new QueryClient();
  client.setQueryData(queryKeys.schema, { tables: [{ name: "users" }] });
  renderFirstRun(client);
  expect(screen.queryByText(/E-commerce/i)).not.toBeInTheDocument();
});

test("does not offer on a second entry (seen flag already set)", () => {
  vi.stubGlobal("fetch", sampleFetch());
  localStorage.setItem(SEEN, "true");
  renderFirstRun();
  expect(screen.queryByText(/E-commerce/i)).not.toBeInTheDocument();
});

test("surfaces a load error from the sample endpoint", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/api/samples")) {
      return new Response(
        JSON.stringify({
          samples: [
            { name: "ecommerce", title: "E-commerce", description: "demo", dialect: "sqlite", table_count: 7 },
          ],
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(JSON.stringify({ message: "boom" }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  renderFirstRun();
  await waitFor(() => expect(screen.getByText(/E-commerce/i)).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /E-commerce/i }));
  await waitFor(() => expect(screen.getByRole("alert")).toBeInTheDocument());
});
