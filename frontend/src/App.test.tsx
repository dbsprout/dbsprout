import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { App } from "./App";

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ status: "ok" }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
    ),
  );
});

afterEach(() => {
  vi.unstubAllGlobals();
});

test("renders heading and reports backend ok", async () => {
  render(<App />);
  expect(screen.getByText("DBSprout Workbench")).toBeInTheDocument();
  await waitFor(() =>
    expect(screen.getByTestId("backend-status")).toHaveTextContent("backend: ok"),
  );
});
