import { screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "./test/renderWithClient";
import { App } from "./App";

afterEach(() => vi.unstubAllGlobals());

test("renders the Workbench shell", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ samples: [] }), { status: 200, headers: { "Content-Type": "application/json" } })),
  );
  renderWithClient(<App />);
  expect(screen.getByText("DBSprout Workbench")).toBeInTheDocument();
});
