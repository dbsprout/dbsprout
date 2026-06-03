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

test("mounts the Configure surface", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ samples: [] }), { status: 200, headers: { "Content-Type": "application/json" } })),
  );
  renderWithClient(<App />);
  expect(screen.getByRole("heading", { name: "Configure" })).toBeInTheDocument();
});

test("mounts the Generate surface", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ samples: [] }), { status: 200, headers: { "Content-Type": "application/json" } })),
  );
  renderWithClient(<App />);
  expect(screen.getByRole("heading", { name: "Generate" })).toBeInTheDocument();
});

test("mounts the Insert surface", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ samples: [] }), { status: 200, headers: { "Content-Type": "application/json" } })),
  );
  renderWithClient(<App />);
  expect(screen.getByRole("heading", { name: "Insert" })).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /preview insert/i })).toBeInTheDocument();
});

test("mounts the Validate surface", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ samples: [] }), { status: 200, headers: { "Content-Type": "application/json" } })),
  );
  renderWithClient(<App />);
  expect(screen.getByRole("heading", { name: "Validate" })).toBeInTheDocument();
});

test("mounts the Runs & Quality surface (P1c-4)", () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify({ samples: [] }), { status: 200, headers: { "Content-Type": "application/json" } })),
  );
  renderWithClient(<App />);
  expect(screen.getByRole("heading", { name: "Runs & Quality" })).toBeInTheDocument();
});
