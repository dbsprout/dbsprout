import { fireEvent, screen } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { StartPanel } from "./StartPanel";

afterEach(() => vi.unstubAllGlobals());

test("renders source tabs and switches to Paste", () => {
  vi.stubGlobal("fetch", vi.fn(async () => new Response(JSON.stringify({ samples: [] }), { status: 200, headers: { "Content-Type": "application/json" } })));
  renderWithClient(<StartPanel onLoaded={() => undefined} />);
  for (const name of [/live database/i, /upload/i, /paste/i, /sample/i]) {
    expect(screen.getByRole("tab", { name })).toBeInTheDocument();
  }
  fireEvent.click(screen.getByRole("tab", { name: /paste/i }));
  expect(screen.getByRole("textbox")).toBeInTheDocument();
});
