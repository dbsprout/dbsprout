import { fireEvent, screen, waitFor, within } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { StartPanel } from "./StartPanel";

afterEach(() => vi.unstubAllGlobals());

function stubFetch(connections: { name: string; url: string }[] = []) {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL) => {
      const url = String(input);
      const body = url.endsWith("/api/connections") ? { connections } : { samples: [] };
      return new Response(JSON.stringify(body), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    }),
  );
}

test("renders source tabs and switches to Paste", () => {
  stubFetch();
  renderWithClient(<StartPanel onLoaded={() => undefined} />);
  for (const name of [/live database/i, /upload/i, /paste/i, /sample/i]) {
    expect(screen.getByRole("tab", { name })).toBeInTheDocument();
  }
  fireEvent.click(screen.getByRole("tab", { name: /paste/i }));
  expect(screen.getByLabelText(/paste schema/i)).toBeInTheDocument();
});

test("mounts SavedConnections (P2a-2)", async () => {
  stubFetch([]);
  renderWithClient(<StartPanel onLoaded={() => undefined} />);
  const region = await screen.findByRole("region", { name: /saved connections/i });
  await waitFor(() =>
    expect(within(region).getByText(/no saved connections/i)).toBeInTheDocument(),
  );
});

test("loading a saved connection surfaces the URL and switches to the Live tab", async () => {
  stubFetch([{ name: "prod", url: "postgresql://u:@db/app" }]);
  renderWithClient(<StartPanel onLoaded={() => undefined} />);

  fireEvent.click(screen.getByRole("tab", { name: /paste/i }));
  const region = await screen.findByRole("region", { name: /saved connections/i });
  await waitFor(() => expect(within(region).getByText("prod")).toBeInTheDocument());
  fireEvent.click(within(region).getByRole("button", { name: /^load$/i }));

  expect(screen.getByRole("status")).toHaveTextContent("postgresql://u:@db/app");
  expect(screen.getByRole("tab", { name: /live database/i })).toHaveAttribute(
    "aria-selected",
    "true",
  );
});
