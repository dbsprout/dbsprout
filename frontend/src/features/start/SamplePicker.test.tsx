import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { SamplePicker } from "./SamplePicker";

afterEach(() => vi.unstubAllGlobals());

test("lists samples and loads one on click", async () => {
  const fetchMock = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input);
    if (url.endsWith("/api/samples")) {
      return new Response(
        JSON.stringify({ samples: [{ name: "ecommerce", title: "E-commerce", description: "demo", dialect: "sqlite", table_count: 7 }] }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      );
    }
    return new Response(JSON.stringify({ source: "sample:ecommerce", table_count: 7, tables: [], dialect: "sqlite" }), {
      status: 200, headers: { "Content-Type": "application/json" },
    });
  });
  vi.stubGlobal("fetch", fetchMock);
  const onLoaded = vi.fn();
  renderWithClient(<SamplePicker onLoaded={onLoaded} />);
  await waitFor(() => expect(screen.getByText("E-commerce")).toBeInTheDocument());
  fireEvent.click(screen.getByRole("button", { name: /E-commerce/i }));
  await waitFor(() => expect(onLoaded).toHaveBeenCalled());
  expect(fetchMock.mock.calls.some(([u]) => String(u).endsWith("/api/schema/sample"))).toBe(true);
});
