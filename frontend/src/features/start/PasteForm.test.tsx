import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { PasteForm } from "./PasteForm";

afterEach(() => vi.unstubAllGlobals());

test("Load button is disabled until text is entered, then pastes the schema", async () => {
  const onLoaded = vi.fn();
  const fetchMock = vi.fn<typeof fetch>(
    async () =>
      new Response(JSON.stringify({ source: "paste", table_count: 1, tables: ["t"], dialect: "sqlite" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", fetchMock);
  renderWithClient(<PasteForm onLoaded={onLoaded} />);

  const button = screen.getByRole("button", { name: /load pasted schema/i });
  expect(button).toBeDisabled();

  fireEvent.change(screen.getByLabelText(/paste schema/i), {
    target: { value: "CREATE TABLE t (id int);" },
  });
  expect(button).toBeEnabled();

  fireEvent.click(button);
  await waitFor(() => expect(onLoaded).toHaveBeenCalled());
  const [, init] = fetchMock.mock.calls[0];
  expect(init?.method).toBe("POST");
  expect(JSON.parse(String(init?.body))).toMatchObject({
    text: "CREATE TABLE t (id int);",
  });
});

test("surfaces a paste error", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: { code: "PARSE_ERROR", message: "cannot parse" } }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }),
    ),
  );
  renderWithClient(<PasteForm onLoaded={() => undefined} />);
  fireEvent.change(screen.getByLabelText(/paste schema/i), { target: { value: "junk" } });
  fireEvent.click(screen.getByRole("button", { name: /load pasted schema/i }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/cannot parse/i));
});
