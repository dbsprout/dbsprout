import { fireEvent, screen, waitFor } from "@testing-library/react";
import { afterEach, expect, test, vi } from "vitest";
import { renderWithClient } from "../../test/renderWithClient";
import { UploadForm } from "./UploadForm";

afterEach(() => vi.unstubAllGlobals());

test("Load file is disabled until a file is chosen, then uploads it", async () => {
  const onLoaded = vi.fn();
  const fetchMock = vi.fn(
    async () =>
      new Response(JSON.stringify({ source: "upload", table_count: 1, tables: ["t"], dialect: "sqlite" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  );
  vi.stubGlobal("fetch", fetchMock);
  renderWithClient(<UploadForm onLoaded={onLoaded} />);

  const button = screen.getByRole("button", { name: /load file/i });
  expect(button).toBeDisabled();

  const file = new File(["CREATE TABLE t (id int);"], "schema.sql", { type: "text/plain" });
  fireEvent.change(screen.getByLabelText(/schema file/i), { target: { files: [file] } });
  expect(button).toBeEnabled();

  fireEvent.click(button);
  await waitFor(() => expect(onLoaded).toHaveBeenCalled());
  expect(fetchMock).toHaveBeenCalledWith("/api/schema/load", expect.objectContaining({ method: "POST" }));
});

test("surfaces an upload error", async () => {
  vi.stubGlobal(
    "fetch",
    vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: { code: "PARSE_ERROR", message: "bad schema" } }), {
          status: 400,
          headers: { "Content-Type": "application/json" },
        }),
    ),
  );
  renderWithClient(<UploadForm onLoaded={() => undefined} />);
  const file = new File(["x"], "bad.sql", { type: "text/plain" });
  fireEvent.change(screen.getByLabelText(/schema file/i), { target: { files: [file] } });
  fireEvent.click(screen.getByRole("button", { name: /load file/i }));
  await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent(/bad schema/i));
});
