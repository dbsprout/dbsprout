import { afterEach, expect, test, vi } from "vitest";
import * as client from "./client";
import { exportData } from "./endpoints";

afterEach(() => vi.restoreAllMocks());

test("exportData posts the chosen format + selected tables with a per-table fallback name", async () => {
  const spy = vi.spyOn(client, "apiDownload").mockResolvedValue(undefined);

  await exportData("csv", ["users"]);

  expect(spy).toHaveBeenCalledOnce();
  const [path, body, fallback] = spy.mock.calls[0];
  expect(path).toBe("/api/export");
  expect(body).toEqual({ format: "csv", tables: ["users"] });
  // Single-table CSV → the server names the file "<table>.csv"; mirror that fallback.
  expect(fallback).toBe("users.csv");
});

test("exportData omits the tables key when no subset is given and uses the bundle fallback", async () => {
  const spy = vi.spyOn(client, "apiDownload").mockResolvedValue(undefined);

  await exportData("sql");

  const [, body, fallback] = spy.mock.calls[0];
  expect(body).toEqual({ format: "sql" });
  expect(body).not.toHaveProperty("tables");
  expect(fallback).toBe("dbsprout-export.sql");
});

test("exportData uses the bundle fallback for a multi-table subset", async () => {
  const spy = vi.spyOn(client, "apiDownload").mockResolvedValue(undefined);

  await exportData("json", ["users", "orders"]);

  const [, body, fallback] = spy.mock.calls[0];
  expect(body).toEqual({ format: "json", tables: ["users", "orders"] });
  expect(fallback).toBe("dbsprout-export.json");
});
