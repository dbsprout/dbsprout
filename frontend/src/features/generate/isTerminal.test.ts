import { expect, test } from "vitest";
import { isTerminal } from "./isTerminal";

test("succeeded / failed / cancelled are terminal", () => {
  expect(isTerminal("succeeded")).toBe(true);
  expect(isTerminal("failed")).toBe(true);
  expect(isTerminal("cancelled")).toBe(true);
});

test("running and undefined are not terminal", () => {
  expect(isTerminal("running")).toBe(false);
  expect(isTerminal(undefined)).toBe(false);
});
