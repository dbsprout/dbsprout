import { expect, test } from "vitest";
import { parseSeed } from "./parseSeed";

test("empty / blank input → null (server materialises a seed)", () => {
  expect(parseSeed("")).toBeNull();
  expect(parseSeed("   ")).toBeNull();
});

test("a non-negative integer string → that number", () => {
  expect(parseSeed("42")).toBe(42);
  expect(parseSeed("0")).toBe(0);
  expect(parseSeed("  7 ")).toBe(7);
});

test("a non-integer / negative / garbage string → null (never sends NaN)", () => {
  expect(parseSeed("3.5")).toBeNull();
  expect(parseSeed("-1")).toBeNull();
  expect(parseSeed("abc")).toBeNull();
});
