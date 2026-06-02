import { expect, test } from "vitest";
import { genLabel } from "./genLabel";

test("joins provider and method with a slash", () => {
  expect(genLabel("mimesis", "email")).toBe("mimesis/email");
});

test("omits the slash when method is null", () => {
  expect(genLabel("mimesis", null)).toBe("mimesis");
});
