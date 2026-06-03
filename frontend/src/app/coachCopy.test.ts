import { expect, test } from "vitest";
import { STEP_IDS } from "./steps";
import { COACH_COPY } from "./coachCopy";

test("coach copy covers every step id with a non-empty title and body", () => {
  for (const id of STEP_IDS) {
    const entry = COACH_COPY[id];
    expect(entry).toBeDefined();
    expect(entry.title.length).toBeGreaterThan(0);
    expect(entry.body.length).toBeGreaterThan(0);
  }
});

test("coach copy has no keys outside the step model", () => {
  expect(Object.keys(COACH_COPY).sort()).toEqual([...STEP_IDS].sort());
});
