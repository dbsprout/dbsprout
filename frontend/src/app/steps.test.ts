import { QueryClient } from "@tanstack/react-query";
import { expect, test } from "vitest";
import { queryKeys } from "../api/endpoints";
import { STEPS, STEP_IDS, hasSchema, hasSpec, hasSucceededJob } from "./steps";

function client() {
  return new QueryClient();
}

test("exposes the 7 guided steps in order", () => {
  expect(STEPS.map((s) => s.title)).toEqual([
    "Start",
    "Schema",
    "Configure",
    "Generate",
    "Validate",
    "Output",
    "Runs & Quality",
  ]);
});

test("hasSchema is false with no schema, true with tables", () => {
  const qc = client();
  expect(hasSchema(qc)).toBe(false);
  qc.setQueryData(queryKeys.schema, { tables: [] });
  expect(hasSchema(qc)).toBe(false);
  qc.setQueryData(queryKeys.schema, { tables: [{ name: "users" }] });
  expect(hasSchema(qc)).toBe(true);
});

test("hasSpec is false with no spec, true with tables", () => {
  const qc = client();
  expect(hasSpec(qc)).toBe(false);
  qc.setQueryData(queryKeys.spec, { tables: [{ table_name: "users" }] });
  expect(hasSpec(qc)).toBe(true);
});

test("hasSucceededJob scans parameterized job keys", () => {
  const qc = client();
  expect(hasSucceededJob(qc)).toBe(false);
  qc.setQueryData(queryKeys.job("a"), { status: "running" });
  expect(hasSucceededJob(qc)).toBe(false);
  qc.setQueryData(queryKeys.job("b"), { status: "succeeded" });
  expect(hasSucceededJob(qc)).toBe(true);
});

test("STEP_IDS lists the 7 step ids in order", () => {
  expect(STEP_IDS).toEqual([
    "start",
    "schema",
    "configure",
    "generate",
    "validate",
    "output",
    "runs",
  ]);
});

test("every STEP id is a member of STEP_IDS", () => {
  for (const s of STEPS) {
    expect(STEP_IDS).toContain(s.id);
  }
});

test("gate truth table: each step's precondition", () => {
  const qc = client();
  const gateOf = (title: string) => STEPS.find((s) => s.title === title)!.gate;
  // Nothing loaded → only the no-precondition steps pass.
  expect(gateOf("Start")(qc)).toBe(false);
  expect(gateOf("Schema")(qc)).toBe(false);
  expect(gateOf("Configure")(qc)).toBe(false);
  expect(gateOf("Generate")(qc)).toBe(false);
  expect(gateOf("Validate")(qc)).toBe(true);
  expect(gateOf("Output")(qc)).toBe(true);
  expect(gateOf("Runs & Quality")(qc)).toBe(true);
  // Schema loaded → Start + Schema pass.
  qc.setQueryData(queryKeys.schema, { tables: [{ name: "t" }] });
  expect(gateOf("Start")(qc)).toBe(true);
  expect(gateOf("Schema")(qc)).toBe(true);
  expect(gateOf("Configure")(qc)).toBe(false);
  // Spec loaded → Configure passes.
  qc.setQueryData(queryKeys.spec, { tables: [{ table_name: "t" }] });
  expect(gateOf("Configure")(qc)).toBe(true);
  expect(gateOf("Generate")(qc)).toBe(false);
  // Succeeded job → Generate passes.
  qc.setQueryData(queryKeys.job("j"), { status: "succeeded" });
  expect(gateOf("Generate")(qc)).toBe(true);
});
