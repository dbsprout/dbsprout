import { act, renderHook } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { ModeProvider, useMode } from "./ModeProvider";

const STORAGE_KEY = "dbsprout.mode";

function wrapper({ children }: { children: ReactNode }) {
  return <ModeProvider>{children}</ModeProvider>;
}

beforeEach(() => localStorage.clear());
afterEach(() => localStorage.clear());

test("defaults to advanced when nothing persisted", () => {
  const { result } = renderHook(() => useMode(), { wrapper });
  expect(result.current.mode).toBe("advanced");
  expect(result.current.currentStep).toBe(0);
});

test("setMode persists to localStorage", () => {
  const { result } = renderHook(() => useMode(), { wrapper });
  act(() => result.current.setMode("guided"));
  expect(result.current.mode).toBe("guided");
  expect(localStorage.getItem(STORAGE_KEY)).toBe("guided");
});

test("hydrates persisted guided mode on mount", () => {
  localStorage.setItem(STORAGE_KEY, "guided");
  const { result } = renderHook(() => useMode(), { wrapper });
  expect(result.current.mode).toBe("guided");
});

test("coerces an invalid persisted value to advanced", () => {
  localStorage.setItem(STORAGE_KEY, "banana");
  const { result } = renderHook(() => useMode(), { wrapper });
  expect(result.current.mode).toBe("advanced");
});

test("setStep clamps to [0, 6]", () => {
  const { result } = renderHook(() => useMode(), { wrapper });
  act(() => result.current.setStep(99));
  expect(result.current.currentStep).toBe(6);
  act(() => result.current.setStep(-5));
  expect(result.current.currentStep).toBe(0);
});

test("useMode throws outside a provider", () => {
  expect(() => renderHook(() => useMode())).toThrow(/ModeProvider/);
});

test("falls back to advanced when localStorage.getItem throws", () => {
  const spy = vi.spyOn(Storage.prototype, "getItem").mockImplementation(() => {
    throw new Error("blocked");
  });
  try {
    const { result } = renderHook(() => useMode(), { wrapper });
    expect(result.current.mode).toBe("advanced");
  } finally {
    spy.mockRestore();
  }
});

test("setMode tolerates a localStorage.setItem failure (memory-only)", () => {
  const spy = vi.spyOn(Storage.prototype, "setItem").mockImplementation(() => {
    throw new Error("quota");
  });
  try {
    const { result } = renderHook(() => useMode(), { wrapper });
    act(() => result.current.setMode("guided"));
    expect(result.current.mode).toBe("guided");
  } finally {
    spy.mockRestore();
  }
});
