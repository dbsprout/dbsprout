import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export type Mode = "advanced" | "guided";

const STORAGE_KEY = "dbsprout.mode";
const STEP_MIN = 0;
const STEP_MAX = 6; // 7 steps, indices 0..6 (see steps.ts)

interface ModeContextValue {
  mode: Mode;
  setMode: (mode: Mode) => void;
  currentStep: number;
  setStep: (step: number) => void;
}

const ModeContext = createContext<ModeContextValue | null>(null);

/**
 * Read the persisted mode, defaulting to "advanced" when absent, invalid, or
 * when localStorage is unavailable (private mode / quota / SSR).
 */
function readPersistedMode(): Mode {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw === "guided" || raw === "advanced" ? raw : "advanced";
  } catch {
    return "advanced";
  }
}

const clampStep = (step: number): number =>
  Math.max(STEP_MIN, Math.min(STEP_MAX, Math.trunc(step)));

/**
 * The first context/store in the SPA. Holds the Guided-vs-Advanced mode
 * (persisted to localStorage) and the guided stepper position. Wrapping the App
 * in this provider does not change advanced-mode rendering — panels are never
 * unmounted on a mode switch, so all workspace/query state is preserved.
 */
export function ModeProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<Mode>(readPersistedMode);
  const [currentStep, setStepState] = useState(0);

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, mode);
    } catch {
      // Storage unavailable (private mode / quota) — keep in-memory state only.
    }
  }, [mode]);

  const value: ModeContextValue = {
    mode,
    setMode: setModeState,
    currentStep,
    setStep: (step: number) => setStepState(clampStep(step)),
  };

  return <ModeContext.Provider value={value}>{children}</ModeContext.Provider>;
}

/** Read the mode context; throws if used outside a <ModeProvider>. */
export function useMode(): ModeContextValue {
  const ctx = useContext(ModeContext);
  if (!ctx) {
    throw new Error("useMode must be used within a <ModeProvider>");
  }
  return ctx;
}
