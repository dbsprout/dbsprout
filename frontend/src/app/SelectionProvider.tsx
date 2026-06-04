import { createContext, useContext, useState, type ReactNode } from "react";

/**
 * A cross-panel focus target: the table (and optionally column) a violation,
 * link, or other surface wants the Configure grid to drill into. `column` is
 * nullable because table-level violations (e.g. row-count / FK) carry no column.
 */
export interface SelectionTarget {
  table: string;
  column: string | null;
  /**
   * P5-7: an optional short reason explaining WHY this cell was focused (e.g. the
   * Validate violation that triggered the drill). Consumers may show it as
   * cross-panel context; it carries no behavior of its own.
   */
  reason?: string;
}

interface SelectionContextValue {
  /** The current focus target, or null when nothing is focused. */
  selection: SelectionTarget | null;
  /** Focus a table+column across panels (e.g. from a Validate drill). */
  setSelection: (target: SelectionTarget) => void;
  /** Clear the focus target once a consumer has applied it. */
  clearSelection: () => void;
}

const SelectionContext = createContext<SelectionContextValue | null>(null);

/**
 * The cross-panel selection store. Mirrors `ModeProvider` but holds ephemeral,
 * in-memory UI focus state — there is nothing worth persisting across reloads, so
 * (unlike the mode) it is not written to localStorage. Wrapping the App in this
 * provider does not change any panel's rendering until a `setSelection` arrives.
 */
export function SelectionProvider({ children }: { children: ReactNode }) {
  const [selection, setSelectionState] = useState<SelectionTarget | null>(null);

  const value: SelectionContextValue = {
    selection,
    setSelection: setSelectionState,
    clearSelection: () => setSelectionState(null),
  };

  return <SelectionContext.Provider value={value}>{children}</SelectionContext.Provider>;
}

/** Read the selection context; throws if used outside a <SelectionProvider>. */
export function useSelection(): SelectionContextValue {
  const ctx = useContext(SelectionContext);
  if (!ctx) {
    throw new Error("useSelection must be used within a <SelectionProvider>");
  }
  return ctx;
}
