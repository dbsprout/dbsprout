import type { ReactNode } from "react";
import { useMode } from "./ModeProvider";
import { Stepper } from "./Stepper";

interface AppShellProps {
  children: ReactNode;
}

/**
 * `[Guided | Advanced]` mode toggle. Only sets the mode context — it never
 * unmounts the workspace panels, so all query/UI state survives a switch.
 */
function ModeToggle() {
  const { mode, setMode } = useMode();
  return (
    <div role="group" aria-label="Mode">
      <button
        type="button"
        aria-pressed={mode === "guided"}
        onClick={() => setMode("guided")}
      >
        Guided
      </button>
      <button
        type="button"
        aria-pressed={mode === "advanced"}
        onClick={() => setMode("advanced")}
      >
        Advanced
      </button>
    </div>
  );
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div>
      <header>
        <h1>
          <span aria-hidden="true">🌱</span>{" "}
          <span>DBSprout Workbench</span>
        </h1>
        <ModeToggle />
        <Stepper />
      </header>
      <main>{children}</main>
    </div>
  );
}
