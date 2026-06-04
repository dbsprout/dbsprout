import type { ReactNode } from "react";
import { useMode } from "./ModeProvider";
import { Stepper } from "./Stepper";
import { CoachPanel } from "./CoachPanel";
import { FirstRunSample } from "./FirstRunSample";

interface AppShellProps {
  children: ReactNode;
}

/**
 * `[Guided | Advanced]` mode toggle, styled as a segmented control. Only sets the
 * mode context — it never unmounts the workspace panels, so all query/UI state
 * survives a switch.
 */
function ModeToggle() {
  const { mode, setMode } = useMode();
  const base =
    "px-3 py-1 text-sm font-medium rounded-md transition cursor-pointer focus:outline-none focus:ring-2 focus:ring-accent-500";
  const active = "bg-white text-accent-700 shadow-sm";
  const inactive = "text-slate-600 hover:text-slate-900";
  return (
    <div
      role="group"
      aria-label="Mode"
      className="inline-flex items-center gap-1 rounded-lg border border-slate-200 bg-slate-100 p-1"
    >
      <button
        type="button"
        aria-pressed={mode === "guided"}
        onClick={() => setMode("guided")}
        className={`${base} ${mode === "guided" ? active : inactive}`}
      >
        Guided
      </button>
      <button
        type="button"
        aria-pressed={mode === "advanced"}
        onClick={() => setMode("advanced")}
        className={`${base} ${mode === "advanced" ? active : inactive}`}
      >
        Advanced
      </button>
    </div>
  );
}

export function AppShell({ children }: AppShellProps) {
  return (
    <div className="min-h-screen bg-slate-50">
      <header className="sticky top-0 z-30 border-b border-slate-200 bg-white/90 backdrop-blur">
        <div className="mx-auto flex max-w-5xl flex-wrap items-center justify-between gap-3 px-6 py-3">
          <h1 className="flex items-center gap-2 text-lg font-semibold text-slate-900">
            <span aria-hidden="true" className="text-xl">
              🌱
            </span>{" "}
            <span>DBSprout Workbench</span>
          </h1>
          <ModeToggle />
        </div>
        {/* ═══ P3-2: guided-only — render null in advanced mode, so the advanced
            header stays byte-identical. ═══ */}
        <div className="mx-auto max-w-5xl px-6 pb-3 empty:hidden">
          <Stepper />
          <FirstRunSample />
          <CoachPanel />
        </div>
      </header>
      <main className="mx-auto flex max-w-5xl flex-col gap-6 px-6 py-8">{children}</main>
    </div>
  );
}
