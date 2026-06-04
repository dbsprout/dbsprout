import { AppShell } from "./app/AppShell";
import { GuidedSection } from "./app/GuidedSection";
import { useMode } from "./app/ModeProvider";
import { useSelection } from "./app/SelectionProvider";
import { STEPS } from "./app/steps";
import { SchemaTree } from "./features/schema/SchemaTree";
import { ConfigurePanel } from "./features/configure/ConfigurePanel";
import { SpecPanel } from "./features/configure/SpecPanel";
import { GenerateSurface } from "./features/generate/GenerateSurface";
import { RunsSurface } from "./features/runs/RunsSurface"; // P1c-4
import { StartPanel } from "./features/start/StartPanel";
import { ExportPanel } from "./features/export/ExportPanel"; // P1c-1
import { InsertPanel } from "./features/insert/InsertPanel"; // P1c-2
import { ValidatePanel } from "./features/validate/ValidatePanel"; // P1c-3

/** Index of the Configure step in the guided sequence (single source of truth). */
const configureStepIndex = STEPS.findIndex((s) => s.id === "configure");

export function App() {
  // ─── P4-7 ─── route a Validate violation's drill into the cross-panel selection
  // store; ConfigurePanel consumes it to focus the offending table+column.
  const { setSelection } = useSelection();
  // ─── P5-5 ─── in the guided wizard only the active step is visible; Configure is
  // hidden while you're on Validate, so the P4-7 selection alone is a visual no-op.
  // Navigate the wizard to the Configure step on drill so the now-visible
  // ConfigurePanel shows the focused cell. Advanced shows every section already, so
  // its behaviour is unchanged (the selection effect focuses the visible cell).
  const { mode, setStep } = useMode();
  return (
    <AppShell>
      {/* ═══ P3-2: each section is wrapped in GuidedSection (advanced = plain
          <section>; guided = dim non-active steps). The Output step owns both the
          Export and Insert sections (design spec §6 "Output (Export+Insert)"). ═══ */}
      <GuidedSection stepId="start">
        <h2 className="db-card-title">Start</h2>
        <StartPanel onLoaded={() => undefined} />
      </GuidedSection>
      <GuidedSection stepId="schema">
        <SchemaTree />
      </GuidedSection>
      <GuidedSection stepId="configure">
        <h2 className="db-card-title">Configure</h2>
        <SpecPanel />
        <ConfigurePanel />
      </GuidedSection>
      <GuidedSection stepId="generate">
        <h2 className="db-card-title">Generate</h2>
        <GenerateSurface />
      </GuidedSection>
      {/* ═══ P1c-1: Export ═══ */}
      <GuidedSection stepId="output">
        <h2 className="db-card-title">Output</h2>
        <ExportPanel />
      </GuidedSection>
      {/* ═══ /P1c-1 ═══ */}
      {/* ═══ P1c-2: Insert ═══ */}
      <GuidedSection stepId="output">
        <h2 className="db-card-title">Insert</h2>
        <InsertPanel />
      </GuidedSection>
      {/* ═══ /P1c-2 ═══ */}
      {/* ═══ P1c-3: Validate ═══ */}
      <GuidedSection stepId="validate">
        <h2 className="db-card-title">Validate</h2>
        {/* P4-7: a violation's "Drill to cell" sets the cross-panel selection,
            which ConfigurePanel applies to focus the offending table+column.
            P5-5: in guided mode also navigate the wizard to the (otherwise hidden)
            Configure step so the focused cell is visible. */}
        <ValidatePanel
          onDrill={({ table, column }) => {
            setSelection({ table, column });
            if (mode === "guided") setStep(configureStepIndex);
          }}
        />
      </GuidedSection>
      {/* ═══ /P1c-3 ═══ */}
      {/* ═══ P1c-4: Runs & Quality ═══ */}
      <GuidedSection stepId="runs">
        <h2 className="db-card-title">Runs & Quality</h2>
        <RunsSurface />
      </GuidedSection>
      {/* ═══ /P1c-4 ═══ */}
    </AppShell>
  );
}
