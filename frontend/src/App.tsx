import { AppShell } from "./app/AppShell";
import { GuidedSection } from "./app/GuidedSection";
import { SchemaTree } from "./features/schema/SchemaTree";
import { ConfigurePanel } from "./features/configure/ConfigurePanel";
import { SpecPanel } from "./features/configure/SpecPanel";
import { GenerateSurface } from "./features/generate/GenerateSurface";
import { RunsSurface } from "./features/runs/RunsSurface"; // P1c-4
import { StartPanel } from "./features/start/StartPanel";
import { ExportPanel } from "./features/export/ExportPanel"; // P1c-1
import { InsertPanel } from "./features/insert/InsertPanel"; // P1c-2
import { ValidatePanel } from "./features/validate/ValidatePanel"; // P1c-3

export function App() {
  return (
    <AppShell>
      {/* ═══ P3-2: each section is wrapped in GuidedSection (advanced = plain
          <section>; guided = dim non-active steps). The Output step owns both the
          Export and Insert sections (design spec §6 "Output (Export+Insert)"). ═══ */}
      <GuidedSection stepId="start">
        <h2>Start</h2>
        <StartPanel onLoaded={() => undefined} />
      </GuidedSection>
      <GuidedSection stepId="schema">
        <SchemaTree />
      </GuidedSection>
      <GuidedSection stepId="configure">
        <h2>Configure</h2>
        <SpecPanel />
        <ConfigurePanel />
      </GuidedSection>
      <GuidedSection stepId="generate">
        <h2>Generate</h2>
        <GenerateSurface />
      </GuidedSection>
      {/* ═══ P1c-1: Export ═══ */}
      <GuidedSection stepId="output">
        <h2>Output</h2>
        <ExportPanel />
      </GuidedSection>
      {/* ═══ /P1c-1 ═══ */}
      {/* ═══ P1c-2: Insert ═══ */}
      <GuidedSection stepId="output">
        <h2>Insert</h2>
        <InsertPanel />
      </GuidedSection>
      {/* ═══ /P1c-2 ═══ */}
      {/* ═══ P1c-3: Validate ═══ */}
      <GuidedSection stepId="validate">
        <h2>Validate</h2>
        {/* Drill seam: no shared cross-panel selection store yet, so wire a no-op
            stub. A future story can route this to focus the Configure ColumnGrid. */}
        <ValidatePanel onDrill={() => undefined} />
      </GuidedSection>
      {/* ═══ /P1c-3 ═══ */}
      {/* ═══ P1c-4: Runs & Quality ═══ */}
      <GuidedSection stepId="runs">
        <h2>Runs & Quality</h2>
        <RunsSurface />
      </GuidedSection>
      {/* ═══ /P1c-4 ═══ */}
    </AppShell>
  );
}
