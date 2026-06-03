import { AppShell } from "./app/AppShell";
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
      <section>
        <h2>Start</h2>
        <StartPanel onLoaded={() => undefined} />
      </section>
      <section>
        <SchemaTree />
      </section>
      <section>
        <h2>Configure</h2>
        <SpecPanel />
        <ConfigurePanel />
      </section>
      <section>
        <h2>Generate</h2>
        <GenerateSurface />
      </section>
      {/* ═══ P1c-1: Export ═══ */}
      <section>
        <h2>Output</h2>
        <ExportPanel />
      </section>
      {/* ═══ /P1c-1 ═══ */}
      {/* ═══ P1c-2: Insert ═══ */}
      <section>
        <h2>Insert</h2>
        <InsertPanel />
      </section>
      {/* ═══ /P1c-2 ═══ */}
      {/* ═══ P1c-3: Validate ═══ */}
      <section>
        <h2>Validate</h2>
        {/* Drill seam: no shared cross-panel selection store yet, so wire a no-op
            stub. A future story can route this to focus the Configure ColumnGrid. */}
        <ValidatePanel onDrill={() => undefined} />
      </section>
      {/* ═══ /P1c-3 ═══ */}
      {/* ═══ P1c-4: Runs & Quality ═══ */}
      <section>
        <h2>Runs & Quality</h2>
        <RunsSurface />
      </section>
      {/* ═══ /P1c-4 ═══ */}
    </AppShell>
  );
}
