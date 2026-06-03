import { AppShell } from "./app/AppShell";
import { SchemaTree } from "./features/schema/SchemaTree";
import { ConfigurePanel } from "./features/configure/ConfigurePanel";
import { SpecPanel } from "./features/configure/SpecPanel";
import { GenerateSurface } from "./features/generate/GenerateSurface";
import { StartPanel } from "./features/start/StartPanel";
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
      {/* ═══ P1c-3: Validate ═══ */}
      <section>
        <h2>Validate</h2>
        {/* Drill seam: no shared cross-panel selection store yet, so wire a no-op
            stub. A future story can route this to focus the Configure ColumnGrid. */}
        <ValidatePanel onDrill={() => undefined} />
      </section>
      {/* ═══ /P1c-3 ═══ */}
    </AppShell>
  );
}
