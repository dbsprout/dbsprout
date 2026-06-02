import { AppShell } from "./app/AppShell";
import { SchemaTree } from "./features/schema/SchemaTree";
import { ConfigurePanel } from "./features/configure/ConfigurePanel";
import { SpecPanel } from "./features/configure/SpecPanel";
import { GenerateSurface } from "./features/generate/GenerateSurface";
import { StartPanel } from "./features/start/StartPanel";

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
    </AppShell>
  );
}
