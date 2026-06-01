import { AppShell } from "./app/AppShell";
import { SchemaTree } from "./features/schema/SchemaTree";
import { SpecPanel } from "./features/configure/SpecPanel";
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
      </section>
    </AppShell>
  );
}
