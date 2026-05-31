import { AppShell } from "./app/AppShell";
import { SchemaTree } from "./features/schema/SchemaTree";
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
    </AppShell>
  );
}
