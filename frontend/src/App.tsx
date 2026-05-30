import { AppShell } from "./app/AppShell";
import { SchemaTree } from "./features/schema/SchemaTree";
import { SamplePicker } from "./features/start/SamplePicker";

export function App() {
  return (
    <AppShell>
      <section>
        <h2>Start</h2>
        <SamplePicker onLoaded={() => undefined} />
      </section>
      <section>
        <SchemaTree />
      </section>
    </AppShell>
  );
}
