import { useState } from "react";
import type { GeneratorConfig } from "../../api/types";
import { genLabel } from "./genLabel";

interface ColumnInspectorProps {
  column: string;
  cfg: GeneratorConfig;
  onSave: (cfg: GeneratorConfig) => void;
}

/**
 * Side panel for the focused column. Keyed on `column` so the inner form's local
 * draft state resets cleanly when the user inspects a different column.
 */
export function ColumnInspector(props: ColumnInspectorProps) {
  return <InspectorForm key={props.column} {...props} />;
}

function InspectorForm({ column, cfg, onSave }: ColumnInspectorProps) {
  const [nullableRate, setNullableRate] = useState(String(cfg.nullable_rate));
  const [unique, setUnique] = useState(cfg.unique);
  const [paramsText, setParamsText] = useState(JSON.stringify(cfg.params));
  const [error, setError] = useState<string | null>(null);

  function handleSave() {
    let params: Record<string, unknown>;
    try {
      params = JSON.parse(paramsText) as Record<string, unknown>;
    } catch {
      setError("Invalid JSON in params");
      return;
    }
    setError(null);
    onSave({ ...cfg, nullable_rate: Number(nullableRate), unique, params });
  }

  return (
    <aside aria-label={`inspector for ${column}`}>
      <h3>{column}</h3>
      <p>{genLabel(cfg.provider, cfg.method)}</p>
      <label>
        null %
        <input
          type="number"
          step="0.01"
          min={0}
          max={1}
          value={nullableRate}
          onChange={(e) => setNullableRate(e.target.value)}
        />
      </label>
      <label>
        unique
        <input
          type="checkbox"
          checked={unique}
          onChange={(e) => setUnique(e.target.checked)}
        />
      </label>
      <label>
        params
        <textarea value={paramsText} onChange={(e) => setParamsText(e.target.value)} />
      </label>
      {error && <p role="alert">{error}</p>}
      <button type="button" onClick={handleSave}>
        Save
      </button>
      <button type="button" onClick={() => onSave(cfg)}>
        Re-roll
      </button>
    </aside>
  );
}
