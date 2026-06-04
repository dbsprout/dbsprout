import { useEffect, useRef, useState } from "react";
import type { GeneratorConfig, GeneratorMethod } from "../../api/types";
import { generatorOptions } from "./columnDtype";
import { genLabel } from "./genLabel";

interface ColumnInspectorProps {
  column: string;
  cfg: GeneratorConfig;
  /**
   * Generator catalogue. When supplied, a dtype-filtered generator picker is
   * rendered (P4-1). Optional so callers without the catalogue stay unchanged.
   */
  methods?: GeneratorMethod[];
  /** Raw SQL type of this column (e.g. `VARCHAR(255)`); drives the dtype filter. */
  columnType?: string;
  onSave: (cfg: GeneratorConfig) => void;
}

/**
 * Curated statistical distributions. The numpy engine special-cases `normal`
 * and falls back to `uniform` for everything else (`spec_driven._dispatch_numpy`);
 * the remaining names are carried through as labels for forward compatibility.
 * The empty string is the sentinel for "no distribution" (`null`).
 */
const DISTRIBUTIONS = [
  "uniform",
  "normal",
  "exponential",
  "zipf",
  "poisson",
  "lognormal",
] as const;

/** A single editable distribution-param row; string buffers tolerate mid-edit. */
interface ParamRow {
  key: string;
  value: string;
}

/**
 * Side panel for the focused column. Keyed on `column` so the inner form's local
 * draft state resets cleanly when the user inspects a different column.
 */
export function ColumnInspector(props: ColumnInspectorProps) {
  return <InspectorForm key={props.column} {...props} />;
}

/** Format a number for an input value buffer ("" for null/NaN). */
function numText(v: number | null): string {
  return v === null || Number.isNaN(v) ? "" : String(v);
}

/** Build editable rows from a numeric distribution_params record. */
function toParamRows(params: Record<string, number>): ParamRow[] {
  return Object.entries(params).map(([key, value]) => ({ key, value: String(value) }));
}

/** Render enum values as a newline-separated list for the textarea buffer. */
function enumText(values: string[] | null): string {
  return values ? values.join("\n") : "";
}

/** Parse the enum editor buffer (newline/comma separated) → string[] | null. */
function parseEnum(text: string): string[] | null {
  const items = text
    .split(/[\n,]/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0);
  return items.length > 0 ? items : null;
}

/** Parse a numeric input buffer: "" → null, finite number, else throw. */
function parseOptionalNumber(text: string, field: string): number | null {
  const trimmed = text.trim();
  if (trimmed === "") {
    return null;
  }
  const n = Number(trimmed);
  if (!Number.isFinite(n)) {
    throw new Error(`${field} must be numeric`);
  }
  return n;
}

/**
 * Reduce param rows → numeric record. Blank-key rows are dropped; a non-numeric
 * value on a keyed row throws so the caller can surface an inline error.
 */
function parseParamRows(rows: ParamRow[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const { key, value } of rows) {
    const k = key.trim();
    if (k === "") {
      continue;
    }
    const n = Number(value.trim());
    if (!Number.isFinite(n)) {
      throw new Error(`Distribution param "${k}" must be numeric`);
    }
    out[k] = n;
  }
  return out;
}

/** The full set of editable buffers derived from a persisted cfg. */
interface DraftBuffers {
  nullableRate: string;
  unique: boolean;
  paramsText: string;
  distribution: string;
  paramRows: ParamRow[];
  minText: string;
  maxText: string;
  enumValuesText: string;
}

/** Seed editable buffers from a persisted cfg (single source of truth). */
function seedFrom(cfg: GeneratorConfig): DraftBuffers {
  return {
    nullableRate: String(cfg.nullable_rate),
    unique: cfg.unique,
    paramsText: JSON.stringify(cfg.params),
    distribution: cfg.distribution ?? "",
    paramRows: toParamRows(cfg.distribution_params),
    minText: numText(cfg.min_value),
    maxText: numText(cfg.max_value),
    enumValuesText: enumText(cfg.enum_values),
  };
}

/** Structural equality of two buffer snapshots (detects an unsaved edit). */
function buffersEqual(a: DraftBuffers, b: DraftBuffers): boolean {
  if (
    a.nullableRate !== b.nullableRate ||
    a.unique !== b.unique ||
    a.paramsText !== b.paramsText ||
    a.distribution !== b.distribution ||
    a.minText !== b.minText ||
    a.maxText !== b.maxText ||
    a.enumValuesText !== b.enumValuesText ||
    a.paramRows.length !== b.paramRows.length
  ) {
    return false;
  }
  return a.paramRows.every(
    (row, i) => row.key === b.paramRows[i].key && row.value === b.paramRows[i].value,
  );
}

function InspectorForm({ column, cfg, methods, columnType, onSave }: ColumnInspectorProps) {
  // ── P4-1: dtype-filtered generator picker ──
  const [generator, setGenerator] = useState(genLabel(cfg.provider, cfg.method));
  const [showAllGenerators, setShowAllGenerators] = useState(false);
  const [nullableRate, setNullableRate] = useState(String(cfg.nullable_rate));
  const [unique, setUnique] = useState(cfg.unique);
  const [paramsText, setParamsText] = useState(JSON.stringify(cfg.params));
  const [distribution, setDistribution] = useState(cfg.distribution ?? "");
  const [paramRows, setParamRows] = useState<ParamRow[]>(() =>
    toParamRows(cfg.distribution_params),
  );
  const [minText, setMinText] = useState(numText(cfg.min_value));
  const [maxText, setMaxText] = useState(numText(cfg.max_value));
  const [enumValuesText, setEnumValuesText] = useState(enumText(cfg.enum_values));
  const [error, setError] = useState<string | null>(null);

  // ── P4-3: reconcile with external (grid-driven) edits to this same column ──
  // `baselineRef` holds the cfg the current buffers were seeded from; `seedRef`
  // holds that seed snapshot so we can tell a clean draft from an unsaved edit.
  const baselineRef = useRef<GeneratorConfig>(cfg);
  const seedRef = useRef<DraftBuffers>(seedFrom(cfg));
  const [externalChange, setExternalChange] = useState(false);

  /** Push a buffer snapshot into every input's state + reset the baselines. */
  function applySeed(next: GeneratorConfig) {
    const seed = seedFrom(next);
    setNullableRate(seed.nullableRate);
    setUnique(seed.unique);
    setParamsText(seed.paramsText);
    setDistribution(seed.distribution);
    setParamRows(seed.paramRows);
    setMinText(seed.minText);
    setMaxText(seed.maxText);
    setEnumValuesText(seed.enumValuesText);
    setGenerator(genLabel(next.provider, next.method));
    baselineRef.current = next;
    seedRef.current = seed;
    setExternalChange(false);
    setError(null);
  }

  // When the focused column's persisted cfg changes identity (e.g. the grid
  // edited the SAME column), reconcile: silently re-seed a clean draft, or warn
  // (preserve the edit) and offer an explicit refresh when there's unsaved work.
  useEffect(() => {
    if (cfg === baselineRef.current) {
      return;
    }
    const current: DraftBuffers = {
      nullableRate,
      unique,
      paramsText,
      distribution,
      paramRows,
      minText,
      maxText,
      enumValuesText,
    };
    if (buffersEqual(current, seedRef.current)) {
      applySeed(cfg);
    } else {
      setExternalChange(true);
    }
    // Reconciliation keys on the cfg reference only; buffer state is read
    // imperatively above and must not retrigger this effect.
  }, [cfg]);

  function patchRow(index: number, partial: Partial<ParamRow>) {
    setParamRows((rows) => rows.map((r, i) => (i === index ? { ...r, ...partial } : r)));
  }

  function addRow() {
    setParamRows((rows) => [...rows, { key: "", value: "" }]);
  }

  function removeRow(index: number) {
    setParamRows((rows) => rows.filter((_, i) => i !== index));
  }

  function buildDraft(): GeneratorConfig {
    let params: Record<string, unknown>;
    try {
      params = JSON.parse(paramsText) as Record<string, unknown>;
    } catch {
      throw new Error("Invalid JSON in params");
    }

    const distributionParams = parseParamRows(paramRows);
    const minValue = parseOptionalNumber(minText, "min");
    const maxValue = parseOptionalNumber(maxText, "max");
    if (minValue !== null && maxValue !== null && minValue > maxValue) {
      throw new Error("min must be ≤ max");
    }

    // P4-1: a "provider/method" (or bare "provider") label → provider + method.
    const [provider, method = null] = generator.split("/");

    return {
      ...cfg,
      provider,
      method,
      params,
      unique,
      nullable_rate: Number(nullableRate),
      distribution: distribution === "" ? null : distribution,
      distribution_params: distributionParams,
      min_value: minValue,
      max_value: maxValue,
      enum_values: parseEnum(enumValuesText),
    };
  }

  function handleSave() {
    let draft: GeneratorConfig;
    try {
      draft = buildDraft();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Invalid input");
      return;
    }
    setError(null);
    onSave(draft);
  }

  const pickerOptions = methods
    ? generatorOptions(methods, columnType, showAllGenerators, generator)
    : null;

  return (
    <aside aria-label={`inspector for ${column}`}>
      <h3>{column}</h3>
      <p>{genLabel(cfg.provider, cfg.method)}</p>

      {pickerOptions && (
        <>
          <label>
            generator method
            <select value={generator} onChange={(e) => setGenerator(e.target.value)}>
              {pickerOptions.map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>
          </label>
          <label>
            <input
              type="checkbox"
              aria-label="show all generators"
              checked={showAllGenerators}
              onChange={(e) => setShowAllGenerators(e.target.checked)}
            />
            show all generators
          </label>
        </>
      )}

      {externalChange && (
        <div role="status">
          <span>This column changed elsewhere — your edits are preserved.</span>
          <button type="button" onClick={() => applySeed(cfg)}>
            Refresh
          </button>
        </div>
      )}

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
        distribution
        <select value={distribution} onChange={(e) => setDistribution(e.target.value)}>
          <option value="">(none)</option>
          {DISTRIBUTIONS.map((d) => (
            <option key={d} value={d}>
              {d}
            </option>
          ))}
        </select>
      </label>

      <fieldset>
        <legend>distribution params</legend>
        {paramRows.map((row, i) => (
          <div key={i}>
            <input
              aria-label={`param name ${i}`}
              value={row.key}
              placeholder="name"
              onChange={(e) => patchRow(i, { key: e.target.value })}
            />
            <input
              type="text"
              inputMode="decimal"
              aria-label={`value for ${row.key || `param ${i}`}`}
              value={row.value}
              placeholder="number"
              onChange={(e) => patchRow(i, { value: e.target.value })}
            />
            <button
              type="button"
              aria-label={`remove param ${row.key || i}`}
              onClick={() => removeRow(i)}
            >
              ×
            </button>
          </div>
        ))}
        <button type="button" onClick={addRow}>
          Add param
        </button>
      </fieldset>

      <label>
        min
        <input
          type="text"
          inputMode="decimal"
          value={minText}
          onChange={(e) => setMinText(e.target.value)}
        />
      </label>

      <label>
        max
        <input
          type="text"
          inputMode="decimal"
          value={maxText}
          onChange={(e) => setMaxText(e.target.value)}
        />
      </label>

      <label>
        enum values
        <textarea
          value={enumValuesText}
          placeholder="one value per line"
          onChange={(e) => setEnumValuesText(e.target.value)}
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
