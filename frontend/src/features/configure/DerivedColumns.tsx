import { useState } from "react";
import type { DerivedColumn } from "../../api/types";

interface DerivedColumnsProps {
  /** Active table name (used for labelling + draft reset). */
  table: string;
  /** Column names available on the active table (for the depends_on picker). */
  columns: string[];
  /** Current derived columns for the active table. */
  derived: DerivedColumn[];
  /** Persist the full new derived list. */
  onSave: (derived: DerivedColumn[]) => void;
}

/**
 * Edits a table's expression-based derived columns. Keyed on `table` so the
 * add-derived draft resets cleanly when the user switches tables.
 */
export function DerivedColumns(props: DerivedColumnsProps) {
  return <Editor key={props.table} {...props} />;
}

function Editor({ table, columns, derived, onSave }: DerivedColumnsProps) {
  const [name, setName] = useState("");
  const [expression, setExpression] = useState("");
  const [dependsOn, setDependsOn] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);

  function handleAdd() {
    if (name.trim() === "" || expression.trim() === "") {
      setError("A derived column needs both a name and expression.");
      return;
    }
    setError(null);
    const col: DerivedColumn = {
      column: name.trim(),
      expression: expression.trim(),
      depends_on: dependsOn,
    };
    onSave([...derived, col]);
    setName("");
    setExpression("");
    setDependsOn([]);
  }

  function handleRemove(index: number) {
    onSave(derived.filter((_, i) => i !== index));
  }

  return (
    <section aria-label={`derived for ${table}`}>
      <h4>Derived columns</h4>
      <ul aria-label={`derived columns for ${table}`}>
        {derived.length === 0 && <li>No derived columns.</li>}
        {derived.map((col, i) => (
          <li key={`${col.column}-${i}`}>
            <span>
              {col.column} = {col.expression}
            </span>
            {col.depends_on.length > 0 && <span> [{col.depends_on.join(", ")}]</span>}
            <button
              type="button"
              aria-label={`remove derived ${col.column}`}
              onClick={() => handleRemove(i)}
            >
              remove
            </button>
          </li>
        ))}
      </ul>

      <fieldset>
        <legend>Add derived column</legend>
        <label>
          name
          <input
            aria-label="derived column name"
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
        </label>
        <label>
          expression
          <input
            aria-label="derived expression"
            value={expression}
            onChange={(e) => setExpression(e.target.value)}
          />
        </label>
        <label>
          depends on
          <select
            multiple
            aria-label="depends on"
            value={dependsOn}
            onChange={(e) =>
              setDependsOn(Array.from(e.target.selectedOptions, (o) => o.value))
            }
          >
            {columns.map((c) => (
              <option key={c} value={c}>
                {c}
              </option>
            ))}
          </select>
        </label>
        {error && <p role="alert">{error}</p>}
        <button type="button" onClick={handleAdd}>
          add derived
        </button>
      </fieldset>
    </section>
  );
}
