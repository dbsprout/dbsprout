import { useMutation } from "@tanstack/react-query";
import { useState } from "react";
import { ApiError } from "../../api/client";
import { insertData, insertPreview } from "../../api/endpoints";
import type { InsertMethod, InsertPreview, InsertResponse } from "../../api/types";
import { InsertProgress } from "./InsertProgress";

const METHODS: readonly InsertMethod[] = ["auto", "batch", "copy"];

interface InsertPanelProps {
  /** Poll interval forwarded to the progress console (kept short in tests). */
  pollMs?: number;
}

/**
 * The direct-insert surface (write generated rows into the connected DB).
 *
 * Step machine:
 *   1. idle      → "Preview insert" requests a write-guard preview (scope + a
 *                  single-use, scope-bound HMAC token, held only in component
 *                  state — never logged, never rendered).
 *   2. previewed → render the scope (per-table row counts) + warnings + a method
 *                  select, then an explicit "Confirm & insert" gate. Confirming
 *                  POSTs the held token + method and starts a background job.
 *   3. running   → mount <InsertProgress> to poll the job to a terminal state,
 *                  with a Cancel button. On success show the inserted-rows
 *                  summary; on failure show the scrubbed typed error.
 *
 * A rejected token / failed start clears the held token and returns to the
 * preview step (the backend token is single-use, so the user must re-preview).
 */
export function InsertPanel({ pollMs }: InsertPanelProps) {
  const [preview, setPreview] = useState<InsertPreview | null>(null);
  const [job, setJob] = useState<InsertResponse | null>(null);
  const [method, setMethod] = useState<InsertMethod>("auto");
  const [succeeded, setSucceeded] = useState(false);

  const previewMutation = useMutation({
    mutationFn: () => insertPreview(),
    onSuccess: (data) => {
      setJob(null);
      setSucceeded(false);
      setPreview(data);
    },
  });

  const insertMutation = useMutation({
    mutationFn: (token: string) =>
      insertData({ tables: null, confirmation_token: token, method }),
    onSuccess: (res) => {
      // Token is single-use; drop the preview (and its token) once consumed.
      setPreview(null);
      setJob(res);
    },
  });

  function handlePreview() {
    setJob(null);
    setSucceeded(false);
    previewMutation.mutate();
  }

  function handleConfirm() {
    if (preview) insertMutation.mutate(preview.confirmation_token);
  }

  return (
    <div className="flex flex-col gap-3">
      <button
        type="button"
        className="db-btn-secondary self-start"
        disabled={previewMutation.isPending}
        onClick={handlePreview}
      >
        Preview insert
      </button>

      {previewMutation.isError && (
        <p role="alert" className="db-notice-alert">{(previewMutation.error as ApiError).message}</p>
      )}

      {preview && !job && (
        <div className="db-subsection flex flex-col gap-2">
          <p className="text-sm text-slate-600">
            Target: <span className="font-mono">{preview.target}</span> · {preview.dialect}
          </p>
          <ul className="flex flex-col gap-1 font-mono text-sm text-slate-700">
            {preview.scope.map((s) => (
              <li key={s.table}>
                {s.table} — {s.row_count.toLocaleString()} rows
              </li>
            ))}
          </ul>
          <p className="text-sm text-slate-600">
            Total: <strong className="text-slate-900">{preview.total_rows.toLocaleString()}</strong> rows
          </p>
          {preview.warnings && preview.warnings.length > 0 && (
            <ul aria-label="insert warnings" className="db-notice-status flex flex-col gap-1">
              {preview.warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
          <label className="db-field mb-0">
            <span className="db-label">insert method</span>
            <select
              aria-label="insert method"
              className="db-input"
              value={method}
              onChange={(e) => setMethod(e.target.value as InsertMethod)}
            >
              {METHODS.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="db-btn-danger self-start"
            disabled={insertMutation.isPending}
            onClick={handleConfirm}
          >
            Confirm &amp; insert
          </button>
          {insertMutation.isError && (
            <p role="alert" className="db-notice-alert">{(insertMutation.error as ApiError).message}</p>
          )}
        </div>
      )}

      {job && (
        <InsertProgress
          jobId={job.job_id}
          pollMs={pollMs}
          onSucceeded={() => setSucceeded(true)}
        />
      )}

      {job && succeeded && (
        <div className="db-subsection">
          <h3 className="db-subsection-title">Inserted</h3>
          <p className="text-sm text-slate-700">
            {job.total_rows.toLocaleString()} rows across {job.scope.length} tables via{" "}
            {job.writer} ({job.method}).
          </p>
          {job.scope_warnings.length > 0 && (
            <ul aria-label="scope warnings" className="db-notice-status mt-2 flex flex-col gap-1">
              {job.scope_warnings.map((w) => (
                <li key={w}>{w}</li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  );
}
