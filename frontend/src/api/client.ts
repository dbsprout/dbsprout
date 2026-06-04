import type { ApiErrorBody } from "./types";

export class ApiError extends Error {
  readonly name = "ApiError";
  readonly status: number;
  readonly code: string;
  readonly hint?: string;

  constructor(status: number, body: Partial<ApiErrorBody>) {
    super(body.message ?? `request failed (${status})`);
    this.status = status;
    this.code = body.code ?? "UNKNOWN";
    this.hint = body.hint;
  }
}

async function parse<T>(resp: Response): Promise<T> {
  let data: unknown = null;
  try {
    data = await resp.json();
  } catch {
    if (!resp.ok) throw new ApiError(resp.status, { message: `request failed (${resp.status})` });
    return null as T;
  }
  if (!resp.ok) {
    const detail = (data as { detail?: Partial<ApiErrorBody> } | null)?.detail ?? {};
    throw new ApiError(resp.status, detail);
  }
  return data as T;
}

export async function apiGet<T>(path: string): Promise<T> {
  return parse<T>(await fetch(path));
}

export async function apiPost<T>(path: string, body?: unknown): Promise<T> {
  return parse<T>(
    await fetch(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

export async function apiPut<T>(path: string, body?: unknown): Promise<T> {
  return parse<T>(
    await fetch(path, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: body === undefined ? undefined : JSON.stringify(body),
    }),
  );
}

export async function apiUpload<T>(path: string, form: FormData): Promise<T> {
  // No JSON content-type: the browser sets the multipart boundary itself.
  return parse<T>(await fetch(path, { method: "POST", body: form }));
}

/** Parse the download filename out of a ``Content-Disposition`` header. */
function filenameFromDisposition(header: string | null): string | undefined {
  if (!header) return undefined;
  // attachment; filename="users.sql"  (also tolerate an unquoted filename)
  const match = /filename\*?=(?:UTF-8'')?"?([^";]+)"?/i.exec(header);
  return match?.[1];
}

/**
 * POST a JSON body and download the response as a file.
 *
 * On a 2xx the response body is read as a `Blob` and handed to the browser via a
 * transient object URL + a synthetic `<a download>` click; the filename comes from
 * the `Content-Disposition` header, falling back to `fallbackName`. On a non-OK
 * response the (JSON) error envelope is parsed and re-thrown as an {@link ApiError}
 * — no download is triggered.
 */
export async function apiDownload(
  path: string,
  body: unknown,
  fallbackName: string,
): Promise<void> {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!resp.ok) {
    let detail: Partial<ApiErrorBody> = {};
    try {
      const data = (await resp.json()) as { detail?: Partial<ApiErrorBody> } | null;
      detail = data?.detail ?? {};
    } catch {
      // Non-JSON error body — fall through to a status-only ApiError.
    }
    throw new ApiError(resp.status, detail);
  }

  const blob = await resp.blob();
  const filename = filenameFromDisposition(resp.headers.get("Content-Disposition")) ?? fallbackName;
  const url = URL.createObjectURL(blob);
  try {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  } finally {
    URL.revokeObjectURL(url);
  }
}
