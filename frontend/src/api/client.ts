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

export async function apiUpload<T>(path: string, form: FormData): Promise<T> {
  // No JSON content-type: the browser sets the multipart boundary itself.
  return parse<T>(await fetch(path, { method: "POST", body: form }));
}
