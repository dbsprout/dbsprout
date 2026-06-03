import { useEffect, useRef, useState } from "react";
import { jobSocketUrl } from "../../api/endpoints";
import type { JobProgress, JobProgressFrame } from "../../api/types";

/** What `useJobSocket` reports to a consumer (e.g. ProgressConsole). */
export interface JobSocketState {
  /** Latest live snapshot, or null before the first frame / when unsupported. */
  progress: JobProgress | null;
  /** True between the socket opening and a disconnect / terminal frame. */
  connected: boolean;
  /** True once the terminal frame has arrived — the run ended over the wire. */
  terminal: boolean;
}

const INITIAL: JobSocketState = { progress: null, connected: false, terminal: false };

function reduceFrame(prev: JobProgress | null, frame: JobProgressFrame): JobProgress {
  if (frame.phase === "terminal") {
    return {
      table: prev?.table ?? null,
      tablesDone: prev?.tablesDone ?? 0,
      tablesTotal: prev?.tablesTotal ?? 0,
      totalRows: prev?.totalRows ?? 0,
      status: frame.status,
      error: frame.error,
    };
  }
  return {
    table: frame.table,
    tablesDone: frame.tables_done,
    tablesTotal: frame.tables_total,
    totalRows: frame.total_rows,
    status: null,
    error: null,
  };
}

/**
 * Subscribe to the live job-progress WebSocket (`GET /ws/jobs/{jobId}`,
 * dbsprout/web/progress.py) and expose a reduced render snapshot.
 *
 * The server streams one event frame per table boundary, then a single terminal
 * frame, then closes. This hook tracks that lifecycle so a console can render
 * per-table progress AND decide whether to keep polling: a close/error *before*
 * the terminal frame surfaces as `connected=false, terminal=false`, the
 * disconnect signal the caller treats as "fall back to polling". A close after
 * the terminal frame is a clean shutdown and leaves `terminal=true`.
 *
 * Resilient by construction: an empty `jobId` opens nothing, a missing global
 * `WebSocket` (SSR / unsupported) degrades to the inert initial state, and a
 * malformed frame is ignored. The socket is always closed on unmount / jobId
 * change (no leaks).
 */
export function useJobSocket(jobId: string): JobSocketState {
  const [state, setState] = useState<JobSocketState>(INITIAL);
  // Tracks whether the terminal frame arrived, so a trailing close stays clean.
  const terminalRef = useRef(false);

  useEffect(() => {
    setState(INITIAL);
    terminalRef.current = false;

    if (!jobId || typeof WebSocket === "undefined") {
      return;
    }

    const socket = new WebSocket(jobSocketUrl(jobId));

    socket.onopen = () => {
      setState((s) => ({ ...s, connected: true }));
    };

    socket.onmessage = (ev: MessageEvent) => {
      let frame: JobProgressFrame;
      try {
        frame = JSON.parse(ev.data as string) as JobProgressFrame;
      } catch {
        return; // ignore a malformed frame
      }
      const isTerminalFrame = frame.phase === "terminal";
      if (isTerminalFrame) terminalRef.current = true;
      setState((s) => ({
        progress: reduceFrame(s.progress, frame),
        connected: s.connected,
        terminal: isTerminalFrame ? true : s.terminal,
      }));
      if (isTerminalFrame) socket.close();
    };

    const handleDisconnect = () => {
      // A close/error after the terminal frame is the server's clean shutdown.
      if (terminalRef.current) {
        setState((s) => ({ ...s, connected: false }));
        return;
      }
      setState((s) => ({ ...s, connected: false, terminal: false }));
    };

    socket.onclose = handleDisconnect;
    socket.onerror = handleDisconnect;

    return () => {
      socket.onopen = null;
      socket.onmessage = null;
      socket.onclose = null;
      socket.onerror = null;
      socket.close();
    };
  }, [jobId]);

  return state;
}
