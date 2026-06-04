import { useState } from "react";
import { ConnectForm } from "./ConnectForm";
import { PasteForm } from "./PasteForm";
import { SamplePicker } from "./SamplePicker";
import { SavedConnections } from "./SavedConnections";
import { UploadForm } from "./UploadForm";

type Tab = "connect" | "upload" | "paste" | "sample";

interface TabDef {
  id: Tab;
  label: string;
}

const TABS: TabDef[] = [
  { id: "connect", label: "Live database" },
  { id: "upload", label: "Upload" },
  { id: "paste", label: "Paste" },
  { id: "sample", label: "Sample" },
];

interface StartPanelProps {
  onLoaded: () => void;
}

export function StartPanel({ onLoaded }: StartPanelProps) {
  const [activeTab, setActiveTab] = useState<Tab>("connect");
  // ═══ P2a-2 ═══ — the URL of the most recently loaded saved connection.
  const [loadedUrl, setLoadedUrl] = useState<string | null>(null);

  function handleLoadConnection(url: string) {
    setLoadedUrl(url);
    setActiveTab("connect");
  }

  return (
    <div>
      <div role="tablist" className="mb-4 inline-flex gap-1 rounded-lg border border-slate-200 bg-slate-100 p-1">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            role="tab"
            aria-selected={activeTab === id}
            onClick={() => setActiveTab(id)}
            className={`cursor-pointer rounded-md px-3 py-1 text-sm font-medium transition focus:outline-none focus:ring-2 focus:ring-accent-500 ${
              activeTab === id
                ? "bg-white text-accent-700 shadow-sm"
                : "text-slate-600 hover:text-slate-900"
            }`}
          >
            {label}
          </button>
        ))}
      </div>

      <div role="tabpanel">
        {activeTab === "connect" && <ConnectForm onLoaded={onLoaded} />}
        {activeTab === "upload" && <UploadForm onLoaded={onLoaded} />}
        {activeTab === "paste" && <PasteForm onLoaded={onLoaded} />}
        {activeTab === "sample" && <SamplePicker onLoaded={onLoaded} />}
      </div>

      {/* ═══ P2a-2 ═══ */}
      {loadedUrl !== null && (
        <p role="status" className="db-notice-status mt-3">{`Loaded connection: ${loadedUrl}`}</p>
      )}
      <SavedConnections onLoad={handleLoadConnection} />
      {/* ═══ end P2a-2 ═══ */}
    </div>
  );
}
