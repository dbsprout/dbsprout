import { useState } from "react";
import { ConnectForm } from "./ConnectForm";
import { PasteForm } from "./PasteForm";
import { SamplePicker } from "./SamplePicker";
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

  return (
    <div>
      <div role="tablist">
        {TABS.map(({ id, label }) => (
          <button
            key={id}
            role="tab"
            aria-selected={activeTab === id}
            onClick={() => setActiveTab(id)}
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
    </div>
  );
}
