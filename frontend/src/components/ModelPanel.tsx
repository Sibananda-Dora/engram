import { useState } from "react";
import { api } from "../lib/api";
import NeuronField from "./NeuronField";
import type { Neuron } from "../lib/types";

type Props = {
  side: "A" | "B";
  accent: string;
  glow: string;
  modelKey: string | null;
  models: any[];
  onLoad: (key: string) => void;
  selected: { layer: number; neuron: number } | null;
  onSelectNeuron: (l: number, n: number) => void;
  onDragStart: (side: "A" | "B", layer: number, neuron: number) => void;
  layers: Record<string, Neuron[]>;
  numLayers: number;
  highlighted?: Set<string>;
  pruned: Set<string>;
  fitness: number | null;
  onActivations: (prompt: string) => void;
  onPrune: (ratio: number) => Promise<void> | void;
  chatLog: { role: string; content: string }[];
  onSend: (text: string) => void;
  busy: boolean;
};

export default function ModelPanel({
  side,
  accent,
  glow,
  modelKey,
  models,
  onLoad,
  selected,
  onSelectNeuron,
  onDragStart,
  layers,
  numLayers,
  highlighted,
  pruned,
  fitness,
  onActivations,
  onPrune,
  chatLog,
  onSend,
  busy,
}: Props) {
  const [input, setInput] = useState("");
  const [actPrompt, setActPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [pruneRatio, setPruneRatio] = useState(0.25);
  const [pruning, setPruning] = useState(false);
  const [exporting, setExporting] = useState(false);

  const handlePrune = async () => {
    if (!modelKey || pruning) return;
    setPruning(true);
    try {
      await onPrune(pruneRatio);
    } finally {
      setPruning(false);
    }
  };

  const handleExport = async () => {
    if (!modelKey || exporting) return;
    const suggested = `${side === "A" ? "model-a" : "model-b"}-${modelKey.split("-").pop()}`;
    const name = window.prompt(
      "Export name (letters, numbers, dot, underscore, hyphen only):",
      suggested
    );
    if (!name) return;
    setExporting(true);
    try {
      const r = await api.saveModel(modelKey, name);
      alert(`Model saved as .safetensors:\n${r.path}`);
    } catch (e: any) {
      alert("Export failed: " + e.message);
    } finally {
      setExporting(false);
    }
  };

  const handleLoad = async (key: string) => {
    setLoading(true);
    try {
      await api.loadModel(key);
      onLoad(key);
    } catch (e: any) {
      alert("Load failed: " + e.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 py-3 border-b border-white/10">
        <div className="flex items-center gap-2">
          <span
            className="w-2.5 h-2.5 rounded-full pulse-glow shrink-0"
            style={{ background: accent, boxShadow: glow }}
            aria-hidden="true"
          />
          <span className="font-semibold text-sm tracking-wide">MODEL {side}</span>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <select
            aria-label={`Select model for Model ${side}`}
            className="bg-black/40 border border-white/15 rounded-md px-2 py-1 text-xs text-white max-w-[190px] focus:outline-none focus:border-white/40"
            value={modelKey || ""}
            onChange={(e) => handleLoad(e.target.value)}
            disabled={loading}
          >
            <option value="">— select model —</option>
            {models.map((m) => (
              <option key={m.key} value={m.key}>
                {m.label} {m.loaded ? "✓" : ""}
              </option>
            ))}
          </select>
          <button
            onClick={handleExport}
            disabled={!modelKey || exporting}
            title="Save this model's current weights as .safetensors"
            aria-label={`Export Model ${side} as safetensors`}
            className="px-2.5 py-1 rounded-md text-[10px] font-medium border border-white/20 hover:bg-white/10 disabled:opacity-40 transition-colors"
          >
            {exporting ? "saving…" : "Export .safetensors"}
          </button>
        </div>
      </div>

      {/* Chat */}
      <div className="flex-1 flex flex-col min-h-[160px]">
        <div className="flex-1 overflow-y-auto scrollbar px-3 py-2 space-y-2">
          {chatLog.length === 0 && (
            <div className="text-xs text-dim text-center py-6 px-4">
              <div className="font-medium text-white/80 mb-1">
                {modelKey ? `Model ${side} Ready` : "No Model Loaded"}
              </div>
              <p className="text-[11px] leading-relaxed">
                {modelKey
                  ? "Chat with this model or inspect neurons below to view its internal activations."
                  : "Choose a model from the dropdown above to begin inspecting and experimenting."}
              </p>
            </div>
          )}
          {chatLog.map((m, i) => (
            <div
              key={i}
              className={`text-sm px-3 py-2 rounded-lg max-w-[90%] ${
                m.role === "user"
                  ? "ml-auto bg-white/5 border border-white/10"
                  : "mr-auto border"
              }`}
              style={m.role !== "user" ? { borderColor: accent + "40", background: accent + "10" } : {}}
            >
              <div className="text-[10px] opacity-60 mb-0.5 uppercase tracking-wider">{m.role}</div>
              <div className="whitespace-pre-wrap leading-relaxed">{m.content}</div>
            </div>
          ))}
          {busy && (
            <div className="text-xs text-dim italic animate-pulse" aria-live="polite">generating…</div>
          )}
        </div>
        <div className="p-2 border-t border-white/10 flex gap-2">
          <input
            className="flex-1 bg-black/40 border border-white/15 rounded-md px-3 py-1.5 text-sm text-white focus:outline-none focus:border-white/40"
            placeholder={`Message Model ${side}…`}
            aria-label={`Message Model ${side}`}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && input.trim()) {
                onSend(input.trim());
                setInput("");
              }
            }}
          />
          <button
            onClick={() => {
              if (input.trim()) {
                onSend(input.trim());
                setInput("");
              }
            }}
            aria-label="Send message"
            className="px-3.5 py-1.5 rounded-md text-xs font-semibold hover:opacity-90 transition-opacity"
            style={{ background: accent, color: "#0a0e1a" }}
          >
            Send
          </button>
        </div>
      </div>

      {/* Activations trigger */}
      <div className="px-3 py-2 border-t border-white/10 flex gap-2">
        <input
          className="flex-1 bg-black/40 border border-white/15 rounded-md px-2.5 py-1.5 text-xs text-white focus:outline-none focus:border-white/40"
          placeholder="Prompt to inspect neurons (e.g. 'What is truth?')"
          aria-label="Prompt to inspect neurons"
          value={actPrompt}
          onChange={(e) => setActPrompt(e.target.value)}
        />
        <button
          onClick={() => actPrompt.trim() && onActivations(actPrompt.trim())}
          aria-label="Inspect neuron activations for this prompt"
          className="px-3 py-1.5 rounded-md text-xs font-medium border border-white/20 hover:bg-white/10 transition-colors"
        >
          Inspect
        </button>
      </div>

      {/* Prune controls */}
      <div className="px-3 py-2 border-t border-white/10 flex items-center gap-2 flex-wrap text-xs">
        <span className="text-[10px] text-dim tracking-wider font-semibold">PRUNE</span>
        {[0.1, 0.25, 0.5].map((r) => (
          <button
            key={r}
            onClick={() => setPruneRatio(r)}
            disabled={pruning}
            aria-label={`Set prune ratio to ${Math.round(r * 100)} percent`}
            className={`px-2 py-0.5 rounded text-[10px] font-mono border transition-colors ${
              pruneRatio === r
                ? "border-[#ffd166] text-[#ffd166] bg-[#ffd16610]"
                : "border-white/15 text-dim hover:bg-white/5"
            }`}
          >
            {Math.round(r * 100)}%
          </button>
        ))}
        <button
          onClick={handlePrune}
          disabled={!modelKey || pruning}
          aria-label="Prune inactive neurons using Wanda metric"
          className="px-2.5 py-1 rounded-md text-[10px] font-medium border border-white/20 hover:bg-white/10 disabled:opacity-40 transition-colors"
        >
          {pruning ? "pruning…" : "Prune Inactive"}
        </button>
        {fitness !== null && (
          <span className="ml-auto text-[10px] text-dim" title="12-probe battery fitness">
            fitness <span className="font-mono text-white font-medium">{fitness.toFixed(4)}</span>
          </span>
        )}
      </div>

      {/* Neuron field */}
      <div className="p-2 border-t border-white/10">
        <NeuronField
          layers={layers}
          numLayers={numLayers}
          accent={accent}
          selected={selected}
          onSelect={onSelectNeuron}
          onDragStart={(l, n) => onDragStart(side, l, n)}
          highlighted={highlighted}
          pruned={pruned}
          width={500}
          height={260}
        />
      </div>
    </div>
  );
}