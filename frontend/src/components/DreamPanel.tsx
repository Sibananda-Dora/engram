import { useState } from "react";
import { api } from "../lib/api";

type Props = {
  modelKey: string | null;
  onApplySurgery?: (s: any) => void;
};

export default function DreamPanel({ modelKey, onApplySurgery }: Props) {
  const [probes, setProbes] = useState<any>(null);
  const [fitness, setFitness] = useState<number | null>(null);
  const [dreaming, setDreaming] = useState(false);
  const [dreamResult, setDreamResult] = useState<any>(null);
  const [runningProbes, setRunningProbes] = useState(false);
  const [history, setHistory] = useState<any[]>([]);

  const runProbes = async () => {
    if (!modelKey) return;
    setRunningProbes(true);
    try {
      const r = await api.probes(modelKey, 40);
      setProbes(r.probes);
      setFitness(r.fitness);
    } catch (e: any) {
      alert("Probes failed: " + e.message);
    } finally {
      setRunningProbes(false);
    }
  };

  const startDream = async () => {
    if (!modelKey) return;
    setDreaming(true);
    setDreamResult(null);
    try {
      // Candidates are drawn from the top-32 active neurons in every layer.
      const r = await api.dream(modelKey, 3, 5, 32);
      setDreamResult(r);
      setHistory(r.history || []);
      if (r.best) setFitness(r.best.fitness);
    } catch (e: any) {
      alert("Dream failed: " + e.message);
    } finally {
      setDreaming(false);
    }
  };

  return (
    <div className="glass p-4 mt-3">
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 mb-3">
        <div>
          <div className="font-bold text-sm tracking-wider flex items-center gap-2" style={{ color: "#ffd166" }}>
            <span>WORLD MODEL · NEURAL DARWINISM</span>
          </div>
          <div className="text-[11px] text-dim mt-0.5 leading-relaxed">
            Treat surgery as a Markov decision process. Simulates surgeries in an active neuron pool and identifies top improvements.
          </div>
        </div>
        <div className="flex items-center gap-2 flex-wrap">
          <button
            onClick={runProbes}
            disabled={!modelKey || runningProbes}
            aria-label="Run 12-probe battery"
            className="px-3 py-1.5 rounded-md text-xs font-medium border border-white/20 hover:bg-white/10 disabled:opacity-40 transition-colors"
          >
            {runningProbes ? "probing…" : "Run Probes"}
          </button>
          <button
            onClick={startDream}
            disabled={!modelKey || dreaming}
            aria-label="Start evolutionary surgery dreaming"
            className="px-3.5 py-1.5 rounded-md text-xs font-semibold disabled:opacity-40 hover:opacity-90 transition-opacity"
            style={{ background: "#ffd166", color: "#0a0e1a" }}
          >
            {dreaming ? "dreaming…" : "Start Dreaming"}
          </button>
        </div>
      </div>

      {fitness !== null && (
        <div className="flex flex-wrap items-center gap-4 mb-3 p-2 rounded-lg bg-black/20 border border-white/5">
          <div className="text-xs">
            <span className="text-dim">Baseline Fitness:</span>{" "}
            <span className="font-mono text-white font-medium">{fitness.toFixed(4)}</span>
          </div>
          {dreamResult && (
            <div className="text-xs">
              <span className="text-dim">Best Discovered:</span>{" "}
              <span className="font-mono font-medium" style={{ color: "#3ddc97" }}>
                {dreamResult.best?.fitness?.toFixed(4)}
              </span>
              <span className="text-dim ml-2">
                (Δ {dreamResult.best?.delta >= 0 ? "+" : ""}{dreamResult.best?.delta?.toFixed(4)})
              </span>
            </div>
          )}
        </div>
      )}

      {probes && (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 xl:grid-cols-4 gap-1.5 mb-3 max-h-48 overflow-y-auto scrollbar">
          {Object.entries(probes).map(([id, p]: any) => (
            <div
              key={id}
              title={`Response: ${p.response}`}
              className="flex items-center justify-between text-[11px] px-2.5 py-1.5 rounded bg-black/40 border border-white/5"
            >
              <span className="truncate text-dim font-medium mr-2">{id}</span>
              <span className="font-mono shrink-0 font-medium" style={{ color: p.score >= 0.7 ? "#3ddc97" : p.score >= 0.4 ? "#ffd166" : "#ff5c7a" }}>
                {p.score.toFixed(2)}
              </span>
            </div>
          ))}
        </div>
      )}

      {dreamResult?.best && (
        <div className="glass-strong p-3 mb-3" style={{ borderColor: "#ffd16650" }}>
          <div className="text-xs font-bold mb-1" style={{ color: "#ffd166" }}>BEST DREAMED RECIPE</div>
          <div className="font-mono text-xs">
            Layer {dreamResult.best.surgery.layer} · Neuron {dreamResult.best.surgery.neuron} · {dreamResult.best.surgery.mode}
            {dreamResult.best.surgery.scale !== undefined && ` ×${dreamResult.best.surgery.scale}`}
            {dreamResult.best.surgery.value !== undefined && ` =${dreamResult.best.surgery.value}`}
          </div>
          <div className="text-[10px] text-dim mt-1">
            baseline {dreamResult.baseline_fitness?.toFixed(4)} → {dreamResult.best.fitness?.toFixed(4)}
          </div>
          {onApplySurgery && (
            <button
              onClick={() => onApplySurgery(dreamResult.best.surgery)}
              className="mt-2 px-3 py-1 rounded-md text-xs"
              style={{ background: "#ffd166", color: "#0a0e1a" }}
            >
              Apply this surgery
            </button>
          )}
        </div>
      )}

      {history.length > 0 && (
        <div className="max-h-32 overflow-y-auto scrollbar">
          <div className="text-[10px] text-dim mb-1">search history ({history.length} candidates)</div>
          {history.slice(-12).map((h, i) => (
            <div key={i} className="flex items-center justify-between text-[10px] px-2 py-0.5">
              <span className="font-mono text-dim">
                gen{h.gen} L{h.candidate.layer} N{h.candidate.neuron} {h.candidate.mode}
              </span>
              <span className="font-mono" style={{ color: h.delta >= 0 ? "#3ddc97" : "#ff5c7a" }}>
                {h.fitness.toFixed(3)} ({h.delta >= 0 ? "+" : ""}{h.delta.toFixed(3)})
              </span>
            </div>
          ))}
        </div>
      )}

      {dreaming && (
        <div className="text-xs text-dim italic mt-2">
          Dreaming… this runs real surgeries in a sandbox and may take 1–2 minutes.
        </div>
      )}
    </div>
  );
}