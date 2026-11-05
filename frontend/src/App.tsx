import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { api } from "./lib/api";
import ModelPanel from "./components/ModelPanel";
import DreamPanel from "./components/DreamPanel";


type SideState = {
  modelKey: string | null;
  chatLog: { role: string; content: string }[];
  layers: Record<string, any[]>;
  numLayers: number;
  intermediate: number;
  selected: { layer: number; neuron: number } | null;
  busy: boolean;
  pruned: Set<string>;
  fitness: number | null;
};

const emptySide: SideState = {
  modelKey: null,
  chatLog: [],
  layers: {},
  numLayers: 0,
  intermediate: 0,
  selected: null,
  busy: false,
  pruned: new Set<string>(),
  fitness: null,
};

export default function App() {
  const [models, setModels] = useState<any[]>([]);
  const [device, setDevice] = useState<string>("cpu");
  const [A, setA] = useState<SideState>({ ...emptySide, pruned: new Set() });
  const [B, setB] = useState<SideState>({ ...emptySide, pruned: new Set() });
  const [highlighted, setHighlighted] = useState<Set<string>>(new Set());
  const [dragging, setDragging] = useState<{ side: "A" | "B"; layer: number; neuron: number } | null>(null);
  const [dropHover, setDropHover] = useState(false);
  const [transplantResult, setTransplantResult] = useState<any>(null);
  const [passport, setPassport] = useState<any>(null);
  const [globalBusy, setGlobalBusy] = useState(false);
  const [showGuide, setShowGuide] = useState(false);
  const [targetSide, setTargetSide] = useState<"A" | "B">("A");
  // Synchronous mirror of `dragging`: lets async handlers check the drag state
  // without waiting for a React re-render.
  const draggingRef = useRef<{ side: "A" | "B"; layer: number; neuron: number } | null>(null);

  useEffect(() => {
    api.health().then((h) => setDevice(h.device + (h.gpu ? " · " + h.gpu : ""))).catch(() => {});
    api.models().then((r) => setModels(r.models)).catch(() => {});
    const handler = () => {
      draggingRef.current = null;
      setTimeout(() => setDragging(null), 120);
    };
    const keyHandler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        setPassport(null);
        setTransplantResult(null);
        setShowGuide(false);
      }
    };
    window.addEventListener("pointerup", handler);
    window.addEventListener("keydown", keyHandler);
    return () => {
      window.removeEventListener("pointerup", handler);
      window.removeEventListener("keydown", keyHandler);
    };
  }, []);

  const refreshModels = () => api.models().then((r) => setModels(r.models)).catch(() => {});

  const patch = (side: "A" | "B", patch: Partial<SideState>) => {
    const setter = side === "A" ? setA : setB;
    setter((s) => ({ ...s, ...patch }));
  };

  const handleLoad = (side: "A" | "B", key: string) => {
    const st = side === "A" ? A : B;
    if (st.modelKey !== key) {
      // A different model means the old pruned/highlight state no longer applies.
      patch(side, { modelKey: key, pruned: new Set(), fitness: null });
    } else {
      patch(side, { modelKey: key });
    }
    refreshModels();
  };

  const handleSend = async (side: "A" | "B", text: string) => {
    const st = side === "A" ? A : B;
    if (!st.modelKey) return;
    const newLog = [...st.chatLog, { role: "user", content: text }];
    patch(side, { chatLog: newLog, busy: true });
    try {
      const res = await api.chat(st.modelKey, newLog);
      patch(side, { chatLog: [...newLog, { role: "assistant", content: res.response }], busy: false });
    } catch (e: any) {
      patch(side, { chatLog: [...newLog, { role: "assistant", content: "Error: " + e.message }], busy: false });
    }
  };

  const handleActivations = async (
    side: "A" | "B",
    prompt: string,
    showPruned?: Record<string, number[]>
  ) => {
    const st = side === "A" ? A : B;
    if (!st.modelKey) return;
    try {
      const res = await api.activations(st.modelKey, prompt, null, 40);
      const layers = res.layers;
      if (showPruned) {
        // Zeroed neurons never rank in the top-k, so add a small sample per
        // layer as zero-activation nodes; NeuronField renders them grey.
        for (const [layer, idxs] of Object.entries(showPruned)) {
          const list = layers[layer] ? [...layers[layer]] : [];
          const existing = new Set(list.map((n: any) => n.neuron));
          const extras = (idxs as number[])
            .filter((n) => !existing.has(n))
            .slice(0, 6)
            .map((n) => ({ neuron: n, activation: 0, abs_activation: 0 }));
          layers[layer] = [...list, ...extras];
        }
      }
      patch(side, {
        layers,
        numLayers: res.num_layers,
        intermediate: res.intermediate,
      });
    } catch (e: any) {
      alert("Activations failed: " + e.message);
    }
  };

  const handlePrune = async (side: "A" | "B", ratio: number) => {
    const st = side === "A" ? A : B;
    if (!st.modelKey) return;
    setGlobalBusy(true);
    try {
      const res = await api.prune(st.modelKey, ratio);
      const next = new Set(st.pruned);
      for (const [layer, neurons] of Object.entries(res.pruned_by_layer || {})) {
        for (const n of neurons as number[]) next.add(`${layer}:${n}`);
      }
      const message =
        `PRUNE: zeroed ${res.pruned_count}/${res.total_neurons} neurons ` +
        `(${(res.sparsity * 100).toFixed(1)}% sparsity). ` +
        `Fitness ${res.baseline_fitness.toFixed(4)} → ${res.new_fitness.toFixed(4)}`;
      patch(side, {
        pruned: next,
        fitness: res.new_fitness,
        chatLog: [...st.chatLog, { role: "assistant", content: message }],
      });
      // Refresh the field so a sample of pruned neurons render grey.
      await handleActivations(
        side,
        "The quick brown fox jumps over the lazy dog.",
        res.pruned_by_layer
      );
    } catch (e: any) {
      alert("Prune failed: " + e.message);
    } finally {
      setGlobalBusy(false);
    }
  };

  const handleSelectNeuron = async (side: "A" | "B", l: number, n: number) => {
    setTargetSide(side);
    const st = side === "A" ? A : B;
    if (!st.modelKey) return;
    patch(side, { selected: { layer: l, neuron: n } });
    try {
      const info = await api.neuronInfo(st.modelKey, l, n);
      // A drag started while the request was in flight: don't cover the UI.
      if (draggingRef.current) return;
      setPassport({ side, ...info });
    } catch (e: any) {
      console.error(e);
    }
  };

  const handleDragStart = (side: "A" | "B", layer: number, neuron: number) => {
    setPassport(null);
    draggingRef.current = { side, layer, neuron };
    setDragging({ side, layer, neuron });
  };

  const handleDrop = async (targetSide: "A" | "B") => {
    if (!dragging) return;
    if (dragging.side === targetSide) {
      draggingRef.current = null;
      setDragging(null);
      return;
    }
    const src = dragging.side === "A" ? A : B;
    const dst = targetSide === "A" ? A : B;
    if (!src.modelKey || !dst.modelKey) {
      draggingRef.current = null;
      setDragging(null);
      return;
    }
    setGlobalBusy(true);
    try {
      // capture baseline response on dst before transplant
      const testPrompt = "Hello, who are you?";
      let baseline = "";
      try {
        const r = await api.chat(dst.modelKey, [{ role: "user", content: testPrompt }], 60, 0.7);
        baseline = r.response;
      } catch {}

      const dstLayer = dst.selected?.layer ?? dragging.layer;
      const dstNeuron = dst.selected?.neuron ?? dragging.neuron;
      const res = await api.transplant(
        src.modelKey,
        dst.modelKey,
        dragging.layer,
        dragging.neuron,
        dstLayer,
        dstNeuron,
        "copy",
        1.0,
        testPrompt
      );
      setTransplantResult({
        src: dragging,
        dst: { side: targetSide, layer: dstLayer, neuron: dstNeuron },
        metrics: res.metrics,
        baseline,
        after: res.response_after,
      });
      setHighlighted((h) => {
        const nh = new Set(h);
        nh.add(`${targetSide}:${dstLayer}:${dstNeuron}`);
        return nh;
      });
      // The destination neuron now has source weights, so it is no longer pruned.
      const nextPruned = new Set(dst.pruned);
      nextPruned.delete(`${dstLayer}:${dstNeuron}`);
      patch(targetSide, { pruned: nextPruned, fitness: null });
      // refresh activations on dst
      await handleActivations(targetSide, "transplant");
    } catch (e: any) {
      alert("Transplant failed: " + e.message);
    } finally {
      setGlobalBusy(false);
      draggingRef.current = null;
      setDragging(null);
    }
  };

  // Intervene demo: clamp selected neuron
  const handleIntervene = async (side: "A" | "B", mode: string) => {
    const st = side === "A" ? A : B;
    if (!st.modelKey || !st.selected) return;
    const prompt = "Tell me a short story about a robot.";
    setGlobalBusy(true);
    try {
      const res = await api.intervene(st.modelKey, prompt, st.selected.layer, st.selected.neuron, mode, 0, 3.0, 80);
      const newLog = [
        ...st.chatLog,
        { role: "user", content: `[intervene ${mode} L${st.selected.layer} N${st.selected.neuron}] ${prompt}` },
        { role: "assistant", content: `BASELINE:\n${res.baseline}\n\nMODIFIED (${mode}):\n${res.modified}` },
      ];
      patch(side, { chatLog: newLog });
    } catch (e: any) {
      alert("Intervene failed: " + e.message);
    } finally {
      setGlobalBusy(false);
    }
  };

  return (
    <div className="h-screen flex flex-col">
      {/* Top bar */}
      <header className="shrink-0 flex flex-wrap items-center justify-between gap-3 px-4 sm:px-6 py-2.5 border-b border-white/10 glass-strong">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg flex items-center justify-center shrink-0" style={{ background: "linear-gradient(135deg,#00d4ff,#ff3ea5)", boxShadow: "0 0 20px rgba(0,212,255,0.4)" }}>
            <span className="text-black font-black text-sm">N</span>
          </div>
          <div>
            <div className="font-bold text-sm tracking-wider flex items-center gap-2">
              <span>NEURON LAB</span>
              <span className="text-[10px] px-1.5 py-0.5 rounded border border-white/10 text-dim font-normal">v1.0</span>
            </div>
            <div className="text-[10px] text-dim hidden sm:block">
              Interactive mechanistic interpretability · surgical neuron transplant · neural Darwinism
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3 text-xs flex-wrap">
          <button
            onClick={() => setShowGuide(true)}
            className="px-2.5 py-1 rounded-md text-xs font-medium border border-cyan-400/30 text-cyan-300 bg-cyan-950/40 hover:bg-cyan-900/40 transition-colors flex items-center gap-1.5"
            aria-label="Open visualizer guide and legend"
          >
            <span className="w-4 h-4 rounded-full bg-cyan-400/20 text-cyan-200 flex items-center justify-center text-[10px] font-bold">?</span>
            <span>Guide & Legend</span>
          </button>
          <span className="text-dim text-[11px]">device: <span className="text-white font-mono">{device}</span></span>
          {dragging && (
            <span className="px-2 py-0.5 rounded text-[11px] font-mono animate-pulse" style={{ background: "#ffd16620", color: "#ffd166", border: "1px solid #ffd16650" }}>
              dragging L{dragging.layer}·N{dragging.neuron} (Model {dragging.side})
            </span>
          )}
          {globalBusy && <span className="text-dim italic text-[11px] animate-pulse">working…</span>}
        </div>
      </header>

      {/* Scrollable workspace: model panels keep a stable height, the dream
          panel grows below and the page scrolls instead of overlapping them. */}
      <div className="flex-1 overflow-y-auto min-h-0 scrollbar">
        {/* Main split: responsive 1 col on mobile/tablet, 2 cols on desktop */}
        <div className="grid grid-cols-1 xl:grid-cols-2 gap-4 p-3 sm:p-4 min-h-fit xl:h-[620px]">
          <div
            onPointerUp={() => { if (dragging && dragging.side !== "A") handleDrop("A"); setDropHover(false); }}
            onPointerEnter={() => { if (dragging && dragging.side !== "A") setDropHover(true); }}
            onPointerLeave={() => setDropHover(false)}
            className={`flex flex-col min-h-[520px] xl:h-full transition-all ${
              dropHover && dragging?.side !== "A" ? "ring-2 ring-[#ffd166] rounded-2xl" : ""
            }`}
          >
            <ModelPanel
              side="A"
              accent="#00d4ff"
              glow="0 0 20px rgba(0,212,255,0.5)"
              modelKey={A.modelKey}
              models={models}
              onLoad={(k) => handleLoad("A", k)}
              selected={A.selected}
              onSelectNeuron={(l, n) => handleSelectNeuron("A", l, n)}
              onDragStart={(side, l, n) => handleDragStart(side, l, n)}
              layers={A.layers}
              numLayers={A.numLayers}
              highlighted={new Set([...highlighted].filter((s) => s.startsWith("A:")).map((s) => s.slice(2)))}
              pruned={A.pruned}
              fitness={A.fitness}
              onPrune={(ratio) => handlePrune("A", ratio)}
              onActivations={(p) => handleActivations("A", p)}
              chatLog={A.chatLog}
              onSend={(t) => handleSend("A", t)}
              busy={A.busy}
            />
          </div>
          <div
            onPointerUp={() => { if (dragging && dragging.side !== "B") handleDrop("B"); setDropHover(false); }}
            onPointerEnter={() => { if (dragging && dragging.side !== "B") setDropHover(true); }}
            onPointerLeave={() => setDropHover(false)}
            className={`flex flex-col min-h-[520px] xl:h-full transition-all ${
              dropHover && dragging?.side !== "B" ? "ring-2 ring-[#ffd166] rounded-2xl" : ""
            }`}
          >
            <ModelPanel
              side="B"
              accent="#ff3ea5"
              glow="0 0 20px rgba(255,62,165,0.5)"
              modelKey={B.modelKey}
              models={models}
              onLoad={(k) => handleLoad("B", k)}
              selected={B.selected}
              onSelectNeuron={(l, n) => handleSelectNeuron("B", l, n)}
              onDragStart={(side, l, n) => handleDragStart(side, l, n)}
              layers={B.layers}
              numLayers={B.numLayers}
              highlighted={new Set([...highlighted].filter((s) => s.startsWith("B:")).map((s) => s.slice(2)))}
              pruned={B.pruned}
              fitness={B.fitness}
              onPrune={(ratio) => handlePrune("B", ratio)}
              onActivations={(p) => handleActivations("B", p)}
              chatLog={B.chatLog}
              onSend={(t) => handleSend("B", t)}
              busy={B.busy}
            />
          </div>
        </div>

        {/* World Model / Dream panel */}
        <div className="px-3 sm:px-4 pb-3">
          <DreamPanel
            modelKey={A.modelKey || B.modelKey}
            onApplySurgery={async (s: any) => {
              const mk = A.modelKey || B.modelKey;
              if (!mk) return;
              try {
                await api.applySurgery(mk, s);
                alert(`Applied persistent weight surgery: L${s.layer} N${s.neuron} ${s.mode}`);
                const side = A.modelKey === mk ? "A" : "B";
                await handleActivations(side, "The quick brown fox jumps over the lazy dog.");
              } catch (e: any) {
                alert("Apply failed: " + e.message);
              }
            }}
          />
        </div>
      </div>

      {/* Bottom action bar */}
      {(() => {
        const active = targetSide === "A" ? A : B;
        const sel = active.selected;
        return (
          <div className="shrink-0 px-4 py-2 border-t border-white/10 flex flex-wrap gap-2 items-center justify-between glass-strong">
            <div className="flex items-center gap-2 flex-wrap">
              <div className="flex items-center gap-1.5 mr-2">
                <span className="text-[11px] text-dim font-medium">Target:</span>
                <div className="inline-flex rounded-md border border-white/15 p-0.5 bg-black/40">
                  <button
                    onClick={() => setTargetSide("A")}
                    className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                      targetSide === "A" ? "bg-cyan-500/30 text-cyan-300" : "text-dim hover:text-white"
                    }`}
                  >
                    Model A {A.selected ? `(L${A.selected.layer}·N${A.selected.neuron})` : ""}
                  </button>
                  <button
                    onClick={() => setTargetSide("B")}
                    className={`px-2 py-0.5 rounded text-[10px] font-semibold transition-colors ${
                      targetSide === "B" ? "bg-pink-500/30 text-pink-300" : "text-dim hover:text-white"
                    }`}
                  >
                    Model B {B.selected ? `(L${B.selected.layer}·N${B.selected.neuron})` : ""}
                  </button>
                </div>
              </div>
              <button
                onClick={() => sel && handleIntervene(targetSide, "clamp")}
                disabled={!active.modelKey || !sel}
                className="px-3 py-1.5 rounded-md text-xs font-medium border border-white/20 hover:bg-white/10 disabled:opacity-40 transition-colors"
                title={sel ? `Clamp L${sel.layer}·N${sel.neuron} to 0` : "Select a neuron dot first"}
              >
                Clamp {sel ? `L${sel.layer}·N${sel.neuron}` : "Selected"} → 0
              </button>
              <button
                onClick={() => sel && handleIntervene(targetSide, "ablate")}
                disabled={!active.modelKey || !sel}
                className="px-3 py-1.5 rounded-md text-xs font-medium border border-red-500/30 text-red-300 hover:bg-red-500/10 disabled:opacity-40 transition-colors"
                title={sel ? `Ablate (zero out) L${sel.layer}·N${sel.neuron}` : "Select a neuron dot first"}
              >
                Ablate {sel ? `L${sel.layer}·N${sel.neuron}` : ""}
              </button>
              <button
                onClick={() => sel && handleIntervene(targetSide, "amplify")}
                disabled={!active.modelKey || !sel}
                className="px-3 py-1.5 rounded-md text-xs font-medium border border-amber-500/30 text-amber-300 hover:bg-amber-500/10 disabled:opacity-40 transition-colors"
                title={sel ? `Amplify L${sel.layer}·N${sel.neuron} ×3` : "Select a neuron dot first"}
              >
                Amplify ×3
              </button>
            </div>
            <div className="text-[11px] text-dim hidden sm:block">
              💡 Tip: Click any neuron to inspect details; drag across panels to transplant.
            </div>
          </div>
        );
      })()}

      {/* Neuron passport modal */}
      <AnimatePresence>
        {passport && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            role="dialog"
            aria-modal="true"
            aria-label="Neuron Passport"
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setPassport(null)}
          >
            <motion.div
              initial={{ scale: 0.95, y: 15 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 15 }}
              className="glass-strong p-5 w-full max-w-[420px] max-h-[85vh] overflow-y-auto scrollbar"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
                <div className="font-bold text-sm tracking-wide">NEURON PASSPORT</div>
                <button
                  onClick={() => setPassport(null)}
                  className="w-7 h-7 rounded-md flex items-center justify-center text-dim hover:text-white hover:bg-white/10 text-base"
                  aria-label="Close dialog"
                >
                  ✕
                </button>
              </div>
              <div className="text-xs space-y-2">
                <div className="flex justify-between">
                  <span className="text-dim">Model</span>
                  <span className="font-semibold text-white">Model {passport.side}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dim">Location</span>
                  <span className="font-mono text-cyan-300">Layer {passport.layer} · Neuron {passport.neuron}</span>
                </div>
                {passport.activation !== undefined && (
                  <div className="flex justify-between">
                    <span className="text-dim">Activation</span>
                    <span className="font-mono font-medium" style={{ color: passport.activation >= 0 ? "#3ddc97" : "#ff5c7a" }}>
                      {passport.activation.toFixed(4)}
                    </span>
                  </div>
                )}
                <div className="flex justify-between">
                  <span className="text-dim">Gate weight norm ‖·‖</span>
                  <span className="font-mono">{passport.gate_weight_norm?.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dim">Up weight norm ‖·‖</span>
                  <span className="font-mono">{passport.up_weight_norm?.toFixed(4)}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-dim">Down weight norm ‖·‖</span>
                  <span className="font-mono">{passport.down_weight_norm?.toFixed(4)}</span>
                </div>
                <div className="flex justify-between border-t border-white/5 pt-1">
                  <span className="text-dim">Total norm ‖·‖</span>
                  <span className="font-mono text-white font-medium">{passport.total_norm?.toFixed(4)}</span>
                </div>
                <div className="mt-3 p-2.5 rounded bg-black/40 border border-white/5 text-[11px] text-dim leading-relaxed">
                  💡 This neuron resides in its model's coordinate system. Transplanting into a model with the same lineage (e.g. Qwen Base ↔ Instruct) transfers weights directly.
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Transplant result modal */}
      <AnimatePresence>
        {transplantResult && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            role="dialog"
            aria-modal="true"
            aria-label="Transplant Result"
            className="fixed inset-0 bg-black/70 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setTransplantResult(null)}
          >
            <motion.div
              initial={{ scale: 0.95, y: 15 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 15 }}
              className="glass-strong p-5 w-full max-w-[560px] max-h-[85vh] overflow-y-auto scrollbar"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-3 border-b border-white/10 pb-2">
                <div className="font-bold text-sm" style={{ color: "#ffd166" }}>TRANSPLANT COMPLETE</div>
                <button
                  onClick={() => setTransplantResult(null)}
                  className="w-7 h-7 rounded-md flex items-center justify-center text-dim hover:text-white hover:bg-white/10 text-base"
                  aria-label="Close dialog"
                >
                  ✕
                </button>
              </div>
              <div className="text-xs space-y-3">
                <div className="grid grid-cols-2 gap-2">
                  <div className="glass p-2.5">
                    <div className="text-dim text-[10px] uppercase tracking-wider mb-1">Source</div>
                    <div className="font-mono text-cyan-300 font-medium">Model {transplantResult.src.side} · L{transplantResult.src.layer}·N{transplantResult.src.neuron}</div>
                  </div>
                  <div className="glass p-2.5">
                    <div className="text-dim text-[10px] uppercase tracking-wider mb-1">Destination</div>
                    <div className="font-mono text-pink-300 font-medium">Model {transplantResult.dst.side} · L{transplantResult.dst.layer}·N{transplantResult.dst.neuron}</div>
                  </div>
                </div>
                <div className="glass p-3 space-y-1.5">
                  <div className="text-dim text-[10px] uppercase tracking-wider mb-1">Metrics</div>
                  <div className="flex justify-between">
                    <span>Cosine similarity (src ↔ dst)</span>
                    <span className="font-mono font-medium text-white">{transplantResult.metrics.cosine_similarity_src_dst.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Source weight norm</span>
                    <span className="font-mono">{transplantResult.metrics.src_norm.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Dest weight norm (before)</span>
                    <span className="font-mono">{transplantResult.metrics.dst_norm_before.toFixed(4)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Mode</span>
                    <span className="font-mono text-cyan-300">{transplantResult.metrics.mode}</span>
                  </div>
                </div>
                {transplantResult.baseline && (
                  <div className="glass p-3">
                    <div className="text-dim text-[10px] uppercase tracking-wider mb-1">Baseline response (before)</div>
                    <div className="whitespace-pre-wrap text-[11px] text-white/80 leading-relaxed max-h-32 overflow-y-auto scrollbar">{transplantResult.baseline}</div>
                  </div>
                )}
                {transplantResult.after && (
                  <div className="glass p-3" style={{ borderColor: "#ffd16650" }}>
                    <div className="text-[10px] uppercase tracking-wider mb-1 font-semibold" style={{ color: "#ffd166" }}>After transplant</div>
                    <div className="whitespace-pre-wrap text-[11px] text-white leading-relaxed max-h-32 overflow-y-auto scrollbar">{transplantResult.after}</div>
                  </div>
                )}
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Quick Guide & Visual Legend Modal */}
      <AnimatePresence>
        {showGuide && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            role="dialog"
            aria-modal="true"
            aria-label="Neuron Lab Quick Guide and Legend"
            className="fixed inset-0 bg-black/75 backdrop-blur-sm flex items-center justify-center z-50 p-4"
            onClick={() => setShowGuide(false)}
          >
            <motion.div
              initial={{ scale: 0.95, y: 15 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 15 }}
              className="glass-strong p-6 w-full max-w-[620px] max-h-[88vh] overflow-y-auto scrollbar"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex items-center justify-between mb-4 border-b border-white/10 pb-3">
                <div className="flex items-center gap-2">
                  <span className="w-5 h-5 rounded-md flex items-center justify-center text-xs font-bold text-black" style={{ background: "linear-gradient(135deg,#00d4ff,#ff3ea5)" }}>N</span>
                  <span className="font-bold text-base tracking-wide">Neuron Lab User Guide</span>
                </div>
                <button
                  onClick={() => setShowGuide(false)}
                  className="w-7 h-7 rounded-md flex items-center justify-center text-dim hover:text-white hover:bg-white/10 text-base"
                  aria-label="Close dialog"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4 text-xs">
                {/* Visual Legend */}
                <div>
                  <h4 className="font-bold text-white mb-2 uppercase text-[10px] tracking-wider text-cyan-300">Neuron Canvas Visual Legend</h4>
                  <div className="grid grid-cols-2 gap-2">
                    <div className="flex items-center gap-2 p-2 rounded bg-black/30 border border-white/5">
                      <span className="w-3.5 h-3.5 rounded-full shrink-0 shadow-sm" style={{ background: "rgba(255, 120, 60, 0.9)", boxShadow: "0 0 8px rgba(255,120,60,0.6)" }} />
                      <div>
                        <div className="font-semibold text-white">Orange (+act)</div>
                        <div className="text-[10px] text-dim">Active neuron exciting next tokens</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 p-2 rounded bg-black/30 border border-white/5">
                      <span className="w-3.5 h-3.5 rounded-full shrink-0 shadow-sm" style={{ background: "rgba(60, 120, 255, 0.9)", boxShadow: "0 0 8px rgba(60,120,255,0.6)" }} />
                      <div>
                        <div className="font-semibold text-white">Blue (-act)</div>
                        <div className="text-[10px] text-dim">Inhibitory neuron suppressing tokens</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 p-2 rounded bg-black/30 border border-white/5">
                      <span className="w-3.5 h-3.5 rounded-full shrink-0 shadow-sm bg-[#ffd166]" style={{ boxShadow: "0 0 8px rgba(255,209,102,0.6)" }} />
                      <div>
                        <div className="font-semibold text-white">Gold (Transplanted)</div>
                        <div className="text-[10px] text-dim">Transplanted from another model</div>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 p-2 rounded bg-black/30 border border-white/5">
                      <span className="w-3.5 h-3.5 rounded-full shrink-0 border border-white/20 bg-slate-700/40" />
                      <div>
                        <div className="font-semibold text-white">Dim Grey (Pruned)</div>
                        <div className="text-[10px] text-dim">Zeroed out via Wanda pruning</div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Step-by-Step Workflow */}
                <div>
                  <h4 className="font-bold text-white mb-2 uppercase text-[10px] tracking-wider text-pink-300">How to Experiment (5 Easy Steps)</h4>
                  <ol className="space-y-2 list-decimal list-inside text-dim">
                    <li className="leading-relaxed">
                      <strong className="text-white">Load Models:</strong> Pick <code className="text-cyan-300">Qwen 2.5 0.5B Base</code> in Model A and <code className="text-pink-300">Qwen 2.5 0.5B Instruct</code> in Model B.
                    </li>
                    <li className="leading-relaxed">
                      <strong className="text-white">Inspect Activations:</strong> Enter a prompt like <em className="text-white/80">"What is consciousness?"</em> in the input box and click <strong className="text-white">Inspect</strong> to light up neurons across all 24 layers.
                    </li>
                    <li className="leading-relaxed">
                      <strong className="text-white">Drag & Drop Transplant:</strong> Click and hold any neuron node in Model A, drag it across into Model B's panel, and release to graft its weight vectors into Model B.
                    </li>
                    <li className="leading-relaxed">
                      <strong className="text-white">Intervene & Prune:</strong> Click a neuron to inspect its weights, then test single-neuron <strong className="text-white">Clamp</strong>, <strong className="text-white">Ablate</strong>, or <strong className="text-white">Amplify (×3)</strong>. Use <strong className="text-white">Prune Inactive</strong> to mute redundant neurons.
                    </li>
                    <li className="leading-relaxed">
                      <strong className="text-white">Neural Darwinism & Export:</strong> In the World Model panel below, click <strong className="text-white">Start Dreaming</strong> to automatically discover beneficial surgeries, apply the winner, and click <strong className="text-white">Export .safetensors</strong> to save your new model!
                    </li>
                  </ol>
                </div>

                <div className="p-3 rounded-lg bg-white/5 border border-white/10 flex items-start gap-2.5">
                  <span className="text-base">💡</span>
                  <p className="text-[11px] text-dim leading-relaxed">
                    <strong className="text-white">Why Base ↔ Instruct?</strong> Because both models share the exact same layer architecture and training lineage, individual neuron indices remain aligned (cosine similarity ~0.997), making surgical transplants meaningful.
                  </p>
                </div>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}