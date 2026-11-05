import { useRef, useEffect, useState, useCallback } from "react";

export type NeuronNode = {
  layer: number;
  neuron: number;
  activation: number;
  x: number;
  y: number;
  r: number;
};

type Props = {
  layers: Record<string, { neuron: number; activation: number; abs_activation: number }[]>;
  numLayers: number;
  accent: string; // css color
  selected: { layer: number; neuron: number } | null;
  onSelect: (l: number, n: number) => void;
  onDragStart: (l: number, n: number, ev: React.PointerEvent) => void;
  highlighted?: Set<string>; // "l:n"
  pruned?: Set<string>; // "l:n" — zeroed neurons
  width?: number;
  height?: number;
};

export default function NeuronField({
  layers,
  numLayers,
  accent,
  selected,
  onSelect,
  onDragStart,
  highlighted,
  pruned,
  width = 520,
  height = 360,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [nodes, setNodes] = useState<NeuronNode[]>([]);
  const [hover, setHover] = useState<NeuronNode | null>(null);
  const [measuredWidth, setMeasuredWidth] = useState(0);
  // A press becomes a drag only after the pointer moves past a small threshold,
  // so a plain click opens the passport without ever starting a transplant.
  const pressRef = useRef<{
    node: NeuronNode;
    startX: number;
    startY: number;
    dragging: boolean;
  } | null>(null);

  // Fill the panel instead of drawing into a fixed-width canvas.
  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const update = () => setMeasuredWidth(el.clientWidth);
    update();
    const ro = new ResizeObserver(update);
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const canvasWidth = measuredWidth > 0 ? measuredWidth : width;

  useEffect(() => {
    const clearPress = () => {
      pressRef.current = null;
    };
    // pointerup outside the canvas (e.g. on the drop target) must end the press
    window.addEventListener("pointerup", clearPress);
    return () => window.removeEventListener("pointerup", clearPress);
  }, []);

  // Build node layout
  useEffect(() => {
    const padX = 40;
    const padY = 30;
    const usableW = canvasWidth - padX * 2;
    const usableH = height - padY * 2;
    const colW = usableW / Math.max(1, numLayers);
    const all: NeuronNode[] = [];
    // Collect all neurons present across layers
    const layerKeys = Object.keys(layers);
    layerKeys.forEach((lk) => {
      const li = parseInt(lk);
      const neurons = layers[lk] || [];
      const x = padX + li * colW + colW / 2;
      neurons.forEach((n, i) => {
        // spread vertically by index within layer
        const frac = neurons.length > 1 ? i / (neurons.length - 1) : 0.5;
        const y = padY + frac * usableH;
        all.push({ layer: li, neuron: n.neuron, activation: n.activation, x, y, r: 4 + Math.min(6, Math.abs(n.activation) * 2) });
      });
    });
    setNodes(all);
  }, [layers, numLayers, canvasWidth, height]);

  // Draw
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(canvasWidth * dpr);
    canvas.height = height * dpr;
    canvas.style.width = "100%";
    canvas.style.height = height + "px";
    ctx.scale(dpr, dpr);

    // background
    ctx.fillStyle = "rgba(8, 12, 24, 0.6)";
    ctx.fillRect(0, 0, canvasWidth, height);

    // grid lines per layer
    const padX = 40;
    const usableW = canvasWidth - padX * 2;
    const colW = usableW / Math.max(1, numLayers);
    ctx.strokeStyle = "rgba(120,160,255,0.08)";
    ctx.lineWidth = 1;
    for (let li = 0; li < numLayers; li++) {
      const x = padX + li * colW;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // layer labels
    ctx.fillStyle = "rgba(150,170,220,0.6)";
    ctx.font = "10px system-ui";
    ctx.textAlign = "center";
    for (let li = 0; li < numLayers; li++) {
      const x = padX + li * colW + colW / 2;
      ctx.fillText(`L${li}`, x, height - 6);
    }

    // draw nodes
    nodes.forEach((node) => {
      const key = `${node.layer}:${node.neuron}`;
      const isSel = selected && selected.layer === node.layer && selected.neuron === node.neuron;
      const isHi = highlighted?.has(key);
      const isPruned = pruned?.has(key);
      const val = node.activation;
      // color: blue negative, orange positive, intensity by magnitude
      const mag = Math.min(1, Math.abs(val));
      let color: string;
      if (val >= 0) {
        color = `rgba(255, ${Math.round(120 + 80 * (1 - mag))}, 60, ${0.5 + 0.5 * mag})`;
      } else {
        color = `rgba(60, ${Math.round(120 + 80 * (1 - mag))}, 255, ${0.5 + 0.5 * mag})`;
      }
      if (isPruned) color = "rgba(130, 140, 160, 0.22)";
      else if (isHi) color = "#ffd166";
      else if (isSel) color = accent;

      // glow (pruned neurons stay dark/grey)
      if (!isPruned && (isSel || isHi)) {
        ctx.shadowColor = isHi ? "#ffd166" : accent;
        ctx.shadowBlur = 16;
      } else {
        ctx.shadowBlur = 0;
      }

      ctx.beginPath();
      ctx.arc(node.x, node.y, node.r, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
      ctx.shadowBlur = 0;

      if (isSel) {
        ctx.strokeStyle = "#ffffff";
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    });

    // hover tooltip
    if (hover) {
      const hoverPruned = pruned?.has(`${hover.layer}:${hover.neuron}`);
      ctx.fillStyle = "rgba(0,0,0,0.85)";
      ctx.fillRect(hover.x + 8, hover.y - 30, 110, 26);
      ctx.fillStyle = "#e6ecff";
      ctx.font = "11px system-ui";
      ctx.textAlign = "left";
      ctx.fillText(`L${hover.layer} · N${hover.neuron}`, hover.x + 12, hover.y - 18);
      ctx.fillStyle = hoverPruned ? "#ff5c7a" : "#9aa6cc";
      ctx.fillText(hoverPruned ? "pruned (zeroed)" : `act ${hover.activation.toFixed(3)}`, hover.x + 12, hover.y - 6);
    }
  }, [nodes, selected, hover, highlighted, pruned, accent, numLayers, canvasWidth, height]);

  // pointer handling
  const findNode = useCallback(
    (e: React.PointerEvent) => {
      const rect = canvasRef.current!.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      let best: NeuronNode | null = null;
      let bestD = 12;
      for (const n of nodes) {
        const d = Math.hypot(n.x - mx, n.y - my);
        if (d < bestD) {
          bestD = d;
          best = n;
        }
      }
      return best;
    },
    [nodes]
  );

  const handlePointerMove = (e: React.PointerEvent) => {
    setHover(findNode(e));
    const press = pressRef.current;
    if (!press || press.dragging) return;
    if (Math.hypot(e.clientX - press.startX, e.clientY - press.startY) > 5) {
      press.dragging = true;
      onDragStart(press.node.layer, press.node.neuron, e);
    }
  };

  const handlePointerDown = (e: React.PointerEvent) => {
    const n = findNode(e);
    if (!n) return;
    pressRef.current = {
      node: n,
      startX: e.clientX,
      startY: e.clientY,
      dragging: false,
    };
  };

  const handlePointerUp = () => {
    const press = pressRef.current;
    pressRef.current = null;
    if (press && !press.dragging) {
      // Plain click: select the neuron and open its passport.
      onSelect(press.node.layer, press.node.neuron);
    }
  };

  const handlePointerLeave = () => {
    setHover(null);
    // Keep an active drag alive; the drop target is outside this canvas.
    if (pressRef.current && !pressRef.current.dragging) {
      pressRef.current = null;
    }
  };

  return (
    <div ref={wrapRef} className="relative w-full">
      <canvas
        ref={canvasRef}
        className="rounded-lg cursor-pointer"
        onPointerMove={handlePointerMove}
        onPointerDown={handlePointerDown}
        onPointerUp={handlePointerUp}
        onPointerCancel={() => { pressRef.current = null; }}
        onPointerLeave={handlePointerLeave}
      />
      {hover && (
        <div className="absolute bottom-2 left-2 text-[10px] text-dim pointer-events-none">
          drag a neuron to transplant
        </div>
      )}
    </div>
  );
}