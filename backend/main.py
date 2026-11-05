"""FastAPI backend for the Neuron Lab visualizer."""
import json
import os
import re
import sys
import threading
import time
from pathlib import Path
sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from model_manager import ModelManager, DEVICE, DTYPE
import world_model as wm

app = FastAPI(title="Neuron Lab", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

mm = ModelManager()

BACKEND_DIR = Path(__file__).resolve().parent
DATA_DIR = BACKEND_DIR / "data"
SURGERY_LOG_PATH = DATA_DIR / "surgeries.jsonl"
EXPORT_DIR = BACKEND_DIR / "exported_models"
_log_lock = threading.Lock()


def _load_surgery_log():
    """Load persisted entries so history survives a server restart."""
    entries = []
    if not SURGERY_LOG_PATH.exists():
        return entries
    try:
        with open(SURGERY_LOG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return entries


def append_surgery_log(entry):
    """Append one entry to memory and backend/data/surgeries.jsonl."""
    entry = dict(entry)
    entry.setdefault("timestamp", time.time())
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    line = json.dumps(entry, ensure_ascii=False, default=str)
    with _log_lock:
        with open(SURGERY_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        SURGERY_LOG.append(entry)
    return entry


SURGERY_LOG = _load_surgery_log()  # persisted across restarts


# ---------- schemas ----------
class LoadReq(BaseModel):
    key: str

class ChatReq(BaseModel):
    model: str
    messages: List[Dict[str, str]]
    max_new_tokens: int = 80
    temperature: float = 0.7
    intervention: Optional[Dict[str, Any]] = None

class ActReq(BaseModel):
    model: str
    prompt: str
    layer: Optional[int] = None
    top_k: int = 48

class NeuronInfoReq(BaseModel):
    model: str
    layer: int
    neuron: int
    prompt: Optional[str] = None

class InterveneReq(BaseModel):
    model: str
    prompt: str
    layer: int
    neuron: int
    mode: str = "clamp"  # clamp | ablate | amplify
    value: float = 0.0
    scale: float = 2.0
    max_new_tokens: int = 80

class TransplantReq(BaseModel):
    src_model: str
    dst_model: str
    src_layer: int
    src_neuron: int
    dst_layer: int
    dst_neuron: int
    mode: str = "copy"  # copy | delta | blend
    scale: float = 1.0
    test_prompt: Optional[str] = None

class SteerReq(BaseModel):
    model: str
    prompt: str
    positive_prompt: str
    negative_prompt: str
    layer: int
    scale: float = 1.0
    max_new_tokens: int = 80


# ---------- routes ----------
@app.get("/api/health")
def health():
    import torch
    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {"status": "ok", "device": DEVICE, "gpu": gpu, "torch": torch.__version__}


@app.get("/api/models")
def list_models():
    return {"models": mm.available()}


@app.post("/api/models/load")
def load_model(req: LoadReq):
    try:
        m = mm.load(req.key)
        return {"key": req.key, "loaded": True, "layers": m.num_layers(),
                "hidden": m.hidden_size(), "intermediate": m.intermediate_size()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/models/unload")
def unload_model(req: LoadReq):
    mm.unload(req.key)
    return {"key": req.key, "loaded": False}


@app.post("/api/chat")
def chat(req: ChatReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # build prompt from messages
    prompt = ""
    for msg in req.messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"<|system|>\n{content}\n"
        elif role == "user":
            prompt += f"<|user|>\n{content}\n"
        elif role == "assistant":
            prompt += f"<|assistant|>\n{content}\n"
    prompt += "<|assistant|>\n"
    # simpler: use tokenizer chat template if available
    try:
        toks = m.tokenizer.apply_chat_template(req.messages, tokenize=False, add_generation_prompt=True)
        prompt = toks
    except Exception:
        pass
    text = m.generate(prompt, max_new_tokens=req.max_new_tokens, temperature=req.temperature,
                      intervention=req.intervention)
    return {"response": text}


@app.post("/api/activations")
def activations(req: ActReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    res = m.get_activations(req.prompt, layer_idx=req.layer, top_k=req.top_k)
    # serialize
    out = {}
    for li, neurons in res.items():
        out[str(li)] = neurons
    return {"model": req.model, "prompt": req.prompt, "layers": out,
            "num_layers": m.num_layers(), "intermediate": m.intermediate_size()}


@app.post("/api/neuron_info")
def neuron_info(req: NeuronInfoReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    info = m.neuron_info(req.layer, req.neuron, prompt=req.prompt)
    return info


@app.post("/api/intervene")
def intervene(req: InterveneReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # baseline
    try:
        toks = m.tokenizer.apply_chat_template([{"role": "user", "content": req.prompt}], tokenize=False, add_generation_prompt=True)
        prompt = toks
    except Exception:
        prompt = req.prompt
    baseline = m.generate(prompt, max_new_tokens=req.max_new_tokens, temperature=0.7)
    modified = m.generate(prompt, max_new_tokens=req.max_new_tokens, temperature=0.7,
                          intervention={"layer": req.layer, "neuron": req.neuron,
                                       "mode": req.mode, "value": req.value, "scale": req.scale})
    append_surgery_log({
        "type": "intervention",
        "model": req.model,
        "layer": req.layer,
        "neuron": req.neuron,
        "mode": req.mode,
        "value": req.value,
        "scale": req.scale,
        "prompt": req.prompt,
        "baseline": baseline,
        "modified": modified,
    })
    return {"baseline": baseline, "modified": modified, "prompt": req.prompt,
            "layer": req.layer, "neuron": req.neuron, "mode": req.mode}


@app.post("/api/transplant")
def transplant(req: TransplantReq):
    try:
        src = mm.get(req.src_model)
        dst = mm.get(req.dst_model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    metrics = dst.transplant_neuron(src, req.src_layer, req.src_neuron, req.dst_layer, req.dst_neuron,
                                   mode=req.mode, scale=req.scale)
    result = {"metrics": metrics}
    after = None
    if req.test_prompt:
        try:
            prompt = dst.tokenizer.apply_chat_template([{"role": "user", "content": req.test_prompt}], tokenize=False, add_generation_prompt=True)
        except Exception:
            prompt = req.test_prompt
        # We can't easily get "before" after transplant since weights changed.
        # Return post-transplant response; frontend captures the baseline beforehand.
        after = dst.generate(prompt, max_new_tokens=80, temperature=0.7)
        result["response_after"] = after
    append_surgery_log({
        "type": "transplant",
        "src_model": req.src_model,
        "dst_model": req.dst_model,
        "src": {"layer": req.src_layer, "neuron": req.src_neuron},
        "dst": {"layer": req.dst_layer, "neuron": req.dst_neuron},
        "mode": req.mode,
        "scale": req.scale,
        "metrics": metrics,
        "test_prompt": req.test_prompt,
        "response_after": after,
    })
    return result


@app.post("/api/steer")
def steer(req: SteerReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    vec = m.compute_steering_vector(req.positive_prompt, req.negative_prompt, req.layer)
    try:
        prompt = m.tokenizer.apply_chat_template([{"role": "user", "content": req.prompt}], tokenize=False, add_generation_prompt=True)
    except Exception:
        prompt = req.prompt
    baseline = m.generate(prompt, max_new_tokens=req.max_new_tokens, temperature=0.7)
    steered = m.generate_with_steering(prompt, vec, req.layer, scale=req.scale, max_new_tokens=req.max_new_tokens)
    append_surgery_log({
        "type": "steer",
        "model": req.model,
        "layer": req.layer,
        "scale": req.scale,
        "prompt": req.prompt,
        "positive_prompt": req.positive_prompt,
        "negative_prompt": req.negative_prompt,
        "baseline": baseline,
        "steered": steered,
        "vector_norm": float(vec.norm().item()),
    })
    return {"baseline": baseline, "steered": steered, "vector_norm": float(vec.norm().item())}


# ---------- world model / probes / dream ----------
class ProbesReq(BaseModel):
    model: str
    max_new_tokens: int = 40

@app.post("/api/probes")
def probes(req: ProbesReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    results, fitness = wm.run_probes(m.model, m.tokenizer, DEVICE, max_new_tokens=req.max_new_tokens)
    m.clear_cache()
    wm.empty_cuda_cache()
    return {"model": req.model, "probes": results, "fitness": fitness}


class DreamReq(BaseModel):
    model: str
    generations: int = 4
    population_size: int = 6
    top_k_per_layer: int = 32

@app.post("/api/dream")
def dream(req: DreamReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if req.generations < 1 or req.population_size < 1:
        raise HTTPException(status_code=400, detail="generations and population_size must be >= 1")
    if not (1 <= req.top_k_per_layer <= m.intermediate_size()):
        raise HTTPException(status_code=400, detail=f"top_k_per_layer must be 1..{m.intermediate_size()}")

    max_new = 40
    _, baseline = wm.run_probes(m.model, m.tokenizer, DEVICE, max_new_tokens=max_new)
    m.clear_cache()
    pool = wm.collect_active_neurons(
        m.model, m.tokenizer, DEVICE,
        top_k_per_layer=req.top_k_per_layer,
    )
    search = wm.EvolutionarySearch(
        m.model, m.tokenizer, DEVICE,
        population_size=req.population_size,
        generations=req.generations,
        candidate_pool=pool,
        top_k_per_layer=req.top_k_per_layer,
        probe_max_new_tokens=max_new,
    )
    best, history = search.run(baseline)
    m.clear_cache()
    wm.empty_cuda_cache()

    append_surgery_log({
        "type": "dream",
        "model": req.model,
        "baseline": baseline,
        "best_fitness": best["fitness"] if best else baseline,
        "best_delta": best["delta"] if best else 0.0,
        "best_surgery": best["surgery"] if best else None,
        "history": history,
        "pool_size": len(pool),
        "top_k_per_layer": req.top_k_per_layer,
        "generations": req.generations,
        "population_size": req.population_size,
    })
    return {
        "model": req.model,
        "baseline_fitness": baseline,
        "best": best,
        "history": history,
        "pool_size": len(pool),
    }


# ---------- pruning ----------
class PruneReq(BaseModel):
    model: str
    layer: Optional[int] = None
    ratio: float = 0.25
    mode: str = "zero_mask"

@app.post("/api/prune")
def prune(req: PruneReq):
    if req.mode != "zero_mask":
        raise HTTPException(status_code=400, detail="only mode='zero_mask' is supported")
    if not (0.0 < req.ratio < 1.0):
        raise HTTPException(status_code=400, detail="ratio must be in (0, 1)")
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if req.layer is not None and not (0 <= req.layer < m.num_layers()):
        raise HTTPException(status_code=400, detail=f"layer must be 0..{m.num_layers() - 1}")

    calibration = [p["prompt"] for p in wm.PROBE_BATTERY]
    _, baseline = wm.run_probes(m.model, m.tokenizer, DEVICE, max_new_tokens=40)
    m.clear_cache()
    try:
        result = m.prune_neurons(req.layer, req.ratio, calibration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prune failed: {e}")
    _, new_fitness = wm.run_probes(m.model, m.tokenizer, DEVICE, max_new_tokens=40)
    m.clear_cache()
    wm.empty_cuda_cache()

    entry = append_surgery_log({
        "type": "prune",
        "model": req.model,
        "layer": req.layer,
        "ratio": req.ratio,
        "mode": req.mode,
        "pruned_count": result["pruned_count"],
        "total_neurons": result["total_neurons"],
        "sparsity": result["sparsity"],
        "baseline_fitness": baseline,
        "new_fitness": new_fitness,
    })
    return {
        **result,
        "baseline_fitness": baseline,
        "new_fitness": new_fitness,
        "mode": req.mode,
        "log_timestamp": entry["timestamp"],
    }


# ---------- weight-surgery application (persistent) ----------
class ApplySurgeryReq(BaseModel):
    model: str
    surgery: Dict[str, Any]

@app.post("/api/apply_surgery")
def apply_surgery(req: ApplySurgeryReq):
    try:
        m = mm.get(req.model)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    s = dict(req.surgery or {})
    mode = s.get("mode")
    if mode not in {"ablate", "amplify", "clamp_set"}:
        raise HTTPException(status_code=400, detail="surgery.mode must be ablate, amplify, or clamp_set")
    try:
        layer = int(s["layer"])
        neuron = int(s["neuron"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(status_code=400, detail="surgery.layer and surgery.neuron are required integers")
    if not (0 <= layer < m.num_layers()):
        raise HTTPException(status_code=400, detail=f"layer must be 0..{m.num_layers() - 1}")
    if not (0 <= neuron < m.intermediate_size()):
        raise HTTPException(status_code=400, detail=f"neuron must be 0..{m.intermediate_size() - 1}")

    s["layer"] = layer
    s["neuron"] = neuron
    s["mode"] = mode
    if mode == "amplify":
        s["scale"] = float(s.get("scale", 2.0))
    elif mode == "clamp_set":
        s["value"] = float(s.get("value", 0.0))

    # Persistent weight edit: deliberately discard the restore snapshot.
    wm.apply_surgery(m.model, s)
    m.clear_cache()
    wm.empty_cuda_cache()
    append_surgery_log({"type": "apply_surgery", "model": req.model, "surgery": s})
    return {"applied": True, "surgery": s}


# ---------- model export ----------
class SaveModelReq(BaseModel):
    model: Optional[str] = None
    model_key: Optional[str] = None
    output_name: str

@app.post("/api/models/save")
def save_model(req: SaveModelReq):
    key = req.model or req.model_key
    if not key:
        raise HTTPException(status_code=400, detail="model is required")
    try:
        m = mm.get(key)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    name = req.output_name.strip()
    if not name or name in {".", ".."} or not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise HTTPException(status_code=400, detail="output_name must contain only letters, numbers, dot, underscore, or hyphen")

    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = (EXPORT_DIR / name).resolve()
    if save_path.parent != EXPORT_DIR.resolve():
        raise HTTPException(status_code=400, detail="invalid output path")

    try:
        m.model.save_pretrained(str(save_path), safe_serialization=True)
        m.tokenizer.save_pretrained(str(save_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"save failed: {e}")

    append_surgery_log({
        "type": "export",
        "model": key,
        "path": str(save_path),
        "format": "safetensors",
    })
    return {"status": "saved", "path": str(save_path), "format": "safetensors", "model": key}


# ---------- surgery log ----------
class SurgeryLogReq(BaseModel):
    entry: Dict[str, Any]

@app.post("/api/surgery_log")
def add_surgery_log(req: SurgeryLogReq):
    entry = append_surgery_log(req.entry)
    return {"ok": True, "count": len(SURGERY_LOG), "timestamp": entry.get("timestamp")}

@app.get("/api/surgery_log")
def get_surgery_log():
    return {"entries": SURGERY_LOG, "count": len(SURGERY_LOG), "path": str(SURGERY_LOG_PATH)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
