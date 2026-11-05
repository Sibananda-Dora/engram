r"""Temporary end-to-end verification for Neuron Lab v1.

Run from neuron-lab/backend:
    ..\.venv\Scripts\python _e2e_test.py

Uses FastAPI's TestClient (in-process, no blocking server). Cleans up the
test export directory and any log lines it added.
"""
import json
import os
import shutil
import sys
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))

from fastapi.testclient import TestClient
import torch
from safetensors import safe_open

import main as app_main

client = TestClient(app_main.app, raise_server_exceptions=True)

BACKEND = Path(__file__).resolve().parent
LOG_PATH = BACKEND / "data" / "surgeries.jsonl"
EXPORT_NAME = "e2e-test-export"
EXPORT_PATH = BACKEND / "exported_models" / EXPORT_NAME

failures = []


def check(label, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {label}" + (f" :: {detail}" if detail else ""))
    if not cond:
        failures.append(label)


# --- backup log so the test leaves no fake entries -------------------------
log_backup = LOG_PATH.read_text(encoding="utf-8") if LOG_PATH.exists() else None
log_lines_before = 0 if log_backup is None else len(
    [ln for ln in log_backup.splitlines() if ln.strip()]
)
export_preexisting = EXPORT_PATH.exists()
if export_preexisting:
    shutil.rmtree(EXPORT_PATH)


def post(path, payload):
    r = client.post(path, json=payload)
    if r.status_code != 200:
        raise AssertionError(f"{path} -> HTTP {r.status_code}: {r.text[:800]}")
    return r.json()


try:
    # 1. health -------------------------------------------------------------
    health = client.get("/api/health").json()
    check("health ok", health.get("status") == "ok", json.dumps(health))

    # 2. model registry -----------------------------------------------------
    models = client.get("/api/models").json()["models"]
    check("registry has 2 models", len(models) == 2, str([m["key"] for m in models]))

    # 3. concurrent load both models (regression: from_pretrained race) -----
    # The UI can fire two dropdown loads within milliseconds. Reproduce that
    # exact pattern and require healthy dtypes plus tied lm_head afterwards.
    mm = app_main.mm
    loaded = {}
    load_errors = []
    barrier = threading.Barrier(2)

    def _load_one(key):
        try:
            barrier.wait(timeout=30)
            loaded[key] = mm.load(key)
        except BaseException as exc:  # noqa: BLE001
            load_errors.append((key, repr(exc)))

    threads = [
        threading.Thread(target=_load_one, args=(k,))
        for k in ("qwen2.5-0.5b-instruct", "qwen2.5-0.5b-base")
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("concurrent model loads succeed", not load_errors, str(load_errors))

    from model_manager import DTYPE

    for key, m in loaded.items():
        bad = [
            name
            for name, param in m.model.named_parameters()
            if param.is_floating_point() and param.dtype != DTYPE
        ]
        in_emb = m.model.get_input_embeddings()
        out_emb = m.model.get_output_embeddings()
        tied = in_emb is not None and out_emb is not None and (
            in_emb.weight.data_ptr() == out_emb.weight.data_ptr()
        )
        check(
            f"{key} concurrent load dtype/tie healthy",
            not bad and tied,
            f"wrong_dtype={len(bad)} tied={tied}",
        )

    inst = post("/api/models/load", {"key": "qwen2.5-0.5b-instruct"})
    check(
        "load instruct",
        inst["loaded"] and inst["layers"] == 24 and inst["intermediate"] == 4864,
        json.dumps(inst),
    )
    base = post("/api/models/load", {"key": "qwen2.5-0.5b-base"})
    check("load base", base["loaded"] and base["layers"] == 24)

    instruct = mm.get("qwen2.5-0.5b-instruct")
    basem = mm.get("qwen2.5-0.5b-base")

    # 4. chat ---------------------------------------------------------------
    chat = post(
        "/api/chat",
        {
            "model": "qwen2.5-0.5b-instruct",
            "messages": [
                {"role": "user", "content": "What is the capital of France? Answer in one word."}
            ],
            "max_new_tokens": 16,
            "temperature": 0.0,
        },
    )
    check(
        "instruct chat coherent",
        "paris" in chat["response"].lower(),
        repr(chat["response"][:80]),
    )
    base_chat = post(
        "/api/chat",
        {
            "model": "qwen2.5-0.5b-base",
            "messages": [
                {"role": "user", "content": "What is the capital of France? Answer in one word."}
            ],
            "max_new_tokens": 16,
            "temperature": 0.0,
        },
    )
    check(
        "base chat coherent",
        "paris" in base_chat["response"].lower(),
        repr(base_chat["response"][:80]),
    )

    # 5. activations --------------------------------------------------------
    act = post(
        "/api/activations",
        {
            "model": "qwen2.5-0.5b-instruct",
            "prompt": "The quick brown fox jumps over the lazy dog.",
            "top_k": 8,
        },
    )
    check(
        "activations 24 layers x 8",
        len(act["layers"]) == 24
        and all(len(v) == 8 for v in act["layers"].values())
        and act["intermediate"] == 4864,
        f"layers={len(act['layers'])}",
    )

    # 6. neuron info --------------------------------------------------------
    info = post(
        "/api/neuron_info",
        {
            "model": "qwen2.5-0.5b-instruct",
            "layer": 0,
            "neuron": 123,
            "prompt": "The quick brown fox jumps over the lazy dog.",
        },
    )
    check(
        "neuron_info norms/activation",
        info["gate_weight_norm"] >= 0
        and info["down_weight_norm"] >= 0
        and "activation" in info,
        f"gate={info['gate_weight_norm']:.3f} act={info.get('activation')}",
    )

    # 7. transplant base -> instruct ---------------------------------------
    src_gate = basem.mlp(0).gate_proj.weight[100].detach().clone()
    tr = post(
        "/api/transplant",
        {
            "src_model": "qwen2.5-0.5b-base",
            "dst_model": "qwen2.5-0.5b-instruct",
            "src_layer": 0,
            "src_neuron": 100,
            "dst_layer": 0,
            "dst_neuron": 100,
            "mode": "copy",
        },
    )
    dst_gate = instruct.mlp(0).gate_proj.weight[100].detach().clone()
    cos = torch.nn.functional.cosine_similarity(
        src_gate.float().unsqueeze(0), dst_gate.float().unsqueeze(0)
    ).item()
    check(
        "transplant copies weights",
        torch.allclose(src_gate, dst_gate) and cos > 0.99,
        f"cos={cos:.6f}",
    )
    check(
        "transplant cosine metric present",
        -1.0 <= tr["metrics"]["cosine_similarity_src_dst"] <= 1.0,
    )

    # 8. probes -------------------------------------------------------------
    pr = post(
        "/api/probes",
        {"model": "qwen2.5-0.5b-instruct", "max_new_tokens": 8},
    )
    check(
        "probes 12 results in [0,1]",
        len(pr["probes"]) == 12
        and all(0.0 <= v["score"] <= 1.0 for v in pr["probes"].values())
        and 0.0 <= pr["fitness"] <= 1.0,
        f"fitness={pr['fitness']:.4f}",
    )

    # 9. prune layer 0 at 10% ----------------------------------------------
    prune = post(
        "/api/prune",
        {
            "model": "qwen2.5-0.5b-instruct",
            "layer": 0,
            "ratio": 0.1,
            "mode": "zero_mask",
        },
    )
    expected_k = int(4864 * 0.1)
    check(
        "prune counts",
        prune["pruned_count"] == expected_k
        and prune["total_neurons"] == 4864
        and abs(prune["sparsity"] - expected_k / 4864) < 1e-9,
        f"count={prune['pruned_count']} total={prune['total_neurons']} sparsity={prune['sparsity']:.4f}",
    )
    check(
        "prune fitness before/after in [0,1]",
        0.0 <= prune["baseline_fitness"] <= 1.0
        and 0.0 <= prune["new_fitness"] <= 1.0,
        f"{prune['baseline_fitness']:.4f} -> {prune['new_fitness']:.4f}",
    )
    idx0 = prune["pruned_by_layer"]["0"]
    check("prune_by_layer only layer 0", set(prune["pruned_by_layer"]) == {"0"}, str(set(prune["pruned_by_layer"])))
    mlp0 = instruct.mlp(0)
    zero_gate = all(torch.count_nonzero(mlp0.gate_proj.weight[n]).item() == 0 for n in idx0)
    zero_up = all(torch.count_nonzero(mlp0.up_proj.weight[n]).item() == 0 for n in idx0)
    zero_down = all(torch.count_nonzero(mlp0.down_proj.weight[:, n]).item() == 0 for n in idx0)
    check("pruned rows/columns are zero", zero_gate and zero_up and zero_down)
    kept = next(n for n in range(4864) if n not in set(idx0))
    check(
        "unpruned neuron untouched",
        torch.count_nonzero(mlp0.gate_proj.weight[kept]).item() > 0,
        f"neuron={kept}",
    )

    # 9b. all-layer prune at 1% (vectorized per-layer path) ---------------
    prune_all = post(
        "/api/prune",
        {
            "model": "qwen2.5-0.5b-instruct",
            "layer": None,
            "ratio": 0.01,
            "mode": "zero_mask",
        },
    )
    expected_all = 24 * int(4864 * 0.01)
    check(
        "all-layer prune counts",
        prune_all["pruned_count"] == expected_all
        and prune_all["total_neurons"] == 24 * 4864
        and abs(prune_all["sparsity"] - expected_all / (24 * 4864)) < 1e-9,
        f"count={prune_all['pruned_count']} total={prune_all['total_neurons']}",
    )
    check(
        "all-layer prune reports every layer",
        len(prune_all["pruned_by_layer"]) == 24,
        f"layers={len(prune_all['pruned_by_layer'])}",
    )

    # 10. dream (1 gen x 2 pop, top-8/layer) ------------------------------
    # Snapshot every possible pool neuron's weights to prove restoration.
    from world_model import collect_active_neurons

    pool = collect_active_neurons(
        instruct.model, instruct.tokenizer, "cuda" if torch.cuda.is_available() else "cpu",
        top_k_per_layer=8,
    )
    pool_keys = {(p["layer"], p["neuron"]) for p in pool}
    snaps = {}
    for layer, neuron in pool_keys:
        mlp = instruct.mlp(layer)
        snaps[(layer, neuron)] = (
            mlp.gate_proj.weight[neuron].detach().clone(),
            mlp.up_proj.weight[neuron].detach().clone(),
            mlp.down_proj.weight[:, neuron].detach().clone(),
        )

    dream = post(
        "/api/dream",
        {
            "model": "qwen2.5-0.5b-instruct",
            "generations": 1,
            "population_size": 2,
            "top_k_per_layer": 8,
        },
    )
    check(
        "dream pool size = 24 layers x 8",
        dream["pool_size"] == 24 * 8,
        f"pool_size={dream['pool_size']}",
    )
    check(
        "dream history 1 gen x 2 pop",
        len(dream["history"]) == 2 and all(h["gen"] == 0 for h in dream["history"]),
    )
    all_in_pool = all(
        (h["candidate"]["layer"], h["candidate"]["neuron"]) in pool_keys
        for h in dream["history"]
    )
    check("all candidates drawn from active pool", all_in_pool)
    check(
        "dream fitness values valid",
        0.0 <= dream["baseline_fitness"] <= 1.0
        and all(0.0 <= h["fitness"] <= 1.0 for h in dream["history"]),
        f"baseline={dream['baseline_fitness']:.4f}",
    )

    restored = True
    bad = None
    for (layer, neuron), (g, u, d) in snaps.items():
        mlp = instruct.mlp(layer)
        if not (
            torch.equal(g, mlp.gate_proj.weight[neuron].detach())
            and torch.equal(u, mlp.up_proj.weight[neuron].detach())
            and torch.equal(d, mlp.down_proj.weight[:, neuron].detach())
        ):
            restored = False
            bad = (layer, neuron)
            break
    check("dream restored every candidate's weights", restored, f"bad={bad}")

    # 11. persistent apply_surgery ----------------------------------------
    layer, neuron = 1, 50
    mlp = instruct.mlp(layer)
    before = (
        mlp.gate_proj.weight[neuron].detach().clone(),
        mlp.up_proj.weight[neuron].detach().clone(),
        mlp.down_proj.weight[:, neuron].detach().clone(),
    )
    applied = post(
        "/api/apply_surgery",
        {
            "model": "qwen2.5-0.5b-instruct",
            "surgery": {"layer": layer, "neuron": neuron, "mode": "amplify", "scale": 1.5},
        },
    )
    after = (
        mlp.gate_proj.weight[neuron].detach().clone(),
        mlp.up_proj.weight[neuron].detach().clone(),
        mlp.down_proj.weight[:, neuron].detach().clone(),
    )
    check("apply_surgery returns applied", applied["applied"] is True)
    check(
        "apply_surgery persisted 1.5x weights",
        torch.allclose(after[0], before[0] * 1.5, atol=1e-3)
        and torch.allclose(after[1], before[1] * 1.5, atol=1e-3)
        and torch.allclose(after[2], before[2] * 1.5, atol=1e-3),
        f"max_delta={float((after[0] - before[0] * 1.5).abs().max()):.4f}",
    )

    # 12. export to safetensors -------------------------------------------
    save = post(
        "/api/models/save",
        {"model": "qwen2.5-0.5b-instruct", "output_name": EXPORT_NAME},
    )
    check(
        "export response",
        save["status"] == "saved"
        and save["format"] == "safetensors"
        and Path(save["path"]) == EXPORT_PATH,
        save["path"],
    )
    check(
        "export dir contents",
        EXPORT_PATH.is_dir()
        and (EXPORT_PATH / "model.safetensors").is_file()
        and (EXPORT_PATH / "config.json").is_file()
        and (EXPORT_PATH / "tokenizer_config.json").is_file(),
        str(sorted(p.name for p in EXPORT_PATH.iterdir())) if EXPORT_PATH.is_dir() else "missing",
    )
    st_file = EXPORT_PATH / "model.safetensors"
    with safe_open(str(st_file), framework="pt", device="cpu") as f:
        keys = list(f.keys())
        gate_shape = tuple(f.get_slice("model.layers.0.mlp.gate_proj.weight").get_shape())
    check(
        "safetensors readable with expected shapes",
        len(keys) > 100 and gate_shape == (4864, 896),
        f"tensors={len(keys)} gate_shape={gate_shape}",
    )
    # Confirm the pruned rows actually made it into the export.
    with safe_open(str(st_file), framework="pt", device="cpu") as f:
        exported_gate0 = f.get_tensor("model.layers.0.mlp.gate_proj.weight")
    check(
        "export contains pruned zero rows",
        all(torch.count_nonzero(exported_gate0[n]).item() == 0 for n in idx0[:20]),
    )
    del exported_gate0
    torch.cuda.empty_cache()

    # 13. surgery log persistence -----------------------------------------
    log = client.get("/api/surgery_log").json()
    types = [e.get("type") for e in log["entries"]]
    for t in ("transplant", "prune", "dream", "apply_surgery", "export"):
        check(f"log contains {t}", t in types)
    on_disk = [
        json.loads(ln)
        for ln in LOG_PATH.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    check(
        "JSONL file has the same entries",
        len(on_disk) == len(log["entries"]) and types == [e.get("type") for e in on_disk],
        f"disk={len(on_disk)} memory={len(log['entries'])}",
    )
    check("log path reported", log["path"].endswith("surgeries.jsonl"), log["path"])
finally:
    # --- cleanup -----------------------------------------------------------
    if EXPORT_PATH.exists() and not export_preexisting:
        shutil.rmtree(EXPORT_PATH, ignore_errors=True)
    if log_backup is None:
        if LOG_PATH.exists():
            LOG_PATH.unlink()
        if LOG_PATH.parent.exists() and not any(LOG_PATH.parent.iterdir()):
            LOG_PATH.parent.rmdir()
    else:
        LOG_PATH.write_text(log_backup, encoding="utf-8")
    # Reload the in-memory log to match the restored file.
    app_main.SURGERY_LOG[:] = app_main._load_surgery_log()
    try:
        app_main.mm.unload("qwen2.5-0.5b-base")
        app_main.mm.unload("qwen2.5-0.5b-instruct")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

print()
if failures:
    print(f"RESULT: {len(failures)} FAILURE(S): {failures}")
    sys.exit(1)
print("RESULT: ALL CHECKS PASSED")
sys.exit(0)
