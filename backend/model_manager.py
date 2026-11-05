"""Model loading and forward-hook management for the Neuron Lab."""
import threading

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Registry of loadable models. Add more here.
MODEL_REGISTRY = {
    "qwen2.5-0.5b-base": {
        "hf_id": "Qwen/Qwen2.5-0.5B",
        "label": "Qwen 2.5 0.5B (base)",
        "family": "qwen2.5-0.5b",
    },
    "qwen2.5-0.5b-instruct": {
        "hf_id": "Qwen/Qwen2.5-0.5B-Instruct",
        "label": "Qwen 2.5 0.5B (instruct)",
        "family": "qwen2.5-0.5b",
    },
}


class LoadedModel:
    def __init__(self, key, hf_id, label, family):
        self.key = key
        self.hf_id = hf_id
        self.label = label
        self.family = family
        self.model = None
        self.tokenizer = None
        self.hooks = []  # list of hook handles
        self.activation_cache = {}  # layer -> tensor (intermediate acts)
        self.loaded = False

    def load(self):
        print(f"[load] {self.key} from {self.hf_id} on {DEVICE}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_id, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.hf_id,
            dtype=DTYPE,
            device_map=DEVICE if DEVICE == "cuda" else None,
            trust_remote_code=True,
        )
        if DEVICE == "cpu":
            self.model = self.model.to(DEVICE)
        self.model.eval()
        self._repair_load_state()
        self._register_hooks()
        self.loaded = True
        print(f"[load] {self.key} ready. layers={self.num_layers()} hidden={self.hidden_size()} intermediate={self.intermediate_size()}")

    def _repair_load_state(self):
        """Guard against mixed-dtype/tied-weight corruption from a bad load.

        Concurrent `from_pretrained` calls can leave some floating parameters
        in float32 and an untied, randomly-initialised `lm_head`, which causes
        `mat1 and mat2 must have the same dtype` during generation. Loads are
        serialized in ModelManager, but this check keeps a bad state from
        silently reaching the UI.
        """
        bad = [
            name
            for name, param in self.model.named_parameters()
            if param.is_floating_point() and param.dtype != DTYPE
        ]
        if bad:
            print(f"[load] repairing dtype for {len(bad)} parameters")
            self.model.to(device=DEVICE, dtype=DTYPE)

        if getattr(self.model.config, "tie_word_embeddings", False):
            in_emb = self.model.get_input_embeddings()
            out_emb = self.model.get_output_embeddings()
            if (
                in_emb is not None
                and out_emb is not None
                and in_emb.weight.data_ptr() != out_emb.weight.data_ptr()
            ):
                print("[load] re-tying lm_head to input embeddings")
                self.model.tie_weights()

        bad = [
            name
            for name, param in self.model.named_parameters()
            if param.is_floating_point() and param.dtype != DTYPE
        ]
        if bad:
            raise RuntimeError(
                f"{self.key}: parameters with wrong dtype after load: {bad[:8]}"
            )
        if getattr(self.model.config, "tie_word_embeddings", False):
            in_emb = self.model.get_input_embeddings()
            out_emb = self.model.get_output_embeddings()
            if (
                in_emb is not None
                and out_emb is not None
                and in_emb.weight.data_ptr() != out_emb.weight.data_ptr()
            ):
                raise RuntimeError(f"{self.key}: tied embeddings are not shared after load")

    def unload(self):
        self.model = None
        self.tokenizer = None
        for h in self.hooks:
            h.remove()
        self.hooks = []
        self.loaded = False
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

    # ---- architecture helpers ----
    def num_layers(self):
        return len(self.model.model.layers)

    def hidden_size(self):
        return self.model.config.hidden_size

    def intermediate_size(self):
        # MLP intermediate dim = number of "neurons" per layer
        return self.model.model.layers[0].mlp.gate_proj.out_features

    def mlp(self, layer_idx):
        return self.model.model.layers[layer_idx].mlp

    # ---- hooks ----
    def _register_hooks(self):
        """Register forward hooks to capture intermediate MLP activations."""
        for li, layer in enumerate(self.model.model.layers):
            mlp = layer.mlp

            def make_hook(layer_idx):
                def hook(module, inputs, output):
                    # output of down_proj is hidden; we want the INPUT to down_proj
                    # which is the intermediate activation. We capture via a pre-hook instead.
                    pass
                return hook

            # Use a pre-forward hook on down_proj to capture intermediate acts
            def make_prehook(layer_idx):
                def prehook(module, args):
                    # args[0] is the intermediate activation tensor (batch, seq, intermediate)
                    act = args[0].detach()
                    self.activation_cache[layer_idx] = act
                return prehook

            h = mlp.down_proj.register_forward_pre_hook(make_prehook(li))
            self.hooks.append(h)

    def clear_cache(self):
        self.activation_cache = {}

    # ---- generation ----
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=80, temperature=0.7, intervention=None):
        """Generate text. intervention: dict with layer, neuron, mode, value."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        # register intervention hook if requested
        handle = None
        if intervention:
            handle = self._register_intervention(intervention)
        try:
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            if handle is not None:
                handle.remove()
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text

    def _register_intervention(self, iv):
        layer_idx = iv["layer"]
        neuron = iv["neuron"]
        mode = iv["mode"]  # clamp | ablate | amplify | set
        value = iv.get("value", 0.0)
        scale = iv.get("scale", 1.0)

        mlp = self.model.model.layers[layer_idx].mlp

        def prehook(module, args):
            act = args[0].clone()
            if mode == "ablate":
                act[..., neuron] = 0.0
            elif mode == "clamp":
                act[..., neuron] = value
            elif mode == "amplify":
                act[..., neuron] = act[..., neuron] * scale
            elif mode == "set":
                act[..., neuron] = value
            return (act,) + args[1:]

        return mlp.down_proj.register_forward_pre_hook(prehook)

    # ---- activations ----
    @torch.no_grad()
    def get_activations(self, prompt, layer_idx=None, top_k=64):
        """Return top-k neurons by mean activation for a prompt."""
        self.clear_cache()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
        _ = self.model(**inputs)
        results = {}
        layers = [layer_idx] if layer_idx is not None else list(range(self.num_layers()))
        for li in layers:
            act = self.activation_cache.get(li)
            if act is None:
                continue
            # act: (batch, seq, intermediate) -> mean over batch & seq
            mean_act = act.float().mean(dim=(0, 1))  # (intermediate,)
            # top-k by absolute magnitude
            vals, idxs = torch.topk(mean_act.abs(), k=min(top_k, mean_act.numel()))
            neurons = []
            for v, i in zip(vals.tolist(), idxs.tolist()):
                neurons.append({
                    "neuron": int(i),
                    "activation": float(mean_act[i].item()),
                    "abs_activation": float(abs(mean_act[i].item())),
                })
            results[li] = neurons
        return results

    # ---- neuron info ----
    @torch.no_grad()
    def neuron_info(self, layer_idx, neuron_idx, prompt=None):
        mlp = self.model.model.layers[layer_idx].mlp
        gate_w = mlp.gate_proj.weight[neuron_idx].detach().float().cpu()
        up_w = mlp.up_proj.weight[neuron_idx].detach().float().cpu()
        down_w = mlp.down_proj.weight[:, neuron_idx].detach().float().cpu()
        info = {
            "layer": layer_idx,
            "neuron": neuron_idx,
            "gate_weight_norm": float(gate_w.norm().item()),
            "up_weight_norm": float(up_w.norm().item()),
            "down_weight_norm": float(down_w.norm().item()),
            "total_norm": float(torch.cat([gate_w, up_w, down_w]).norm().item()),
            "gate_weight_sample": gate_w[:8].tolist(),
            "down_weight_sample": down_w[:8].tolist(),
        }
        if prompt:
            acts = self.get_activations(prompt, layer_idx=layer_idx, top_k=self.intermediate_size())
            # find this neuron's activation
            found = None
            for n in acts.get(layer_idx, []):
                if n["neuron"] == neuron_idx:
                    found = n["activation"]
                    break
            # recompute exact (topk may have dropped it)
            self.clear_cache()
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            _ = self.model(**inputs)
            act = self.activation_cache.get(layer_idx)
            if act is not None:
                mean_act = act.float().mean(dim=(0, 1))
                info["activation"] = float(mean_act[neuron_idx].item())
            else:
                info["activation"] = found
        return info

    # ---- pruning ----
    @torch.no_grad()
    def compute_neuron_importance(self, calibration_prompts, layer_idx=None):
        """Wanda-style neuron importance:

            I_n = ||W_gate[n]||_2 * E[|activation[n]|]

        The expectation is taken over the calibration prompts (the 12-probe
        battery by default). Returns a dict of CPU float32 tensors keyed by
        layer index.
        """
        layers = [layer_idx] if layer_idx is not None else list(range(self.num_layers()))
        sums = {
            li: torch.zeros(self.intermediate_size(), dtype=torch.float32)
            for li in layers
        }
        used = 0
        for prompt in calibration_prompts:
            self.clear_cache()
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=128).to(DEVICE)
            # Run the base model only: hooks still fire, but we skip the lm_head
            # and its full-vocabulary logits, saving VRAM.
            _ = self.model.model(**inputs)
            for li in layers:
                act = self.activation_cache.get(li)
                if act is None:
                    continue
                sums[li] += act.float().abs().mean(dim=(0, 1)).cpu()
            used += 1
        if used:
            for li in layers:
                sums[li] /= used

        importance = {}
        for li in layers:
            gate = self.mlp(li).gate_proj.weight.detach().float().cpu()
            importance[li] = gate.norm(dim=1) * sums[li]
        self.clear_cache()
        return importance

    @torch.no_grad()
    def prune_neurons(self, layer_idx, ratio, calibration_prompts):
        """Zero the bottom `ratio` fraction of neurons by importance.

        If layer_idx is None the ratio is applied independently to every layer.
        Zeroing gate/up rows and the down column makes the neuron permanently
        inactive. Returns counts plus the pruned indices per layer.
        """
        if not (0.0 < ratio < 1.0):
            raise ValueError("ratio must be in (0, 1)")

        importance = self.compute_neuron_importance(calibration_prompts, layer_idx=layer_idx)
        pruned_by_layer = {}
        pruned_count = 0
        total_neurons = 0

        for li, imp in importance.items():
            n_total = int(imp.numel())
            total_neurons += n_total
            k = int(n_total * ratio)
            if k <= 0:
                pruned_by_layer[str(li)] = []
                continue
            idxs = torch.topk(imp, k, largest=False).indices.tolist()
            mlp = self.mlp(li)
            # Vectorized zero-mask: one indexed write per tensor instead of a
            # Python loop over every neuron (matters for all-layer 50% prunes).
            idx_t = torch.as_tensor(
                idxs, dtype=torch.long, device=mlp.gate_proj.weight.device
            )
            mlp.gate_proj.weight.data[idx_t] = 0.0
            mlp.up_proj.weight.data[idx_t] = 0.0
            mlp.down_proj.weight.data[:, idx_t] = 0.0
            pruned_by_layer[str(li)] = [int(n) for n in idxs]
            pruned_count += len(idxs)

        self.clear_cache()
        return {
            "pruned_count": pruned_count,
            "total_neurons": total_neurons,
            "sparsity": (pruned_count / total_neurons) if total_neurons else 0.0,
            "pruned_by_layer": pruned_by_layer,
        }

    # ---- transplant ----
    @torch.no_grad()
    def transplant_neuron(self, src, src_layer, src_neuron, dst_layer, dst_neuron, mode="copy", scale=1.0):
        """Extract from src model, apply to self (dst). Returns metrics."""
        src_mlp = src.mlp(src_layer)
        dst_mlp = self.mlp(dst_layer)

        # source weights
        src_gate = src_mlp.gate_proj.weight[src_neuron].clone()
        src_up = src_mlp.up_proj.weight[src_neuron].clone()
        src_down = src_mlp.down_proj.weight[:, src_neuron].clone()

        # destination weights (before)
        dst_gate_before = dst_mlp.gate_proj.weight[dst_neuron].clone()
        dst_up_before = dst_mlp.up_proj.weight[dst_neuron].clone()
        dst_down_before = dst_mlp.down_proj.weight[:, dst_neuron].clone()

        # cosine similarity between src and dst (input-side)
        src_vec = torch.cat([src_gate, src_up])
        dst_vec = torch.cat([dst_gate_before, dst_up_before])
        cos = torch.nn.functional.cosine_similarity(src_vec.unsqueeze(0), dst_vec.unsqueeze(0)).item()

        if mode == "copy":
            dst_mlp.gate_proj.weight.data[dst_neuron] = src_gate.to(dst_mlp.gate_proj.weight.dtype).to(DEVICE)
            dst_mlp.up_proj.weight.data[dst_neuron] = src_up.to(dst_mlp.up_proj.weight.dtype).to(DEVICE)
            dst_mlp.down_proj.weight.data[:, dst_neuron] = src_down.to(dst_mlp.down_proj.weight.dtype).to(DEVICE)
        elif mode == "delta":
            # add scaled delta
            dst_mlp.gate_proj.weight.data[dst_neuron] += (src_gate - dst_gate_before).to(dst_mlp.gate_proj.weight.dtype).to(DEVICE) * scale
            dst_mlp.up_proj.weight.data[dst_neuron] += (src_up - dst_up_before).to(dst_mlp.up_proj.weight.dtype).to(DEVICE) * scale
            dst_mlp.down_proj.weight.data[:, dst_neuron] += (src_down - dst_down_before).to(dst_mlp.down_proj.weight.dtype).to(DEVICE) * scale
        elif mode == "blend":
            alpha = scale
            dst_mlp.gate_proj.weight.data[dst_neuron] = ((1 - alpha) * dst_gate_before + alpha * src_gate).to(dst_mlp.gate_proj.weight.dtype).to(DEVICE)
            dst_mlp.up_proj.weight.data[dst_neuron] = ((1 - alpha) * dst_up_before + alpha * src_up).to(dst_mlp.up_proj.weight.dtype).to(DEVICE)
            dst_mlp.down_proj.weight.data[:, dst_neuron] = ((1 - alpha) * dst_down_before + alpha * src_down).to(dst_mlp.down_proj.weight.dtype).to(DEVICE)

        return {
            "src": {"layer": src_layer, "neuron": src_neuron},
            "dst": {"layer": dst_layer, "neuron": dst_neuron},
            "mode": mode,
            "cosine_similarity_src_dst": float(cos),
            "src_norm": float(src_vec.norm().item()),
            "dst_norm_before": float(dst_vec.norm().item()),
        }

    # ---- activation steering ----
    @torch.no_grad()
    def compute_steering_vector(self, positive_prompt, negative_prompt, layer_idx):
        """Compute mean activation difference (positive - negative) in residual stream."""
        # Hook residual stream
        captured = {"pos": None, "neg": None}

        def make_hook(store_key):
            def hook(module, inputs, output):
                # output is (hidden_states,) or tensor
                hs = output[0] if isinstance(output, tuple) else output
                captured[store_key] = hs.detach().float().mean(dim=1)  # (batch, hidden)
            return hook

        layer = self.model.model.layers[layer_idx]
        h1 = layer.register_forward_hook(make_hook("pos"))
        inputs = self.tokenizer(positive_prompt, return_tensors="pt").to(DEVICE)
        _ = self.model(**inputs)
        h1.remove()

        h2 = layer.register_forward_hook(make_hook("neg"))
        inputs = self.tokenizer(negative_prompt, return_tensors="pt").to(DEVICE)
        _ = self.model(**inputs)
        h2.remove()

        vec = (captured["pos"] - captured["neg"]).squeeze(0)  # (hidden,)
        return vec

    @torch.no_grad()
    def generate_with_steering(self, prompt, steering_vec, layer_idx, scale=1.0, max_new_tokens=80):
        vec = steering_vec.to(DEVICE).to(DTYPE)

        def hook(module, inputs, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs + scale * vec.unsqueeze(0).unsqueeze(0)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        handle = self.model.model.layers[layer_idx].register_forward_hook(hook)
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(DEVICE)
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        finally:
            handle.remove()
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


class ModelManager:
    def __init__(self):
        self.models = {}  # key -> LoadedModel
        # `from_pretrained` is not thread-safe: two near-simultaneous UI
        # dropdown loads can interleave weight loading and leave mixed dtypes
        # plus an untied/random lm_head. Serialise the whole load.
        self._load_lock = threading.Lock()

    def available(self):
        return [
            {"key": k, "label": v["label"], "family": v["family"], "hf_id": v["hf_id"],
             "loaded": k in self.models and self.models[k].loaded}
            for k, v in MODEL_REGISTRY.items()
        ]

    def load(self, key):
        if key in self.models and self.models[key].loaded:
            return self.models[key]
        if key not in MODEL_REGISTRY:
            raise ValueError(f"Unknown model: {key}")
        with self._load_lock:
            # Another request may have loaded it while we waited.
            if key in self.models and self.models[key].loaded:
                return self.models[key]
            info = MODEL_REGISTRY[key]
            m = LoadedModel(key, info["hf_id"], info["label"], info["family"])
            m.load()
            self.models[key] = m
            return m

    def get(self, key):
        if key not in self.models or not self.models[key].loaded:
            raise ValueError(f"Model not loaded: {key}")
        return self.models[key]

    def unload(self, key):
        if key in self.models:
            self.models[key].unload()
            del self.models[key]
