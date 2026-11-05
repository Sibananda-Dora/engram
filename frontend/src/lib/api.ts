const BASE = "/api";

async function post<T>(path: string, body: any): Promise<T> {
  const res = await fetch(BASE + path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const err = await res.text();
    throw new Error(err || res.statusText);
  }
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(res.statusText);
  return res.json();
}

export const api = {
  health: () => get<any>("/health"),
  models: () => get<{ models: any[] }>("/models"),
  loadModel: (key: string) => post<any>("/models/load", { key }),
  unloadModel: (key: string) => post<any>("/models/unload", { key }),
  chat: (model: string, messages: any[], max_new_tokens = 80, temperature = 0.7, intervention?: any) =>
    post<{ response: string }>("/chat", { model, messages, max_new_tokens, temperature, intervention }),
  activations: (model: string, prompt: string, layer?: number | null, top_k = 48) =>
    post<any>("/activations", { model, prompt, layer, top_k }),
  neuronInfo: (model: string, layer: number, neuron: number, prompt?: string) =>
    post<any>("/neuron_info", { model, layer, neuron, prompt }),
  intervene: (model: string, prompt: string, layer: number, neuron: number, mode: string, value = 0, scale = 2, max_new_tokens = 80) =>
    post<any>("/intervene", { model, prompt, layer, neuron, mode, value, scale, max_new_tokens }),
  transplant: (src_model: string, dst_model: string, src_layer: number, src_neuron: number, dst_layer: number, dst_neuron: number, mode = "copy", scale = 1, test_prompt?: string) =>
    post<any>("/transplant", { src_model, dst_model, src_layer, src_neuron, dst_layer, dst_neuron, mode, scale, test_prompt }),
  steer: (model: string, prompt: string, positive_prompt: string, negative_prompt: string, layer: number, scale = 1, max_new_tokens = 80) =>
    post<any>("/steer", { model, prompt, positive_prompt, negative_prompt, layer, scale, max_new_tokens }),
  probes: (model: string, max_new_tokens = 40) =>
    post<any>("/probes", { model, max_new_tokens }),
  dream: (model: string, generations = 3, population_size = 5, top_k_per_layer = 32) =>
    post<any>("/dream", { model, generations, population_size, top_k_per_layer }),
  prune: (model: string, ratio: number, layer: number | null = null, mode = "zero_mask") =>
    post<any>("/prune", { model, ratio, layer, mode }),
  applySurgery: (model: string, surgery: any) =>
    post<any>("/apply_surgery", { model, surgery }),
  saveModel: (model: string, outputName: string) =>
    post<any>("/models/save", { model, output_name: outputName }),
  surgeryLog: () => get<any>("/surgery_log"),
};