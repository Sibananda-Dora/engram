export type ModelInfo = {
  key: string;
  label: string;
  family: string;
  hf_id: string;
  loaded: boolean;
};

export type Neuron = {
  neuron: number;
  activation: number;
  abs_activation: number;
};

export type NeuronInfo = {
  layer: number;
  neuron: number;
  gate_weight_norm: number;
  up_weight_norm: number;
  down_weight_norm: number;
  total_norm: number;
  activation?: number;
  gate_weight_sample: number[];
  down_weight_sample: number[];
};

export type TransplantMetrics = {
  src: { layer: number; neuron: number };
  dst: { layer: number; neuron: number };
  mode: string;
  cosine_similarity_src_dst: number;
  src_norm: number;
  dst_norm_before: number;
};