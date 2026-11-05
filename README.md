# Neuron Lab 🧠⚡

> **An interactive mechanistic interpretability workbench for surgical neuron inspection, drag-and-drop weight transplantation, Wanda pruning, and neural Darwinism.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![React 19](https://img.shields.io/badge/React-19-cyan.svg)](https://react.dev)
[![VRAM](https://img.shields.io/badge/VRAM-4GB%20Friendly-green.svg)](#hardware-requirements)

---

## Overview

Most neural network visualizers are **read-only museum exhibits**—they show token flow and attention matrices, but don't let you alter the network. Most interpretability frameworks (TransformerLens, NNsight) are **headless Python libraries** locked inside Jupyter notebooks.

**Neuron Lab bridges the microscope and the scalpel.** It is an interactive browser application where you can:
1. Inspect live SwiGLU MLP neuron activations across all 24 layers of an LLM.
2. Physically **drag a neuron** from Model A and drop it into Model B, grafting its weights in real-time GPU memory.
3. Perform surgical **Clamp**, **Ablate**, and **Amplify (×3)** interventions on individual neurons.
4. Apply in-place **Wanda (Weights and Activations) Pruning** to zero out dead neurons and evaluate probe fitness before/after.
5. Use an **Evolutionary World Model** to simulate hundreds of candidate surgeries in an active-neuron sandbox ("Neural Darwinism") and apply the winning recipe.
6. **Export your modified model** directly to `.safetensors` format for downstream use in Hugging Face or Ollama.

---

## Key Features

* **Side-by-Side Live Inspection**: Compare two models simultaneously (e.g., `Qwen 2.5 0.5B Base` vs. `Qwen 2.5 0.5B Instruct`).
* **Interactive Canvas 2D Field**:
  * 🟠 **Orange Nodes**: Positive Activation (+act, firing to boost tokens).
  * 🔵 **Blue Nodes**: Negative Activation (-act, inhibitory).
  * 🟡 **Gold Nodes**: Transplanted neurons from another model.
  * ⚪ **Dim Grey Nodes**: Pruned / Inactive neurons (zeroed out).
* **Drag-and-Drop Transplant**: Drag any neuron node across panels to graft `gate_proj`, `up_proj`, and `down_proj` weights with live cosine similarity metrics.
* **Single-Neuron Surgery**:
  * `Clamp → 0`: Lock a neuron to a constant value.
  * `Ablate`: Zero out a neuron's weights entirely.
  * `Amplify ×3`: Multiply activation by 3 (reproducing viral behavioral switch experiments like "Evil Qwen").
* **Wanda In-Place Pruning**: Select 10%, 25%, or 50% ratios to mute lowest-importance neurons ($||W||_2 \times |A|$) and measure degradation on the 12-probe benchmark.
* **Active-Pool Evolutionary Search**: Rather than searching 116,000 neurons at random, candidates are sampled strictly from neurons that actively fired during baseline probes.
* **One-Click Export**: Save the pruned or transplanted model directly to disk as `.safetensors` + tokenizer configs.

---

## Architecture & Scientific Principles

```
┌─────────────────────────────────────────────────────────────┐
│ Browser UI (localhost:5173)                                 │
│ ┌────────────────┐  ┌────────────────┐  ┌────────────────┐ │
│ │  ModelPanel A  │  │  ModelPanel B  │  │   DreamPanel   │ │
│ │  (Chat+Canvas) │  │  (Chat+Canvas) │  │  (Evolutionary)│ │
│ └────────┬───────┘  └────────┬───────┘  └────────┬───────┘ │
└──────────┼───────────────────┼───────────────────┼──────────┘
           │ fetch /api/*      │                   │
┌──────────┼───────────────────┼───────────────────┼──────────┐
│ FastAPI Backend (localhost:8000)                             │
│ ┌────────────────────────────┴─────────────────────────────┐ │
│ │ main.py — REST API (17 routes)                           │ │
│ └──────────────┬───────────────────────────┬───────────────┘ │
│ ┌──────────────┴──────────────┐ ┌──────────┴───────────────┐ │
│ │ model_manager.py            │ │ world_model.py           │ │
│ │ (Hooks, Wanda, Transplant)  │ │ (12 Probes, Darwinism)   │ │
│ └──────────────┬──────────────┘ └──────────────────────────┘ │
│ ┌──────────────┴──────────────┐                              │
│ │ Qwen 2.5 0.5B (CUDA fp16)   │                              │
│ │ 24 Layers · 4,864 Neurons   │                              │
│ └─────────────────────────────┘                              │
└──────────────────────────────────────────────────────────────┘
```

### Why Same-Lineage Transplant Works
Neural networks have arbitrary coordinate rotation and **permutation invariance**—copying a neuron between two arbitrary models (e.g. Llama to Mistral) results in mathematical noise. 

Neuron Lab operates on **same-lineage models** (`Qwen 2.5 0.5B Base` ↔ `Qwen 2.5 0.5B Instruct`). Because Instruct was fine-tuned directly from Base, the neuron coordinates remain strictly aligned (cosine similarity ~0.997), making surgical weight transplants meaningful.

---

## Hardware Requirements

* **GPU**: NVIDIA GPU with ≥ 4 GB VRAM (verified on NVIDIA GeForce RTX 3050 Laptop GPU).
* **RAM**: ≥ 8 GB System Memory.
* **Disk**: ~2 GB (for cached model weights).
* **OS**: Windows, Linux, or macOS.

---

## Quickstart Guide

### 1. Prerequisites
- Python 3.11+ (Python 3.13 verified)
- Node.js 18+ & npm
- (Recommended) NVIDIA GPU with CUDA drivers

### 2. Backend Setup
```bash
cd backend
python -m venv .venv

# Activate virtual environment
# Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# Linux/macOS:
source .venv/bin/activate

# Install PyTorch with CUDA (e.g., cu124 or cu132 matching your GPU):
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu124

# Install backend dependencies:
pip install -r requirements.txt
```

### 3. Frontend Setup
```bash
cd ../frontend
npm install
```

### 4. Running the Application

**Terminal 1 — Backend:**
```bash
cd backend
python main.py
# Server runs on http://localhost:8000
```

**Terminal 2 — Frontend:**
```bash
cd frontend
npm run dev
# Vite dev server runs on http://localhost:5173
```

Open **`http://localhost:5173`** in your browser.

---

## Project Structure

```
neuron-lab/
├── backend/
│   ├── main.py               # FastAPI application & REST routes
│   ├── model_manager.py      # Hooks, activation cache, transplant & pruning
│   ├── world_model.py        # 12-probe battery & active-pool evolutionary search
│   ├── requirements.txt      # Pinned, verified dependencies
│   ├── _e2e_test.py          # 39-point in-process test suite
│   ├── data/                 # JSONL surgery persistence
│   └── exported_models/      # Safetensors export target
├── frontend/
│   ├── src/
│   │   ├── App.tsx           # Layout, drag-and-drop, modals & responsive grid
│   │   ├── components/
│   │   │   ├── ModelPanel.tsx  # Model chat, inspect trigger, prune controls
│   │   │   ├── NeuronField.tsx # Canvas 2D dynamic neuron activation field
│   │   │   └── DreamPanel.tsx  # World model probe scores & evolutionary search
│   │   ├── lib/              # API fetch client & TypeScript interfaces
│   │   └── index.css         # Dark glassmorphism styling
│   ├── package.json
│   └── vite.config.ts
├── .gitignore                # Excludes .venv, node_modules, weights & caches
└── README.md
```

---

## Verification & Testing

To run the full 39-point backend test suite (testing concurrency, activations, transplant, Wanda pruning, Darwinian search, and safetensors export):

```bash
cd backend
python _e2e_test.py
```

To run the frontend production build:
```bash
cd frontend
npm run build
```

---

## License

This project is open-source under the [MIT License](LICENSE).
