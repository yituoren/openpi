# OpenPI Codebase Structure

## Top-Level Directory Layout

```
openpi/
├── src/openpi/              # Main Python package (models, training, serving, data)
├── scripts/                 # Entry-point scripts (train, serve, compute norm stats)
├── examples/                # Robot/simulator-specific inference & data conversion
├── packages/openpi-client/  # Lightweight client library for robot-side inference
├── third_party/             # Git submodules (aloha, libero support files)
├── docs/                    # Documentation
├── pyproject.toml           # Dependencies and project metadata
└── uv.lock                  # Locked dependency manifest
```

---

## `src/openpi/` Package Structure

### `models/` — Model Definitions (JAX/Flax NNX)

| File | Purpose |
|---|---|
| `model.py` | `BaseModelConfig` / `BaseModel` ABCs, `Observation` / `Actions` dataclasses, `ModelType` enum |
| `pi0_config.py` / `pi0.py` | Pi0 flow-matching diffusion policy (two Gemma experts + SigLIP) |
| `pi0_fast.py` | Pi0FAST autoregressive policy (single Gemma + pluggable tokenizer) |
| `gemma.py` / `gemma_fast.py` | Gemma 2B/300M language model modules (Linen), with KV-cache variant |
| `siglip.py` | SigLIP So400m/14 vision encoder |
| `lora.py` | LoRA adapter for Einsum layers |
| `tokenizer.py` | All tokenizers: `PaligemmaTokenizer`, `FASTTokenizer`, `BinningTokenizer`, `FSQTokenizer` |
| `utils/fsq_tokenizer.py` | FSQ attention tokenizer (Linen) |

**Model pattern**: each model has a `Config` dataclass (hyperparams + factory methods) and a `Model` NNX module (loss + sampling). `Pi0FASTConfig.fast_model_tokenizer` is pluggable — defaults to `FASTTokenizer` but can be swapped to `BinningTokenizer`.

### `training/` — Training Infrastructure

| File | Purpose |
|---|---|
| `config.py` | `TrainConfig`, all `DataConfigFactory` subclasses, named config registry (`_CONFIGS`) |
| `data_loader.py` | `TorchDataLoader` (LeRobot), `RLDSDataLoader` (DROID), batch → JAX arrays |
| `optimizer.py` | `CosineDecaySchedule`, `AdamW`, `SGD` (Optax wrappers) |
| `checkpoints.py` | Orbax save/restore for train state and params |
| `weight_loaders.py` | `CheckpointWeightLoader`, `PaliGemmaWeightLoader` |
| `sharding.py` | FSDP mesh setup and automatic tensor sharding |
| `utils.py` | `TrainState` dataclass, logging helpers |

### `policies/` — Inference Adapters

| File | Purpose |
|---|---|
| `policy.py` | `Policy` class: wraps model + transforms for `infer(obs) -> actions` |
| `policy_config.py` | `create_trained_policy()` factory (loads checkpoint → Policy) |
| `aloha_policy.py` | Aloha-specific input/output transforms |
| `droid_policy.py` | DROID-specific transforms |
| `libero_policy.py` | LIBERO-specific transforms |

### `serving/`

- `websocket_policy_server.py` — asyncio WebSocket server; receives msgpack obs, returns actions.

### `shared/` — Utilities

| File | Purpose |
|---|---|
| `normalize.py` | `NormStats`, `RunningStats`, save/load as JSON |
| `download.py` | `maybe_download()` with GCS/fsspec + file locking |
| `nnx_utils.py` | `module_jit()`, `PathRegex` NNX filter |
| `image_tools.py` | `resize_with_pad` |
| `array_typing.py` | JAX/PyTorch type aliases, `@typecheck` |

### `transforms.py` — Data Transform Primitives

Key transforms used across training and inference:

| Transform | Description |
|---|---|
| `RepackTransform` | Remaps dict keys using a template |
| `Normalize` / `Unnormalize` | Z-score or quantile normalization |
| `ResizeImages` | Pad and resize to target dimensions |
| `TokenizePrompt` | For Pi0: encodes prompt with `PaligemmaTokenizer` |
| `TokenizeFASTInputs` | For Pi0FAST: encodes prompt + state + actions with pluggable tokenizer |
| `ExtractFASTActions` | Post-inference: decodes tokens back to continuous actions |
| `DeltaActions` / `AbsoluteActions` | Convert between absolute and delta action spaces |
| `PromptFromLeRobotTask` | Looks up task string from dataset metadata |

---

## `scripts/` — Entry Points

| Script | Purpose |
|---|---|
| `train.py` | JAX training loop (config parsing, data loading, gradient updates, checkpointing) |
| `train_pytorch.py` | PyTorch DDP training (mirrors JAX trainer) |
| `serve_policy.py` | Loads checkpoint, starts WebSocket policy server |
| `compute_norm_stats.py` | Iterates dataset, computes `NormStats`, saves to assets dir |

---

## `examples/` — Robot-Specific Examples

Each example follows the pattern: connect to policy server via `WebsocketClientPolicy` → run env loop → `policy.infer(obs)` → execute actions.

- `aloha_real/` — Physical Aloha robot
- `aloha_sim/` — gym-aloha simulation
- `libero/` — LIBERO benchmark (includes data conversion + training scripts)
- `droid/` — DROID manipulation dataset
- `simple_client/` — Minimal latency benchmark

---

## Training Flow (End-to-End)

```
1. Config:      tyro parses TrainConfig from CLI args
2. Data:        LeRobotDataset → repack transforms → robot transforms
                → Normalize(norm_stats) → model transforms (tokenize, resize, pad)
                → TorchDataLoader → JAX sharded batches
3. Model:       config.model.create(rng) → weight_loader.load(params)
4. Train loop:  for each step:
                  loss, grads = value_and_grad(model.compute_loss)(batch)
                  params = optax.apply_updates(params, optimizer.update(grads))
                  ema_params = 0.99 * ema + 0.01 * params
5. Checkpoint:  assets/<asset_id>/norm_stats.json + params/ (EMA) + train_state/
```

**Pi0**: flow-matching MSE loss on predicted velocity field.
**Pi0FAST**: cross-entropy next-token prediction loss on action token postfix.

## Inference Flow (End-to-End)

```
1. Load:        get_config(name) → model.load(checkpoint/params)
2. Policy:      Policy(model, input_transforms, output_transforms)
3. Server:      WebsocketPolicyServer(policy).serve_forever()
4. Per request: recv obs → input transforms → model.sample_actions(obs)
                → output transforms (decode tokens, unnormalize) → send actions
```

---

## Tokenizer Architecture (Pi0FAST)

All Pi0FAST tokenizers share the same interface and output format:

- **Prefix**: `Task: {prompt}, State: {discretized_state};\n` — bidirectional attention (`ar_mask=0`)
- **Postfix**: `Action: {action_tokens}|<eos>` — causal attention (`ar_mask=1`), loss computed here only

| Tokenizer | Action Encoding | Token Count |
|---|---|---|
| `FASTTokenizer` | Learned FAST encoder (HuggingFace) | ~16 tokens |
| `BinningTokenizer` | Uniform 256-bin discretization over [-1,1] | `action_dim * action_horizon` tokens (e.g., 70) |
| `FSQTokenizer` | Learned FSQ encoder (JAX checkpoint) | Variable |

All action tokens are mapped to the upper region of the PaliGemma vocabulary via `vocab_size - 1 - 128 - token_id`.

---

## Checkpoint Structure

```
checkpoints/<config_name>/<exp_name>/<step>/
├── assets/<asset_id>/norm_stats.json   # normalization statistics
├── train_state/                        # full optimizer state (for resuming)
└── params/                             # EMA params (for inference)
```

## Key Dependencies

- **JAX + Flax NNX** — model definition and training
- **Optax** — optimizers and LR schedules
- **Orbax** — checkpoint management
- **LeRobot** — dataset format
- **SentencePiece** — PaliGemma tokenizer
- **Transformers** — FAST tokenizer (HuggingFace)
- **PyTorch** — data loading (and optional PyTorch model port)
- **tyro** — CLI config parsing
- **W&B** — experiment tracking
