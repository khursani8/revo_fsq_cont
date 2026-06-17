# Onboarding: New Machine Setup

Resuming the FSQ Knowledge Memory Bank project on a fresh machine.
Estimated setup time: 10-20 minutes (dominated by `uv sync` and HF model download).

## 1. Prerequisites

- Python 3.11 or 3.12 (`python --version`)
- [uv](https://docs.astral.sh/uv/) (`curl -LsSf https://astral.sh/install.sh | sh` or `pipx install uv`)
- [git](https://git-scm.com/)
- [gh CLI](https://cli.github.com/) authenticated (`gh auth login`)
- [HuggingFace CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) authenticated (`huggingface-cli login`) — Gemma-4 requires license acceptance on HF Hub
- NVIDIA GPU with **at least 16GB free VRAM** (smoke test peak is ~14-16GB; production Phase 2 extraction will need more)
- ~50GB disk for repo + venv + model weights cache

## 2. Clone & install

```bash
git clone https://github.com/khursanirevo/fsq_cont.git
cd fsq_cont
uv sync
```

This pulls `transformers>=5.12,<6.0`, `torch>=2.3`, `faiss-gpu>=1.8`, plus everything else pinned in `pyproject.toml`. First run downloads ~3GB of wheels.

Verify install:
```bash
uv run python -c "import transformers, torch; print('transformers', transformers.__version__); print('torch', torch.__version__); print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"
```

Expected: `transformers 5.12.1`, `torch 2.x`, `cuda True 1+`.

## 3. Verify env sanity

Confirm Gemma-4 architecture is registered:
```bash
uv run python -c "from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES as m; assert 'gemma4_text' in m, 'gemma4_text missing'; print('gemma4_text registered OK')"
```

Confirm the verified HF candidates file is sane:
```bash
cat configs/_hf_candidates.json | head -10
# Should show canonical_candidate: google/gemma-4-E4B
```

## 4. Check GPU state

```bash
nvidia-smi --query-gpu=index,memory.free,utilization.gpu --format=csv
```

You need a GPU with at least 16GB free. If multiple GPUs are available and the busiest one has room, note its index for `CUDA_VISIBLE_DEVICES`.

## 5. Run smoke tests (Phase 0)

### 5a. Pilot regression (gemma-2-2b-it) — fast, ~2GB download

```bash
uv run python scripts/smoke_test_model.py --model pilot
```

Expected: exit 0, `logs/phase0a_smoke_test_pilot.json` written with `passed=true`, `logits_bitwise_equal=true`, `max_abs_diff=0.0`. This re-verifies the env didn't regress across machines.

### 5b. Final smoke test (gemma-4-E4B) — slow first time, ~8GB download

```bash
# Pin to a GPU with >=16GB free (replace 0 with your chosen index)
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_test_model.py --model final
```

Expected: exit 0, `logs/phase0b_smoke_test_final.json` written with:
- `passed=true`
- `determinism.logits_bitwise_equal=true`
- `layer_module_path: "model.model.language_model.layers"` (Gemma-4 triple-nested layout)
- 12 tap points captured at {8,16,24,32} × {residual,mlp_out,attention_out}
- `num_hidden_layers=42`, `hidden_size=2560`, `multimodal=true`

Capture run logs:
```bash
CUDA_VISIBLE_DEVICES=0 uv run python scripts/smoke_test_model.py --model final 2>&1 | tee logs/phase0b_smoke_run.txt
```

## 6. Acceptance gates (Phase 0 → Phase 1)

Phase 0b MUST pass before any Phase 1 work. If it fails:

| Failure | Likely cause | Action |
|---------|-------------|--------|
| `transformers` version mismatch | env not synced | `uv sync` |
| `gemma4_text not in CONFIG_MAPPING` | transformers <5.12 | check `pyproject.toml` pin |
| `tokenizer AttributeError` | extra_special_tokens regression | check `scripts/smoke_test_model.py:177` |
| `resolve_layer_module` crash | code regression | check `resolve_layer_module()` in `smoke_test_model.py:46` |
| CUDA OOM | GPU memory < 16GB free | use different GPU or wait |
| Non-deterministic forward pass | GPU non-determinism | check `torch.use_deterministic_algorithms` in script |

## 7. Once Phase 0b passes

The plan-v3.md gates for Phase 1 work are:
- **OQ-2** (wall-clock budget for Phases 0-4) — user must answer
- **OQ-3** (Magpie license compatibility) — user must confirm

These are tracked in `.omc/plans/open-questions.md`. Surface them to the user before starting Phase 1 dataset generation.

## 8. Re-establishing team orchestration (optional)

If you want to resume with `/oh-my-claudecode:team`:

```bash
# In Claude Code on the new machine:
/oh-my-claudecode:team 2:executor "Phase 0b completion: run smoke test --model final, verify acceptance criteria, report"
```

The team state from the old machine is NOT carried over (gitignored). You start fresh.

## 9. Key file map

| Path | Purpose |
|------|---------|
| `PLAN.md` | Source spec (10 phases) — read first |
| `.omc/plans/plan-v3.md` | RALPLAN-approved canonical plan |
| `.omc/plans/CONSENSUS.md` | Approval record + blocking OQs |
| `.omc/plans/open-questions.md` | 8 open questions (OQ-1/2/7 user-blocking) |
| `pyproject.toml` | Dependencies (transformers 5.12, torch, faiss-gpu, etc.) |
| `configs/model.yaml` | Pilot (gemma-2-2b-it) + Final (gemma-4-E4B) model configs |
| `configs/fsq_sweep.yaml` | Two single-variable FSQ sweeps (levels + latent-dim) |
| `configs/_hf_candidates.json` | Verified canonical HF model record |
| `scripts/verify_hf_model_id.py` | HF Hub candidate prober |
| `scripts/smoke_test_model.py` | Phase 0 determinism + hook-tap smoke test |
| `logs/phase0a_smoke_test_pilot.json` | ✅ Passing pilot result (carried from old machine) |
| `logs/phase0b_smoke_run.txt` | ❌ Failed final run (OOM) — for reference only |

## 10. Quick verification snippet

After setup, before running anything else:

```bash
uv run python - <<'EOF'
import yaml, torch, transformers
cfg = yaml.safe_load(open('configs/model.yaml'))
print('pilot:', cfg['pilot']['model_id'])
print('final:', cfg['final']['model_id'])
print('transformers:', transformers.__version__)
print('cuda available:', torch.cuda.is_available())
print('cuda devices:', torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(f'  GPU {i}: {torch.cuda.get_device_name(i)}, {torch.cuda.mem_get_info(i)[0]/1e9:.1f}GB free')
EOF
```
