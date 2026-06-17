# Plan v1 — FSQ Knowledge Memory Bank for Gemma-4-E4B

Source spec: `/mnt/data/work/fsq_cont/PLAN.md` (10 phases)
Status: `pending approval` — no execution commands embedded
Mode: RALPLAN consensus, DELIBERATE
Drafted: 2026-06-17

---

## 1. RALPLAN-DR Summary

### Principles (govern every downstream decision)

1. **Resumability over throughput** — every long-running stage writes incremental JSONL + checkpoint files and can be killed / restarted without data loss. The 4x H200 GPUs are shared (11–24GB free each); preemption is expected, not exceptional.
2. **Real model activations over synthetic substitutes** — no mock activations, no toy proxy models for the final bank. A small proxy is allowed ONLY as a Phase-0 de-risking smoke test, never as the deliverable.
3. **Discrete codes must carry semantic load before any regularization use** — Phase 4 must demonstrate that FSQ codes align with latent clusters (Phase 3) and pass semantic nearest-neighbor sanity (Phase 5) before any Phase 8 retention loss is trained. Semantic signal gates regularization.
4. **Single-source-of-truth config** — one `configs/paths.yaml` + `configs/fsq_sweep.yaml` drives every stage. No hardcoded paths in scripts.
5. **One variable per stage** — during sweeps change exactly one of {layer, PCA dim, FSQ levels, signal type}. Multi-factor studies come only after single-factor coverage curves are stable (PLAN.md §11 compliance).

### Decision Drivers (ranked, hardest constraints first)

1. **GPU memory ceiling under contention** — only 11–24GB free per H200 (other workloads running). A 4B-param model in bf16 weights ≈ 8GB + KV cache + activation buffers; cannot fit the full extraction pipeline on one GPU. Forces device-map=auto, gradient-checkpointed hooks, streaming activations to disk per-prompt, and rules out training the FSQ encoder in-memory on the same GPUs doing extraction.
2. **Disk ceiling: 5.2TB free vs. 100GB–2TB raw activations projected** — at 4096-dim × float16 × 4 layers × 5M prompts the raw footprint ⚠️ WARNING: THEORETICAL CALCULATION ≈ 1.6TB before PCA; the coverage loop (Phase 3) multiplies this if iterations are not pruned. Forces PCA-before-write and a strict retention policy (raw shards deleted after PCA + checksum verification).
3. **No access to original pretraining data** — PLAN.md is explicit. Knowledge manifold must be approximated by Magpie + FineWeb + synthetic QA. Coverage is only as good as the prompt mixture; this drives the Phase 3 coverage loop as a hard gate, not an optimization.

### Viable Options

**Architectural choice: which layers/signals to capture and how to compress.**

#### Option A — Single-layer residual @ mid-depth, PCA-256, FSQ overcapacity sweep
- Capture: residual stream @ layer 16 only (4B model has ~32–48 layers; layer 16 ≈ mid-network where semantic abstraction is strongest per Gemma-2 literature).
- Compress: 4096 → 256 via incremental PCA (fitted on a 500k-prompt sample, persisted as `pca_layer16.joblib`).
- FSQ: start at 256 dims × 16 levels (PLAN.md "overpowered" config), sweep down to 32 dims × 8 levels.
- **Pros**: (1) Smallest disk footprint (~100–200GB projected ⚠️ THEORETICAL); (2) Fastest sweep turnaround; (3) Cleanest causal story (one layer, one signal).
- **Cons**: (1) Single layer may miss knowledge distributed across depths; (2) Undermines PLAN.md's explicit "compare layers 8/16/24/32" directive; (3) Lower ceiling on breakthrough success.

#### Option B — Multi-layer {8, 16, 24, 32} residual + MLP, no PCA, FSQ per layer *(chosen baseline)*
- Capture: residual stream AND MLP output at layers 8, 16, 24, 32 (matches PLAN.md §Phase 2 verbatim).
- Compress: PCA 4096 → 256 per (layer, signal) pair; 8 PCA models total.
- FSQ: independent FSQ bank per (layer, signal); capacity sweep per bank.
- **Pros**: (1) Faithful to PLAN.md spec (no scope downgrade); (2) Enables cross-layer drift fingerprinting (Phase 6); (3) Future-proofs against the "knowledge is distributed" hypothesis.
- **Cons**: (1) ~8× disk and compute of Option A (~800GB–1.6TB projected ⚠️ THEORETICAL CALCULATION); (2) Sweep matrix is 8× larger — risk of blowing wall-clock budget; (3) Requires stricter checkpoint discipline.
- **Mitigations**: Phase 2 streams directly to PCA'd float16 shards (never writes raw 4096-dim to disk in bulk); Phase 4 sweep is ordered by PLAN.md's capacity ladder and can early-stop when reconstruction MSE plateaus.

#### Option C — All layers, residual + attn + MLP
- **Invalidated**: disk footprint ⚠️ THEORETICAL ≈ 4–6TB at 5M prompts exceeds the 5.2TB free ceiling with no headroom for checkpoints, FAISS index, or FSQ artifacts. Rejected for this hardware budget. Revisit if/when GPU contention clears and additional storage is provisioned.

**Chosen**: **Option B** — matches user intent, fits within disk budget under the streaming+PCA-on-write discipline, and preserves the multi-layer drift fingerprint that gates Phase 6/7 research value.

#### Sub-decision: FSQ implementation
- **Chosen**: `torchfree` / hand-rolled FSQ op (the operation is ~20 lines: `z_q = tanh(z) * levels_inv; round; dequant`). No external FSQ package is canonical as of 2026-06-17; depends on `mcp__plugin_oh-my-claude:context7` confirmation during Phase 0. Alternatives (`vector-quantize-pytorch`'s FSQ) are tracked but not committed.

#### Sub-decision: Training framework
- **Chosen**: plain `torch` + `accelerate` for extraction; plain `torch` for the FSQ autoencoder (small MLP, no need for full Lightning). `peft` reserved for Phase 9 LoRA track.

---

## 2. Pre-Mortem — DELIBERATE MODE

### Scenario 1 — FSQ codebook collapse
- **Trigger**: Encoder lands in a degenerate solution where >80% of prompts map to <5% of codes. Reconstruction MSE looks fine (mean tracks the dominant cluster) but codes carry no semantic signal.
- **Early warning signal**: per-phase metric `codebook_utilization` (active codes / total codes) drops below 20% on the held-out validation shard; `code_entropy` < 1.0 nat when uniform baseline is ln(num_codes).
- **Mitigation**: (a) L2-normalize activations before FSQ; (b) entropy regularizer on code distribution weighted 0.01–0.1; (c) EMA code-usage tracker that upweights rare codes during training; (d) if still collapsed, fall back to VQ-VIBE / affinity-normalized FSQ variant; (e) the semantic-NN check (Phase 5) is a HARD GATE before any Phase 8 work — collapse is caught here, not at Phase 10.

### Scenario 2 — Activation distribution drift between collection sessions
- **Trigger**: Extraction runs on different days with different GPU contention. Non-determinism in cuBLAS / flash-attn kernels + different batch shapes shift the activation distribution enough that PCA fitted on Day-1 is misaligned with Day-2 shards.
- **Early warning signal**: per-shard activation mean/std logged to `activations/_drift_log.jsonl`; alarm if shard-mean L2 from running global mean > 2σ. Reconstruction MSE on a fixed probe set degrades > 20% across sessions.
- **Mitigation**: (a) fixed `torch.manual_seed` + `use_deterministic_algorithms=True` where supported; (b) batch size pinned to 1 for extraction (latency cost accepted; throughput is not the driver per Principle 1); (c) refit PCA incrementally with `IncrementalPCA` as new shards arrive — never freeze PCA on Day-1 only; (d) maintain a 10k-prompt probe set replayed every session to measure drift; (e) if drift is unavoidable, document it as a finding (it itself becomes a knowledge-fingerprint signal in Phase 6).

### Scenario 3 — 5M Magpie generation intractable on shared GPUs
- **Trigger**: Only 11–24GB free per GPU; the model used for Magpie generation needs weights + KV cache. At ~8GB weights for a 4B model, only ~3–13GB remains for KV cache → small batch sizes → projected throughput ⚠️ WARNING: EXPECTED/PROJECTED ~50–200 prompts/sec/GPU → 5M prompts needs 7–28 hours of dedicated GPU time, which is unavailable while other workloads run.
- **Early warning signal**: Phase-1 throughput probe (10k prompts) reports < 30 prompts/sec aggregate across all 4 GPUs; projected wall-clock for 5M exceeds the user's stated budget (Open Question b).
- **Mitigation**: (a) start with a 500k-prompt pilot for Phases 2–5 to prove the pipeline end-to-end, then scale to 5M once the pipeline is validated — this is the de-risking path in Open Question (e); (b) Magpie generation can run at lower priority / nice level to coexist with other workloads; (c) if even 500k is intractable, use FineWeb-edu as the dominant source and Magpie as augmentation only (PLAN.md Source B is already a listed primary); (d) stream prompts to disk as they're generated so partial progress survives preemption.

---

## 3. Phased Execution Plan

All paths under `/mnt/data/work/fsq_cont/`. All long-running stages checkpoint to JSONL + JSON per PLAN.md §6.6.

### Top 3 gating deliverables (in priority order)

1. **Activation dataset (Phase 2 output)** — `activations/pca_shards/*.npz` (PCA'd, float16, per-prompt rows keyed by prompt_id). **Why it gates everything**: Phase 3 (coverage), Phase 4 (FSQ), Phase 6 (fingerprint), Phase 7 (drift measurement) all consume this artifact. Without it, no downstream phase can start.
2. **FSQ memory bank (Phase 4 output)** — `fsq/bank_layer16_residual.pt` + `fsq/encoder.pt` + `fsq/decoder.pt` + `fsq/sweep_results.jsonl`. **Why it gates everything**: Phase 5 (semantic validation), Phase 6 (fingerprint stats over codes), Phase 8 (retention loss), Phase 10 (drift correlation) all depend on a trained, stable FSQ bank. A collapsed bank invalidates all four.
3. **Knowledge fingerprint (Phase 6 output)** — `memory_snapshots/baseline_fingerprint.json` (P(code), P(code_i, code_j), entropy, rare-code freq). **Why it gates everything**: Phase 7 (baseline forgetting), Phase 8 (retention loss target), Phase 10 (drift as forgetting predictor) all require a frozen baseline fingerprint to diff against. This is the actual research artifact — without it, "drift" is undefined.

### Phase 0 — Environment & Model Verification
- **Deliverable**: `pyproject.toml`, `configs/paths.yaml`, `configs/model.yaml`, `scripts/smoke_test_model.py` (loads model, runs one forward pass, prints hidden-state shapes per layer), `scripts/verify_hf_model_id.py` (probes HuggingFace Hub for `google/gemma-4-E4B` and lists candidate alternatives), `.gitignore`.
- **Acceptance criteria**:
  - `uv run python scripts/smoke_test_model.py` prints the actual layer count and hidden size (e.g., `num_hidden_layers=48, hidden_size=4096` — currently speculative until model is loaded).
  - `scripts/verify_hf_model_id.py` confirms the canonical HF model ID or surfaces a ranked list of alternatives (gemma-3-4B, gememma-2-9B-it, etc.) with parameter counts.
  - `activations/ datasets/ clusters/ fsq/ checkpoints/ memory_snapshots/ evaluation/` directories created and listed in `configs/paths.yaml`.
  - `uv pip list` shows: `transformers>=4.45`, `accelerate`, `torch`, `faiss-gpu`, `scikit-learn`, `joblib`, `numpy`, `pyarrow`, `tqdm`, `peft` (pinned versions recorded).
- **Owner**: executor (sonnet)
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~1–3 GPU-hours (single forward pass + package install; model download is the dominant cost at ~8–16GB weights).
- **Disk footprint**: ~16GB (model cache) + ~0 (code only).
- **Hard gate to Phase 1**: smoke test prints deterministic hidden-state shapes on two consecutive runs; HF model ID is confirmed and recorded in `configs/model.yaml:model_id`.

### Phase 1 — Knowledge Coverage Dataset
- **Deliverable**:
  - `scripts/gen_magpie.py` (resumable, JSONL streaming output to `datasets/magpie/raw/*.jsonl` with per-shard checkpoint).
  - `scripts/sample_fineweb.py` (streams FineWeb / C4 / Dolma samples to `datasets/web/*.parquet`).
  - `scripts/gen_synthetic_qa.py` (per-domain generators: coding, math, science, medicine, law, philosophy, history, languages, reasoning, writing — PLAN.md Source C).
  - `scripts/gen_self_explore.py` (self-prompted task creation / critique loop — PLAN.md Source D).
  - `scripts/build_unified_corpus.py` (merges all sources, dedupes by exact MinHash, emits `datasets/unified/prompts.parquet` with columns `prompt_id, source, domain, prompt_text`).
- **Acceptance criteria**:
  - At least 500k prompts (pilot scale) present in `datasets/unified/prompts.parquet` with non-null `prompt_text` and no duplicate `prompt_id`.
  - Each generator script is resumable: re-running after kill skips already-generated `prompt_id`s (loaded from existing shards).
  - Sampling mixture logged: 20% greedy / 30% T=0.8 / 30% T=1.2 / 20% T=1.5 for Magpie (PLAN.md §Phase 1 Source A).
  - Source mixture recorded: per-source prompt counts in `datasets/unified/_manifest.json`.
- **Owner**: data-engineer (sonnet)
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~50–200 GPU-hours for 5M Magpie prompts; ~5–10 GPU-hours for the 500k pilot (see Scenario 3 pre-mortem).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~20–80GB text (pilot) / ~200–800GB text (full 5M).
- **Hard gate to Phase 2**: `datasets/unified/prompts.parquet` exists with ≥500k rows; per-source manifest shows all 4 sources (A, B, C, D) represented.

### Phase 2 — Activation Extraction
- **Deliverable**:
  - `scripts/extract_activations.py` — for each prompt: forward pass with hooks on layers {8, 16, 24, 32}, signals {residual, mlp_out}; write PCA'd float16 vectors to `activations/pca_shards/shard_{NNNN}.npz` with `prompt_id` array; delete raw 4096-dim in-memory immediately.
  - `scripts/fit_pca_incremental.py` — pre-pass over first 500k prompts to fit 8 `IncrementalPCA(n_components=256)` models (one per (layer, signal)); persisted as `activations/pca_models/(layer|signal)_pca.joblib`.
  - `scripts/build_drift_log.py` — per-shard activation mean/std appended to `activations/_drift_log.jsonl` (feeds Scenario 2 mitigation).
- **Acceptance criteria**:
  - For each (layer, signal) pair, `activations/pca_shards/` contains float16 arrays of shape `(N_prompt_shard, 256)` with paired `prompt_id` arrays.
  - Reconstruction sanity: `InversePCA(PCA(x)) ≈ x` within relative error < 0.05 on a 1k-prompt probe (logged to `activations/_pca_recon_sanity.json`).
  - Resumability: script checks existing shards' `prompt_id`s and skips them.
  - Per-shard drift log shows no shard-mean > 2σ from running global mean (Scenario 2 alarm).
  - Hidden-size assumption (4096) verified against actual `model.config.hidden_size` — abort if mismatch.
- **Owner**: data-engineer (sonnet) for extraction harness; scientist (opus) for hook placement validation.
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~100–400 GPU-hours (500k prompts, 8 hooks each, batch=1 to ensure determinism).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~150–250GB (500k prompts × 8 signals × 256 dims × float16 ≈ 200GB).
- **Hard gate to Phase 3**: total prompt count in `activations/pca_shards/` matches `datasets/unified/prompts.parquet` within 0.1%; drift log shows no Scenario-2 alarm.

### Phase 3 — Latent Coverage Analysis
- **Deliverable**:
  - `scripts/build_faiss_index.py` — FAISS-GPU index per (layer, signal) over the 256-dim PCA space; persisted to `clusters/faiss_(layer)_(signal).index`.
  - `scripts/fit_clusters.py` — fits MiniBatchKMeans (K=10000) per (layer, signal); HDBSCAN on a 100k subsample for cross-check; outputs `clusters/labels_(layer)_(signal).parquet`.
  - `scripts/cluster_metrics.py` — per-cluster: count, entropy, mean intra-cluster distance, prompt-topic diversity (via simple lexical overlap), activation diversity. Writes `clusters/coverage_report.json`.
  - `scripts/coverage_loop.py` — orchestrates the PLAN.md coverage loop: identify sparse clusters, request more prompts from the Phase-1 generators targeted at sparse-cluster regions (use FAISS nearest-centroid to filter candidate prompts), repeat. Stops when KL(cluster_dist_t || cluster_dist_{t-1}) < 0.01 over 3 iterations.
- **Acceptance criteria**:
  - `clusters/coverage_report.json` lists K=10000 clusters per (layer, signal) with all five PLAN.md metrics populated.
  - Coverage loop convergence criterion met: cluster-distribution KL < 0.01 across 3 consecutive iterations.
  - No cluster has < 10 prompts (sparse-cluster alarm) at convergence; if some do, they are flagged with a "long-tail" annotation, not silently dropped.
- **Owner**: scientist (sonnet/opus for coverage-loop design).
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~20–60 GPU-hours (FAISS-GPU build + MiniBatchKMeans; coverage loop iterations compound this).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~50–100GB (FAISS indexes + label parquets).
- **Hard gate to Phase 4**: coverage loop converged OR user override documenting why convergence is infeasible; sparse clusters enumerated.

### Phase 4 — FSQ Memory Bank
- **Deliverable**:
  - `scripts/train_fsq_autoencoder.py` — encoder `256→128→64`, FSQ op, decoder `64→128→256`; loss = MSE + cosine-similarity; trains one model per (layer, signal). Saves `fsq/encoder_(layer)_(signal).pt`, `fsq/decoder_(layer)_(signal).pt`, `fsq/bank_(layer)_(signal).pt` (code lookup table).
  - `scripts/fsq_capacity_sweep.py` — runs PLAN.md's ladder: 256×16, 128×16, 64×16, 64×8, 32×8. Outputs `fsq/sweep_results.jsonl` (one row per config with reconstruction MSE, cosine, codebook utilization, code entropy).
  - `scripts/render_sweep_curves.py` — produces `evaluation/fsq_capacity_curve.png` (compression-ratio vs MSE).
- **Acceptance criteria**:
  - Sweep complete for layer 16 residual at all 5 capacity points; `fsq/sweep_results.jsonl` has 5 rows for that (layer, signal).
  - **Collapse check**: codebook utilization > 20% at every capacity point (Scenario 1 alarm); if below, training halts and triggers Scenario-1 mitigations.
  - **Reconstruction gate**: best capacity point achieves MSE < 0.1 (normalized) AND cosine > 0.7 on the held-out probe (numbers pending Phase 2 actuals; will be re-calibrated once pilot PCA is fit).
  - `evaluation/fsq_capacity_curve.png` shows a monotonic quality-vs-compression tradeoff (no anomalies that suggest a training bug).
- **Owner**: scientist (opus — FSQ encoder design is research-grade).
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~30–100 GPU-hours (small MLP, but 8 banks × 5 configs × multi-seed for the headline results).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~5–20GB (model checkpoints are small; sweep logs dominate).
- **Hard gate to Phase 5**: a single (layer, signal, capacity) configuration is nominated as the "canonical" FSQ bank and recorded in `configs/fsq_canonical.yaml`. Semantic validation (Phase 5) runs against this canonical bank only.

### Phases 5–10 — one-liners with dependencies

- **Phase 5 — Semantic Validation** (depends: Phase 4 canonical bank). `scripts/semantic_nn_eval.py` + `scripts/code_intervention.py`. Acceptance: prompts sharing codes have topic-consistency > chance (p < 0.01 via permutation test); code edits produce measurable activation drift.
- **Phase 6 — Knowledge Fingerprint** (depends: Phase 5 passes). `scripts/build_fingerprint.py` → `memory_snapshots/baseline_fingerprint.json`. Acceptance: P(code), P(code_i, code_j), entropy, rare-code frequency populated for the canonical bank; fingerprint is deterministic across two recomputation runs.
- **Phase 7 — Fine-Tuning Baseline** (depends: Phase 6; Open Question d for benchmark scope). `scripts/finetune_baseline.py` (plain SFT on coding/instruction/domain tasks using `peft` or full FT) + `scripts/run_benchmarks.py` (MMLU, GSM8K, HumanEval, HellaSwag, TruthfulQA via `lm-eval-harness`). Acceptance: post-FT benchmark delta and post-FT fingerprint delta both recorded in `evaluation/baseline_forgetting.json`.
- **Phase 8 — FSQ-Guided Continual Learning** (depends: Phase 7 baseline). `scripts/finetune_with_memory.py` implementing `L = L_task + λ · L_memory` with L_memory candidates A/B/C/D from PLAN.md. Acceptance: sweep over λ and retention levels {1.0, 0.75, 0.50, 0.25} produces `evaluation/retention_sweep.jsonl`; memory_rewrite_rate tracked at {1%, 5%, 10%, 25%, 50%} targets.
- **Phase 9 — LoRA Validation Track** (depends: Phase 8). Same scripts as Phase 8 but with LoRA adapter; produces `evaluation/lora_comparison.json`. Acceptance: FSQ drift under LoRA vs full FT compared head-to-head; PLAN.md hypothesis ("easier to interpret under LoRA") answered with explicit metric.
- **Phase 10 — Memory Preservation Benchmark** (depends: Phase 9). `scripts/preservation_benchmark.py` correlates parameter-distance vs FSQ-memory-distance against benchmark retention. Acceptance: Spearman ρ reported for both predictors; success criterion (FSQ ρ > parameter ρ) answered yes/no with CI.

---

## 4. Expanded Test Plan — DELIBERATE MODE

### Unit
- **FSQ round-trip**: `tests/test_fsq_roundtrip.py` — quantize → dequantize a known vector; assert L2 error ≤ sqrt(d)/levels (analytic bound).
- **FSQ gradient**: `tests/test_fsq_grad.py` — straight-through gradient passes through; grad shape matches input shape.
- **PCA fit/save/load**: `tests/test_pca_io.py` — fit `IncrementalPCA` on 10k random vectors; save to `.joblib`; reload; assert `transform()` output bitwise-equal.
- **Activation hook determinism**: `tests/test_hook_determinism.py` — same prompt, same seed, two forward passes; assert hidden states bitwise-equal (catches nondeterministic cuBLAS paths).
- **Checkpoint resumability**: `tests/test_resume.py` — start extraction on 100 prompts, kill after 50, restart; assert all 100 eventually processed exactly once.

### Integration
- **End-to-end pipeline probe**: `tests/test_integration_pipeline.py` — 1k-prompt mini-corpus → Phase 2 extraction → Phase 3 cluster → Phase 4 FSQ train (1 epoch) → reconstruction MSE computed. Assert MSE is finite, non-NaN, < 1.0.
- **PCA inverse fidelity**: `tests/test_pca_inverse.py` — on real activations (100-prompt probe), `inverse(transform(x))` within rel-error 0.05 of `x`.

### E2E
- **Knowledge fingerprint reproducibility**: `tests/test_fingerprint_repro.py` — run Phase 6 twice from the same activation shards; assert fingerprint JSON bitwise-equal (or within float32 tolerance).
- **Semantic neighbor sharing**: `tests/test_semantic_neighbors.py` — for 100 probe prompts, compute nearest neighbors in (a) prompt-embedding space and (b) FSQ-code space; assert Jaccard overlap > random baseline at p < 0.01.

### Observability
Every long-running script emits structured JSONL to `logs/{phase}_{script}.jsonl` with:
- per-shard: `prompt_ids`, `mean_norm`, `std_norm`, `duration_sec`, `gpu_mem_peak_gb`.
- per-FSQ-training-epoch: `epoch`, `train_mse`, `val_mse`, `codebook_utilization`, `code_entropy_nat`, `cluster_silhouette`.
- per-coverage-iteration: `kl_to_prev`, `num_sparse_clusters`, `total_clusters`.

Checkpoint-resumability tests run in CI on a 100-prompt toy dataset every commit. Determinism tests run once per model-load (Phase 0 acceptance).

---

## 5. Open Questions for the User (≤5, all blocking)

1. **Exact HuggingFace model ID for `google/gemma-4-E4B`.** As of 2026-06-17 the existence of this model on HF Hub is unverified. **Why it blocks Phase 0**: every downstream path (layer count, hidden size, tokenizer, attention impl) depends on the actual model. `scripts/verify_hf_model_id.py` will surface candidates (gemma-3-4B-it, gemma-2-9B-it, etc.) but the user must confirm which to canonize. Without this, hidden-size assumptions (4096) and layer indices {8, 16, 24, 32} are speculative.

2. **Acceptable wall-clock budget for Phases 0–4.** At 11–24GB free per GPU under contention, Phase 2 alone is projected ⚠️ WARNING: EXPECTED/PROJECTED at 100–400 GPU-hours. **Why it blocks Phase 1 scaling**: determines whether the pilot is 500k prompts (de-risk first) or jump straight to 5M. Directly trades against the user's other running workloads.

3. **Is Magpie available and licensed for this use?** Magpie-UA / Magpie-Align terms vary by release; commercial-research splits matter. **Why it blocks Phase 1 Source A**: if Magpie is unavailable or license-incompatible, Phase 1 must rely on FineWeb + synthetic QA only, which weakens latent coverage and may force a larger FineWeb sample to compensate.

4. **Acceptable benchmark suites for Phase 7 baseline.** PLAN.md lists MMLU, GSM8K, HumanEval, HellaSwag, TruthfulQA. Are any deprecated/replaced as of 2026-06-17? Is `lm-eval-harness` the runner, or an internal harness? **Why it blocks Phase 7**: defines what "forgetting" means operationally and gates the Phase 10 drift-correlation claim.

5. **Should we start on a smaller proxy model first (e.g., gemma-2-2B) to de-risk the pipeline?** A 2B model fits comfortably in 11GB free and runs the entire Phase 0–5 pipeline in hours, not days. **Why it blocks Phase 0**: if "yes", Phase 0 targets the proxy and the gemma-4-E4B run becomes Phase 0b. If "no", we accept the contention risk on the 4B model from day one (Principle 2 still holds — proxy is never the final deliverable, only a pipeline validator).

---

## 6. ADR — Architectural Decision Record

### Decision
Adopt **Option B** (multi-layer {8,16,24,32} residual + MLP, PCA-256 per (layer, signal), independent FSQ banks with capacity sweep) as the canonical architecture for the FSQ Knowledge Memory Bank.

### Drivers
1. GPU memory contention (11–24GB free per H200) forces streaming + small in-memory footprint.
2. 5.2TB disk free caps raw activation storage; PCA-on-write is mandatory, not optional.
3. PLAN.md explicitly lists layers {8,16,24,32} and signals {residual, attention, MLP} — scope downgrade is disallowed by user intent ("Do NOT downgrade scope").

### Alternatives Considered
- **Option A** (single layer 16, residual only, PCA-256, FSQ sweep): rejected because it drops the multi-layer drift fingerprint that makes Phase 6/10 research-valuable, and downgrades PLAN.md scope.
- **Option C** (all layers, all signals): invalidated by disk ceiling (4–6TB > 5.2TB free at 5M prompts).
- **External FSQ package** vs hand-rolled: deferred to Phase 0 verification via `context7`; hand-rolled chosen as fallback because the op is ~20 LOC and avoids an unverified dependency.
- **Lightning** vs plain `torch`: plain torch chosen; the FSQ autoencoder is a small MLP and Lightning's overhead is unjustified. `peft` reserved for Phase 9.

### Why Chosen
Option B is the only option that simultaneously (a) respects PLAN.md's explicit layer/signal list, (b) fits within the 5.2TB disk budget under the streaming+PCA discipline, (c) preserves the cross-layer drift signal needed for Phase 6/10, and (d) has a tractable wall-clock under GPU contention when combined with the 500k-prompt pilot de-risking path.

### Consequences
- **Positive**: faithful to user intent; multi-layer drift fingerprint is the highest-research-value artifact; PCA-on-write keeps disk tractable.
- **Negative**: 8× the sweep matrix of Option A — Phase 4 wall-clock is the longest single stage; requires strict checkpoint discipline (mitigated by PLAN.md §6.6 compliance built into every script).
- **Risk**: if the chosen model's actual `num_hidden_layers < 32`, the layer indices {8, 16, 24, 32} must be re-scaled (Phase 0 smoke test catches this).

### Follow-ups
- Re-evaluate Option C if GPU contention clears and additional storage is provisioned.
- If the FSQ sweep (Phase 4) shows that one layer dominates reconstruction quality, propose a Phase-4b ablation reducing to that layer only — but only after the multi-layer result is on record.
- Proxy-model pilot (Open Question 5): if approved, becomes Phase 0a; ADR is re-opened if pilot surfaces a fundamental pipeline issue.
