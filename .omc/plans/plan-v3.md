# Plan v3 — FSQ Knowledge Memory Bank for Gemma-4-E4B

Source spec: `/mnt/data/work/fsq_cont/PLAN.md` (10 phases)
Previous draft: `/mnt/data/work/fsq_cont/.omc/plans/plan-v2.md`
Status: `USER-APPROVED via /ralplan --interactive (2026-06-17)`. Execution pathway: `/oh-my-claudecode:team` for Phases 0-4. OQ-1/2/7 to be resolved inline as each phase gates.
Mode: RALPLAN consensus, DELIBERATE
Drafted: 2026-06-17
Iteration: v3 — applies Critic RC-v3-1..RC-v3-4 + OQ-8 + OQ-6/7 consolidation. All other sections preserved verbatim from v2.

---

## 1. RALPLAN-DR Summary

### Principles (govern every downstream decision)

1. **Resumability over throughput** — every long-running stage writes incremental JSONL + checkpoint files and can be killed / restarted without data loss. The 4x H200 GPUs are shared (11–24GB free each); preemption is expected, not exceptional.
2. **Real model activations over synthetic substitutes** — no mock activations, no toy proxy models for the final bank. A small proxy is allowed ONLY as a Phase-0 de-risking smoke test, never as the deliverable.
3. **Discrete codes must carry semantic load before any regularization use** — Phase 4 must demonstrate that FSQ codes align with latent clusters (Phase 3) and pass semantic nearest-neighbor sanity (Phase 5) before any Phase 8 retention loss is trained. Semantic signal gates regularization.
4. **Single-source-of-truth config** — one `configs/paths.yaml` + `configs/fsq_sweep.yaml` drives every stage. No hardcoded paths in scripts.
5. **One variable per stage** — during sweeps change exactly one of {layer, PCA dim, FSQ levels, FSQ latent dim, signal type}. Multi-factor studies come only after single-factor coverage curves are stable (PLAN.md §11 compliance).

### Decision Drivers (ranked, hardest constraints first)

1. **GPU memory ceiling under contention** — only 11–24GB free per H200 (other workloads running). A 4B-param model in bf16 weights ≈ 8GB + KV cache + activation buffers; cannot fit the full extraction pipeline on one GPU. Forces device-map=auto, gradient-checkpointed hooks, streaming activations to disk per-prompt, and rules out training the FSQ encoder in-memory on the same GPUs doing extraction.
2. **Disk ceiling: 5.2TB free vs. 100GB–2TB raw activations projected** — at 4096-dim × float16 × 4 layers × 3 signals (residual + MLP + attention) × 5M prompts the raw footprint ⚠️ WARNING: THEORETICAL CALCULATION ≈ 1.1–2.1TB before PCA (revised per RC4; includes attention signal, still within 5.2TB budget); the coverage loop (Phase 3) multiplies this if iterations are not pruned. Forces PCA-before-write, a strict retention policy (raw shards deleted after PCA + checksum verification), and a bounded raw-activation ring buffer (see RC1) to enable PCA refits without re-extraction.
3. **No access to original pretraining data** — PLAN.md is explicit. Knowledge manifold must be approximated by Magpie + FineWeb + synthetic QA. Coverage is only as good as the prompt mixture; this drives the Phase 3 coverage loop as a hard gate, not an optimization.

### Viable Options

**Architectural choice: which layers/signals to capture and how to compress.**

#### Option A (single-shot) — Single-layer residual @ mid-depth, PCA-256, FSQ overcapacity sweep
- Capture: residual stream @ layer 16 only (4B model has ~32–48 layers; layer 16 ≈ mid-network where semantic abstraction is strongest per Gemma-2 literature).
- Compress: 4096 → 256 via one-shot PCA (fitted on a 500k-prompt pre-pass over a bounded raw ring buffer; persisted as `pca_layer16.joblib`).
- FSQ: start at 256 dims × 16 levels (PLAN.md "overpowered" config), sweep down to 32 dims × 8 levels.
- **Pros**: (1) Smallest disk footprint (~100–200GB projected ⚠️ THEORETICAL); (2) Fastest sweep turnaround; (3) Cleanest causal story (one layer, one signal).
- **Cons**: (1) Single layer may miss knowledge distributed across depths; (2) Undermines PLAN.md's explicit "compare layers 8/16/24/32" directive; (3) Lower ceiling on breakthrough success.

#### Option A (staged) — Layer-16 residual first, expand to Option B matrix after Phase 5 (steelman, per RC3)
- Begin with Option A (single-shot) — layer 16 residual only — and complete Phases 2–5 at ~1/8 the compute and disk cost of Option B. Expand to the full Option B layer/signal matrix ONLY after Phase 5 semantic validation passes.
- **Pros**: (1) ~8× faster de-risk; first research result in days, not weeks; (2) If layer-16 alone produces semantic codes, the multi-layer sweep becomes confirmatory rather than exploratory, so Option B is then run with a known-good baseline; (3) Lower day-one disk commitment; (4) Natural fallback if Magpie generation slips (Scenario 3) — see ADR.
- **Cons**: (1) Anchoring risk — early layer-16 findings may bias the multi-layer interpretation; (2) PLAN.md's cross-layer drift fingerprint (Phase 6) requires multiple layers by construction, so the staged path defers the highest-research-value artifact; (3) Two extraction passes (layer 16 first, then {8, 24, 32} later) means re-running hooks on prompts already processed for layer 16 — mitigated by prompt_id-keyed resumability.
- **Rejected because**: PLAN.md's cross-layer drift fingerprint requires multiple layers by construction, and a staged approach risks anchoring on layer-16 findings before the multi-layer picture is available.
- **However**: if Phase 1 Magpie generation slips (Scenario 3 trigger fires), the staged Option A becomes the fallback and the ADR is re-opened (see §6 ADR Follow-ups).

#### Option B — Multi-layer {8, 16, 24, 32} residual + MLP + attention, PCA-256, FSQ per (layer, signal) *(chosen baseline — revised per RC4)*
- Capture: residual stream, MLP output, AND attention output at layers 8, 16, 24, 32 (matches PLAN.md §Phase 2 verbatim, which lists all three signals).
- Compress: PCA 4096 → 256 per (layer, signal) pair; 12 PCA models total (4 layers × 3 signals).
- FSQ: independent FSQ bank per (layer, signal); capacity sweep per bank.
- Sweep matrix: 12 (layer, signal) pairs × 5 capacity points × multi-seed = the headline Phase-4 grid.
- **Pros**: (1) Faithful to PLAN.md spec (no scope downgrade; attention signal is explicitly listed in PLAN.md §Phase 2); (2) Enables cross-layer AND cross-signal drift fingerprinting (Phase 6); (3) Future-proofs against the "knowledge is distributed" hypothesis.
- **Cons**: (1) ~12× disk and compute of Option A (~1.1–2.1TB projected ⚠️ THEORETICAL CALCULATION — still within the 5.2TB budget); (2) Sweep matrix is 12× larger than the layer-16-only baseline — wall-clock risk; (3) Requires stricter checkpoint discipline.
- **Mitigations**: Phase 2 streams directly to PCA'd float16 shards (never writes raw 4096-dim to disk in bulk); Phase 4 sweep is ordered by PLAN.md's capacity ladder and can early-stop when reconstruction MSE plateaus; bounded raw ring buffer (RC1) supports PCA refit without full re-extraction.

#### Option C — All layers × {residual, MLP, attention} at full scale *(reclassified per RC4 — viable but deferred)*
- Earlier draft rejected Option C on disk grounds citing ~4–6TB. That math conflated two scope axes: "all layers" and "all signals". Recomputed at the actual sweep envelope — 4 layers × 3 signals = 12 (layer, signal) pairs, identical to Option B — the footprint ⚠️ WARNING: THEORETICAL CALCULATION ≈ 1.1–2.1TB at 5M prompts, which is within the 5.2TB budget.
- The real rejection grounds are wall-clock, not disk: sweeping all 48 layers (assuming a 48-layer model) × 3 signals = 144 (layer, signal) pairs produces a sweep matrix 12× larger than the Option B baseline and is projected to exceed the Phase 4 budget even at 500k-prompt pilot scale.
- **Classification**: viable but deferred to Phase 4b ablation. Option B's 4-layer × 3-signal subset is the committed first pass; if Phase 4 reveals that coverage of layers {8, 16, 24, 32} misses substantial drift signal, Option C becomes the expansion target (see ADR Follow-ups).

**Chosen**: **Option B** — matches PLAN.md's explicit layer and signal list (including attention, which was previously omitted), fits within the revised 1.1–2.1TB disk projection under the streaming+PCA-on-write discipline, and preserves the cross-layer AND cross-signal drift fingerprint that gates Phase 6/7 research value.

#### Sub-decision: FSQ implementation
- **Chosen**: `torchfree` / hand-rolled FSQ op (the operation is ~20 lines: `z_q = tanh(z) * levels_inv; round; dequant`). No external FSQ package is canonical as of 2026-06-17; depends on `mcp__plugin_oh-my-claude:context7` confirmation during Phase 0. Alternatives (`vector-quantize-pytorch`'s FSQ) are tracked but not committed.

#### Sub-decision: Training framework
- **Chosen**: plain `torch` + `accelerate` for extraction; plain `torch` for the FSQ autoencoder (small MLP, no need for full Lightning). `peft` reserved for Phase 9 LoRA track.

---

## 2. Pre-Mortem — DELIBERATE MODE

### Scenario 1 — FSQ codebook collapse
- **Trigger**: Encoder lands in a degenerate solution where most prompts map to a small subset of codes. Reconstruction MSE looks fine (mean tracks the dominant cluster) but codes carry no semantic signal.
- **Early warning signal** (revised per RC2 — capacity-relative utilization, not a flat codebook-utilization percentage):
  ```
  code_entropy_nat = H(code_frequencies)           # empirical code distribution entropy
  perplexity       = exp(code_entropy_nat)         # effective number of codes in use
  utilization      = perplexity / theoretical_codebook_size
  alarm when utilization < 0.20                    # comparable across capacity points
  ```
  The alarm fires on a held-out validation shard per (layer, signal, capacity) cell.
- **Mitigation**: (a) L2-normalize activations before FSQ; (b) entropy regularizer on code distribution weighted 0.01–0.1; (c) EMA code-usage tracker that upweights rare codes during training; (d) if still collapsed, fall back to VQ-VIBE / affinity-normalized FSQ variant (see Open Question 7); (e) the semantic-NN check (Phase 5) is a HARD GATE before any Phase 8 work — collapse is caught here, not at Phase 10.

  **FSQ collapse differs from VQ-VAE collapse (note per RC2).** FSQ has no learned codebook embeddings: the codebook is the Cartesian product of per-dimension quantization levels, so most codes are empty even in a healthy bank (e.g., a 64-dim × 16-level FSQ has 16^64 theoretical codes, of which only a vanishing fraction can ever be visited by a finite prompt set). A flat "active codes / total codes" ratio is therefore not comparable across capacity points. Perplexity-based utilization — effective-active-codes normalized by theoretical codebook size — is the standard cross-capacity diagnostic and is used everywhere in this plan (Scenario 1 alarm, Phase 4 acceptance, §4 Observability).

### Scenario 2 — Activation distribution drift between collection sessions *(revised per RC1)*
- **Trigger**: Extraction runs on different days with different GPU contention. Non-determinism in cuBLAS / flash-attn kernels + different batch shapes shift the activation distribution enough that PCA fitted on Day-1 is misaligned with Day-2 shards.
- **Early warning signal**: per-shard activation mean/std logged to `activations/_drift_log.jsonl`; alarm if shard-mean L2 from running global mean > 2σ. Reconstruction MSE on a fixed probe set degrades > 20% across sessions.
- **Mitigation** (revised — one-shot PCA with full-refit fallback, no incremental PCA drift accommodation):
  - (a) fixed `torch.manual_seed` + `use_deterministic_algorithms=True` where supported.
  - (b) batch size pinned to 1 for extraction (latency cost accepted; throughput is not the driver per Principle 1).
  - (c) **One-shot PCA, frozen after the Phase-2 500k-prompt pre-pass.** The PCA fit is NOT incrementally updated as new shards arrive — this avoids a moving quantization basis that would silently invalidate cross-session code comparisons.
  - **If the drift alarm fires** (shard-mean > 2σ from running global mean): run `scripts/refit_pca_full.py`, which reads the bounded raw-activation ring buffer at `/mnt/data/work/fsq_cont/activations/_raw_ring/` (last N=200k prompts, FIFO), refits PCA from scratch on that buffer, rewrites every previously written PCA'd shard in `activations/pca_shards/` using the new basis, and emits a new `activations/_pca_refit_log.jsonl` entry. Downstream FAISS indexes (Phase 3) are then rebuilt from the re-transformed shards (see Phase 3 "FAISS-index rebuild on PCA refit").
  - (d) maintain a 10k-prompt probe set replayed every session to measure drift.
  - (e) if drift is unavoidable, document it as a finding (it itself becomes a knowledge-fingerprint signal in Phase 6).

  **Bounded raw ring buffer (new, per RC1).** `/mnt/data/work/fsq_cont/activations/_raw_ring/` holds the most recent N=200k prompts' raw 4096-dim float16 activations for each (layer, signal) pair. At 12 (layer, signal) pairs × 200k prompts × 4096 dim × 2 bytes ⚠️ WARNING: THEORETICAL CALCULATION ≈ 196GB — within the 5.2TB disk budget. Old prompts are evicted FIFO. This bounded buffer exists solely to enable PCA refits without re-extraction; it is NOT the canonical artifact (the PCA'd shards are).

### Scenario 3 — 5M Magpie generation intractable on shared GPUs
- **Trigger**: Only 11–24GB free per GPU; the model used for Magpie generation needs weights + KV cache. At ~8GB weights for a 4B model, only ~3–13GB remains for KV cache → small batch sizes → projected throughput ⚠️ WARNING: EXPECTED/PROJECTED ~50–200 prompts/sec/GPU → 5M prompts needs 7–28 hours of dedicated GPU time, which is unavailable while other workloads run.
- **Early warning signal**: Phase-1 throughput probe (10k prompts) reports < 30 prompts/sec aggregate across all 4 GPUs; projected wall-clock for 5M exceeds the user's stated budget (Open Question 2).
- **Mitigation**: (a) start with a 500k-prompt pilot for Phases 2–5 to prove the pipeline end-to-end, then scale to 5M once the pipeline is validated — this is the de-risking path in Open Question 5; (b) Magpie generation can run at lower priority / nice level to coexist with other workloads; (c) if even 500k is intractable, use FineWeb-edu as the dominant source and Magpie as augmentation only (PLAN.md Source B is already a listed primary); (d) stream prompts to disk as they're generated so partial progress survives preemption; (e) **if Scenario 3 trigger fires, re-open the ADR and fall back to staged Option A** (layer-16 residual first) per RC3 — this is the pre-committed contingency, not an ad-hoc downgrade.

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
  - `scripts/verify_hf_model_id.py` confirms the canonical HF model ID OR surfaces a ranked list of alternatives (gemma-3-4B-it, gemma-2-9B-it, etc.) with parameter counts.
  - **If `verify_hf_model_id.py` returns NO candidates** (per Critic minor finding): the script MUST exit non-zero with an explicit `NO_CANDIDATES` status, log to `logs/phase0_model_probe.json`, and halt Phase 0 for user escalation. The script must NOT fall back to any default model silently — model identity is user-owned per Open Question 1.
  - `activations/ datasets/ clusters/ fsq/ checkpoints/ memory_snapshots/ evaluation/` directories created and listed in `configs/paths.yaml`.
  - `uv pip list` shows: `transformers>=4.45`, `accelerate`, `torch`, `faiss-gpu`, `scikit-learn`, `joblib`, `numpy`, `pyarrow`, `tqdm`, `peft` (pinned versions recorded).
- **Owner**: executor (sonnet)
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~1–3 GPU-hours (single forward pass + package install; model download is the dominant cost at ~8–16GB weights).
- **Disk footprint**: ~16GB (model cache) + ~0 (code only).
- **Hard gate to Phase 1**: smoke test prints deterministic hidden-state shapes on two consecutive runs; HF model ID is confirmed and recorded in `configs/model.yaml:model_id`.

### Phase 1 — Knowledge Coverage Dataset *(revised per RC6)*
- **Deliverable**:
  - `scripts/gen_magpie.py` (resumable, JSONL streaming output to `datasets/magpie/raw/*.jsonl` with per-shard checkpoint).
  - `scripts/sample_fineweb.py` (streams FineWeb / C4 / Dolma samples to `datasets/web/*.parquet`).
  - `scripts/gen_synthetic_qa.py` (per-domain generators: coding, math, science, medicine, law, philosophy, history, languages, reasoning, writing — PLAN.md Source C).
  - `scripts/gen_self_explore.py` (self-prompted task creation / critique loop — PLAN.md Source D).
  - `scripts/build_unified_corpus.py` (merges all sources, dedupes via MinHash ACROSS sources — not just within — so that Magpie/FineWeb overlap is removed; emits `datasets/unified/prompts.parquet` with columns `prompt_id, source, domain, prompt_text`; logs cross-source overlap statistics to `datasets/unified/_overlap.json`).
- **Acceptance criteria**:
  - At least 500k prompts (pilot scale) present in `datasets/unified/prompts.parquet` with non-null `prompt_text` and no duplicate `prompt_id`.
  - Each generator script is resumable: re-running after kill skips already-generated `prompt_id`s (loaded from existing shards).
  - Sampling mixture logged: 20% greedy / 30% T=0.8 / 30% T=1.2 / 20% T=1.5 for Magpie (PLAN.md §Phase 1 Source A).
  - Source mixture recorded: per-source prompt counts in `datasets/unified/_manifest.json`.
  - **Source mixture for the 500k pilot (per RC6)**: >=40% FineWeb, >=20% Magpie, >=20% synthetic QA, <=20% self-explore (by prompt count). Rationale: FineWeb provides broad latent coverage; Magpie provides instruction-following coverage; synthetic QA fills under-represented domains (coding, math, science, medicine, law, philosophy, history, languages, reasoning, writing); self-explore is exploratory and capped to avoid mode collapse on self-referential prompts.
  - **Cross-source overlap (per RC6)**: `datasets/unified/_overlap.json` exists and reports pairwise MinHash Jaccard between every (Source A, Source B/C/D) pair. If any cross-source Jaccard > 0.30 the build is flagged but not halted (high overlap is a finding about the corpus, not a blocker); near-duplicate prompts are removed by MinHash dedupe before the final prompt count is tallied.
- **Owner**: data-engineer (sonnet)
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~50–200 GPU-hours for 5M Magpie prompts; ~5–10 GPU-hours for the 500k pilot (see Scenario 3 pre-mortem).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~20–80GB text (pilot) / ~200–800GB text (full 5M).
- **Hard gate to Phase 2**: `datasets/unified/prompts.parquet` exists with >=500k rows; per-source manifest shows all 4 sources (A, B, C, D) represented AND meets the mixture floor above; `_overlap.json` exists.

### Phase 2 — Activation Extraction *(revised per RC1 and RC4)*
- **Deliverable**:
  - `scripts/extract_activations.py` — for each prompt: forward pass with hooks on layers {8, 16, 24, 32}, signals {residual, mlp_out, attention_out} (per RC4 attention signal added); write PCA'd float16 vectors to `activations/pca_shards/shard_{NNNN}.npz` with `prompt_id` array; ALSO write raw 4096-dim vectors to the bounded ring buffer `activations/_raw_ring/` (FIFO, N=200k prompts per (layer, signal)) to enable future PCA refits (RC1); delete the in-memory raw tensor immediately after both writes.
  - `scripts/fit_pca_incremental.py` — **one-shot** pre-pass over the first 500k prompts: reads raw activations into memory in batches of 10k (NOT from disk shards — the raw activations are streamed through memory batch-by-batch during the pre-pass), fits 12 `IncrementalPCA(n_components=256)` models (one per (layer, signal); 4 layers × 3 signals per RC4), and persists each as `activations/pca_models/(layer)_(signal)_pca.joblib`. Explicitly: IncrementalPCA fits on raw in-memory activations batch-by-batch during this pre-pass, NOT on persisted raw shards on disk. After the fit, the same script transforms every prompt's raw activation and writes the PCA'd shard.
  - `scripts/build_drift_log.py` — per-shard activation mean/std appended to `activations/_drift_log.jsonl` (feeds Scenario 2 alarm).
  - `scripts/refit_pca_full.py` (new, per RC1) — invoked manually or by the Phase 3 drift alarm handler when Scenario 2 triggers. Reads raw activations from `activations/_raw_ring/` (bounded 200k-prompt buffer per (layer, signal)), refits PCA from scratch, rewrites every PCA'd shard in `activations/pca_shards/` under the new basis, and logs to `activations/_pca_refit_log.jsonl`. Does NOT incrementally update — full refit only.
- **Acceptance criteria**:
  - For each (layer, signal) pair, `activations/pca_shards/` contains float16 arrays of shape `(N_prompt_shard, 256)` with paired `prompt_id` arrays.
  - 12 PCA models persisted (4 layers × 3 signals; per RC4 attention signal added).
  - Reconstruction sanity: `InversePCA(PCA(x)) ≈ x` within relative error < 0.05 on a 1k-prompt probe (logged to `activations/_pca_recon_sanity.json`).
  - Resumability: script checks existing shards' `prompt_id`s and skips them.
  - Per-shard drift log shows no shard-mean > 2σ from running global mean (Scenario 2 alarm). If alarm fires, `refit_pca_full.py` is the documented response (NOT incremental PCA drift accommodation).
  - Hidden-size assumption (4096) verified against actual `model.config.hidden_size` — abort if mismatch.
  - Bounded ring buffer `activations/_raw_ring/` is populated and FIFO eviction confirmed (oldest 200k–N prompt evicted as prompt N+1 arrives).
- **Owner**: data-engineer (sonnet) for extraction harness; scientist (opus) for hook placement validation (now includes attention-output hook).
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~100–400 GPU-hours (500k prompts, 12 hooks each [was 8 pre-RC4], batch=1 to ensure determinism).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~200–350GB PCA'd shards (500k prompts × 12 signals × 256 dims × float16 ≈ 300GB) + ~196GB bounded raw ring buffer (per RC1 calculation in Scenario 2) = ~400–550GB total. Within 5.2TB budget. **(RC-v3-1 — PCA-refit transient disk peak):** at pilot scale, the refit transient peaks at old PCA shards (~30GB) + new shards being written (~30GB) + ring buffer (~196GB) + safety margin (~3GB) ⚠️ WARNING: THEORETICAL CALCULATION ≈ 0.26 TB peak vs 5.2 TB available → ≥4.9 TB headroom. Atomic-rename-per-shard is the write discipline during refit so a mid-refit crash cannot corrupt the canonical artifact (each shard is fully written to a temp path then `os.rename`'d into place; a crash leaves either the old or new shard, never a torn write).
- **Hard gate to Phase 3**: total prompt count in `activations/pca_shards/` matches `datasets/unified/prompts.parquet` within 0.1%; drift log shows no Scenario-2 alarm OR any alarm has been resolved via `refit_pca_full.py` (refit log entry recorded).

### Phase 3 — Latent Coverage Analysis *(revised per RC1 and Critic minor findings)*
- **Deliverable**:
  - `scripts/build_faiss_index.py` — FAISS-GPU index per (layer, signal) over the 256-dim PCA space; persisted to `clusters/faiss_(layer)_(signal).index`. Now 12 indexes (per RC4 attention signal addition).
  - `scripts/fit_clusters.py` — fits MiniBatchKMeans (K=10000) per (layer, signal); HDBSCAN on a 100k subsample for cross-check; outputs `clusters/labels_(layer)_(signal).parquet`.
  - `scripts/cluster_metrics.py` — per-cluster: count, entropy, mean intra-cluster distance, prompt-topic diversity (via simple lexical overlap), activation diversity. Writes `clusters/coverage_report.json`.
  - `scripts/rebuild_faiss_index.py` (new, per RC1) — invoked when `refit_pca_full.py` (Phase 2) completes, OR when the coverage loop requests a re-cluster. Drops the existing FAISS index for a given (layer, signal), reads the re-transformed PCA'd shards, and builds a fresh index. Linked to RC1's resolution: this script is the Phase-3 counterpart of `refit_pca_full.py` — PCA refit triggers FAISS rebuild, which triggers re-clustering, which triggers a fresh coverage report.
  - `scripts/coverage_loop.py` — orchestrates the PLAN.md coverage loop: identify sparse clusters, request more prompts from the Phase-1 generators targeted at sparse-cluster regions (use FAISS nearest-centroid to filter candidate prompts), repeat. Stops when KL(cluster_dist_t || cluster_dist_{t-1}) < 0.01 over 3 iterations. **When the coverage loop requests new prompts (per Critic minor finding)**, it invokes Phase 1's `gen_magpie.py` / `sample_fineweb.py` / `gen_synthetic_qa.py` with a targeted prompt spec; the new prompts then trigger Phase 2 re-extraction (`extract_activations.py` resumable — picks up only new prompt_ids), then a FAISS index incremental update, then re-clustering. The coverage loop does NOT skip Phase 2 for new prompts — activation extraction is mandatory for every prompt entering the unified corpus.
- **Acceptance criteria**:
  - `clusters/coverage_report.json` lists K=10000 clusters per (layer, signal) with all five PLAN.md metrics populated.
  - Coverage loop convergence criterion met: cluster-distribution KL < 0.01 across 3 consecutive iterations.
  - No cluster has < 10 prompts (sparse-cluster alarm) at convergence; if some do, they are flagged with a "long-tail" annotation, not silently dropped.
  - If a PCA refit occurred in Phase 2, `rebuild_faiss_index.py` has been run for every affected (layer, signal) and the coverage report reflects post-refit shards.
- **Owner**: scientist (sonnet/opus for coverage-loop design).
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~20–60 GPU-hours (FAISS-GPU build + MiniBatchKMeans; coverage loop iterations compound this; now 12 (layer, signal) indexes vs 8 pre-RC4).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~75–150GB (12 FAISS indexes + label parquets).
- **Hard gate to Phase 4**: coverage loop converged OR user override documenting why convergence is infeasible; sparse clusters enumerated.

### Phase 4 — FSQ Memory Bank *(revised per RC2, RC5, and RC-v3-2/3/4)*
- **Deliverable**:
  - `scripts/train_fsq_autoencoder.py` — encoder `256→128→64`, FSQ op, decoder `64→128→256`; loss = MSE + cosine-similarity; trains one model per (layer, signal). Saves `fsq/encoder_(layer)_(signal).pt`, `fsq/decoder_(layer)_(signal).pt`, `fsq/bank_(layer)_(signal).pt` (code lookup table). **Determinism posture (per Critic minor finding)**: FSQ autoencoder training uses `torch.manual_seed` and `torch.use_deterministic_algorithms(True)` for the forward pass and FSQ rounding op (which is deterministic by construction — rounding has no nondeterminism). The backward pass MAY use nondeterministic cuDNN convolution algorithms for the encoder/decoder MLPs; we accept this nondeterminism because (a) it is small relative to the FSQ rounding signal, (b) it is bounded by multi-seed runs at the headline capacity point, and (c) forcing deterministic cuDNN would impose a 2–3x slowdown that violates the GPU-contention budget. Multi-seed runs (3 seeds) are the mitigation, not deterministic backward.
  - Note: this two-sweep design (levels sweep + latent-dim sweep) supersedes PLAN.md's single 5-point capacity ladder (PLAN.md:313–320) which conflates latent dim and levels (256→128→64 alongside 16→16→16→8→8). The split is justified by Principle 5 (one variable per stage) and is recorded as a deliberate spec deviation. If the user prefers PLAN.md's original ladder preserved, this is reversible — note in OQ-8 (below).
  - `scripts/fsq_capacity_sweep.py` — runs TWO single-variable sweeps per Principle 5 (per RC5; replaces the v1 single ladder). Terms defined explicitly (per RC5):
    - **"FSQ latent dim"** = encoder output dim (the dimensionality of the vector that FSQ quantizes).
    - **"FSQ levels"** = per-dimension quantization levels (the number of discrete values each latent dim can take).
    - **Levels sweep** (latent dim fixed at 64): levels ∈ {16, 8, 4, 2}. Theoretical codebook size = levels^64.
    - **Latent-dim sweep** (levels fixed at 16): latent dim ∈ {256, 128, 64, 32}. Theoretical codebook size = 16^latent_dim.
    - Each sweep changes exactly one variable; the cross-product (e.g., latent_dim=128 AND levels=8) is NOT in this sweep and is reserved for Phase 4b if single-variable curves suggest an interaction.
  - `fsq/sweep_results.jsonl` schema (revised per RC5) — one row per (sweep_axis, layer, signal, latent_dim, levels, seed) with fields: `reconstruction_mse`, `cosine_sim`, `code_entropy_nat`, `perplexity`, `theoretical_codebook_size`, `utilization` (= `perplexity / theoretical_codebook_size`), `sweep_axis` (one of `"levels"` | `"latent_dim"`), `seed`, `early_stopped_at_epoch` (per RC-v3-3; null if early-stop did not fire).
  - `scripts/render_sweep_curves.py` — produces `evaluation/fsq_capacity_curve_levels.png` and `evaluation/fsq_capacity_curve_latent_dim.png` (two panels — one per sweep axis, per RC5).
- **Acceptance criteria**:
  - Both sweeps complete for layer 16 residual at all capacity points (4 levels-sweep cells + 4 latent-dim-sweep cells = 8 rows for that (layer, signal)).
  - **Collapse check (per RC2 — capacity-relative utilization)**: `utilization = perplexity / theoretical_codebook_size` > 0.20 at every capacity point in both sweeps (Scenario 1 alarm); if below, training halts and triggers Scenario-1 mitigations. The 0.20 threshold is comparable across capacity points because it is normalized by theoretical codebook size.
  - **Reconstruction gate**: best capacity point in EACH sweep achieves MSE < 0.1 (normalized) AND cosine > 0.7 on the held-out probe (numbers pending Phase 2 actuals; re-calibrated once pilot PCA is fit).
  - **Early-stop rule (enforced, per RC-v3-3)**: each FSQ training cell uses `patience=10 epochs` with `delta < 0.001 MSE` on the held-out probe as the early-stop trigger; hard ceiling at `max_epochs=200` per cell. Early-stop firing is logged to `fsq/sweep_results.jsonl` with `early_stopped_at_epoch`. A cell that hits max_epochs=200 without early-stop is flagged (not failed) for analysis.
  - `evaluation/fsq_capacity_curve_levels.png` and `evaluation/fsq_capacity_curve_latent_dim.png` each show a monotonic quality-vs-compression tradeoff (no anomalies that suggest a training bug).
- **Owner**: scientist (opus — FSQ encoder design is research-grade).
- **GPU-hours**: ⚠️ WARNING: EXPECTED/PROJECTED ~40–120 GPU-hours (small MLP, but 12 banks × 8 sweep cells × 3 seeds for the headline results; increased from v1's 8 banks × 5 cells due to RC4 attention addition and RC5 sweep split).
- **Disk footprint**: ⚠️ WARNING: THEORETICAL ~10–30GB (model checkpoints are small; sweep logs dominate).

#### Phase 4 wall-clock gate (per RC-v3-2)

- **Projected wall-clock (full Phase 4)**: 12 banks × 8 cells × 3 seeds = 288 runs at ~30 min/run ⚠️ WARNING: EXPECTED/PROJECTED ≈ 144 GPU-hours (lower bound; the 40–120 range in the GPU-hours line above is the tighter estimate assuming early-stop fires promptly).
- **Pre-Phase-4 gate**: before kickoff, read the user's answer to OQ-2 (acceptable wall-clock for Phases 0–4). If the projected Phase-4 wall-clock exceeds the remaining budget, trigger fallback in this strict priority order:
  1. **Reduce seed count from 3 → 1** (saves ~96 GPU-hours; statistical robustness degraded but sweep coverage preserved).
  2. **Activate Staged-Option-A** (RC3 Scenario-3 path): sweep layer-16 residual only at 12 cells × 3 seeds = 36 runs (~18 GPU-hours ⚠️ WARNING: EXPECTED/PROJECTED), documented as Phase 4a; defer full Option B to Phase 4b pending additional budget.
  3. **If even 18 GPU-hours exceeds budget**: halt and escalate to user (do NOT silently proceed).
- **Record-keeping**: the user's OQ-2 answer and the chosen branch (full / seeds-reduced / staged-Option-A / halted) MUST be written to `logs/phase4_wallclock_gate.json` BEFORE any Phase 4 training starts. The Phase-4 hard gate (next) cannot fire without this file present.

- **Hard gate to Phase 5**: a single (layer, signal, latent_dim, levels) configuration is nominated as the "canonical" FSQ bank and recorded in `configs/fsq_canonical.yaml`. Semantic validation (Phase 5) runs against this canonical bank only. The file `logs/phase4_wallclock_gate.json` exists and records the chosen branch.

### Phases 5–10 — one-liners with dependencies

- **Phase 5 — Semantic Validation** (depends: Phase 4 canonical bank). `scripts/semantic_nn_eval.py` + `scripts/code_intervention.py`. Acceptance: prompts sharing codes have topic-consistency > chance (p < 0.01 via permutation test); code edits produce measurable activation drift.
- **Phase 6 — Knowledge Fingerprint** (depends: Phase 5 passes). `scripts/build_fingerprint.py` → `memory_snapshots/baseline_fingerprint.json`. Acceptance: P(code), P(code_i, code_j), entropy, rare-code frequency populated for the canonical bank; fingerprint is deterministic across two recomputation runs.
- **Phase 7 — Fine-Tuning Baseline** (depends: Phase 6; Open Question 4 for benchmark scope). `scripts/finetune_baseline.py` (plain SFT on coding/instruction/domain tasks using `peft` or full FT) + `scripts/run_benchmarks.py` (MMLU, GSM8K, HumanEval, HellaSwag, TruthfulQA via `lm-eval-harness`). **`lm-eval-harness` version (per Critic minor finding)**: pinned in `pyproject.toml` as `lm-eval>=0.4.5,<0.5` (or the latest stable as of Phase 7 kickoff — confirm via `context7`); the exact version and commit hash are recorded in `evaluation/baseline_forgetting.json:lm_eval_version`. Acceptance: post-FT benchmark delta and post-FT fingerprint delta both recorded in `evaluation/baseline_forgetting.json`.
- **Phase 8 — FSQ-Guided Continual Learning** (depends: Phase 7 baseline). `scripts/finetune_with_memory.py` implementing `L = L_task + λ · L_memory` with L_memory candidates A/B/C/D from PLAN.md. Acceptance: sweep over λ and retention levels {1.0, 0.75, 0.50, 0.25} produces `evaluation/retention_sweep.jsonl`; `memory_rewrite_rate` tracked at {1%, 5%, 10%, 25%, 50%} targets. **`memory_rewrite_rate` acceptance/fallback (per Critic minor finding)**: the target rates above are EXPLORATORY sweep points, not hard acceptance criteria — the rewrite rate is an emergent property of (λ, retention, task), not directly controllable. Acceptance criterion is instead: "for every (λ, retention) cell, `memory_rewrite_rate` is measured and logged; the SWEEP COVERAGE is the gate (all cells have data), not a specific rate value". If measured rates cluster outside the {1%…50%} envelope (e.g., all cells < 0.1%), the plan documents this as a finding about L_memory strength, extends the λ sweep upward, and re-runs — the fallback is λ re-sweep, not plan invalidation.
- **Phase 9 — LoRA Validation Track** (depends: Phase 8). Same scripts as Phase 8 but with LoRA adapter; produces `evaluation/lora_comparison.json`. Acceptance: FSQ drift under LoRA vs full FT compared head-to-head; PLAN.md hypothesis ("easier to interpret under LoRA") answered with explicit metric.
- **Phase 10 — Memory Preservation Benchmark** (depends: Phase 9). `scripts/preservation_benchmark.py` correlates parameter-distance vs FSQ-memory-distance against benchmark retention. Acceptance: Spearman ρ reported for both predictors; success criterion (FSQ ρ > parameter ρ) answered yes/no with CI.

---

## 4. Expanded Test Plan — DELIBERATE MODE *(revised per Critic minor findings)*

### Unit
- **FSQ round-trip**: `tests/test_fsq_roundtrip.py` — quantize → dequantize a known vector; assert L2 error ≤ sqrt(d)/levels (analytic bound).
- **FSQ gradient**: `tests/test_fsq_grad.py` — straight-through gradient passes through; grad shape matches input shape.
- **PCA fit/save/load**: `tests/test_pca_io.py` — fit `IncrementalPCA` on 10k random vectors; save to `.joblib`; reload; assert `transform()` output bitwise-equal.
- **Activation hook determinism**: `tests/test_hook_determinism.py` — same prompt, same seed, two forward passes; assert hidden states bitwise-equal (catches nondeterministic cuBLAS paths).
- **Checkpoint resumability**: `tests/test_resume.py` — start extraction on 100 prompts, kill after 50, restart; assert all 100 eventually processed exactly once.
- **Codebook-collapse alarm (per Critic minor finding)**: `tests/test_codebook_collapse_alarm.py` — synthetic activation set where 95% of mass concentrates on one cluster; train FSQ 1 epoch; assert `utilization = perplexity / theoretical_codebook_size < 0.20` triggers the Scenario 1 alarm; also verify a healthy uniform activation set does NOT trigger the alarm.
- **Coverage-loop convergence detector (per Critic minor finding)**: `tests/test_coverage_convergence.py` — synthetic sequence of cluster distributions with monotone-decreasing KL to a fixed point; assert the convergence criterion (KL < 0.01 across 3 iterations) fires exactly at the expected iteration; also assert a non-converging sequence (KL oscillating) does NOT fire.
- **Sweep-monotonicity check (per Critic minor finding)**: `tests/test_sweep_monotonicity.py` — synthetic `sweep_results.jsonl` where reconstruction MSE is monotone in capacity for each sweep axis; assert the checker passes; also feed a non-monotone (buggy) sweep and assert the checker flags it. This guards against training-bug artifacts being misread as real capacity-curve structure.

### Integration
- **End-to-end pipeline probe**: `tests/test_integration_pipeline.py` — 1k-prompt mini-corpus → Phase 2 extraction → Phase 3 cluster → Phase 4 FSQ train (1 epoch) → reconstruction MSE computed. Assert MSE is finite, non-NaN, < 1.0.
- **PCA inverse fidelity**: `tests/test_pca_inverse.py` — on real activations (100-prompt probe), `inverse(transform(x))` within rel-error 0.05 of `x`.
- **PCA refit + FAISS rebuild chain (per RC1)**: `tests/test_pca_refit_chain.py` — seed a drift alarm by injecting an out-of-distribution shard; assert `refit_pca_full.py` runs; assert downstream FAISS indexes are rebuilt by `rebuild_faiss_index.py` and reflect the new PCA basis (probe-prompt nearest-neighbor sets change post-refit).

### E2E
- **Knowledge fingerprint reproducibility**: `tests/test_fingerprint_repro.py` — run Phase 6 twice from the same activation shards; assert fingerprint JSON bitwise-equal (or within float32 tolerance).
- **Semantic neighbor sharing**: `tests/test_semantic_neighbors.py` — for 100 probe prompts, compute nearest neighbors in (a) prompt-embedding space and (b) FSQ-code space; assert Jaccard overlap > random baseline at p < 0.01.

### Observability
Every long-running script emits structured JSONL to `logs/{phase}_{script}.jsonl` with:
- per-shard: `prompt_ids`, `mean_norm`, `std_norm`, `duration_sec`, `gpu_mem_peak_gb`.
- per-FSQ-training-epoch: `epoch`, `train_mse`, `val_mse`, `code_entropy_nat`, `perplexity`, `theoretical_codebook_size`, `utilization` (= `perplexity / theoretical_codebook_size`, per RC2), `cluster_silhouette`.
- per-coverage-iteration: `kl_to_prev`, `num_sparse_clusters`, `total_clusters`.

Checkpoint-resumability tests run in CI on a 100-prompt toy dataset every commit. Determinism tests run once per model-load (Phase 0 acceptance).

---

## 5. Open Questions for the User (now 7, all blocking)

1. **Exact HuggingFace model ID for `google/gemma-4-E4B`.** As of 2026-06-17 the existence of this model on HF Hub is unverified. **Why it blocks Phase 0**: every downstream path (layer count, hidden size, tokenizer, attention impl) depends on the actual model. `scripts/verify_hf_model_id.py` will surface candidates (gemma-3-4B-it, gemma-2-9B-it, etc.) but the user must confirm which to canonize. Without this, hidden-size assumptions (4096) and layer indices {8, 16, 24, 32} are speculative.

2. **Acceptable wall-clock budget for Phases 0–4.** At 11–24GB free per GPU under contention, Phase 2 alone is projected ⚠️ WARNING: EXPECTED/PROJECTED at 100–400 GPU-hours. **Why it blocks Phase 1 scaling**: determines whether the pilot is 500k prompts (de-risk first) or jump straight to 5M. Directly trades against the user's other running workloads. Also feeds the Phase 4 wall-clock gate (see §3 Phase 4).

3. **Is Magpie available and licensed for this use?** Magpie-UA / Magpie-Align terms vary by release; commercial-research splits matter. **Why it blocks Phase 1 Source A**: if Magpie is unavailable or license-incompatible, Phase 1 must rely on FineWeb + synthetic QA only, which weakens latent coverage and may force a larger FineWeb sample to compensate.

4. **Acceptable benchmark suites for Phase 7 baseline.** PLAN.md lists MMLU, GSM8K, HumanEval, HellaSwag, TruthfulQA. Are any deprecated/replaced as of 2026-06-17? Is `lm-eval-harness` the runner, or an internal harness? **Why it blocks Phase 7**: defines what "forgetting" means operationally and gates the Phase 10 drift-correlation claim.

5. **Should we start on a smaller proxy model first (e.g., gemma-2-2B) to de-risk the pipeline?** A 2B model fits comfortably in 11GB free and runs the entire Phase 0–5 pipeline in hours, not days. **Why it blocks Phase 0**: if "yes", Phase 0 targets the proxy and the gemma-4-E4B run becomes Phase 0b. If "no", we accept the contention risk on the 4B model from day one (Principle 2 still holds — proxy is never the final deliverable, only a pipeline validator).

6. **Should PLAN.md's original single 5-point capacity ladder (256×16, 128×16, 64×16, 64×8, 32×8) be preserved verbatim, or is the v3 two-sweep design (levels sweep + latent-dim sweep) acceptable?** The v3 split isolates FSQ levels from encoder latent-dim per Principle 5 but deviates from PLAN.md §Phase 4. **Why it blocks Phase 4**: determines the sweep grid shape. Default in v3: two-sweep design. User-veto path: revert to PLAN.md ladder with a documented Principle-5 caveat.

7. **(merged OQ-6/OQ-7) Is the user committed to FSQ specifically, or would VQ-VAE / LFQ / BSVQ be acceptable same-family alternatives if FSQ collapse proves intractable at this scale?** Rationale: the entire plan (encoder shape, sweep axes, theoretical-codebook-size math, perplexity-based utilization) is FSQ-specific. Scenario 1 mitigation (d) lists a "VQ-VIBE / affinity-normalized FSQ variant" fallback. If the user is FSQ-only committed, that fallback is a documented dead-end and the pre-mortem needs a stronger escape path (e.g., reduce capacity to a single layer, or accept collapse as a finding rather than a failure). If LFQ / BSVQ are acceptable, the Scenario 1 escape is concrete and the pre-mortem stands. This is a user preference (quantization-family commitment), not a codebase fact. **Why it blocks Phase 4 fallback design**: determines whether Scenario 1 mitigation (d) is a real branch or a documented dead-end. (Canonical record: OQ-7 in `open-questions.md`.)

---

## 6. ADR — Architectural Decision Record *(revised per RC3 and RC4)*

### Decision
Adopt **Option B** (multi-layer {8, 16, 24, 32} × signals {residual, MLP, attention}, PCA-256 per (layer, signal), independent FSQ banks with capacity sweep) as the canonical architecture for the FSQ Knowledge Memory Bank.

### Drivers
1. GPU memory contention (11–24GB free per H200) forces streaming + small in-memory footprint.
2. 5.2TB disk free caps raw activation storage; PCA-on-write is mandatory, not optional.
3. PLAN.md explicitly lists layers {8, 16, 24, 32} and signals {residual, attention, MLP} — scope downgrade is disallowed by user intent ("Do NOT downgrade scope").

### Alternatives Considered

- **Option A (single-shot)** (single layer 16, residual only, PCA-256, FSQ sweep): rejected because it drops the multi-layer drift fingerprint that makes Phase 6/10 research-valuable, and downgrades PLAN.md scope.

- **Option A (staged)** (per RC3 steelman): begin with layer-16 residual only; complete Phases 2–5 at ~1/8 the cost; expand to the full Option B matrix ONLY after Phase 5 semantic validation passes.
  - **Pros**: (1) ~8× faster de-risk; first research result in days, not weeks. (2) If layer-16 alone produces semantic codes, the multi-layer sweep is confirmatory rather than exploratory — Option B then runs with a known-good baseline instead of being the first experiment. (3) Lower day-one disk commitment. (4) Cleanly absorbs Scenario 3 (Magpie generation slip) as the pre-committed fallback rather than an ad-hoc downgrade.
  - **Rejected because**: PLAN.md's cross-layer drift fingerprint requires multiple layers by construction, and a staged approach risks anchoring on layer-16 findings before the multi-layer picture is available. The Option B multi-layer result is the headline artifact; deferring it by even one Phase risks the user interpreting the layer-16 interim as the answer.
  - **However**: if Phase 1 Magpie generation slips (Scenario 3 trigger fires — projected 5M-prompt wall-clock exceeds the user's budget), the staged Option A becomes the fallback and this ADR is re-opened. This is the pre-committed contingency path, documented here so that invoking it is not a plan-change but a plan-activation. (Note: the Phase-4 wall-clock gate's fallback tier 2 — RC-v3-2 — also invokes staged Option A as the Phase-4a path under budget pressure, even if Scenario 3 has not fired at the Phase-1 level.)

- **Option C** (all layers × {residual, MLP, attention} at full scale): earlier draft rejected this on disk grounds citing ⚠️ THEORETICAL ~4–6TB. That math was wrong — it conflated "all layers" with "all signals". At the actual sweep envelope of 4 layers × 3 signals = 12 (layer, signal) pairs (identical to Option B), the footprint ⚠️ WARNING: THEORETICAL CALCULATION ≈ 1.1–2.1TB at 5M prompts, which is within the 5.2TB budget. The real rejection grounds are wall-clock, not disk: sweeping ALL layers (e.g., 48 for a 48-layer model) × 3 signals = 144 (layer, signal) pairs produces a sweep matrix 12× larger than the Option B baseline and is projected to exceed the Phase 4 budget even at 500k-prompt pilot scale.
  - **Reclassification (per RC4)**: Option C is **viable but deferred** to a Phase-4b ablation target. If Phase 4's Option B sweep reveals that layers {8, 16, 24, 32} miss substantial drift signal (e.g., inter-layer drift is concentrated outside the sampled layers), Option C becomes the expansion target. The Option B result is the precondition for deciding whether Option C is worth its wall-clock cost.

- **External FSQ package** vs hand-rolled: deferred to Phase 0 verification via `context7`; hand-rolled chosen as fallback because the op is ~20 LOC and avoids an unverified dependency.
- **Lightning** vs plain `torch`: plain torch chosen; the FSQ autoencoder is a small MLP and Lightning's overhead is unjustified. `peft` reserved for Phase 9.

### Why Chosen
Option B is the only option that simultaneously (a) respects PLAN.md's explicit layer/signal list including attention (per RC4), (b) fits within the 5.2TB disk budget under the streaming+PCA discipline even with the revised 1.1–2.1TB footprint projection, (c) preserves the cross-layer AND cross-signal drift signal needed for Phase 6/10, and (d) has a tractable wall-clock under GPU contention when combined with the 500k-prompt pilot de-risking path and the pre-committed staged-Option-A fallback under Scenario 3.

### Consequences
- **Positive**: faithful to user intent; multi-layer multi-signal drift fingerprint is the highest-research-value artifact; PCA-on-write keeps disk tractable; pre-committed Scenario-3 fallback means Magpie slip is absorbed without plan renegotiation.
- **Negative**: 12× the sweep matrix of Option A (single-shot) — Phase 4 wall-clock is the longest single stage; requires strict checkpoint discipline (mitigated by PLAN.md §6.6 compliance built into every script).
- **Risk**: if the chosen model's actual `num_hidden_layers < 32`, the layer indices {8, 16, 24, 32} must be re-scaled (Phase 0 smoke test catches this).

### Follow-ups
- Re-evaluate Option C (expansion to all layers) after Phase 4 Option B results are in — Option C is now "viable but deferred" not "invalidated".
- If the FSQ sweep (Phase 4) shows that one (layer, signal) dominates reconstruction quality, propose a Phase-4b ablation reducing to that cell only — but only after the multi-layer multi-signal result is on record.
- Proxy-model pilot (Open Question 5): if approved, becomes Phase 0a; ADR is re-opened if pilot surfaces a fundamental pipeline issue.
- **Scenario 3 re-trigger**: if Magpie generation slips, staged Option A is activated and this ADR records the activation event in `logs/adr_reopen_sceanrio3.json`.
- **Phase-4 wall-clock budget pressure**: if the pre-Phase-4 gate (RC-v3-2) escalates to staged Option A as Phase 4a, the ADR records the activation in `logs/phase4_wallclock_gate.json` and Option B full sweep is deferred to Phase 4b — this is plan-activation, not plan-change, mirroring the Scenario-3 contingency pattern.

---

## Plan v3 Addendum — OQ-1 and OQ-5 Resolution (2026-06-17)

### OQ-1 RESOLVED
`google/gemma-4-E4B` confirmed on HuggingFace Hub. Config: `model_type = gemma4_text`, `hidden_size = 2560`, `num_hidden_layers = 42`, `num_attention_heads = 8`, multimodal = true (text + vision towers; hooks must target `text_config` / `model.language_model.model.layers[i]`), last modified 2026-06-03. This is the canonical model for the final FSQ Knowledge Memory Bank.

### OQ-5 RESOLVED
YES proxy pilot first. Pilot model: `google/gemma-2-2b-it` (`hidden_size = 2304`, `num_hidden_layers = 26`, `num_attention_heads = 8`, non-multimodal). Pilot `hook_layers` re-scaled to `{5, 10, 15, 20}` — evenly-spaced fractions (~19/38/57/76%) of the original `{8, 16, 24, 32}` over the final model's 42 layers, projected onto the pilot's 26-layer stack.

### New phase structure
- **Phase 0a** = pilot smoke test (`scripts/smoke_test_model.py --model pilot` → `logs/phase0a_smoke_test_pilot.json`) + Phases 1–5 executed on the pilot to de-risk the pipeline end-to-end.
- **Phase 0b** = final smoke test (`scripts/smoke_test_model.py --model final` → `logs/phase0b_smoke_test_final.json`) + re-extraction on `google/gemma-4-E4B` for the canonical bank.

### Plan-v3.md original 4096-dim assumption: SUPERSEDED
Actual hidden sizes: 2304 (pilot) and 2560 (final). All code must read from `model.config` (or `model.config.text_config` for the multimodal final model) at runtime per the existing Phase 0 acceptance criteria. The `expected_hidden_size: 4096` / `expected_num_hidden_layers: 48` speculative entries in `configs/model.yaml` have been replaced by per-model sections.

### Ring-buffer disk math revised
12 (layer, signal) pairs × 200k prompts × float16:
- Pilot (2304-dim): ⚠️ WARNING: THEORETICAL CALCULATION ≈ 110 GB.
- Final (2560-dim): ⚠️ WARNING: THEORETICAL CALCULATION ≈ 122 GB.

Both fit comfortably inside the 5.2 TB disk budget; the RC1 bounded ring-buffer design is unchanged.

### PCA input dim
- Pilot: 2304 → 256 (~9× reduction).
- Final:  2560 → 256 (~10× reduction).
Both retain the PCA-256 target from §3 Phase 2.

### Multimodal hook paths (final model only)
`scripts/smoke_test_model.py` detects multimodality via `hasattr(model.config, 'text_config')` and resolves the decoder layer container at `model.language_model.model.layers[i]` (falling back to `model.language_model.layers[i]` or `model.model.layers[i]` for non-multimodal). The resolved path is printed to stdout and recorded in the structured log so downstream extraction scripts can consume it.

### Config schema change
`configs/model.yaml` is now a two-section structure (`pilot:` / `final:`) plus `active_model:` selector and `candidates_file:`. The flat `model_id`, `expected_hidden_size`, `expected_num_hidden_layers`, `hook_layers`, `hook_signals`, `dtype`, `attention_impl`, `device_map` top-level keys are removed.
