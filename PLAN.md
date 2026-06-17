# PLAN.md

# FSQ Knowledge Memory Bank for Gemma-4-E4B

## Goal

Create an FSQ-based knowledge memory bank for Gemma-4-E4B that:

1. Compresses a significant portion of the model's internal knowledge into discrete memory states.
2. Can be preserved during future fine-tuning.
3. Enables measurement of knowledge drift without requiring access to the original pretraining dataset.
4. Serves as the foundation for a future retention loss.

This project is NOT attempting continual learning initially.

The first milestone is proving that a stable and meaningful FSQ memory representation can be extracted from an existing model.

---

# Core Hypothesis

Current LLM knowledge is distributed across parameters.

If we can learn an FSQ bottleneck that reconstructs important internal activations, then:

- FSQ codes become compressed knowledge states.
- Knowledge drift becomes measurable in code space.
- Future fine-tuning can preserve codes rather than replay datasets.

The FSQ memory bank should be viewed as a Learned Knowledge Preservation Space, not merely a compression layer.

---

# Research Questions

RQ1:
Can an FSQ bottleneck reconstruct internal activations with acceptable loss?

RQ2:
Do FSQ codes exhibit semantic clustering?

RQ3:
Do similar prompts activate similar codes?

RQ4:
Can code distributions be used as a knowledge fingerprint?

RQ5:
Does preserving FSQ codes reduce forgetting during fine-tuning?

RQ6:
Can FSQ memory drift predict forgetting better than parameter drift?

---

# Phase 0: Environment

## Model

Target:

google/gemma-4-E4B

Requirements:

- HuggingFace transformers
- accelerate
- flash-attn (if supported)
- FAISS
- PyTorch
- FSQ implementation
- PEFT/LoRA support

---

## Storage

Expect:

- Activation collection: 100GB–2TB

Prepare:

activations/
datasets/
clusters/
fsq/
checkpoints/
memory_snapshots/
evaluation/

---

# Phase 1: Knowledge Coverage Dataset

## Goal

Create a broad activation dataset that approximates the model's internal knowledge manifold.

Do NOT rely only on human-written prompts.

Need maximum latent-space coverage.

---

## Dataset Sources

### Source A: Magpie-generated instructions

Generate:

5M–20M instructions

Sampling mixture:

- 20% greedy
- 30% temperature=0.8
- 30% temperature=1.2
- 20% temperature=1.5

Avoid relying exclusively on min_p=0.1.

Explore long-tail capabilities.

---

### Source B: Open Web Text

Possible datasets:

- FineWeb
- C4
- Dolma subsets

Sample broadly and uniformly.

---

### Source C: Synthetic QA

Generate content covering:

- coding
- math
- science
- medicine
- law
- philosophy
- history
- languages
- reasoning
- writing

---

### Source D: Self-Generated Exploration

Prompt the model to:

- create tasks
- answer tasks
- critique answers
- generate alternatives
- generate edge cases

Purpose:

explore latent capabilities.

---

# Phase 2: Activation Extraction

## Goal

Capture internal knowledge states.

---

## Candidate Layers

Collect activations from:

- Layer 8
- Layer 16
- Layer 24
- Layer 32

Compare later.

---

## Candidate Signals

Store:

1. Residual stream
2. Attention output
3. MLP output

Priority:

Residual stream first.

---

## Compression

Before storage:

Apply PCA.

Example:

4096 → 256 dimensions

Store:

float16

---

# Phase 3: Latent Coverage Analysis

## Goal

Discover underexplored regions of the model.

---

## Index

Build FAISS index.

---

## Clustering

Evaluate:

- KMeans
- MiniBatchKMeans
- HDBSCAN

Initial experiment:

K = 10,000

---

## Metrics

For each cluster:

- sample count
- entropy
- average distance
- prompt diversity
- activation diversity

---

## Coverage Loop

Iterate:

generate prompts
→ collect activations
→ cluster assignment
→ identify sparse clusters
→ generate more prompts
→ repeat

Stop when cluster distribution stabilizes.

---

# Phase 4: FSQ Memory Bank

## Goal

Create discrete knowledge representations.

---

## Encoder

Input:

activation vector

Example:

256 dimensions

Encoder:

256
→ 128
→ 64

---

## FSQ

Initial overpowered experiment:

256 dimensions
16 levels

Then progressively reduce capacity.

Test:

- 256 dims × 16 levels
- 128 dims × 16 levels
- 64 dims × 16 levels
- 64 dims × 8 levels
- 32 dims × 8 levels

Generate compression-quality curves.

---

## Decoder

FSQ codes
→ activation reconstruction

Loss:

- MSE
- Cosine similarity

---

# Phase 5: Semantic Validation

## Goal

Verify that FSQ codes contain meaningful knowledge.

---

## Nearest Neighbor Tests

Find prompts with:

- identical codes
- similar codes

Evaluate:

- topic consistency
- capability consistency
- reasoning consistency

---

## Code Interventions

Manually modify codes.

Measure:

- activation drift
- output drift

---

## Cluster Alignment

Measure:

FSQ codes ↔ latent clusters

Desired:

strong alignment.

---

# Phase 6: Knowledge Fingerprint

## Goal

Represent model knowledge using FSQ statistics.

---

Collect:

- P(code)
- P(code_i, code_j)
- code entropy
- code frequency
- rare code frequency

---

Store:

memory_fingerprint.json

Contains baseline knowledge statistics.

---

# Phase 7: Fine-Tuning Baseline

## Goal

Measure forgetting without memory preservation.

---

Fine-tune normally.

Tasks:

- coding
- instruction following
- domain adaptation

---

Benchmarks:

- MMLU
- GSM8K
- HumanEval
- HellaSwag
- TruthfulQA

---

Collect:

new memory fingerprint

Measure:

- code drift
- KL divergence
- entropy shift

---

# Phase 8: FSQ-Guided Continual Learning

## Goal

Use FSQ memory space as a regularization target during fine-tuning.

---

# Motivation

Cross-entropy optimizes learning new information.

Cross-entropy does not explicitly preserve old information.

Current approaches depend on:

- replay datasets
- distillation datasets
- stored examples

FSQ memory should provide a compact preservation target.

---

# Memory Snapshot

Before fine-tuning:

Collect:

- FSQ codes
- code distributions
- cluster statistics
- entropy statistics

Store:

memory_snapshot.pt

This becomes the reference knowledge state.

---

# Fine-Tuning Objective

Baseline:

L = L_task

Proposed:

L = L_task + λ * L_memory

Where:

L_task = task-specific objective

L_memory = knowledge-preservation objective

---

# Candidate Memory Losses

## A. Code Preservation

Hamming distance between old and new codes.

Pros:

- Simple
- Direct

Cons:

- Potentially too restrictive

---

## B. Quantized Distance

Allow small movement.

Examples:

|old_code - new_code|

or

(old_code - new_code)^2

---

## C. Distribution Preservation

KL(P_old(code) || P_new(code))

Pros:

- Flexible
- Scales well

---

## D. Mutual Information Preservation

Maximize:

I(old_codes ; new_codes)

Pros:

- Most information-theoretic

Cons:

- Harder optimization problem

---

# Retention Strength

Configurable preservation levels.

Examples:

Retention = 1.0

Preserve nearly everything.

---

Retention = 0.75

Moderate adaptation.

---

Retention = 0.50

Allow significant rewriting.

---

Retention = 0.25

Prioritize task learning.

---

# Memory Rewrite Budget

Interpret FSQ memory as finite knowledge storage.

Track:

memory_rewrite_rate

Definition:

changed_codes / total_codes

Targets:

- 1%
- 5%
- 10%
- 25%
- 50%

Measure correlation with forgetting.

---

# Phase 9: LoRA Validation Track

## Goal

Validate FSQ retention before attempting full fine-tuning.

---

Run:

1. LoRA fine-tuning without memory loss
2. LoRA fine-tuning with memory loss

Compare:

- forgetting
- memory drift
- benchmark retention

Hypothesis:

FSQ drift will be easier to interpret under LoRA than full parameter updates.

If unsuccessful under LoRA, unlikely to succeed under full fine-tuning.

---

# Phase 10: Memory Preservation Benchmark

## Goal

Determine whether FSQ memory predicts forgetting.

---

Measure:

## Task Metrics

- task accuracy
- task loss
- benchmark scores

---

## Retention Metrics

- MMLU retention
- GSM8K retention
- HumanEval retention
- TruthfulQA retention

---

## Memory Metrics

- code drift
- KL divergence
- entropy shift
- cluster migration
- rewrite rate

---

## Core Research Question

Compare:

Parameter Distance

vs

FSQ Memory Distance

as predictors of forgetting.

Success if FSQ memory drift correlates more strongly with capability loss.

---

# Future Work

If validation succeeds:

Integrate memory directly into pretraining.

Architecture:

Memory Tokens
→ FSQ
→ Transformer

Training:

CrossEntropy
+ PredictiveMemoryLoss
+ RetentionLoss

Long-term vision:

CrossEntropy learns new information.

MemoryLoss preserves previously compressed knowledge.

---

# Success Criteria

## Minimum Success

FSQ reconstructs activations and forms stable semantic clusters.

---

## Strong Success

FSQ fingerprints correlate with forgetting.

---

## Major Success

Retention loss improves continual learning without replay datasets.

---

## Breakthrough Success

Knowledge drift can be measured and controlled entirely in FSQ memory space.

---

# Most Important Deliverables

1. Activation dataset
2. Latent coverage map
3. FSQ memory bank
4. Knowledge fingerprint system
5. Memory drift metrics
6. Fine-tuning regularization experiments
7. Retention-vs-compression scaling curves
8. FSQ-guided continual learning prototype
