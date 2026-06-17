## Handoff: ralplan → team-exec (Phase 0 only)

- **Decided**: Plan v3 APPROVED by RALPLAN consensus (Planner + Architect + Critic, 2 iterations) and USER-APPROVED via /ralplan --interactive UI on 2026-06-17. Execution pathway: team. Phase 0 scope only for this team — no model download, no GPU work, no activation extraction until OQs answered.
- **Rejected**: Full Phases 0-4 in one team invocation (resource commitment too large without OQ-1/2/7 answered). Single-agent execution (user picked team pathway explicitly). Default Staged-Option-A (Architect's steelman — PLAN.md directive wins).
- **Risks**:
  - `google/gemma-4-E4B` may not exist on HF Hub — `verify_hf_model_id.py` MUST handle NO_CANDIDATES case and halt + escalate (per plan-v3.md Phase 0 RC).
  - Workers may be tempted to "just download the model to test it" — explicitly forbidden in worker prompts; smoke_test_model.py is WRITE-ONLY this phase.
  - uv sync will fetch torch (~2GB) + faiss-gpu — non-trivial network/disk but reversible.
- **Files**:
  - `/mnt/data/work/fsq_cont/.omc/plans/plan-v3.md` — canonical approved plan
  - `/mnt/data/work/fsq_cont/.omc/plans/CONSENSUS.md` — approval record
  - `/mnt/data/work/fsq_cont/.omc/plans/open-questions.md` — 8 OQs (OQ-1/2/7 user-blocking)
- **Remaining (out of scope for this team)**:
  - Phase 1 dataset generation (needs OQ-2 budget + OQ-3 Magpie license)
  - Phase 2 activation extraction (needs Phase 1 outputs)
  - Phase 3-4 (downstream of Phase 2)
  - Phase 4 FSQ bank (needs OQ-7 FSQ commitment for fallback design)
  - All model loading / GPU work (needs OQ-1 canonical model ID)
