"""Phase 0: probe HuggingFace Hub for the canonical Gemma model ID.

Resolves Open Question 1 (plan-v3.md) by surfacing which `google/gemma-*`
model IDs exist on the Hub. Fetches config.json (~1KB per model) only —
does NOT download model weights.

Exit codes:
    0  — at least one candidate found; results in configs/_hf_candidates.json
    2  — NO candidates found (halt-and-escalate per plan-v3.md Phase 0)
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub import HfApi
from huggingface_hub.errors import RepositoryNotFoundError, RevisionNotFoundError

REPO_ROOT = Path("/mnt/data/work/fsq_cont")
OUTPUT_PATH = REPO_ROOT / "configs" / "_hf_candidates.json"
LOG_PATH = REPO_ROOT / "logs" / "phase0_model_probe.json"

CANONICAL_CANDIDATE = "google/gemma-4-E4B"
FALLBACK_CANDIDATES = [
    "google/gemma-3-4b-it",
    "google/gemma-3-4b-pt",
    "google/gemma-2-9b-it",
    "google/gemma-2-9b",
    "google/gemma-2-2b-it",
    "google/gemma-2-2b",
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("verify_hf_model_id")


def probe_model(api: HfApi, model_id: str) -> dict | None:
    """Fetch config.json metadata for a single model_id.

    Returns None if the repo does not exist or config.json is missing.
    Surfaces other errors to the caller (no silent swallowing).
    """
    try:
        info = api.repo_info(model_id, repo_type="model", timeout=30)
    except RepositoryNotFoundError:
        logger.info("model_id=%s NOT FOUND on Hub", model_id)
        return None
    except RevisionNotFoundError:
        logger.warning("model_id=%s exists but main revision missing", model_id)
        return None

    try:
        config_path = api.hf_hub_download(
            repo_id=model_id,
            filename="config.json",
            repo_type="model",
        )
    except Exception as exc:
        logger.warning(
            "model_id=%s exists but config.json unavailable: %s", model_id, exc
        )
        return {"id": model_id, "exists": True, "config_available": False}

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)

    # Gemma-3+ are multimodal: text-config lives under `text_config`.
    text_cfg = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    effective = {**config, **text_cfg} if text_cfg else config
    effective["multimodal"] = bool(text_cfg)

    license_str = None
    siblings = getattr(info, "siblings", None) or []
    for sib in siblings:
        rfname = getattr(sib, "rfilename", "")
        if rfname in ("LICENSE", "LICENSE.md", "LICENSE.txt"):
            license_str = rfname
            break

    card_data = getattr(info, "cardData", None) or {}
    card_license = card_data.get("license") if isinstance(card_data, dict) else None

    params = (
        effective.get("num_hidden_layers")
        and effective.get("hidden_size")
        and _estimate_params(effective)
    )

    return {
        "id": model_id,
        "exists": True,
        "config_available": True,
        "params": params,
        "num_hidden_layers": effective.get("num_hidden_layers"),
        "hidden_size": effective.get("hidden_size"),
        "num_attention_heads": effective.get("num_attention_heads"),
        "model_type": effective.get("model_type"),
        "multimodal": effective.get("multimodal", False),
        "has_text_config": bool(text_cfg),
        "license_file": license_str,
        "card_license": card_license,
        "sha": getattr(info, "sha", None),
        "last_modified": _iso(getattr(info, "lastModified", None)),
        "private": getattr(info, "private", None),
    }


def _iso(value) -> str | None:
    """Coerce HfApi lastModified (datetime or str) to ISO-8601 string."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.isoformat()
    return str(value)


def _estimate_params(config: dict) -> str | None:
    nl = config.get("num_hidden_layers")
    hs = config.get("hidden_size")
    vsize = config.get("vocab_size")
    if nl is None or hs is None:
        return None
    approx_layers = nl * hs * hs * 12
    embed = (vsize or 0) * hs
    approx = approx_layers + embed
    billion = approx / 1e9
    if billion >= 1.0:
        return f"~{billion:.1f}B (config-derived estimate)"
    return f"~{approx/1e6:.1f}M (config-derived estimate)"


def main() -> int:
    api = HfApi()
    candidates: list[dict] = []

    probe_order = [CANONICAL_CANDIDATE] + FALLBACK_CANDIDATES
    for model_id in probe_order:
        logger.info("probing model_id=%s", model_id)
        try:
            result = probe_model(api, model_id)
        except Exception:
            logger.exception("unexpected error probing model_id=%s", model_id)
            raise
        if result is not None:
            candidates.append(result)

    canonical_found = any(c["id"] == CANONICAL_CANDIDATE for c in candidates)

    if not candidates:
        payload = {
            "status": "NO_CANDIDATES",
            "canonical_candidate": CANONICAL_CANDIDATE,
            "fallback_candidates": FALLBACK_CANDIDATES,
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.error(
            "NO candidates found; wrote %s and exiting 2 per plan-v3.md Phase 0",
            OUTPUT_PATH,
        )
        return 2

    recommendation = (
        CANONICAL_CANDIDATE
        if canonical_found
        else _recommend_fallback(candidates)
    )

    payload = {
        "status": "OK",
        "canonical_candidate": CANONICAL_CANDIDATE,
        "canonical_found": canonical_found,
        "recommendation": recommendation,
        "candidates": candidates,
    }
    OUTPUT_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    LOG_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(
        "wrote %d candidates to %s; recommendation=%s",
        len(candidates),
        OUTPUT_PATH,
        recommendation,
    )
    for c in candidates:
        logger.info(
            "candidate %s: layers=%s hidden=%s license=%s",
            c["id"],
            c.get("num_hidden_layers"),
            c.get("hidden_size"),
            c.get("card_license") or c.get("license_file"),
        )
    return 0


def _recommend_fallback(candidates: list[dict]) -> str | None:
    """Prefer Gemma-3 4B-it (instruction-tuned, closest match to E4B intent)."""
    priority = [
        "google/gemma-3-4b-it",
        "google/gemma-3-4b-pt",
        "google/gemma-2-9b-it",
        "google/gemma-2-2b-it",
        "google/gemma-2-9b",
        "google/gemma-2-2b",
    ]
    candidate_ids = {c["id"] for c in candidates}
    for pid in priority:
        if pid in candidate_ids:
            return pid
    return candidates[0]["id"]


if __name__ == "__main__":
    sys.exit(main())
