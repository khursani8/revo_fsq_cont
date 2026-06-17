"""Phase 0a/0b smoke test: load pilot or final model, run two deterministic
forward passes, verify hidden-state shapes and bitwise determinism.

Per plan-v3.md Phase 0 acceptance criteria + 2026-06-17 OQ-1/OQ-5 addendum:
  - selects model via --model {pilot|final} (default pilot)
  - prints num_hidden_layers, hidden_size, attention impl, hook tap points
  - asserts num_hidden_layers matches configs/model.yaml
  - runs forward pass TWICE under torch.manual_seed(0); asserts bitwise-equal
  - logs structured JSON to logs/phase0a_smoke_test_pilot.json or
    logs/phase0b_smoke_test_final.json
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

ROOT = Path("/mnt/data/work/fsq_cont")
MODEL_CONFIG = ROOT / "configs" / "model.yaml"
LOG_DIR = ROOT / "logs"
PROBE_PROMPT = "The quick brown fox"

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def resolve_layer_module(model, multimodal):
    """Return the decoder-layer container and a human-readable module path.

    Gemma-2 (non-multimodal): model.model.layers
    Gemma-4-E4B (multimodal):  model.model.language_model.layers
        (Gemma4ForConditionalGeneration.model is Gemma4Model, which wraps
        Gemma4TextModel at .language_model; .layers lives on the text model.)
    """
    if multimodal:
        inner = getattr(model, "model", None)
        if inner is not None:
            lm = getattr(inner, "language_model", None)
            if lm is not None and hasattr(lm, "layers"):
                return lm.layers, "model.model.language_model.layers"
        if hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "model") and hasattr(lm.model, "layers"):
                return lm.model.layers, "model.language_model.model.layers"
            if hasattr(lm, "layers"):
                return lm.layers, "model.language_model.layers"
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers, "model.model.layers"
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h, "model.transformer.h"
    raise AttributeError(
        f"Could not locate decoder layers container on model of type "
        f"{type(model).__name__} (multimodal={multimodal})"
    )


def resolve_config(model, multimodal):
    """Return the text config that carries hidden_size / num_hidden_layers."""
    if multimodal and hasattr(model.config, "text_config"):
        return model.config.text_config
    return model.config


def attach_tap_hooks(layers_module, layers_path, hook_layers, hook_signals):
    """Register forward hooks on the chosen tap points.

    Tap point conventions (per plan-v3.md Option B):
      - residual:      decoder block i (residual stream post-block)
      - mlp_out:       MLP sublayer of block i
      - attention_out: self-attention sublayer of block i
    """
    captured = {}
    handles = []

    def make_hook(layer_idx, signal):
        def hook_fn(module, inputs, output):
            tensor = output[0] if isinstance(output, tuple) else output
            captured[(layer_idx, signal)] = {
                "shape": tuple(tensor.shape),
                "dtype": str(tensor.dtype),
            }
        return hook_fn

    for layer_idx in hook_layers:
        if layer_idx >= len(layers_module):
            raise IndexError(
                f"hook layer {layer_idx} >= num_hidden_layers "
                f"{len(layers_module)}; re-scale hook_layers in configs/model.yaml"
            )
        block = layers_module[layer_idx]
        signal_to_attr = {
            "residual": None,
            "mlp_out": "mlp",
            "attention_out": "self_attn",
        }
        for signal in hook_signals:
            attr = signal_to_attr.get(signal)
            module = block if attr is None else getattr(block, attr, None)
            if module is None:
                logger.warning(
                    "Signal %s not available on block %s; skipping",
                    signal, type(block).__name__,
                )
                continue
            handles.append(module.register_forward_hook(make_hook(layer_idx, signal)))
    return captured, handles


def run_once(model, inputs):
    """Forward pass under inference_mode; returns dict of raw last-layer logits."""
    with torch.inference_mode():
        out = model(**inputs)
    logits = out.logits if hasattr(out, "logits") else out[0]
    return logits.detach().cpu().clone()


def main():
    parser = argparse.ArgumentParser(description="Phase 0a/0b smoke test for FSQ plan-v3.")
    parser.add_argument("--model", choices=["pilot", "final"], default="pilot",
                        help="Which model section to load from configs/model.yaml.")
    parser.add_argument("--log-path", type=str, default=None,
                        help="Override structured JSON log path.")
    args = parser.parse_args()

    cfg = load_yaml(MODEL_CONFIG)
    section = cfg[args.model]
    model_id = section["model_id"]
    expected_hidden_size = int(section["hidden_size"])
    expected_num_hidden_layers = int(section["num_hidden_layers"])
    hook_layers = list(section["hook_layers"])
    hook_signals = list(section["hook_signals"])
    multimodal = bool(section.get("multimodal", False))
    dtype_str = str(section.get("dtype", "bfloat16"))
    attention_impl = str(section.get("attention_impl", "sdpa"))
    device_map = section.get("device_map", "auto")
    torch_dtype = DTYPE_MAP.get(dtype_str, torch.bfloat16)

    if args.log_path:
        log_path = Path(args.log_path)
    else:
        suffix = "pilot" if args.model == "pilot" else "final"
        prefix = "phase0a" if args.model == "pilot" else "phase0b"
        log_path = LOG_DIR / f"{prefix}_smoke_test_{suffix}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    result = {
        "model_section": args.model,
        "model_id": model_id,
        "multimodal": multimodal,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "probe_prompt": PROBE_PROMPT,
        "expected_hidden_size": expected_hidden_size,
        "expected_num_hidden_layers": expected_num_hidden_layers,
        "hook_layers": hook_layers,
        "hook_signals": hook_signals,
        "requested_dtype": dtype_str,
        "requested_attention_impl": attention_impl,
        "device_map": device_map,
        "seed": 0,
        "passed": False,
        "errors": [],
        "warnings": [],
    }

    logger.info("Loading tokenizer for %s", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id, extra_special_tokens={})

    logger.info("Loading model %s (dtype=%s, attn_impl=%s, device_map=%s, multimodal=%s)",
                model_id, dtype_str, attention_impl, device_map, multimodal)
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        attn_implementation=attention_impl,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    result["model_load_sec"] = time.perf_counter() - t0

    text_config = resolve_config(model, multimodal)
    actual_hidden_size = int(text_config.hidden_size)
    actual_num_hidden_layers = int(text_config.num_hidden_layers)
    actual_attention_impl = getattr(model.config, "attn_implementation", None)
    actual_multimodal = hasattr(model.config, "text_config")

    result["actual_hidden_size"] = actual_hidden_size
    result["actual_num_hidden_layers"] = actual_num_hidden_layers
    result["actual_attention_impl"] = actual_attention_impl
    result["actual_multimodal_detected"] = actual_multimodal

    if actual_num_hidden_layers != expected_num_hidden_layers:
        msg = (f"num_hidden_layers mismatch: actual={actual_num_hidden_layers} "
               f"expected={expected_num_hidden_layers}. Update configs/model.yaml.")
        result["errors"].append(msg)
        logger.error(msg)

    if actual_hidden_size != expected_hidden_size:
        msg = (f"hidden_size mismatch: actual={actual_hidden_size} "
               f"expected={expected_hidden_size}. Update configs/model.yaml.")
        result["warnings"].append(msg)
        logger.warning(msg)

    try:
        layers_module, layers_path = resolve_layer_module(model, multimodal)
    except AttributeError as exc:
        result["errors"].append(f"layer-module resolution failed: {exc}")
        logger.exception("layer-module resolution failed")
        _finalize(log_path, result)
        sys.exit(1)
    result["layer_module_path"] = layers_path
    result["layer_count_on_module"] = len(layers_module)

    if len(layers_module) != actual_num_hidden_layers:
        msg = (f"resolved module {layers_path} has {len(layers_module)} layers "
               f"but config reports num_hidden_layers={actual_num_hidden_layers}")
        result["warnings"].append(msg)
        logger.warning(msg)

    max_hook = max(hook_layers)
    if max_hook >= len(layers_module):
        msg = (f"hook_layers={hook_layers} exceed module layer count "
               f"{len(layers_module)} at {layers_path}. Re-scale hook_layers.")
        result["errors"].append(msg)
        logger.error(msg)

    if result["errors"]:
        _finalize(log_path, result)
        logger.error("Aborting before forward pass due to config errors. See %s", log_path)
        sys.exit(1)

    inputs = tokenizer(PROBE_PROMPT, return_tensors="pt").to(model.device)
    captured, handles = attach_tap_hooks(
        layers_module, layers_path, hook_layers, hook_signals,
    )

    try:
        torch.manual_seed(0)
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            logger.warning("use_deterministic_algorithms(True) not supported; continuing")

        t1 = time.perf_counter()
        logits1 = run_once(model, inputs)
        t2 = time.perf_counter()
        logits2 = run_once(model, inputs)
        result["forward_pass_1_sec"] = t2 - t1
        result["forward_pass_2_sec"] = time.perf_counter() - t2
    finally:
        for h in handles:
            h.remove()

    bitwise_equal = bool(torch.equal(logits1, logits2))
    max_abs_diff = float((logits1 - logits2).abs().max())
    result["determinism"] = {
        "logits_bitwise_equal": bitwise_equal,
        "logits_max_abs_diff": max_abs_diff,
    }
    if not bitwise_equal:
        msg = (f"Nondeterministic forward pass: logits differ (max_abs_diff="
               f"{max_abs_diff}). Scenario 2 pre-mortem applies.")
        result["errors"].append(msg)
        logger.error(msg)

    tap_points = []
    for layer_idx in hook_layers:
        for signal in hook_signals:
            entry = captured.get((layer_idx, signal))
            if entry is None:
                result["warnings"].append(
                    f"No capture for (layer={layer_idx}, signal={signal})"
                )
                tap_points.append({
                    "layer": layer_idx, "signal": signal, "captured": False,
                    "module_path": None,
                })
            else:
                submodule_attr = {"residual": None,
                                  "mlp_out": "mlp",
                                  "attention_out": "self_attn"}[signal]
                mod_path = (f"{layers_path}[{layer_idx}]"
                            if submodule_attr is None
                            else f"{layers_path}[{layer_idx}].{submodule_attr}")
                tap_points.append({
                    "layer": layer_idx, "signal": signal, "captured": True,
                    "shape": entry["shape"], "dtype": entry["dtype"],
                    "module_path": mod_path,
                })
    result["tap_points"] = tap_points

    logger.info("model_id=%s multimodal=%s num_hidden_layers=%d hidden_size=%d attention_impl=%s",
                model_id, actual_multimodal, actual_num_hidden_layers,
                actual_hidden_size, actual_attention_impl)
    logger.info("layer_module_path=%s", layers_path)
    logger.info("Tap points (layer, signal, shape):")
    for tp in tap_points:
        if tp["captured"]:
            logger.info("  layer=%d signal=%s shape=%s dtype=%s path=%s",
                        tp["layer"], tp["signal"], tp["shape"], tp["dtype"],
                        tp["module_path"])

    result["passed"] = (
        actual_num_hidden_layers == expected_num_hidden_layers
        and all(tp["captured"] for tp in tap_points)
        and bitwise_equal
        and not result["errors"]
    )

    _finalize(log_path, result)
    logger.info("Wrote structured result to %s (passed=%s)", log_path, result["passed"])
    sys.exit(0 if result["passed"] else 1)


def _finalize(log_path, result):
    result["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    with open(log_path, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
