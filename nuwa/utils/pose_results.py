"""Comparison reports, provenance validation and ablation definitions."""


import csv
import io
import math
from pathlib import Path
import re
from .io import atomic_json, atomic_text
from ..species import normalize_species

ABLATIONS = {
    "activation": {"skp": False, "attention_svd": False, "reconstruction": False},
    "reconstruction": {"skp": False, "attention_svd": False, "reconstruction": True},
    "ofp_without_skp": {"skp": False, "attention_svd": True, "reconstruction": True},
    "without_compensation": {"skp": True, "attention_svd": True, "reconstruction": False},
    "full_nuwa": {"skp": True, "attention_svd": True, "reconstruction": True},
}


def valid_label(label):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", label):
        raise ValueError("Model names must use letters, digits, underscores or hyphens")
    return label


def parse_edges(values):
    cases, names = [], {"baseline", "skp_anchor"}
    for value in values:
        if "=" in value:
            label, path = value.split("=", 1)
        else:
            path = value
            label = Path(path).parent.name
        valid_label(label)
        if label.casefold() in names:
            raise ValueError("Duplicate/reserved model name: {}".format(label))
        names.add(label.casefold())
        cases.append({"label": label, "checkpoint": path})
    return cases


def rate_tag(rate):
    value = float(rate)
    if not math.isfinite(value) or not 0 < value < 1:
        raise ValueError("Pruning rates must be strictly between 0 and 1")
    return format(value, ".9g").replace(".", "p")


def ablation_cases(rates, variants):
    result, labels = [], set()
    for rate in rates:
        tag = rate_tag(rate)
        for variant in variants:
            if variant not in ABLATIONS:
                raise ValueError("Unknown ablation variant: {}".format(variant))
            label = "{}_r{}".format(variant, tag)
            if label in labels:
                raise ValueError("Repeated pruning rate or variant")
            labels.add(label)
            result.append(dict(ABLATIONS[variant], label=label, variant=variant, rate=float(rate)))
    if not result:
        raise ValueError("Select at least one pruning rate and variant")
    return result


def validate_provenance(info, role, species, split_id, base_ids, anchor_ids):
    stage = info.get("stage", "ViTPose-base")
    expected = {"baseline": "ViTPose-base", "anchor": "NuWa-SKP-anchor", "edge": "NuWa-OFP"}[role]
    if stage != expected:
        raise ValueError("Wrong checkpoint type for {}: {}".format(role, stage))
    if role == "baseline":
        return
    if normalize_species(info.get("species")) != normalize_species(species):
        raise ValueError("Checkpoint species differs from the comparison species")
    if info.get("split") != split_id:
        raise ValueError("Checkpoint training split differs from the comparison split ID")
    base_hash = info.get("base_checkpoint_sha256")
    anchor_hash = info.get("source_anchor_sha256")
    if base_hash is not None and base_hash not in base_ids:
        raise ValueError("Checkpoint was derived from different base weights")
    if role == "anchor" and base_hash not in base_ids:
        raise ValueError("Anchor base-checkpoint identity cannot be verified")
    if role == "edge":
        if anchor_hash is not None and anchor_hash not in anchor_ids:
            raise ValueError("OFP checkpoint was derived from a different anchor")
        if base_hash not in base_ids and anchor_hash not in anchor_ids:
            raise ValueError("OFP source checkpoint identity cannot be verified")


def comparison_rows(rows):
    baseline = next((row for row in rows if row["label"] == "baseline"), None)
    if baseline is None:
        raise ValueError("A baseline result is required")
    result = []
    for row in rows:
        if row["latency"].get("device", "cpu") != baseline["latency"].get("device", "cpu"):
            raise ValueError("Latency comparisons require the same device for every model")
        for key in ("parameters", "pose_macs", "checkpoint_bytes", "state_bytes"):
            if row[key] <= 0:
                raise ValueError("Invalid {} for {}".format(key, row["label"]))
        ap, reference_ap = row.get("ap"), baseline.get("ap")
        if ap is not None and (not math.isfinite(ap) or not 0 <= ap <= 1):
            raise ValueError("AP must be a fraction in [0, 1]")
        median = row["latency"]["median_ms"]
        if not math.isfinite(median) or median <= 0:
            raise ValueError("Invalid latency")
        result.append({
            "model": row["label"], "AP": None if ap is None else 100 * ap,
            "delta_AP_points": None if ap is None or reference_ap is None else 100 * (ap - reference_ap),
            "parameters_M": row["parameters"] / 1e6,
            "parameter_reduction_percent": 100 * (1 - row["parameters"] / baseline["parameters"]),
            "checkpoint_MiB": row["checkpoint_bytes"] / 2**20,
            "state_MiB": row["state_bytes"] / 2**20,
            "GMACs": row["pose_macs"] / 1e9,
            "matrix_GFLOPs_2_per_MAC": 2 * row["pose_macs"] / 1e9,
            "MAC_reduction_percent": 100 * (1 - row["pose_macs"] / baseline["pose_macs"]),
            "median_ms": median, "p90_ms": row["latency"]["p90_ms"],
            "speedup": baseline["latency"]["median_ms"] / median,
            "device": row["latency"].get("device", "cpu"),
            "peak_memory_GiB": row["latency"].get("peak_memory_GiB"),
            "smoke_source": bool(row.get("smoke_source", False)),
            "SKP": row.get("components", {}).get("skp"),
            "attention_SVD": row.get("components", {}).get("attention_svd"),
            "reconstruction": row.get("components", {}).get("reconstruction"),
            "target_pruning_rate": row.get("target_pruning_rate"),
        })
    return result


def write_comparison(directory, rows, protocol, status):
    directory = Path(directory)
    summary = comparison_rows(rows)
    atomic_json(directory / "comparison.json", {"schema_version": 2, "status": status,
                "protocol": protocol, "summary": summary, "models": rows})
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
    writer.writeheader()
    writer.writerows(summary)
    atomic_text(directory / "comparison.csv", stream.getvalue())
    def number(value, suffix=""):
        return "--" if value is None else "{:.2f}{}".format(value, suffix)
    device = protocol.get("device", summary[0]["device"])
    lines = ["# ViTPose / NuWa comparison", "", "Status: {}".format(status), "",
             "Device: {}".format(device), "",
             "| Model | AP | Delta AP (points) | Params (M) | Checkpoint (MiB) | GMACs | MAC reduction (%) | Median (ms) | Speedup |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in summary:
        name = row["model"] + (" [smoke source]" if row["smoke_source"] else "")
        lines.append("| " + " | ".join([name, number(row["AP"]), number(row["delta_AP_points"]),
                     number(row["parameters_M"]), number(row["checkpoint_MiB"]),
                     number(row["GMACs"]), number(row["MAC_reduction_percent"]),
                     number(row["median_ms"]), number(row["speedup"], "x")]) + " |")
    lines += ["", "AP uses the same species/split and the original AP-10K evaluator.",
              "Latency is batch-1 heatmap forward on {}: fixed CPU threads, warmup, no flip testing,".format(device),
              "and no data loading, preprocessing or keypoint decoding. AP keeps config flip testing.",
              "GMACs count matrix operations; matrix GFLOPs in CSV/JSON use two FLOPs per MAC.",
              "Checkpoint size uses the exported model weights/metadata and excludes optimizer state.",
              "CUDA timings synchronize the selected device around each measured forward.",
              "A speedup below 1x means the pruned model was slower on the selected device."]
    if protocol.get("smoke_test"):
        lines += ["", "Smoke run: AP was skipped. Timing samples are only a connection check."]
    atomic_text(directory / "comparison.md", "\n".join(lines) + "\n")
    return summary
