#!/usr/bin/env python3
import argparse
from pathlib import Path

import h5py
import numpy as np
import matplotlib.pyplot as plt


SCALAR_KEYS = [
    "h_star",
    "h_star_hard",
    "h_star_robust",
    "d_obs_hard",
]

CATEGORICAL_KEYS = [
    "proposal_type",
    "safety_region",
]


def collect_dataset_values(h5_path: Path):
    values = {k: [] for k in SCALAR_KEYS + CATEGORICAL_KEYS}

    with h5py.File(h5_path, "r") as f:
        tasks = f["tasks"]

        for task_name in tasks.keys():
            scenes = tasks[task_name]["scenes"]

            for scene_name in scenes.keys():
                scene = scenes[scene_name]

                for key in values.keys():
                    if key in scene:
                        arr = np.asarray(scene[key]).reshape(-1)
                        values[key].append(arr)

        attrs = dict(f.attrs)

    merged = {}
    for key, chunks in values.items():
        if len(chunks) == 0:
            merged[key] = np.asarray([])
        else:
            merged[key] = np.concatenate(chunks, axis=0)

    return merged, attrs


def print_summary(values, attrs):
    print("\n========== DATASET SUMMARY ==========")
    print(f"num_samples attr: {attrs.get('num_samples', 'N/A')}")
    print(f"num_scenes attr:  {attrs.get('num_scenes', 'N/A')}")
    print(f"safety_margin:   {attrs.get('safety_margin', 'N/A')}")
    print(f"h_scale:         {attrs.get('h_scale', 'N/A')}")
    print(f"sampling_mode:   {attrs.get('sampling_mode', 'N/A')}")
    print("=====================================\n")

    for key in SCALAR_KEYS:
        x = values[key]
        if x.size == 0:
            print(f"[WARN] missing key: {key}")
            continue

        finite = x[np.isfinite(x)]
        print(f"{key}")
        print(f"  count: {finite.size}")
        print(f"  min:   {np.min(finite): .5f}")
        print(f"  p01:   {np.percentile(finite, 1): .5f}")
        print(f"  p05:   {np.percentile(finite, 5): .5f}")
        print(f"  p25:   {np.percentile(finite, 25): .5f}")
        print(f"  mean:  {np.mean(finite): .5f}")
        print(f"  p50:   {np.percentile(finite, 50): .5f}")
        print(f"  p75:   {np.percentile(finite, 75): .5f}")
        print(f"  p95:   {np.percentile(finite, 95): .5f}")
        print(f"  p99:   {np.percentile(finite, 99): .5f}")
        print(f"  max:   {np.max(finite): .5f}")
        print()

    for key in CATEGORICAL_KEYS:
        x = values[key]
        if x.size == 0:
            print(f"[WARN] missing key: {key}")
            continue

        unique, counts = np.unique(x.astype(int), return_counts=True)
        total = counts.sum()

        print(f"{key}")
        for u, c in zip(unique, counts):
            print(f"  {u}: {c} samples ({100.0 * c / total:.2f}%)")
        print()


def plot_histograms(values, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)

    for key in SCALAR_KEYS:
        x = values[key]
        if x.size == 0:
            continue

        x = x[np.isfinite(x)]

        plt.figure(figsize=(8, 5))
        plt.hist(x, bins=80)
        plt.axvline(0.0, linestyle="--", linewidth=2)
        plt.xlabel(key + " [m]")
        plt.ylabel("count")
        plt.title(f"Histogram of {key}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}_hist.png", dpi=160)
        plt.close()

    for key in CATEGORICAL_KEYS:
        x = values[key]
        if x.size == 0:
            continue

        x = x.astype(int)
        unique, counts = np.unique(x, return_counts=True)

        plt.figure(figsize=(7, 5))
        plt.bar([str(u) for u in unique], counts)
        plt.xlabel(key)
        plt.ylabel("count")
        plt.title(f"Counts of {key}")
        plt.tight_layout()
        plt.savefig(out_dir / f"{key}_counts.png", dpi=160)
        plt.close()

def print_cross_stats(values):
    h = values["h_star"]
    proposal = values["proposal_type"]
    region = values["safety_region"]

    n = min(len(h), len(proposal), len(region))
    h = h[:n]
    proposal = proposal[:n].astype(int)
    region = region[:n].astype(int)

    print("\n========== CROSS STATS ==========")
    for p in sorted(np.unique(proposal)):
        mask_p = proposal == p
        print(f"\nproposal_type={p} count={mask_p.sum()}")

        for r in [0, 1, 2]:
            c = np.logical_and(mask_p, region == r).sum()
            print(f"  safety_region={r}: {c} ({100*c/max(mask_p.sum(),1):.2f}%)")

        hp = h[mask_p]
        hp = hp[np.isfinite(hp)]
        print(f"  h mean={hp.mean():.4f}")
        print(f"  h min ={hp.min():.4f}")
        print(f"  h max ={hp.max():.4f}")


def print_counts(values, key, label_map=None):
    if key not in values:
        print(f"\n{key}")
        print("  MISSING")
        return

    arr = np.asarray(values[key]).reshape(-1)
    arr = arr[np.isfinite(arr)]

    if arr.size == 0:
        print(f"\n{key}")
        print("  empty")
        return

    arr = arr.astype(int)
    total = int(arr.size)

    print(f"\n{key}")
    for value in sorted(np.unique(arr)):
        count = int(np.sum(arr == value))
        pct = 100.0 * count / max(total, 1)

        if label_map is not None and int(value) in label_map:
            name = label_map[int(value)]
            print(f"  {value} ({name}): {count} samples ({pct:.2f}%)")
        else:
            print(f"  {value}: {count} samples ({pct:.2f}%)")


def print_cross_stats_for_region(values, region_key="safety_region"):
    if "h_star" not in values or "proposal_type" not in values or region_key not in values:
        print(f"\n========== CROSS STATS {region_key} ==========")
        print("missing required keys")
        return

    h = np.asarray(values["h_star"]).reshape(-1)
    proposal = np.asarray(values["proposal_type"]).reshape(-1).astype(int)
    region = np.asarray(values[region_key]).reshape(-1).astype(int)

    n = min(len(h), len(proposal), len(region))
    h = h[:n]
    proposal = proposal[:n]
    region = region[:n]

    print(f"\n========== CROSS STATS {region_key} ==========")
    for p in sorted(np.unique(proposal)):
        mask_p = proposal == p
        print(f"\nproposal_type={p} count={mask_p.sum()}")

        for r in [0, 1, 2]:
            c = np.logical_and(mask_p, region == r).sum()
            print(f"  {region_key}={r}: {c} ({100*c/max(mask_p.sum(),1):.2f}%)")

        hp = h[mask_p]
        hp = hp[np.isfinite(hp)]
        if hp.size > 0:
            print(f"  h mean={hp.mean():.4f}")
            print(f"  h min ={hp.min():.4f}")
            print(f"  h max ={hp.max():.4f}")


def add_derived_safety_region_hard(values, boundary_band):
    if "safety_region_hard" in values:
        return

    if "h_star_hard" not in values:
        return

    h = np.asarray(values["h_star_hard"]).reshape(-1)
    region = np.zeros_like(h, dtype=np.int8)

    region[h < 0.0] = 2
    region[(h >= 0.0) & (h < float(boundary_band))] = 1
    region[h >= float(boundary_band)] = 0

    values["safety_region_hard"] = region

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("h5_path", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("debug/cbf_histograms"))
    args = parser.parse_args()

    values, attrs = collect_dataset_values(args.h5_path)
    print_summary(values, attrs)
    plot_histograms(values, args.out_dir)
    print_cross_stats(values)
    boundary_band = float(attrs.get("boundary_band", 0.06))
    add_derived_safety_region_hard(values, boundary_band)
    print_counts(
    values,
    "proposal_type",
    label_map={
        0: "random_pose_or_random_action",
        1: "near_boundary",
    },
)

    print_counts(
        values,
        "safety_region",
        label_map={
            0: "safe_far",
            1: "near_boundary_safe",
            2: "unsafe_margin",
        },
    )

    print_counts(
        values,
        "safety_region_hard",
        label_map={
            0: "safe_far",
            1: "near_boundary_safe",
            2: "unsafe_margin",
        },
    )
    print(f"\nSaved plots to: {args.out_dir.resolve()}\n")
    print_cross_stats_for_region(values, "safety_region")
    print_cross_stats_for_region(values, "safety_region_hard")

if __name__ == "__main__":
    main()