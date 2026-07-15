from __future__ import annotations

import io
import json
import math
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests

PHI = (1.0 + math.sqrt(5.0)) / 2.0
H = 1.318859065462452
DATA_URL = "https://opendata.auger.org/catalog/data.zip"
OUT = Path("results_catalog")
OUT.mkdir(exist_ok=True)


def solve(design: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    x_train = design[train]
    y_train = target[train]
    scales = np.std(x_train, axis=0)
    scales[scales < 1e-12] = 1.0
    scales[0] = 1.0
    xs = x_train / scales
    ridge = 1e-8 * np.eye(xs.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(xs.T @ xs + ridge, xs.T @ y_train)
    return (design[test] / scales) @ beta


def folds(n: int, seed: int, repeats: int = 3, parts: int = 5):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(repeats):
        order = rng.permutation(n)
        for test in np.array_split(order, min(parts, n)):
            train = np.setdiff1d(np.arange(n), test)
            result.append((train, test))
    return result


def event_arrays(data: dict):
    sdrec = data.get("sdrec", {})
    stations = data.get("stations") or sdrec.get("stations") or []
    selected = [station for station in stations if int(station.get("isSelected", 1)) == 1]
    if len(selected) < 12:
        return None
    required = ("theta", "phi", "x", "y")
    if any(sdrec.get(key) is None for key in required):
        return None

    rows = []
    for station in selected:
        try:
            rows.append([
                float(station["x"]), float(station["y"]), float(station.get("z", 0.0)),
                float(station["t"]), float(station.get("spDistance", "nan")),
            ])
        except (KeyError, TypeError, ValueError):
            continue
    array = np.asarray(rows, dtype=float)
    if len(array) < 12:
        return None

    pos = array[:, :3]
    target = array[:, 3]
    target -= np.median(target)

    core = np.array([
        float(sdrec["x"]), float(sdrec["y"]),
        float(sdrec.get("z") or np.median(pos[:, 2])),
    ])
    theta = math.radians(float(sdrec["theta"]))
    azimuth = math.radians(float(sdrec["phi"]))
    direction = np.array([
        math.sin(theta) * math.cos(azimuth),
        math.sin(theta) * math.sin(azimuth),
        math.cos(theta),
    ])
    direction /= np.linalg.norm(direction)
    e1 = np.cross(direction, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(e1) < 1e-9:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    e2 /= np.linalg.norm(e2)

    displacement = pos - core
    xp = displacement @ e1
    yp = displacement @ e2
    computed_radius = np.sqrt(xp * xp + yp * yp)
    supplied_radius = array[:, 4]
    radius = supplied_radius if np.isfinite(supplied_radius).all() else computed_radius
    if np.any(radius <= 0):
        return None

    angle_quarters = 2.0 * np.arctan2(yp, xp) / math.pi
    reference = float(np.exp(np.mean(np.log(radius))))
    q = angle_quarters - np.log(radius / reference) / math.log(PHI)

    centered = pos - pos.mean(axis=0, keepdims=True)
    radius_scale = max(float(np.median(radius)), 1.0)
    r_scaled = radius / radius_scale
    base = np.column_stack([
        np.ones(len(pos)),
        centered / 1000.0,
        r_scaled * r_scaled,
    ])
    event_id = str(data.get("info", {}).get("id") or data.get("meta", {}).get("id") or "unknown")
    return event_id, target, q, base, radius, sdrec


def periodic(q: np.ndarray, spacing: float) -> np.ndarray:
    columns = []
    for harmonic in (1, 2):
        phase = 2.0 * math.pi * harmonic * q / spacing
        columns.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(columns)


def cv_rmse(design: np.ndarray, target: np.ndarray, split_list) -> float:
    errors = []
    for train, test in split_list:
        prediction = solve(design, target, train, test)
        errors.extend(target[test] - prediction)
    return float(np.sqrt(np.mean(np.square(errors))))


def main() -> None:
    response = requests.get(DATA_URL, timeout=180)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    members = [name for name in archive.namelist() if name.endswith(".json")]

    spacing_grid = np.unique(np.sort(np.append(np.linspace(0.40, 2.40, 101), H)))
    records = []
    curves = []
    cached_events = []

    for member_index, member in enumerate(members):
        data = json.loads(archive.read(member))
        arrays = event_arrays(data)
        if arrays is None:
            continue
        event_id, target, q, base, radius, sdrec = arrays
        seed = 1000 + member_index
        split_list = folds(len(target), seed)
        baseline_rmse = cv_rmse(base, target, split_list)
        if not np.isfinite(baseline_rmse) or baseline_rmse <= 0:
            continue

        event_curve = []
        for spacing in spacing_grid:
            design = np.column_stack([base, periodic(q, float(spacing))])
            event_curve.append(cv_rmse(design, target, split_list))
        event_curve = np.asarray(event_curve)
        ratio = event_curve / baseline_rmse
        best_index = int(np.argmin(event_curve))
        h_index = int(np.where(np.isclose(spacing_grid, H))[0][0])

        records.append({
            "event_id": event_id,
            "member": member,
            "stations": len(target),
            "energy_EeV": sdrec.get("energy"),
            "theta_deg": sdrec.get("theta"),
            "baseline_rmse_ns": baseline_rmse,
            "h_rmse_ns": float(event_curve[h_index]),
            "h_ratio": float(ratio[h_index]),
            "best_spacing": float(spacing_grid[best_index]),
            "best_ratio": float(ratio[best_index]),
        })
        curves.append(ratio)
        cached_events.append((target, q, base, split_list, baseline_rmse))

    frame = pd.DataFrame(records)
    ratio_matrix = np.vstack(curves)
    log_ratio = np.log(ratio_matrix)
    aggregate = np.exp(np.mean(log_ratio, axis=0))
    median_ratio = np.median(ratio_matrix, axis=0)
    best_index = int(np.argmin(aggregate))
    h_index = int(np.where(np.isclose(spacing_grid, H))[0][0])

    # Cross-event validation: spacing chosen on training showers, evaluated on unseen showers.
    rng = np.random.default_rng(20260711)
    selected_spacings = []
    test_ratios = []
    h_test_ratios = []
    event_count = len(frame)
    for _ in range(500):
        order = rng.permutation(event_count)
        cut = max(2, int(0.7 * event_count))
        train = order[:cut]
        test = order[cut:]
        train_score = np.mean(log_ratio[train], axis=0)
        chosen = int(np.argmin(train_score))
        selected_spacings.append(float(spacing_grid[chosen]))
        test_ratios.append(float(np.exp(np.mean(log_ratio[test, chosen]))))
        h_test_ratios.append(float(np.exp(np.mean(log_ratio[test, h_index]))))

    # Permutation null for the fixed h coordinate: scramble q within each event.
    observed_h_log_ratio = float(np.mean(log_ratio[:, h_index]))
    permutation_scores = []
    for _ in range(250):
        event_scores = []
        for target, q, base, split_list, baseline_rmse in cached_events:
            q_permuted = rng.permutation(q)
            design = np.column_stack([base, periodic(q_permuted, H)])
            event_scores.append(math.log(cv_rmse(design, target, split_list) / baseline_rmse))
        permutation_scores.append(float(np.mean(event_scores)))
    permutation_scores = np.asarray(permutation_scores)
    p_value = float((1 + np.sum(permutation_scores <= observed_h_log_ratio)) / (len(permutation_scores) + 1))

    frame.to_csv(OUT / "event_results.csv", index=False)
    curve_frame = pd.DataFrame(ratio_matrix, columns=[f"s_{value:.9f}" for value in spacing_grid])
    curve_frame.insert(0, "event_id", frame.event_id)
    curve_frame.to_csv(OUT / "event_spacing_curves.csv", index=False)
    pd.DataFrame({
        "spacing": spacing_grid,
        "geometric_mean_rmse_ratio": aggregate,
        "median_rmse_ratio": median_ratio,
    }).to_csv(OUT / "aggregate_spacing_scan.csv", index=False)

    report = {
        "events_analyzed": int(event_count),
        "minimum_station_count": 12,
        "phi": PHI,
        "h": H,
        "aggregate": {
            "best_spacing": float(spacing_grid[best_index]),
            "best_geometric_mean_rmse_ratio": float(aggregate[best_index]),
            "h_geometric_mean_rmse_ratio": float(aggregate[h_index]),
            "h_median_rmse_ratio": float(median_ratio[h_index]),
            "events_improved_by_h": int(np.sum(frame.h_ratio < 1.0)),
            "events_worsened_by_h": int(np.sum(frame.h_ratio > 1.0)),
            "h_rank": int(np.sum(aggregate < aggregate[h_index]) + 1),
            "spacing_candidates": int(len(spacing_grid)),
        },
        "cross_event_validation": {
            "median_selected_spacing": float(np.median(selected_spacings)),
            "selected_spacing_iqr": [float(np.quantile(selected_spacings, 0.25)), float(np.quantile(selected_spacings, 0.75))],
            "median_unseen_event_ratio_at_selected_spacing": float(np.median(test_ratios)),
            "fraction_splits_selected_spacing_improved_unseen_events": float(np.mean(np.asarray(test_ratios) < 1.0)),
            "median_unseen_event_ratio_at_h": float(np.median(h_test_ratios)),
            "fraction_splits_h_improved_unseen_events": float(np.mean(np.asarray(h_test_ratios) < 1.0)),
        },
        "h_permutation_test": {
            "observed_mean_log_rmse_ratio": observed_h_log_ratio,
            "null_mean": float(np.mean(permutation_scores)),
            "null_standard_deviation": float(np.std(permutation_scores, ddof=1)),
            "one_sided_p_value_for_improvement": p_value,
        },
    }
    (OUT / "catalog_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(spacing_grid, aggregate, linewidth=2, label="Geometric mean across events")
    ax.plot(spacing_grid, median_ratio, linewidth=1.5, label="Median across events")
    ax.axhline(1.0, linestyle="--", linewidth=1)
    ax.axvline(H, linestyle="--", linewidth=1.2, label=f"h={H:.4f}")
    ax.axvline(spacing_grid[best_index], linestyle=":", linewidth=1.2, label="Global minimum")
    ax.set_xlabel("Log-polar spacing s")
    ax.set_ylabel("Held-out RMSE / baseline RMSE")
    ax.set_title(f"Pierre Auger catalog: universal spacing test across {event_count} showers")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "aggregate_spacing_scan.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(frame.h_ratio, bins=24)
    ax.axvline(1.0, linestyle="--", linewidth=1.2)
    ax.set_xlabel("Held-out RMSE ratio after adding h features")
    ax.set_ylabel("Number of showers")
    ax.set_title("Per-event effect of the fixed h coordinate")
    fig.tight_layout()
    fig.savefig(OUT / "h_event_distribution.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(selected_spacings, bins=30)
    ax.axvline(H, linestyle="--", linewidth=1.2, label="h")
    ax.set_xlabel("Spacing selected from 70% of showers")
    ax.set_ylabel("Cross-event split count")
    ax.set_title("Does one spacing recur across independent showers?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "cross_event_selected_spacing.png", dpi=180)
    plt.close(fig)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
