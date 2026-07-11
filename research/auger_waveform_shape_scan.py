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
OUT = Path("results_waveforms")
OUT.mkdir(exist_ok=True)


def solve(design: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
    x_train = design[train]
    y_train = target[train]
    scales = np.std(x_train, axis=0)
    scales[scales < 1e-12] = 1.0
    scales[0] = 1.0
    xs = x_train / scales
    ridge = 1e-5 * np.eye(xs.shape[1])
    ridge[0, 0] = 0.0
    beta = np.linalg.solve(xs.T @ xs + ridge, xs.T @ y_train)
    return (design[test] / scales) @ beta


def folds(n: int, seed: int, repeats: int = 4, parts: int = 5):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(repeats):
        order = rng.permutation(n)
        for test in np.array_split(order, min(parts, n)):
            train = np.setdiff1d(np.arange(n), test)
            result.append((train, test))
    return result


def pulse_features(station: dict):
    arrays = [np.asarray(station.get(key, []), dtype=float) for key in ("pmt1", "pmt2", "pmt3")]
    arrays = [array for array in arrays if array.size]
    if not arrays:
        return None
    trace = np.mean(np.vstack(arrays), axis=0)
    baseline = float(np.median(trace[:150]))
    noise = float(np.median(np.abs(trace[:150] - baseline)) * 1.4826 + 1e-9)
    signal = trace - baseline
    smooth = np.convolve(signal, np.ones(5) / 5.0, mode="same")
    peak_bin = int(np.argmax(smooth))
    peak = float(smooth[peak_bin])
    if peak < 8.0 * noise:
        return None

    threshold = max(0.10 * peak, 5.0 * noise)
    left = peak_bin
    while left > 0 and smooth[left] >= threshold:
        left -= 1
    right = peak_bin
    while right < len(smooth) - 1 and smooth[right] >= threshold:
        right += 1
    left += 1
    right -= 1
    if right - left < 2 or right - left > 50:
        return None

    pulse = np.maximum(smooth[left:right + 1], 0.0)
    total = float(pulse.sum())
    if total <= 0:
        return None
    cumulative = np.cumsum(pulse)
    q10 = left + int(np.searchsorted(cumulative, 0.10 * total))
    q50 = left + int(np.searchsorted(cumulative, 0.50 * total))
    q90 = left + int(np.searchsorted(cumulative, 0.90 * total))
    width_bins = max(q90 - q10, 1)
    skew_fraction = float((q90 + q10 - 2 * q50) / width_bins)
    return {
        "width_ns": float(width_bins * 25.0),
        "skew_fraction": skew_fraction,
        "peak_offset_ns": float((peak_bin - float(station.get("signalStartBin", peak_bin))) * 25.0),
        "peak_snr": peak / noise,
    }


def event_rows(data: dict):
    sdrec = data.get("sdrec", {})
    stations = data.get("stations") or sdrec.get("stations") or []
    if any(sdrec.get(key) is None for key in ("theta", "phi", "x", "y")):
        return None

    core_z_values = [float(station.get("z", 0.0)) for station in stations if station.get("z") is not None]
    core = np.array([
        float(sdrec["x"]), float(sdrec["y"]),
        float(sdrec.get("z") or (np.median(core_z_values) if core_z_values else 0.0)),
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

    rows = []
    for station in stations:
        if int(station.get("isSelected", 1)) != 1:
            continue
        features = pulse_features(station)
        if features is None:
            continue
        try:
            position = np.array([float(station["x"]), float(station["y"]), float(station.get("z", 0.0))])
            displacement = position - core
            xp = float(displacement @ e1)
            yp = float(displacement @ e2)
            radius = float(station.get("spDistance") or math.hypot(xp, yp))
            signal = float(station.get("signal", 0.0))
            if radius <= 0 or signal <= 0:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        rows.append({
            "xp": xp,
            "yp": yp,
            "radius": radius,
            "signal": signal,
            "sat": float(station.get("sat", 0.0)),
            **features,
        })
    if len(rows) < 12:
        return None
    frame = pd.DataFrame(rows)
    angle_quarters = 2.0 * np.arctan2(frame.yp.to_numpy(), frame.xp.to_numpy()) / math.pi
    radius = frame.radius.to_numpy()
    reference = float(np.exp(np.mean(np.log(radius))))
    frame["q"] = angle_quarters - np.log(radius / reference) / math.log(PHI)
    return frame, sdrec


def periodic(q: np.ndarray, spacing: float) -> np.ndarray:
    phase = 2.0 * math.pi * q / spacing
    return np.column_stack([np.sin(phase), np.cos(phase)])


def evaluate_response(frame: pd.DataFrame, response: str, spacing_grid: np.ndarray, seed: int):
    radius_log = np.log(frame.radius.to_numpy())
    signal_log = np.log(frame.signal.to_numpy())
    radius_centered = radius_log - radius_log.mean()
    signal_centered = signal_log - signal_log.mean()
    base = np.column_stack([
        np.ones(len(frame)),
        radius_centered,
        radius_centered * radius_centered,
        signal_centered,
        frame.sat.to_numpy(),
    ])
    if response == "width_ns":
        target = np.log(frame[response].to_numpy())
    else:
        target = frame[response].to_numpy()
    target -= target.mean()
    split_list = folds(len(frame), seed)

    def rmse(design):
        errors = []
        for train, test in split_list:
            prediction = solve(design, target, train, test)
            errors.extend(target[test] - prediction)
        return float(np.sqrt(np.mean(np.square(errors))))

    baseline = rmse(base)
    curve = []
    q = frame.q.to_numpy()
    for spacing in spacing_grid:
        curve.append(rmse(np.column_stack([base, periodic(q, float(spacing))])) / baseline)
    return baseline, np.asarray(curve), q, base, target, split_list


def main() -> None:
    response = requests.get(DATA_URL, timeout=180)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    members = [name for name in archive.namelist() if name.endswith(".json")]
    spacing_grid = np.unique(np.sort(np.append(np.linspace(0.40, 2.40, 101), H)))
    h_index = int(np.where(np.isclose(spacing_grid, H))[0][0])

    results = {"width_ns": [], "skew_fraction": [], "peak_offset_ns": []}
    cache = {key: [] for key in results}

    for member_index, member in enumerate(members):
        data = json.loads(archive.read(member))
        extracted = event_rows(data)
        if extracted is None:
            continue
        frame, sdrec = extracted
        event_id = str(data.get("info", {}).get("id") or member)
        for response_name in results:
            baseline, curve, q, base, target, split_list = evaluate_response(
                frame, response_name, spacing_grid, 5000 + member_index
            )
            results[response_name].append({
                "event_id": event_id,
                "stations": len(frame),
                "energy_EeV": sdrec.get("energy"),
                "theta_deg": sdrec.get("theta"),
                "baseline_rmse": baseline,
                "curve": curve,
            })
            cache[response_name].append((q, base, target, split_list, baseline))

    report = {"phi": PHI, "h": H, "responses": {}}
    rng = np.random.default_rng(20260711)

    for response_name, event_list in results.items():
        matrix = np.vstack([event["curve"] for event in event_list])
        aggregate = np.exp(np.mean(np.log(matrix), axis=0))
        median = np.median(matrix, axis=0)
        best_index = int(np.argmin(aggregate))

        # Cross-event validation for a universal spacing.
        selected = []
        unseen_ratio = []
        h_unseen_ratio = []
        count = len(event_list)
        for _ in range(500):
            order = rng.permutation(count)
            cut = max(2, int(0.7 * count))
            train = order[:cut]
            test = order[cut:]
            chosen = int(np.argmin(np.mean(np.log(matrix[train]), axis=0)))
            selected.append(float(spacing_grid[chosen]))
            unseen_ratio.append(float(np.exp(np.mean(np.log(matrix[test, chosen])))))
            h_unseen_ratio.append(float(np.exp(np.mean(np.log(matrix[test, h_index])))))

        # Permutation test for h.
        observed = float(np.mean(np.log(matrix[:, h_index])))
        null = []
        for _ in range(200):
            scores = []
            for q, base, target, split_list, baseline in cache[response_name]:
                q_perm = rng.permutation(q)
                design = np.column_stack([base, periodic(q_perm, H)])
                errors = []
                for train, test in split_list:
                    pred = solve(design, target, train, test)
                    errors.extend(target[test] - pred)
                scores.append(math.log(np.sqrt(np.mean(np.square(errors))) / baseline))
            null.append(float(np.mean(scores)))
        null = np.asarray(null)
        p_value = float((1 + np.sum(null <= observed)) / (len(null) + 1))

        event_frame = pd.DataFrame({
            "event_id": [event["event_id"] for event in event_list],
            "stations": [event["stations"] for event in event_list],
            "baseline_rmse": [event["baseline_rmse"] for event in event_list],
            "h_ratio": matrix[:, h_index],
            "best_spacing": spacing_grid[np.argmin(matrix, axis=1)],
            "best_ratio": np.min(matrix, axis=1),
        })
        event_frame.to_csv(OUT / f"{response_name}_events.csv", index=False)
        pd.DataFrame({
            "spacing": spacing_grid,
            "geometric_mean_ratio": aggregate,
            "median_ratio": median,
        }).to_csv(OUT / f"{response_name}_spacing_scan.csv", index=False)

        report["responses"][response_name] = {
            "events_analyzed": int(count),
            "best_spacing": float(spacing_grid[best_index]),
            "best_geometric_mean_ratio": float(aggregate[best_index]),
            "h_geometric_mean_ratio": float(aggregate[h_index]),
            "h_median_ratio": float(median[h_index]),
            "events_improved_by_h": int(np.sum(matrix[:, h_index] < 1.0)),
            "events_worsened_by_h": int(np.sum(matrix[:, h_index] > 1.0)),
            "h_rank": int(np.sum(aggregate < aggregate[h_index]) + 1),
            "spacing_candidates": int(len(spacing_grid)),
            "cross_event_median_selected_spacing": float(np.median(selected)),
            "cross_event_selected_spacing_iqr": [float(np.quantile(selected, 0.25)), float(np.quantile(selected, 0.75))],
            "cross_event_median_unseen_ratio": float(np.median(unseen_ratio)),
            "cross_event_fraction_improved": float(np.mean(np.asarray(unseen_ratio) < 1.0)),
            "h_cross_event_median_unseen_ratio": float(np.median(h_unseen_ratio)),
            "h_cross_event_fraction_improved": float(np.mean(np.asarray(h_unseen_ratio) < 1.0)),
            "h_permutation_p_value": p_value,
        }

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(spacing_grid, aggregate, linewidth=2, label="Geometric mean")
        ax.plot(spacing_grid, median, linewidth=1.5, label="Median")
        ax.axhline(1.0, linestyle="--", linewidth=1)
        ax.axvline(H, linestyle="--", linewidth=1.2, label=f"h={H:.4f}")
        ax.axvline(spacing_grid[best_index], linestyle=":", linewidth=1.2, label="Global minimum")
        ax.set_xlabel("Log-polar spacing s")
        ax.set_ylabel("Held-out RMSE / baseline RMSE")
        ax.set_title(f"Real PMT pulse feature: {response_name} ({count} showers)")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT / f"{response_name}_spacing_scan.png", dpi=180)
        plt.close(fig)

    (OUT / "waveform_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
