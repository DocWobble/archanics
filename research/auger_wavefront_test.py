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
from scipy.stats import pearsonr

PHI = (1.0 + math.sqrt(5.0)) / 2.0
H = 1.318859065462452
EVENT = "PAO110416"
DATA_URL = "https://opendata.auger.org/catalog/data.zip"
OUT = Path("results")
OUT.mkdir(exist_ok=True)


def download_event() -> dict:
    response = requests.get(DATA_URL, timeout=180)
    response.raise_for_status()
    archive = zipfile.ZipFile(io.BytesIO(response.content))
    member = next(name for name in archive.namelist() if name.endswith(f"{EVENT}.json"))
    data = json.loads(archive.read(member))
    (OUT / f"{EVENT}.json").write_text(json.dumps(data), encoding="utf-8")
    return data


def extract_stations(data: dict) -> tuple[pd.DataFrame, dict, list[np.ndarray]]:
    sdrec = data.get("sdrec", {})
    stations = data.get("stations") or sdrec.get("stations") or []
    rows: list[dict] = []
    traces: list[np.ndarray] = []

    for station in stations:
        if int(station.get("isSelected", 1)) != 1:
            continue
        row = {
            key: station.get(key)
            for key in (
                "id", "x", "y", "z", "t", "dt", "signalStartBin",
                "signalStopBin", "signal", "curveResidual",
                "dcurveResidual", "risetime", "spDistance", "sat"
            )
        }
        pmt_arrays = []
        for key in ("pmt1", "pmt2", "pmt3"):
            arr = np.asarray(station.get(key, []), dtype=float)
            if arr.size:
                pmt_arrays.append(arr)
        if pmt_arrays:
            trace = np.mean(np.vstack(pmt_arrays), axis=0)
            baseline = float(np.median(trace[:100]))
            pulse = np.maximum(trace - baseline, 0.0)
            total = pulse.sum()
            cumulative = np.cumsum(pulse)
            if total > 0:
                q10 = int(np.searchsorted(cumulative, 0.10 * total))
                q50 = int(np.searchsorted(cumulative, 0.50 * total))
                q90 = int(np.searchsorted(cumulative, 0.90 * total))
                row.update(
                    waveform_peak_bin=int(np.argmax(pulse)),
                    waveform_centroid_bin=float(np.dot(np.arange(pulse.size), pulse) / total),
                    waveform_q10_bin=q10,
                    waveform_q50_bin=q50,
                    waveform_q90_bin=q90,
                    waveform_width_ns=float((q90 - q10) * 25.0),
                )
            traces.append(trace)
        else:
            traces.append(np.zeros(768))
        rows.append(row)

    frame = pd.DataFrame(rows)
    numeric = [
        "x", "y", "z", "t", "dt", "signal", "curveResidual",
        "dcurveResidual", "risetime", "spDistance", "waveform_width_ns"
    ]
    for column in numeric:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["x", "y", "z", "t"]).reset_index(drop=True)
    return frame, sdrec, traces


def shower_plane_coordinates(frame: pd.DataFrame, sdrec: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    core = np.array([
        float(sdrec.get("x", frame.x.mean())),
        float(sdrec.get("y", frame.y.mean())),
        float(sdrec.get("z", frame.z.mean())),
    ])
    theta = math.radians(float(sdrec.get("theta", 0.0)))
    azimuth = math.radians(float(sdrec.get("phi", 0.0)))
    direction = np.array([
        math.sin(theta) * math.cos(azimuth),
        math.sin(theta) * math.sin(azimuth),
        math.cos(theta),
    ])
    direction /= np.linalg.norm(direction)

    vertical = np.array([0.0, 0.0, 1.0])
    e1 = np.cross(direction, vertical)
    if np.linalg.norm(e1) < 1e-8:
        e1 = np.array([1.0, 0.0, 0.0])
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(direction, e1)
    e2 /= np.linalg.norm(e2)

    positions = frame[["x", "y", "z"]].to_numpy(float) - core
    xp = positions @ e1
    yp = positions @ e2
    radius = np.sqrt(xp * xp + yp * yp)
    return xp, yp, radius


def solve_linear(design: np.ndarray, target: np.ndarray, train: np.ndarray, test: np.ndarray) -> np.ndarray:
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


def folds(n: int, repeats: int = 12, parts: int = 5, seed: int = 110416):
    rng = np.random.default_rng(seed)
    result = []
    for _ in range(repeats):
        order = rng.permutation(n)
        chunks = np.array_split(order, parts)
        for test in chunks:
            train = np.setdiff1d(np.arange(n), test)
            result.append((train, test))
    return result


def design_plane(pos: np.ndarray) -> np.ndarray:
    centered = pos - pos.mean(axis=0, keepdims=True)
    return np.column_stack([np.ones(len(pos)), centered / 1000.0])


def hyper_feature(radius: np.ndarray, length: float) -> np.ndarray:
    return (np.sqrt(radius * radius + length * length) - length) / 1000.0


def choose_hyper_length(base: np.ndarray, radius: np.ndarray, target: np.ndarray, train: np.ndarray) -> float:
    candidates = np.geomspace(100.0, 30000.0, 32)
    rng = np.random.default_rng(int(train.sum()) + len(train))
    local_order = rng.permutation(train)
    chunks = np.array_split(local_order, 4)
    scores = []
    for length in candidates:
        design = np.column_stack([base, hyper_feature(radius, length)])
        errors = []
        for test_inner in chunks:
            train_inner = np.setdiff1d(train, test_inner)
            pred = solve_linear(design, target, train_inner, test_inner)
            errors.extend(target[test_inner] - pred)
        scores.append(np.sqrt(np.mean(np.square(errors))))
    return float(candidates[int(np.argmin(scores))])


def q_coordinate(xp: np.ndarray, yp: np.ndarray, radius: np.ndarray) -> np.ndarray:
    angle_quarters = 2.0 * np.arctan2(yp, xp) / math.pi
    positive = radius[radius > 0]
    reference = float(np.exp(np.mean(np.log(positive))))
    return angle_quarters - np.log(np.maximum(radius, 1e-9) / reference) / math.log(PHI)


def periodic_columns(q: np.ndarray, spacing: float, harmonics: int = 2) -> np.ndarray:
    columns = []
    for harmonic in range(1, harmonics + 1):
        phase = 2.0 * math.pi * harmonic * q / spacing
        columns.extend([np.sin(phase), np.cos(phase)])
    return np.column_stack(columns)


def evaluate(frame: pd.DataFrame, sdrec: dict) -> dict:
    pos = frame[["x", "y", "z"]].to_numpy(float)
    target = frame["t"].to_numpy(float)
    target -= np.median(target)
    xp, yp, radius_computed = shower_plane_coordinates(frame, sdrec)
    radius = frame["spDistance"].to_numpy(float) if frame.get("spDistance") is not None and frame["spDistance"].notna().all() else radius_computed
    q = q_coordinate(xp, yp, radius)
    base = design_plane(pos)
    split_list = folds(len(frame))

    model_names = ["plane", "hyperbolic", "hyperbolic+h", "hyperbolic+free-spacing"]
    errors = {name: [] for name in model_names}
    predictions = {name: np.zeros(len(frame)) for name in model_names}
    counts = np.zeros(len(frame))
    chosen_lengths = []
    chosen_spacings = []

    spacing_grid = np.linspace(0.45, 2.40, 100)

    for train, test in split_list:
        length = choose_hyper_length(base, radius, target, train)
        chosen_lengths.append(length)
        hyper = np.column_stack([base, hyper_feature(radius, length)])

        pred_plane = solve_linear(base, target, train, test)
        pred_hyper = solve_linear(hyper, target, train, test)

        h_design = np.column_stack([hyper, periodic_columns(q, H)])
        pred_h = solve_linear(h_design, target, train, test)

        # Choose the spacing only from training data using a deterministic inner split.
        inner_order = train.copy()
        inner_chunks = np.array_split(inner_order, 4)
        spacing_scores = []
        for spacing in spacing_grid:
            design = np.column_stack([hyper, periodic_columns(q, float(spacing))])
            inner_errors = []
            for inner_test in inner_chunks:
                inner_train = np.setdiff1d(train, inner_test)
                inner_pred = solve_linear(design, target, inner_train, inner_test)
                inner_errors.extend(target[inner_test] - inner_pred)
            spacing_scores.append(np.sqrt(np.mean(np.square(inner_errors))))
        spacing = float(spacing_grid[int(np.argmin(spacing_scores))])
        chosen_spacings.append(spacing)
        free_design = np.column_stack([hyper, periodic_columns(q, spacing)])
        pred_free = solve_linear(free_design, target, train, test)

        fold_predictions = {
            "plane": pred_plane,
            "hyperbolic": pred_hyper,
            "hyperbolic+h": pred_h,
            "hyperbolic+free-spacing": pred_free,
        }
        for name, prediction in fold_predictions.items():
            errors[name].extend((target[test] - prediction).tolist())
            predictions[name][test] += prediction
        counts[test] += 1

    for name in predictions:
        predictions[name] /= np.maximum(counts, 1.0)

    metrics = {}
    for name in model_names:
        residual = np.asarray(errors[name])
        metrics[name] = {
            "rmse_ns": float(np.sqrt(np.mean(residual * residual))),
            "mae_ns": float(np.mean(np.abs(residual))),
            "median_abs_error_ns": float(np.median(np.abs(residual))),
        }

    # Fixed-spacing diagnostic: every candidate gets the same folds and same baseline.
    fixed_length = float(np.median(chosen_lengths))
    hyper_fixed = np.column_stack([base, hyper_feature(radius, fixed_length)])
    scan_rmse = []
    for spacing in spacing_grid:
        design = np.column_stack([hyper_fixed, periodic_columns(q, float(spacing))])
        scan_errors = []
        for train, test in split_list:
            pred = solve_linear(design, target, train, test)
            scan_errors.extend(target[test] - pred)
        scan_rmse.append(float(np.sqrt(np.mean(np.square(scan_errors)))))
    scan_rmse = np.asarray(scan_rmse)
    best_index = int(np.argmin(scan_rmse))
    h_rmse = float(np.interp(H, spacing_grid, scan_rmse))
    h_rank = int(np.sum(scan_rmse < h_rmse) + 1)

    # Simple permutation null for the incremental h feature using fixed splits.
    rng = np.random.default_rng(202606)
    observed_improvement = metrics["hyperbolic"]["rmse_ns"] - metrics["hyperbolic+h"]["rmse_ns"]
    null_improvements = []
    for _ in range(300):
        q_perm = rng.permutation(q)
        design_perm = np.column_stack([hyper_fixed, periodic_columns(q_perm, H)])
        perm_errors = []
        base_errors = []
        for train, test in split_list[:10]:
            pred_perm = solve_linear(design_perm, target, train, test)
            pred_base = solve_linear(hyper_fixed, target, train, test)
            perm_errors.extend(target[test] - pred_perm)
            base_errors.extend(target[test] - pred_base)
        null_improvements.append(
            np.sqrt(np.mean(np.square(base_errors))) - np.sqrt(np.mean(np.square(perm_errors)))
        )
    permutation_p = float((1 + np.sum(np.asarray(null_improvements) >= observed_improvement)) / 301.0)

    result = {
        "event": EVENT,
        "station_count": int(len(frame)),
        "phi": PHI,
        "h": H,
        "metrics": metrics,
        "median_selected_hyperbolic_length_m": fixed_length,
        "free_spacing_median": float(np.median(chosen_spacings)),
        "free_spacing_iqr": [float(np.quantile(chosen_spacings, 0.25)), float(np.quantile(chosen_spacings, 0.75))],
        "fixed_spacing_scan": {
            "best_spacing": float(spacing_grid[best_index]),
            "best_rmse_ns": float(scan_rmse[best_index]),
            "h_rmse_ns": h_rmse,
            "h_rank_of_100": h_rank,
        },
        "h_incremental_improvement_ns": float(observed_improvement),
        "permutation_p_value": permutation_p,
    }

    frame = frame.copy()
    frame["xp_m"] = xp
    frame["yp_m"] = yp
    frame["radius_m"] = radius
    frame["q"] = q
    frame["q_mod_h"] = np.mod(q, H)
    frame["t_centered_ns"] = target
    for name, prediction in predictions.items():
        frame[f"pred_{name}_ns"] = prediction
        frame[f"resid_{name}_ns"] = target - prediction
    frame.to_csv(OUT / "stations_derived.csv", index=False)
    pd.DataFrame({"spacing": spacing_grid, "cv_rmse_ns": scan_rmse}).to_csv(OUT / "spacing_scan.csv", index=False)

    make_figure(frame, traces_global, result, spacing_grid, scan_rmse, predictions)
    return result


def make_figure(frame: pd.DataFrame, traces: list[np.ndarray], result: dict, spacing_grid: np.ndarray, scan_rmse: np.ndarray, predictions: dict) -> None:
    target = frame["t_centered_ns"].to_numpy()
    fig, axes = plt.subplots(2, 3, figsize=(18, 11), constrained_layout=True)

    scatter = axes[0, 0].scatter(frame.xp_m, frame.yp_m, c=frame.resid_plane_ns, s=55)
    axes[0, 0].scatter([0], [0], marker="+", s=100)
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].set_title("Shower-plane stations\ncolor = held-out plane residual")
    axes[0, 0].set_xlabel("x' (m)")
    axes[0, 0].set_ylabel("y' (m)")
    fig.colorbar(scatter, ax=axes[0, 0], label="ns")

    names = ["plane", "hyperbolic", "hyperbolic+h", "hyperbolic+free-spacing"]
    labels = ["Plane", "Hyperbolic", "Hyperbolic + h", "Hyperbolic + free s"]
    values = [result["metrics"][name]["rmse_ns"] for name in names]
    axes[0, 1].bar(labels, values)
    axes[0, 1].tick_params(axis="x", rotation=25)
    axes[0, 1].set_ylabel("Repeated held-out RMSE (ns)")
    axes[0, 1].set_title("Out-of-sample timing error")

    axes[0, 2].plot(spacing_grid, scan_rmse, linewidth=2)
    axes[0, 2].axvline(H, linestyle="--", label=f"h={H:.4f}")
    axes[0, 2].axvline(result["fixed_spacing_scan"]["best_spacing"], linestyle=":", label="best fixed spacing")
    axes[0, 2].set_xlabel("Log-polar spacing s")
    axes[0, 2].set_ylabel("Held-out RMSE (ns)")
    axes[0, 2].set_title("Blind spacing scan")
    axes[0, 2].legend()

    axes[1, 0].scatter(target, predictions["hyperbolic"], s=38, label="Hyperbolic")
    axes[1, 0].scatter(target, predictions["hyperbolic+h"], s=38, label="Hyperbolic+h")
    low, high = float(target.min()), float(target.max())
    axes[1, 0].plot([low, high], [low, high], linestyle="--")
    axes[1, 0].set_xlabel("Observed centered start time (ns)")
    axes[1, 0].set_ylabel("Mean held-out prediction (ns)")
    axes[1, 0].set_title("Observed vs predicted")
    axes[1, 0].legend()

    axes[1, 1].scatter(frame.q_mod_h, frame.resid_hyperbolic_ns, s=38)
    axes[1, 1].set_xlabel("q modulo h")
    axes[1, 1].set_ylabel("Hyperbolic held-out residual (ns)")
    axes[1, 1].set_title("Residual timing field in h-phase")

    selected = np.argsort(frame.signal.to_numpy())[-min(8, len(frame)):]
    for index in selected:
        if index < len(traces):
            trace = traces[index]
            if trace.size:
                trace = trace - np.median(trace[:100])
                scale = max(np.max(np.abs(trace)), 1e-9)
                axes[1, 2].plot(np.arange(trace.size) * 25.0, trace / scale, alpha=0.75)
    axes[1, 2].set_xlabel("FADC time (ns)")
    axes[1, 2].set_ylabel("Normalized PMT amplitude")
    axes[1, 2].set_title("Strongest real station waveforms")

    fig.suptitle(f"Pierre Auger {EVENT}: real irregular-array wavefront test", fontsize=16)
    fig.savefig(OUT / "auger_wavefront_test.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    data = download_event()
    frame, sdrec, traces_global = extract_stations(data)
    if len(frame) < 12:
        raise RuntimeError(f"Expected a many-station event; got {len(frame)} selected stations")
    report = evaluate(frame, sdrec)
    report["sdrec"] = {key: sdrec.get(key) for key in ("theta", "phi", "x", "y", "z", "R", "dR", "nbstat", "energy")}
    report["waveform_width_correlation_with_radius"] = None
    if "waveform_width_ns" in frame and frame.waveform_width_ns.notna().sum() >= 5:
        valid = frame.waveform_width_ns.notna() & frame.spDistance.notna()
        correlation, p_value = pearsonr(frame.loc[valid, "waveform_width_ns"], frame.loc[valid, "spDistance"])
        report["waveform_width_correlation_with_radius"] = {"r": float(correlation), "p": float(p_value)}
    (OUT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
