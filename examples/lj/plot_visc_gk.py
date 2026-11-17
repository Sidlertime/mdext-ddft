"""
Green-Kubo viscosity convergence plots for a multi-replica measurement
(lj.py --run_type visc_gk --N_replicas 16).

Writes two figures into the run's results folder:
  1. visc_gk_mean_band.png: mean running viscosity over all replicas with a
     +/- 1 std-dev band, plus the final mean_eta +/- stderr_eta.
  2. visc_gk_single_replica.png: the running viscosity of one representative
     replica (final value closest to the mean), without any averaging.

eta is given in reduced LJ units.

Usage:
    python plot_visc_gk.py results/<date>_lj_visc_gk_langevin_01
"""
import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from diffusion import analyze_visc_gk

def load_all_replicas(results_dir):
    with open(os.path.join(results_dir, "config.json")) as f:
        config = json.load(f)
    n_replicas = config["N_replicas"]
    T = config["T"]
    dt = config["dt"]
    volume = 10.0 * 10.0 * 10.0  # box volume of the visc_gk runs in lj.py

    curves = []
    time_actual = None
    for i in range(n_replicas):
        filepath = os.path.join(results_dir, "replicas", f"{i:02d}", "visc_data.txt")
        final_visc, running_visc, t = analyze_visc_gk(filepath, T, volume, dt)
        assert final_visc is not None, f"replica {i} failed to parse"
        curves.append(running_visc)
        if time_actual is None:
            time_actual = t
        else:
            assert len(t) == len(time_actual), f"replica {i} has mismatched length ({len(t)} vs {len(time_actual)})"

    matrix = np.stack(curves)  # (n_replicas, n_lags)
    return time_actual, matrix


def main():
    parser = argparse.ArgumentParser(description="Plot Green-Kubo viscosity convergence of a multi-replica visc_gk run.")
    parser.add_argument("results_dir", help="results folder of an lj.py --run_type visc_gk run")
    results_dir = parser.parse_args().results_dir

    with open(os.path.join(results_dir, "visc_gk_summary.json")) as f:
        summary = json.load(f)
    mean_eta = summary["mean_eta"]
    stderr_eta = summary["stderr_eta"]

    time_actual, matrix = load_all_replicas(results_dir)
    n = matrix.shape[0]
    mean_curve = matrix.mean(axis=0)
    std_curve = matrix.std(axis=0)

    plt.rcParams["font.size"] = 14

    # --- Figure 1: mean + std-dev band ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_actual, mean_curve, color="C0", lw=2, label=f"mean across {n} replicas")
    ax.fill_between(time_actual, mean_curve - std_curve, mean_curve + std_curve,
                     color="C0", alpha=0.25, label=r"$\pm 1$ std-dev across replicas")
    ax.axhline(mean_eta, color="black", lw=1.5, ls=":", label=rf"$\eta$ = {mean_eta:.4f} $\pm$ {stderr_eta:.4f} (LJ units)")
    ax.axhspan(mean_eta - stderr_eta, mean_eta + stderr_eta, color="black", alpha=0.1)
    ax.set_xlabel(r"Correlation lag time $t/\tau$")
    ax.set_ylabel(r"Running viscosity $\eta$ (LJ units)")
    ax.set_title(f"Green-Kubo viscosity: mean across {n} replicas")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "visc_gk_mean_band.png"), dpi=200)
    plt.close(fig)

    # --- Figure 2: single representative replica, no averaging ---
    # Representative replica: final value closest to mean_eta.
    final_vals = matrix[:, -int(matrix.shape[1] * 0.2):].mean(axis=1)
    rep_idx = int(np.argmin(np.abs(final_vals - mean_eta)))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(time_actual, matrix[rep_idx], color="C1", lw=1.6, label=f"replica {rep_idx} (single measurement, no averaging)")
    ax.axhline(mean_eta, color="black", lw=1.5, ls=":", label=rf"$\eta$ = {mean_eta:.4f} ({n}-replica mean, LJ units)")
    ax.set_xlabel(r"Correlation lag time $t/\tau$")
    ax.set_ylabel(r"Running viscosity $\eta$ (LJ units)")
    ax.set_title(f"Green-Kubo viscosity: single replica #{rep_idx}")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(os.path.join(results_dir, "visc_gk_single_replica.png"), dpi=200)
    plt.close(fig)

    print(f"Saved both figures to {results_dir}/")
    print(f"mean_eta = {mean_eta:.4f} +/- {stderr_eta:.4f} LJ units")
    print(f"Representative single replica: #{rep_idx}")


if __name__ == "__main__":
    main()
