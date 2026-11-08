"""Single-site LJ fluid test with roughly the density of water at STP."""
import mdext
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from datetime import datetime
import os
import random

from lammps import PyLammps
from mdext import log
from utils import plot, plot_movie, avg_results_parallel
from diffusion import analyze_msd, analyze_vacf, analyze_visc_gk

def get_next_result_path(base_name="results/lj"):
    """
    Finds the next available directory name.
    If lj_01 exists, it returns lj_02, etc.
    """
    counter = 1
    while True:
        # Format the path with leading zeros for better sorting (01, 02, etc.)
        path = f"{base_name}_{counter:02d}"
        if not os.path.exists(path):
            return path
        counter += 1

def main(args) -> None:

    # Current simulation parameters:
    T = args["T"]  # K
    P = args["P"]  # atm
    U0 = args["U0"]   # Amplitude of the external potential (kcal/mol)
    sigma = args["sigma"] # Width of the external potential (A)
    mu = args["mu"]
    equilibration_steps = args["equilibration_steps"] # equilibration cycles
    collection_steps = args["collection_steps"] # collection cycles
    Tdamp = args["Tdamp"]
    h5_file = args.get("h5_file")

    # Determine which potential to use for the active collection phase
    if h5_file and os.path.exists(h5_file):
        log.info(f"Using InterpolatedPotential from {h5_file}")
        active_potential = mdext.potential.InterpolatedPotential(h5_file)
    else:
        log.info(f"Using Gaussian potential (U0={U0}, sigma={sigma}, mu={mu})")
        active_potential = mdext.potential.Gaussian(U0, sigma, mu)

    zero_potential = mdext.potential.Gaussian(0, mu, sigma)

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    current_day = datetime.now().strftime("%Y-%m-%d")
    results_path = get_next_result_path(f"results/{current_day}_{args["run_id"]}_{args["integrator"]}")
    os.makedirs(results_path, exist_ok=True)

    # config
    if args["run_type"] == "collection":
        del args["diffusion_steps"]
    else:
        del args["trials"]
        del args["collection_steps"]

    if args["integrator"] == "nve":
        # Pure nve has no T, P, or Tdamp
        del args["T"]
        del args["P"]
        del args["Tdamp"]
    args['time'] = current_time
    save_path = os.path.join(results_path, "config.json")
    with open(save_path, 'w') as f:
        json.dump(args, f, indent=4)

    if args["run_type"] == "collection":
        # Initialize and run simulation:
        for i in range(0, args["trials"]):
            md = mdext.md.MD(
                setup=setup,
                integrator=args["integrator"],
                T=T,
                P=P,
                seed=i+1,
                potential=zero_potential,
                geometry_type=mdext.geometry.Planar,
                n_atom_types=1,
                potential_type=1,
                steps_per_thermo=10,
                thermo_per_cycle=1,
                units="lj",
                timestep=args["dt"],
                dr=args["dr"],
                Tdamp=Tdamp,
            )

            os.makedirs(f"{results_path}/trials/{i}", exist_ok=True)
            md.run(equilibration_steps, "equilibration", f"{results_path}/trials/{i}/equilibration")
            md.reset_stats()
            md.reset_vext(active_potential)
            md.run(collection_steps, "collection", f"{results_path}/trials/{i}/collection")
            plot(f"{results_path}/trials/{i}")
        
        avg_results_parallel(results_path, args["trials"], collection_steps)
        plot_movie(f"{results_path}/avg_{args["trials"]}")
    elif args["run_type"] == "visc_gk":
        # Multiple independent replicas (different seed each time, same per-replica
        # equilibration + NVE-swap + Green-Kubo sampling pattern as the single-run path
        # that used to live here) to get an honest mean +/- stderr instead of one run's
        # naive "mean of the final 20%" plateau with no uncertainty quantification -
        # see reference_mdext_ddft.md for why this was flagged as the real gap in the
        # methodology (NVE-after-Langevin itself is confirmed correct per LAMMPS's own
        # Green-Kubo example and the literature, so that part is unchanged).
        n_replicas = args["N_replicas"]
        s, p, d = 2, 5000, 10000  # unchanged from the original single-run sampling/correlation/dump params
        replica_etas = []
        replica_curves = []  # (time_actual, running_viscosity) per valid replica, for the combined plot
        for rep in range(n_replicas):
            replica_path = f"{results_path}/replicas/{rep:02d}"
            os.makedirs(replica_path, exist_ok=True)
            md = mdext.md.MD(
                setup=setup,
                integrator=args["integrator"],
                T=T,
                P=P,
                seed=rep + 1,
                potential=zero_potential,
                geometry_type=mdext.geometry.Planar,
                n_atom_types=1,
                potential_type=1,
                steps_per_thermo=10,
                thermo_per_cycle=1,
                units="lj",
                timestep=args["dt"],
                dr=args["dr"],
                Tdamp=Tdamp,
            )
            md.run(args["equilibration_steps"], "equilibration")
            md.lmp.command("reset_timestep 0")
            # Disable thermostat und use NVE
            md.lmp.command("unfix Ensemble")
            md.lmp.command("fix NVE all nve")

            # Force evaluation of pressure tensor components:
            md.lmp.command("compute myTemp all temp")
            md.lmp.command("compute myPress all pressure myTemp")
            md.lmp.command("variable pxy equal c_myPress[4]")
            md.lmp.command("variable pxz equal c_myPress[5]")
            md.lmp.command("variable pyz equal c_myPress[6]")

            # Run the autocorrelation over NVE trajectory
            md.lmp.command(
                f"fix GK_visc all ave/correlate {s} {p} {d} v_pxy v_pxz v_pyz "
                f"type auto file {replica_path}/visc_data.txt ave running"
            )
            md.run(args["diffusion_steps"], "viscosity_test")

            # Analyze results using actual simulation box volume
            boxlo, boxhi = md.lps.extract_box()[:2]
            volume = float(np.prod(np.array(boxhi) - np.array(boxlo)))
            final_viscosity, running_viscosity, time_actual = analyze_visc_gk(
                f"{replica_path}/visc_data.txt", T, volume, args["dt"]
            )
            if final_viscosity is None:
                print(f"[replica {rep+1}/{n_replicas}] FAILED (no valid data block) - skipping")
                continue
            print(f"[replica {rep+1}/{n_replicas}] eta = {final_viscosity:.4f}")
            replica_etas.append(final_viscosity)
            replica_curves.append((time_actual, running_viscosity))

        replica_etas = np.array(replica_etas)
        mean_eta = float(np.mean(replica_etas))
        stderr_eta = float(np.std(replica_etas, ddof=1) / np.sqrt(len(replica_etas))) if len(replica_etas) > 1 else float("nan")
        print(f"--- Green-Kubo Viscosity, {len(replica_etas)}/{n_replicas} valid replicas ---")
        print(f"eta = {mean_eta:.4f} +/- {stderr_eta:.4f} (stderr)")

        with open(f"{results_path}/visc_gk_summary.json", "w") as f:
            json.dump({
                "n_replicas_requested": n_replicas,
                "n_replicas_valid": len(replica_etas),
                "per_replica_eta": replica_etas.tolist(),
                "mean_eta": mean_eta,
                "stderr_eta": stderr_eta,
            }, f, indent=4)

        plt.figure(figsize=(8, 5))
        for time_actual, running_viscosity in replica_curves:
            plt.plot(time_actual, running_viscosity, alpha=0.4)
        plt.axhline(mean_eta, color="red", lw=2, ls="--", label=f"mean = {mean_eta:.4f} +/- {stderr_eta:.4f}")
        plt.axhspan(mean_eta - stderr_eta, mean_eta + stderr_eta, color="red", alpha=0.15)
        plt.xlabel("Time")
        plt.ylabel("Running Viscosity ($\\eta$)")
        plt.title(f"Green-Kubo Viscosity, {len(replica_etas)} replicas")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{results_path}/visc_gk_combined.png", dpi=300, bbox_inches="tight")
        plt.close()
    else:
        # Run Diffusion Measures for DDFT-gamma calculation
        md = mdext.md.MD(
                setup=setup,
                integrator=args["integrator"],
                T=T,
                P=P,
                seed=1,
                potential=zero_potential,
                geometry_type=mdext.geometry.Planar,
                n_atom_types=1,
                potential_type=1,
                steps_per_thermo=10,
                thermo_per_cycle=1,
                units="lj",
                timestep=args["dt"],
                dr=args["dr"],
                Tdamp=Tdamp,
            )
        # Equilibration run
        md.run(args["equilibration_steps"], "equilibration")
        md.lmp.command("reset_timestep 0")
        # Disable thermostat und use NVE
        md.lmp.command("unfix Ensemble")
        md.lmp.command("fix NVE all nve")
        if args["run_type"] == "msd":
            # Collect MSD values
            md.lmp.command("compute myMSD all msd")
            #md.lmp.command(f"fix msd_saver all ave/time 10 1 10 c_myMSD[4] file {results_path}/msd_data.txt")
            md.lmp.command(f"fix msd_saver all ave/time 1 1 1 c_myMSD[4] file {results_path}/msd_data.txt")
            # Diffusion run
            md.run(args["diffusion_steps"], "diffusion_test")
            analyze_msd(f"{results_path}/msd_data.txt", T, 1/Tdamp, args["dt"])
        elif args["run_type"] == "vacf":
            # Collect VACF values
            md.lmp.command("compute myVACF all vacf")
            md.lmp.command(f"fix vacf_saver all ave/time 1 1 1 c_myVACF[4] file {results_path}/vacf_data.txt")
            # Diffusion run
            md.run(args["diffusion_steps"], "diffusion_test")
            analyze_vacf(f"{results_path}/vacf_data.txt", T, 1/Tdamp, args["dt"])
        # "visc_gk" is handled above (multi-replica path, before this single-MD-object branch).




def setup(lmp: PyLammps, seed: int) -> int:
    """Setup initial atomic configuration and interaction potential."""
    
    # Construct simulation box:
    L = np.array([10., 10., 10.])  # overall box dimensions
    lmp.region(
        f"sim_box block {0} {L[0]} 0 {L[1]} 0 {L[2]}"
        " units box"
    )
    lmp.create_box("1 sim_box")
    n_bulk = 0.6  # bulk density of water at STP (in A^-3), the bulk density value
    n_atoms = int(np.round(n_bulk * L.prod()))
    lmp.create_atoms(f"1 random {n_atoms} {seed} sim_box")
    lmp.mass("1 1.")  # pretending this is a united-atom model of water, dz

    # Interaction potential:
    lmp.pair_style("lj/cut 4") # lj cutoff
    # lmp.pair_coeff("1 1 1.0 2.97")  # Adjusted to get water bulk density
    lmp.pair_coeff("1 1 1.0 1.0")
    lmp.pair_modify("tail yes")  # Critical for NPT to be stable

    # Initial minimize:
    log.info("Minimizing initial structure")
    lmp.minimize("1E-4 1E-6 10000 100000")


if __name__ == "__main__":
    PARAM_SCHEME = {
        "run_id": (str, "lj"),
        "integrator": (str, "nve"), # one of [nve, langevin, nvt]
        "trials": (int, 10),
        "equilibration_steps": (int, 200),
        "collection_steps": (int, 400),
        "run_type": (str, "collection"), # one of [collection, msd, vacf, visc_gk]
        "diffusion_steps": (int, 5000),
        "N_replicas": (int, 16), # visc_gk only: independent seeded replicas for mean +/- stderr
        "dt": (float, 0.001),
        "Tdamp": (float, 100.0),
        "dr": (float, 1/32),
        "T": (float, 2.0),
        "P": (float, None),
        "U0": (float, 5.0),
        "mu": (float, 5.0),
        "sigma": (float, 2.0),
        "h5_file": (str, None), # Target HDF5 file containing external potential
    }

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to Config file (Params get overwritten with CLI arguments)")
    for name, (arg_type, default) in PARAM_SCHEME.items():
        if arg_type is bool:
            # Booleans usually need 'store_true'
            parser.add_argument(f"--{name}", action="store_true")
        else:
            parser.add_argument(f"--{name}", type=arg_type)
    args = parser.parse_args()

    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)

    final_args = {}
    for name, (arg_type, default) in PARAM_SCHEME.items():
        cli_val = getattr(args, name)

        if cli_val is not None:
            final_args[name] = cli_val
        elif name in config:
            final_args[name] = config[name]
        else:
            final_args[name] = default
    print(f"Config:\n{json.dumps(final_args, indent=2)}\n")
    main(final_args)
