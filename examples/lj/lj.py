"""Single-site LJ fluid test with roughly the density of water at STP."""
import mdext
import numpy as np
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
                potential=mdext.potential.Gaussian(0, mu, sigma),
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
            md.reset_vext(mdext.potential.Gaussian(U0, sigma, mu))
            md.run(collection_steps, "collection", f"{results_path}/trials/{i}/collection")
            plot(f"{results_path}/trials/{i}")
        
        avg_results_parallel(results_path, args["trials"], collection_steps)
        plot_movie(f"{results_path}/avg_{args["trials"]}")
    else:
        # Run Diffusion Measures for DDFT-gamma calculation
        md = mdext.md.MD(
                setup=setup,
                integrator=args["integrator"],
                T=T,
                P=P,
                seed=1,
                potential=mdext.potential.Gaussian(0, mu, sigma),
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
        elif args["run_type"] == "visc_gk":
            # Green-Kubo parameters
            s = 5  # sample interval (every 5 steps)
            p = args["diffusion_steps"] // 10  # correlation length
            if p == 0: p = 1
            d = s * p  # dump interval
            
            s=1
            d=1000
            p=1000
            md.lmp.command("thermo 1")
            
            # Use LAMMPS native global pressure tensor components directly!
            md.lmp.command("variable pxy equal pxy")
            md.lmp.command("variable pxz equal pxz")
            md.lmp.command("variable pyz equal pyz")
            
            # Run the autocorrelation
            # 'ave running' averages the correlation windows over the entire run
            md.lmp.command(f"fix GK_visc all ave/correlate {s} {p} {d} v_pxy v_pxz v_pyz type auto file {results_path}/visc_data.txt ave running")

            # Create a dedicated compute to force tensor evaluation every 's' steps
            #md.lmp.command("compute myTemp all temp")
            #md.lmp.command("compute pTG all pressure myTemp")
            
            # Extract components (4=pxy, 5=pxz, 6=pyz)
            #md.lmp.command("variable vpxy equal c_pTG[4]")
            #md.lmp.command("variable vpxz equal c_pTG[5]")
            #md.lmp.command("variable vpyz equal c_pTG[6]")
            
            # Run the autocorrelation using our forced variables
            #md.lmp.command(f"fix GK_visc all ave/correlate {s} {p} {d} v_vpxy v_vpxz v_vpyz type auto file {results_path}/visc_data.txt ave running")
            
            # Run the simulation
            md.run(args["diffusion_steps"], "viscosity_test")
            
            # Analyze results (Assuming V = 10*10*10 = 1000 based on your setup)
            volume = 10.0 * 10.0 * 10.0
            analyze_visc_gk(f"{results_path}/visc_data.txt", T, volume, args["dt"])

    


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
        "dt": (float, 0.001),
        "Tdamp": (float, 100.0),
        "dr": (float, 1/32),
        "T": (float, 2.0),
        "P": (float, None),
        "U0": (float, 5.0),
        "mu": (float, 5.0),
        "sigma": (float, 2.0),
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
