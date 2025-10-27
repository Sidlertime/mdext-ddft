"""Single-site LJ fluid test with roughly the density of water at STP."""
import mdext
import numpy as np
import argparse
from datetime import datetime
import os

from lammps import PyLammps
from mdext import log
from utils import plot, plot_movie, avg_results_parallel


def main(args) -> None:

    # Current simulation parameters:
    T = 2  # K
    P = None  # atm
    U0 = 5.   # Amplitude of the external potential (kcal/mol)
    sigma = 2. # Width of the external potential (A)
    mu = 5
    equilibration_steps = args.equilibration_steps # equilibration cycles
    collection_steps = args.collection_steps # collection cycles

    current_time = datetime.now().strftime("%Y-%m-%d")
    results_path = f"results/{current_time}-{args.run_id}"
    os.makedirs(results_path, exist_ok=True)

    # Initialize and run simulation:
    for i in range(0, args.trials):
        md = mdext.md.MD(
            setup=setup,
            T=T,
            P=P,
            seed=i+1,
            potential=mdext.potential.Gaussian(0, 5, 1),
            geometry_type=mdext.geometry.Planar,
            n_atom_types=1,
            potential_type=1,
            steps_per_thermo=10,
            thermo_per_cycle=1,
            units="lj",
            timestep=args.dt,
            dr=1/32,
        )

        os.makedirs(f"{results_path}/trials/{i}", exist_ok=True)
        md.run(equilibration_steps, "equilibration", f"{results_path}/trials/{i}/equilibration")
        md.reset_stats()
        md.reset_vext(mdext.potential.Gaussian(U0, sigma, mu))
        md.run(collection_steps, "collection", f"{results_path}/trials/{i}/collection")
        plot(f"{results_path}/trials/{i}")
    
    avg_results_parallel(results_path, args.trials, collection_steps)
    plot_movie(f"{results_path}/avg_{args.trials}")


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="test")
    parser.add_argument("--trials", type=int, default="10")
    parser.add_argument("--equilibration_steps", type=int, default=200)
    parser.add_argument("--collection_steps", type=int, default=400)
    parser.add_argument("--dt", type=float, default=0.001)
    main(parser.parse_args())
