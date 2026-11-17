# mdext-ddft

Molecular dynamics in external potentials using PyLAMMPS. This is a fork of
[mdext](https://github.com/shankar1729/mdext), extended to generate the reference data
for **underdamped dynamical density functional theory (DDFT)**: time-resolved density
and velocity profiles of a Lennard-Jones fluid in 1D external potentials, simulated
with a Langevin thermostat.

The data is used by [neural-ddft-pcm](https://github.com/Sidlertime/neural-ddft-pcm) to
train and evaluate a hybrid DDFT model (PCM free-energy functional + learned
superadiabatic force). Changes compared to upstream:
- velocity profiles `vz` in the collected HDF5 files;
- tabulated external potentials (`InterpolatedPotential`, `lj.py --h5_file`);
- Green-Kubo viscosity with independent replicas;
- the potential definitions and run scripts in `examples/lj/`.

## Installation

The three project repositories share one conda environment and are expected next to each
other:

```
workspace/
├── neural-ddft-pcm/   git clone https://github.com/Sidlertime/neural-ddft-pcm.git
├── goddard/           git clone https://github.com/Sidlertime/goddard-ddft.git goddard
└── mdext-ddft/        git clone https://github.com/Sidlertime/mdext-ddft.git
```

Create the environment from neural-ddft-pcm. It includes LAMMPS with its Python module
from conda-forge. Then install this package in editable mode:

```bash
cd neural-ddft-pcm
mamba env create -f environment.yml      # or: conda env create -f environment.yml
mamba activate neural-ddft
cd ../mdext-ddft
pip install -e .
```

If you use mdext-ddft on its own, any environment works that has `lammps` (with the Python
module), `mpi4py`, `numpy`, `h5py` and `pyyaml`.

### Quick test

There is no unit-test suite. A short simulation (1 trial, 10 cycles, a few seconds) checks
the whole chain: LAMMPS, the external potential and the HDF5 output.

```bash
cd examples/lj
python potentials.py mg1
TRIALS=1 EQUIL=10 COLLECT=10 TAG=smoketest ./run_collection.sh mg1
```

The log should contain `Using InterpolatedPotential from mg1.h5`, and
`results/<date>_smoketest_mg1_langevin_01/avg_1/` should contain `collection.000.h5` to
`collection.009.h5`.

## Reproducing the reference data

All commands run in `examples/lj/`. Units are reduced Lennard-Jones units (T = 2,
bulk density 0.6, periodic box of 10σ). The density grid spacing is σ/32, and the MD time
step is 0.001τ. Every saved frame is one collection cycle of 10 MD steps (0.01τ).

### 1. External potentials

```bash
python potentials.py          # writes mg1, mg2, s1, s2, s3, gausin1 (.h5)
python potentials.py --list   # prints all names and value ranges
```

- `gaussian` is built into `lj.py` (U0 = 5, μ = 5, σ = 2) and needs no file.
- `3a`–`3d` are the tabulated potentials of Ram et al. (2025); place their `3a.h5`–`3d.h5`
  files in `examples/lj/`.

| Name | Definition V(z), with g(μ,s) = exp(−(z−μ)²/2s²) and L = 10 |
|---|---|
| mg1 | 4 g(2.5, 0.8) + 6 g(7, 1.2) |
| mg2 | 4 g(1.5, 0.6) + 6 g(5, 0.9) + 5 g(8.5, 0.7) |
| s1 | 3 [1 − cos(4πz/L)] |
| s2 | 2 [1 − cos(4πz/L)] + 1.5 [1 − cos(10πz/L + 0.7)] |
| s3 | 1.5 [1 − cos(2πz/L)] + 1.5 [1 − cos(8πz/L + 1.1)] + [1 − cos(14πz/L + 2.3)] |
| gausin1 | 4 g(3, 1) + 4 g(7, 1) + 0.75 [1 − cos(12πz/L)] |

### 2. Langevin reference trajectories

`run_collection.sh` simulates one or more potentials with the settings of the reference data:
- each trial equilibrates for 200 cycles (2τ) without the external potential;
- the potential is then switched on, and 400 frames (4τ) are collected;
- the trials are averaged.

```bash
./run_collection.sh gaussian mg1 mg2 s1 s2 s3 gausin1     # 500 trials each
TRIALS=2000 ./run_collection.sh 3a 3b 3c 3d               # 2000 trials each

# friction study, gamma = 1/Tdamp
TDAMP=20 TAG=ref_gamma005 ./run_collection.sh 3a gaussian # gamma = 0.05
TDAMP=2  TAG=ref_gamma05  ./run_collection.sh 3a gaussian # gamma = 0.5
```

- **Runtime:** one trial takes about 17 s on one CPU core, so 500 trials take about 2.4 h
  per potential. Trials run sequentially inside one process. To parallelize, start
  separate `run_collection.sh` calls (one per potential).
- **Output:**
  `results/<date>_<TAG>_<potential>_langevin_01/avg_<trials>/collection.NNN.h5`. Each file is
  one frame with the keys `r` (grid), `n` (density), `vz` (velocity) and `V` (external
  potential). The trials themselves are kept in `trials/`.
- **For neural-ddft-pcm:** link each `avg_<trials>` folder to
  `neural-ddft-pcm/data/lj_data/ref_<potential>_langevin_01/avg_<trials>`, the path used by
  its configs, e.g.

  ```bash
  mkdir -p ../../../neural-ddft-pcm/data/lj_data/ref_mg1_langevin_01
  ln -s "$PWD/results/<date>_ref_mg1_langevin_01/avg_500" ../../../neural-ddft-pcm/data/lj_data/ref_mg1_langevin_01/avg_500
  ```

### 3. Viscosity (Green-Kubo)

The viscosity η of the bulk fluid enters the DDFT momentum equation. It is measured from the
stress autocorrelation:
- 16 independent replicas;
- equilibration under the Langevin thermostat (Tdamp = 10);
- sampling in NVE.

```bash
python lj.py --run_type visc_gk --integrator langevin --Tdamp 10 \
    --equilibration_steps 5000 --diffusion_steps 10000 --N_replicas 16
python plot_visc_gk.py results/<date>_lj_visc_gk_langevin_01
```

The result is η = 0.81 ± 0.07 (mean ± standard error over 16 replicas). Single replicas
scatter strongly (0.33–1.24), so use the mean, not a single run.

## Additional information

- The friction of the Langevin thermostat is γ = m/Tdamp. `lj.py`'s own default Tdamp is
  100, while the reference data uses Tdamp = 10 (γ = 0.1).
- `lj.py --h5_file` silently falls back to the Gaussian potential if the file does not exist.
  `run_collection.sh` checks for the file first.
- The original mdext examples (water, molten salt) are kept unchanged in `examples/`.

## License and attribution

BSD 3-Clause, see [LICENSE](LICENSE) (original mdext, © 2022 BEAST collaboration) and
[NOTICE.md](NOTICE.md) for the modifications.
