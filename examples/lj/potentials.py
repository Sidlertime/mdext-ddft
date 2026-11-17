"""
External potentials used for the underdamped DDFT reference data.

Writes tabulated potentials as HDF5 files (keys ``r`` and ``V``) for
``lj.py --h5_file <name>.h5``, on the same 1000-point grid (dr = 0.01,
r in [0, 9.99], periodic box L = 10) as the reference potentials 3a-3d.

Not defined here:
  - ``gaussian``: built into lj.py (U0 = 5, mu = 5, sigma = 2, no --h5_file).
  - ``3a``-``3d``: tabulated potentials from Ram et al. (2025),
    arXiv:2505.09543; their HDF5 files are taken as given.

Usage:
    python potentials.py              # write all potentials below
    python potentials.py mg1 s1       # write only the named ones
    python potentials.py --list       # print names and value ranges
"""
import argparse

import h5py
import numpy as np

L = 10.0
R = np.arange(1000) * 0.01


def gauss(r, U0, mu, sigma):
    """Gaussian barrier U0 * exp(-(r - mu)^2 / (2 sigma^2))."""
    return U0 * np.exp(-0.5 * ((r - mu) / sigma) ** 2)


def cos_mode(r, A, n, phase=0.0):
    """Non-negative cosine mode A/2 * (1 - cos(2 pi n r / L + phase)), range [0, A]."""
    return 0.5 * A * (1.0 - np.cos(2.0 * np.pi * n * r / L + phase))


POTENTIALS = {
    # mixed Gaussians
    "mg1": lambda r: gauss(r, 4.0, 2.5, 0.8) + gauss(r, 6.0, 7.0, 1.2),
    "mg2": lambda r: gauss(r, 4.0, 1.5, 0.6) + gauss(r, 6.0, 5.0, 0.9) + gauss(r, 5.0, 8.5, 0.7),
    # sinusoids
    "s1": lambda r: cos_mode(r, 6.0, 2),
    "s2": lambda r: cos_mode(r, 4.0, 2) + cos_mode(r, 3.0, 5, 0.7),
    "s3": lambda r: cos_mode(r, 3.0, 1) + cos_mode(r, 3.0, 4, 1.1) + cos_mode(r, 2.0, 7, 2.3),
    # Gaussian + sinusoid
    "gausin1": lambda r: gauss(r, 4.0, 3.0, 1.0) + gauss(r, 4.0, 7.0, 1.0) + cos_mode(r, 1.5, 6),
}


def write(name, path=None):
    V = POTENTIALS[name](R)
    path = path or f"{name}.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("r", data=R)
        f.create_dataset("V", data=V)
    print(f"{name}: V in [{V.min():.3f}, {V.max():.3f}] -> {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*", help=f"potentials to write (default: all of {', '.join(POTENTIALS)})")
    parser.add_argument("--list", action="store_true", help="only print the available potentials")
    args = parser.parse_args()

    unknown = [n for n in args.names if n not in POTENTIALS]
    if unknown:
        parser.error(f"unknown potential(s): {', '.join(unknown)}; available: {', '.join(POTENTIALS)}")

    for name in args.names or POTENTIALS:
        if args.list:
            V = POTENTIALS[name](R)
            print(f"{name}: V in [{V.min():.3f}, {V.max():.3f}]")
        else:
            write(name)


if __name__ == "__main__":
    main()
