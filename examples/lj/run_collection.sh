#!/bin/bash
# Langevin MD reference trajectories for the underdamped DDFT data.
#
# Usage:  ./run_collection.sh <potential> [<potential> ...]
#   gaussian      built-in Gaussian potential of lj.py (U0=5, mu=5, sigma=2)
#   <name>        tabulated potential <name>.h5 in this folder, e.g.
#                 3a-3d (from Ram et al. 2025, arXiv:2505.09543) or
#                 mg1, mg2, s1, s2, s3, gausin1 (create with: python potentials.py)
#
# Settings (environment variables, defaults = values used in the report):
#   TRIALS=500   independent trials averaged per potential (3a-3d used 2000)
#   TDAMP=10     Langevin damping time, friction gamma = 1/TDAMP (gamma = 0.1)
#   EQUIL=200    equilibration cycles without potential (2 tau)
#   COLLECT=400  collection cycles = saved frames, 0.01 tau apart (4 tau)
#   DR=0.03125   density grid spacing (sigma/32)
#   TAG=ref      run_id prefix; results go to results/<date>_<TAG>_<potential>_langevin_<NN>
#
# Example: TRIALS=2000 ./run_collection.sh 3a 3b 3c 3d
#          TDAMP=20 TAG=ref_gamma005 ./run_collection.sh gaussian   # gamma = 0.05

set -e
cd "$(dirname "$0")"

if [ "$#" -eq 0 ]; then
    sed -n '2,20p' "$0"
    exit 1
fi

TRIALS=${TRIALS:-500}
TDAMP=${TDAMP:-10}
EQUIL=${EQUIL:-200}
COLLECT=${COLLECT:-400}
DR=${DR:-0.03125}
TAG=${TAG:-ref}

for pot in "$@"; do
    POT_ARGS=()
    if [ "$pot" != "gaussian" ]; then
        # lj.py silently falls back to the Gaussian potential if the file is missing
        if [ ! -f "${pot}.h5" ]; then
            echo "Error: ${pot}.h5 not found (create it with: python potentials.py ${pot})" >&2
            exit 1
        fi
        POT_ARGS=(--h5_file "${pot}.h5")
    fi

    echo "=== ${pot}: trials=${TRIALS}, Tdamp=${TDAMP}, equilibration=${EQUIL}, collection=${COLLECT}"
    python lj.py \
        --run_id "${TAG}_${pot}" \
        --integrator langevin \
        --dr "${DR}" \
        --trials "${TRIALS}" \
        --equilibration_steps "${EQUIL}" \
        --collection_steps "${COLLECT}" \
        --Tdamp "${TDAMP}" \
        "${POT_ARGS[@]}"
done

echo "All simulations completed."
