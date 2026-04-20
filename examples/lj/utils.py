import numpy as np
import os
import glob
import h5py
import multiprocessing as mp
from functools import partial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


def plot(path: str) -> None:
    file_pattern = f"{path}" + '/collection.*.h5'
    file_list = sorted(glob.glob(file_pattern))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[2,1]})
    ax_top2 = ax_top.twinx()

    # Top Panel: DENSITY
    ax_top.set_ylabel('Density')
    ax_top.set_ylim(0, 1.1)
    ax_top.tick_params(axis='y', labelcolor="red")
    ax_top2.set_ylabel('External potential', color="#D1D3D4")
    ax_top2.tick_params(axis='y', labelcolor="#D1D3D4")

    # Bottom Panel: VELOCITY
    ax_bot.set_ylabel('Velocity ($v_z$)', color='blue')
    ax_bot.set_xlabel('Position ($z$)')
    ax_bot.axhline(0, color='black', lw=1, ls=':', alpha=0.5) # Zero-line reference
    ax_bot.tick_params(axis='y', labelcolor='blue')

    with h5py.File(file_list[-1], "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])
        vz = np.array(fp["vz"])

    with h5py.File(file_list[0], "r") as fp:
        initial_n = np.array(fp["n"])
        initial_vz = np.array(fp["vz"])

    line_n_init, = ax_top.plot(r, initial_n, color='red', ls='--', alpha=.3, label='Initial')
    line_n, = ax_top.plot(r, n, lw=2, color="red", label="MD")
    line_V, = ax_top2.plot(r, V, label="Vext", color="#D1D3D4")
    line_vz_init, = ax_bot.plot(r, initial_vz, label="Initial", color="blue", ls='--', alpha=.3)
    line_vz, = ax_bot.plot(r, vz, label="Velocity", color="blue")
    ax_top.set_xlim(r.min(), r.max())
    ax_top2.set_ylim(0, V.max())

    v_max = max(abs(vz.min()), abs(vz.max()), 0.1) # Avoid 0 limit
    ax_bot.set_ylim(-v_max * 2.0, v_max * 2.0)

    lines_top = [line_n, line_n_init, line_V]
    labels_top = [l.get_label() for l in lines_top]
    ax_top.legend(lines_top, labels_top, loc='upper right')
    ax_bot.legend(loc='upper right')
    plt.tight_layout()

    plt.savefig(f"{path}/0plot.png")


def plot_movie(path: str) -> None:
    file_pattern = f"{path}" + '/collection.*.h5'
    file_list = sorted(glob.glob(file_pattern))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios':[2,1]})
    ax_top2 = ax_top.twinx()

    # Top Panel: DENSITY
    ax_top.set_ylim(0, 1.1)
    ax_top.set_yticks(np.arange(0, 1.1, 0.1))
    ax_top.set_ylabel('Density')
    ax_top.tick_params(axis='y', labelcolor="red")
    ax_top2.set_ylabel('External potential', color="#D1D3D4")
    ax_top2.tick_params(axis='y', labelcolor="#D1D3D4")

    # Bottom Panel: VELOCITY
    ax_bot.set_ylabel('Velocity ($v_z$)', color='blue')
    ax_bot.set_xlabel('Position ($z$)')
    ax_bot.axhline(0, color='black', lw=1, ls=':', alpha=0.5) # Zero-line reference
    ax_bot.tick_params(axis='y', labelcolor='blue')

    with h5py.File(file_list[0], "r") as fp:
        r = np.array(fp["r"])
        n = np.array(fp["n"])
        V = np.array(fp["V"])
        vz = np.array(fp["vz"])

    line_n_init, = ax_top.plot(r, n, color='red', ls='--', alpha=.3, label='Initial')
    line_n, = ax_top.plot(r, n, lw=2, color="red", label="MD")
    line_V, = ax_top2.plot(r, V, label="Vext", color="#D1D3D4")
    line_vz, = ax_bot.plot(r, vz, label="Velocity", color="blue")
    ax_top.set_xlim(r.min(), r.max())
    ax_top2.set_ylim(0, V.max())

    v_max = max(abs(vz.min()), abs(vz.max()), 0.25) # Avoid 0 limit
    ax_bot.set_ylim(-v_max * 2.0, v_max * 2.0)

    lines_top = [line_n, line_n_init, line_V]
    labels_top = [l.get_label() for l in lines_top]
    ax_top.legend(lines_top, labels_top, loc='upper right')
    ax_bot.legend(loc='upper right')

    plt.tight_layout()

    def update(frame):
        with h5py.File(frame, "r") as fp:
            n = np.array(fp["n"])
            V = np.array(fp["V"])
            vz = np.array(fp["vz"])

        line_n.set_ydata(n)
        line_V.set_ydata(V)
        line_vz.set_ydata(vz)

        # Auto-scale velocity axis if waves grow
        current_v_max = max(abs(vz.min()), abs(vz.max()))
        if current_v_max > ax_bot.get_ylim()[1]:
            ax_bot.set_ylim(-current_v_max*1.1, current_v_max*1.1)
            ax_bot.figure.canvas.draw()

        return line_n, line_V, line_vz

    ani = FuncAnimation(fig, update, frames=file_list, blit=False, repeat=False)
    ani.save(f"{path}/0plot_movie.mp4", fps=10, extra_args=['-vcodec', 'libx264'])
    plt.savefig(f"{path}/0plot.png")


def process_simulation(sim_index: int, base_path: str, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Process a single simulation and return its density and velocity arrays"""
    
    file_pattern = f"{base_path}/trials/{sim_index}/collection.*.h5"
    file_list = sorted(glob.glob(file_pattern))
    with h5py.File(file_list[0], "r") as fp:
        r_shape = np.array(fp["n"]).shape
        vz_shape = np.array(fp["vz"]).shape
    
    result_n = np.zeros((M, r_shape[0], 1))
    result_vz = np.zeros((M, vz_shape[0], 1))
   
    for j, file in enumerate(file_list):
        try:
            with h5py.File(file, "r") as fp:
                result_n[j] = np.array(fp["n"])
                result_vz[j] = np.array(fp["vz"])

        except OSError as e:
            print(f"Failed to open {file}: {e}")
            result_n[j] = result_n[j-1]
            result_vz[j] = result_vz[j-1]

    return result_n, result_vz


def avg_results_parallel(path: str, N: int, M: int) -> None:
    '''
    Average the density results of N simulations of M steps each using parallel processing
    '''
    # Save average M-step density profile
    save_path = f"{path}/avg_{N}"
    os.makedirs(save_path, exist_ok=True)

    # Read first file to get r and V
    first_file = f"{path}/trials/0/collection.000.h5"
    with h5py.File(first_file, "r") as fp:
        r = np.array(fp["r"])
        V = np.array(fp["V"])
        vz = np.array(fp["vz"])

    # Initialize the result array
    avg_n = np.zeros((M, r.shape[0], 1))
    avg_vz = np.zeros((M, vz.shape[0], 1))

    # Set up parallel processing
    num_cores = mp.cpu_count() - 1  # Leave one core free
    process_sim = partial(process_simulation, base_path=path, M=M)

    # Process simulations in parallel
    with mp.Pool(num_cores) as pool:
        results = pool.map(process_sim, range(N))
    
    # Sum all results
    for result_n, result_vz in results:
        avg_n += result_n
        avg_vz += result_vz

    # Calculate average
    avg_n /= N
    avg_vz /= N

    # Save results
    for i in range(M):
        filename = f"{save_path}/collection.{i:03d}.h5"
        with h5py.File(filename, "w") as fp:
            fp["n"] = avg_n[i]
            fp["r"] = r
            fp["V"] = V
            fp["vz"] = avg_vz[i]


if __name__ == "__main__":
    results_path = "results/2025-05-29-test"
    avg_results_parallel(results_path, 5, 10)
    plot_movie(f"{results_path}/avg_5")