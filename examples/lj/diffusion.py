import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.integrate import trapezoid
import argparse
import os

def load_data(filename, dt):
    try:
        # Load data (skipping the header lines)
        # Column 0: TimeStep, Column 1: MSD or VACF
        data = np.loadtxt(filename, skiprows=2)
        timestep_col = data[:, 0]
        data_col = data[:, 1]
        time = timestep_col * dt
        return time, data_col
    except OSError:
        print(f"Error: Could not find file '{filename}'. Run the bulk simulation first.")
        return

def analyze_msd(filename, T, gamma_theory, dt):
    """
    Reads LAMMPS MSD output, plots it, and calculates diffusion coefficient.
    """
    time, msd_col = load_data(filename, dt)

    # --- 1. Calculate D0 from Simulation (Slope/6) ---
    # Maybe fit only the second half of the data to ensure we are in the diffusive regime
    #start_idx = len(time) // 2
    start_idx = 0
    slope, intercept, r_value, p_value, std_err = linregress(time[start_idx:], msd_col[start_idx:])
    
    # D0 = Slope / (2 * dimensions)
    dimension = 3.0
    D0_sim = slope / (2*dimension)

    # --- Theoretical D0 (Einstein Relation) ---
    kB = 1.0
    m = 1.0
    D0_theory = (kB * T) / (m * gamma_theory)
    gamma_sim = (kB * T) / (m * D0_sim)

    print(f"--- RESULTS ---")
    print(f"Theoretical D0 (kB*T/(m*gamma)):    {D0_theory:.4f}")
    print(f"Measured D0 (Slope/(2*dimension)):  {D0_sim:.4f}")
    print(f"Error:                              {abs(D0_sim - D0_theory)/D0_theory * 100:.2f}%")
    print(f"Measured gamma (kB*T/(m*D0)):       {gamma_sim:.4f}")

    # --- Plotting ---
    plt.figure(figsize=(6, 5))
    
    # Plot raw data
    plt.plot(time, msd_col, 'o', color='black', alpha=0.3, markersize=3, label='Simulation MSD')
    
    # Plot best fit line
    fit_line = slope * time + intercept
    plt.plot(time, fit_line, 'r--', lw=2, label=f'Fit (Slope={slope:.2f})')
    
    plt.xlabel("Time ($t$)")
    plt.ylabel("Mean Squared Displacement $\\langle r^2 \\rangle$")
    plt.title(f"Diffusion Check\n$T={T}, \\gamma={gamma_theory}$")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(f"{'/'.join(filename.split('/')[:-1])}/check_diffusion_plot.png")

def analyze_vacf(filename, T, gamma_theory, dt):
    """
    Reads LAMMPS VACF output, plots it, and calculates diffusion coefficient.
    """
    time, vacf_col = load_data(filename, dt)

    # Diffusion is time integration of velocity auto-correlation function (normalized by dimension)
    dimension = 3.0
    D0_sim = trapezoid(vacf_col, time) / dimension

    # --- Theoretical D0 (Einstein Relation) ---
    kB = 1.0
    m = 1.0
    D0_theory = (kB * T) / (m * gamma_theory)
    gamma_sim = (kB * T) / (m * D0_sim)

    print(f"--- RESULTS ---")
    print(f"Theoretical D0 (kB*T/(m*gamma)):    {D0_theory:.4f}")
    print(f"Measured D0 (integral/dimension):   {D0_sim:.4f}")
    print(f"Error:                              {abs(D0_sim - D0_theory)/D0_theory * 100:.2f}%")
    print(f"Measured gamma (kB*T/(m*D0)):       {gamma_sim:.4f}")

def analyze_visc_gk(filepath, T, V, dt):
    """
    Parses LAMMPS ave/correlate output for Green-Kubo viscosity.
    In LJ units, k_B = 1.0.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            
        # Find the start of the LAST data block
        # LAMMPS block headers contain exactly 2 numbers (Timestep and N-rows)
        block_start = -1
        for i in range(len(lines)-1, -1, -1):
            if len(lines[i].split()) == 2 and not lines[i].startswith('#'):
                block_start = i
                break
                
        if block_start == -1:
            print(f"Error: Could not find valid data blocks in {filepath}")
            return
            
        # Extract just the data lines from the final block
        num_rows = int(lines[block_start].split()[1])
        data_lines = lines[block_start + 1 : block_start + 1 + num_rows]
        data = np.array([list(map(float, line.split())) for line in data_lines])
        
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return

    # Column 1 is TimeDelta (in steps), Columns 3, 4, 5 are the Pxy, Pxz, Pyz auto-correlations
    time_steps = data[:, 0]
    time_actual = time_steps * dt
    
    pxy_corr = data[:, 2]
    pxz_corr = data[:, 3]
    pyz_corr = data[:, 4]
    
    # Average the three independent shear directions to reduce noise
    avg_corr = (pxy_corr + pxz_corr + pyz_corr) / 3.0
    
    # Calculate the running integral (cumulative trapezoidal rule)
    running_integral = np.zeros_like(avg_corr)
    running_integral[1:] = np.cumsum((avg_corr[:-1] + avg_corr[1:]) / 2.0 * np.diff(time_actual))
    
    # Calculate running viscosity (in LJ units, k_B = 1)
    k_B = 1.0
    running_viscosity = (V / (k_B * T)) * running_integral
    final_viscosity = running_viscosity[-1]
    
    print(f"--- Green-Kubo Viscosity Analysis ---")
    print(f"Temperature: {T} | Volume: {V}")
    print(f"Calculated Viscosity (LJ units): {final_viscosity:.4f}")
    
    plt.rcParams['text.usetex'] = False
    plt.rcParams['font.size'] = 12
    fig, (ax_corr, ax_int) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    # Top Panel: Auto-Correlation Decay
    # We want to see this curve decay to exactly 0 and stay there.
    ax_corr.set_ylabel('$\\langle P_{\\alpha\\beta}(t) P_{\\alpha\\beta}(0) \\rangle$')
    ax_corr.axhline(0, color='black', lw=1, ls='--')
    ax_corr.plot(time_actual, pxy_corr, alpha=0.3, label='Pxy')
    ax_corr.plot(time_actual, pxz_corr, alpha=0.3, label='Pxz')
    ax_corr.plot(time_actual, pyz_corr, alpha=0.3, label='Pyz')
    ax_corr.plot(time_actual, avg_corr, color='black', lw=2, label='Average')
    ax_corr.legend(loc='upper right')
    ax_corr.set_title("Stress Auto-Correlation Decay")
    
    # Bottom Panel: Running Viscosity Convergence
    # We want to see this curve rise and then plateau to a flat line.
    ax_int.set_ylabel('Running Viscosity ($\\eta$)')
    ax_int.set_xlabel('Time')
    ax_int.plot(time_actual, running_viscosity, color='blue', lw=2)
    ax_int.axhline(final_viscosity, color='red', ls='--', label=f'Final Plateau: {final_viscosity:.4f}')
    ax_int.legend(loc='lower right')
    ax_int.set_title("Green-Kubo Integral Convergence")
    
    plt.tight_layout()
    
    # Determine save path based on the input filepath 
    save_dir = os.path.dirname(filepath)
    save_path = os.path.join(save_dir, "viscosity_plot.png")
    
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved viscosity plot to: {save_path}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", type=str, default="results/lj_blank/msd_text.txt")
    parser.add_argument("--T", type=float, default=2)
    parser.add_argument("--gamma", type=float, default=2)
    args = parser.parse_args()
    analyze_msd(args.filename, args.T, args.gamma)