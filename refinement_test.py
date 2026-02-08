# 6.4

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy.interpolate import RegularGridInterpolator
import time

# Import the solver
try:
    from hjbi_solver import solve_hjbi
except ImportError:
    print("ERROR: Could not import solve_hjbi from hjbi_solver.py")
    print("Please ensure hjbi_solver.py is in the same directory or PYTHONPATH")
    raise


def compute_diagnostics(V, u_star, sigma_star, s_grid, x_grid, K, u_min, u_max, sig_min, sig_max):
    """
    Compute diagnostic metrics for a single grid solution.

    Returns:
        dict with diagnostics
    """
    # Value at the money: v(0, K, 0)
    s_idx = np.argmin(np.abs(s_grid - K))
    x_idx = np.argmin(np.abs(x_grid))
    v_atm = V[s_idx, x_idx]

    # Min and max of V at t=0
    min_V = np.min(V)
    max_V = np.max(V)

    # Fraction using sigma_max and sigma_min (consistent tolerance definition)
    tol_sigma = 0.05 * (sig_max - sig_min)
    frac_sigma_max = np.sum(np.abs(sigma_star - sig_max) < tol_sigma) / sigma_star.size
    frac_sigma_min = np.sum(np.abs(sigma_star - sig_min) < tol_sigma) / sigma_star.size

    # Fraction of u* near bounds
    tol_u = 0.05 * (u_max - u_min)
    frac_u_upper = np.sum(np.abs(u_star - u_max) < tol_u) / u_star.size
    frac_u_lower = np.sum(np.abs(u_star - u_min) < tol_u) / u_star.size

    return {
        'v_atm': v_atm,
        'min_V': min_V,
        'max_V': max_V,
        'frac_sigma_max': frac_sigma_max,
        'frac_sigma_min': frac_sigma_min,
        'frac_u_upper': frac_u_upper,
        'frac_u_lower': frac_u_lower
    }


def interpolate_to_reference(V, s_grid, x_grid, s_ref, x_ref, method='linear'):
    """
    Interpolate V(s,x) onto reference grid.

    Args:
        V: 2D array on original grid
        s_grid, x_grid: 1D arrays defining original grid
        s_ref, x_ref: 1D arrays defining reference grid
        method: 'linear' or 'nearest'

    Returns:
        V_interp: 2D array on reference grid (may contain NaN if out of bounds)
    """
    # Create interpolator with NaN for out-of-bounds (prevents extrapolation artifacts)
    interp = RegularGridInterpolator(
        (s_grid, x_grid),
        V,
        method=method,
        bounds_error=False,
        fill_value=np.nan
    )

    # Create reference mesh
    S_ref, X_ref = np.meshgrid(s_ref, x_ref, indexing='ij')
    points = np.column_stack([S_ref.ravel(), X_ref.ravel()])

    # Interpolate
    V_interp = interp(points).reshape(S_ref.shape)

    # Check for NaNs (should not occur if domains match)
    n_nan = np.sum(np.isnan(V_interp))
    if n_nan > 0:
        print(f"  WARNING: {n_nan} NaN values after interpolation (out of bounds)")

    return V_interp


def compute_switching_mismatch(sigma_A, sigma_B, sig_max, sig_min):
    """
    Compute mismatch rate between switching indicators on same grid.

    Args:
        sigma_A, sigma_B: 2D arrays of sigma controls (same shape)
        sig_max, sig_min: volatility bounds

    Returns:
        mismatch_rate: fraction of points where indicators differ
    """
    # Consistent tolerance definition
    tol_sigma = 0.05 * (sig_max - sig_min)

    # Define switching indicators
    I_A = (np.abs(sigma_A - sig_max) < tol_sigma).astype(int)
    I_B = (np.abs(sigma_B - sig_max) < tol_sigma).astype(int)

    # Ignore NaN points if present
    mask = np.isfinite(sigma_A) & np.isfinite(sigma_B)

    # Compute mismatch
    mismatch = np.sum((I_A != I_B) & mask) / np.sum(mask)

    return mismatch


def run_refinement_study(run_fine=True):
    """
    Run grid refinement study for HJBI solver.

    Args:
        run_fine: if True, run fine grid (may be slow)
    """
    print("=" * 70)
    print("Section 6.4: Numerical Stability and Grid Refinement Tests")
    print("=" * 70)

    # Common parameters
    T = 1.0
    r = 0.02
    K = 1.0
    sig_min = 0.1
    sig_max = 0.4
    u_min = -2.0
    u_max = 2.0
    s_min, s_max = 0.2, 3.0
    x_min, x_max = -2.0, 2.0

    # Grid configurations
    grids = [
        (40, 40, 40, "Coarse", 1e-6, 50),
        (80, 80, 80, "Medium", 1e-6, 50),
    ]

    if run_fine:
        # Fine grid with relaxed tolerance for computational efficiency
        grids.append((100, 100, 100, "Fine", 1e-5, 30))

    # Storage for results
    results = []

    # Open file for writing summary
    with open('refinement_summary.txt', 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("Grid Refinement Study: HJBI Robust Hedging Solver\n")
        f.write("=" * 70 + "\n")
        f.write(f"\nParameters:\n")
        f.write(f"  T={T}, r={r}, K={K}\n")
        f.write(f"  σ ∈ [{sig_min}, {sig_max}], u ∈ [{u_min}, {u_max}]\n")
        f.write(f"  Domain: s ∈ [{s_min}, {s_max}], x ∈ [{x_min}, {x_max}]\n")
        f.write("\n" + "=" * 70 + "\n\n")

        # Run solver on each grid
        for Ns, Nx, Nt, label, tol, max_iter in grids:
            print(f"\n{'=' * 70}")
            print(f"Running {label} grid: {Ns}×{Nx}×{Nt}")
            print(f"  Solver parameters: tol={tol:.0e}, max_iter={max_iter}")
            print(f"{'=' * 70}")

            f.write(f"{label} Grid: {Ns}×{Nx}×{Nt}\n")
            f.write(f"Solver parameters: tol={tol:.0e}, max_iter={max_iter}\n")
            f.write("-" * 70 + "\n")

            start_time = time.time()

            # Solve
            t_grid, s_grid, x_grid, V_all, u_star, sigma_star = solve_hjbi(
                T=T, r=r, K=K,
                sig_min=sig_min, sig_max=sig_max,
                u_min=u_min, u_max=u_max,
                s_min=s_min, s_max=s_max,
                x_min=x_min, x_max=x_max,
                Ns=Ns, Nx=Nx, Nt=Nt,
                tol=tol,
                max_iter=max_iter
            )

            elapsed = time.time() - start_time

            # Extract solution at t=0
            V_0 = V_all[0]

            # Compute diagnostics
            diag = compute_diagnostics(V_0, u_star, sigma_star, s_grid, x_grid,
                                       K, u_min, u_max, sig_min, sig_max)

            # Store results
            results.append({
                'label': label,
                'grid': (Ns, Nx, Nt),
                's_grid': s_grid,
                'x_grid': x_grid,
                'V_0': V_0,
                'u_star': u_star,
                'sigma_star': sigma_star,
                'diag': diag,
                'time': elapsed,
                'tol': tol,
                'max_iter': max_iter
            })

            # Print diagnostics
            print(f"\nDiagnostics for {label} grid:")
            print(f"  Computation time: {elapsed:.2f} seconds")
            print(f"  v(0,K,0) = {diag['v_atm']:.6f}")
            print(f"  min(V)   = {diag['min_V']:.2e}")
            print(f"  max(V)   = {diag['max_V']:.2e}")
            print(f"  Fraction σ* = σ_max: {diag['frac_sigma_max']:.3f}")
            print(f"  Fraction σ* = σ_min: {diag['frac_sigma_min']:.3f}")
            print(f"  Fraction u* ≈ u_max: {diag['frac_u_upper']:.3f}")
            print(f"  Fraction u* ≈ u_min: {diag['frac_u_lower']:.3f}")

            # Write to file
            f.write(f"Computation time: {elapsed:.2f} seconds\n")
            f.write(f"v(0,K,0)         : {diag['v_atm']:.6f}\n")
            f.write(f"min(V)           : {diag['min_V']:.2e}\n")
            f.write(f"max(V)           : {diag['max_V']:.2e}\n")
            f.write(f"Frac σ* = σ_max  : {diag['frac_sigma_max']:.3f}\n")
            f.write(f"Frac σ* = σ_min  : {diag['frac_sigma_min']:.3f}\n")
            f.write(f"Frac u* ≈ u_max  : {diag['frac_u_upper']:.3f}\n")
            f.write(f"Frac u* ≈ u_min  : {diag['frac_u_lower']:.3f}\n")
            f.write("\n")

        # Stability analysis: use finest grid as reference
        print(f"\n{'=' * 70}")
        print("Stability Under Refinement Analysis")
        print(f"{'=' * 70}")

        f.write("=" * 70 + "\n")
        f.write("Stability Under Refinement Analysis\n")
        f.write("=" * 70 + "\n\n")

        # Reference grid (finest)
        ref_idx = len(results) - 1
        s_ref = results[ref_idx]['s_grid']
        x_ref = results[ref_idx]['x_grid']
        V_ref = results[ref_idx]['V_0']
        sigma_ref = results[ref_idx]['sigma_star']
        u_ref = results[ref_idx]['u_star']

        f.write(f"Reference grid: {results[ref_idx]['label']}\n")
        f.write(f"Note: We assess stability under refinement rather than formal convergence rates.\n")
        f.write(f"The control set u is discretized for optimization, so reported norms reflect\n")
        f.write(f"stability of the discrete feedback policy.\n\n")

        # Compare each grid to reference
        for i in range(len(results) - 1):
            label = results[i]['label']
            s_grid = results[i]['s_grid']
            x_grid = results[i]['x_grid']
            V = results[i]['V_0']
            sigma = results[i]['sigma_star']
            u = results[i]['u_star']

            print(f"\n{label} vs {results[ref_idx]['label']}:")
            f.write(f"{label} vs {results[ref_idx]['label']}:\n")
            f.write("-" * 70 + "\n")

            # Interpolate to reference grid
            # Use linear for V, nearest for sigma and u (controls are discrete-valued)
            V_interp = interpolate_to_reference(V, s_grid, x_grid, s_ref, x_ref, method='linear')
            sigma_interp = interpolate_to_reference(sigma, s_grid, x_grid, s_ref, x_ref, method='nearest')
            u_interp = interpolate_to_reference(u, s_grid, x_grid, s_ref, x_ref, method='nearest')

            # Mask for finite values (ignore NaN from out-of-bounds)
            mask = np.isfinite(V_interp) & np.isfinite(V_ref)

            # Compute errors for V
            error_V_inf = np.nanmax(np.abs(V_interp - V_ref)[mask])
            error_V_L2 = np.sqrt(np.nanmean((V_interp - V_ref)[mask] ** 2))
            rel_error_V = error_V_inf / max(1e-12, np.nanmax(np.abs(V_ref)))

            # Value at money comparison
            s_idx = np.argmin(np.abs(s_ref - K))
            x_idx = np.argmin(np.abs(x_ref))
            v_atm_interp = V_interp[s_idx, x_idx]
            v_atm_ref = V_ref[s_idx, x_idx]
            delta_atm = abs(v_atm_interp - v_atm_ref)
            rel_delta_atm = delta_atm / max(1e-12, abs(v_atm_ref))

            # Switching boundary mismatch for sigma
            mismatch_sigma = compute_switching_mismatch(sigma_interp, sigma_ref, sig_max, sig_min)

            # Control errors for u
            mask_u = np.isfinite(u_interp) & np.isfinite(u_ref)
            error_u_inf = np.nanmax(np.abs(u_interp - u_ref)[mask_u])
            error_u_L2 = np.sqrt(np.nanmean((u_interp - u_ref)[mask_u] ** 2))

            print(f"  ||V_interp - V_ref||_∞  = {error_V_inf:.2e}")
            print(f"  ||V_interp - V_ref||_2  = {error_V_L2:.2e}")
            print(f"  Relative error          = {rel_error_V:.2e}")
            print(f"  |v_ATM_interp - v_ATM_ref| = {delta_atm:.2e} ({rel_delta_atm:.2%})")
            print(f"  σ* switching mismatch   = {mismatch_sigma:.3%}")
            print(f"  ||u_interp - u_ref||_∞  = {error_u_inf:.2e}")
            print(f"  ||u_interp - u_ref||_2  = {error_u_L2:.2e}")

            f.write(f"||V_interp - V_ref||_∞  = {error_V_inf:.2e}\n")
            f.write(f"||V_interp - V_ref||_2  = {error_V_L2:.2e}\n")
            f.write(f"Relative error          = {rel_error_V:.2e}\n")
            f.write(f"|Δv_ATM|                = {delta_atm:.2e} ({rel_delta_atm:.2%})\n")
            f.write(f"σ* switching mismatch   = {mismatch_sigma:.3%}\n")
            f.write(f"||u_interp - u_ref||_∞  = {error_u_inf:.2e}\n")
            f.write(f"||u_interp - u_ref||_2  = {error_u_L2:.2e}\n")
            f.write("\n")

        # Empirical convergence estimate (only if all grids use same tolerance)
        all_same_tol = all(res['tol'] == results[0]['tol'] for res in results)

        if len(results) >= 3 and all_same_tol:
            print(f"\nEmpirical Convergence Estimate:")
            f.write("Empirical Convergence Estimate:\n")
            f.write("-" * 70 + "\n")
            f.write("Note: All grids solved with same tolerance, enabling rate estimation.\n\n")

            # Compute rates between consecutive grids
            for i in range(len(results) - 2):
                label1 = results[i]['label']
                label2 = results[i + 1]['label']

                # Interpolate both to finest
                V1_interp = interpolate_to_reference(
                    results[i]['V_0'], results[i]['s_grid'], results[i]['x_grid'],
                    s_ref, x_ref, method='linear'
                )
                V2_interp = interpolate_to_reference(
                    results[i + 1]['V_0'], results[i + 1]['s_grid'], results[i + 1]['x_grid'],
                    s_ref, x_ref, method='linear'
                )

                mask = np.isfinite(V1_interp) & np.isfinite(V2_interp) & np.isfinite(V_ref)
                e1 = np.nanmax(np.abs(V1_interp - V_ref)[mask])
                e2 = np.nanmax(np.abs(V2_interp - V_ref)[mask])

                # Correct grid spacing (Ns-1 for number of intervals)
                h1 = (s_max - s_min) / (results[i]['grid'][0] - 1)
                h2 = (s_max - s_min) / (results[i + 1]['grid'][0] - 1)

                if e1 > 1e-10 and e2 > 1e-10:
                    rate = np.log(e1 / e2) / np.log(h1 / h2)
                    print(f"  {label1} → {label2}: rate ≈ {rate:.2f}")
                    f.write(f"{label1} → {label2}: rate ≈ {rate:.2f}\n")

            f.write("\n")
        elif len(results) >= 3 and not all_same_tol:
            print(f"\nNote: Convergence rates not computed (varying solver tolerances)")
            f.write("Note: Convergence rates not computed due to varying solver tolerances.\n")
            f.write("The Fine grid uses relaxed tolerance for computational efficiency.\n\n")

    # Save numerical arrays for reproducibility
    save_dict = {}
    for res in results:
        prefix = res['label'].lower()
        save_dict[f'{prefix}_s_grid'] = res['s_grid']
        save_dict[f'{prefix}_x_grid'] = res['x_grid']
        save_dict[f'{prefix}_V_0'] = res['V_0']
        save_dict[f'{prefix}_u_star'] = res['u_star']
        save_dict[f'{prefix}_sigma_star'] = res['sigma_star']

    np.savez('refinement_outputs.npz', **save_dict)
    print("\nSaved numerical arrays to refinement_outputs.npz")

    # Create refinement figure
    print(f"\n{'=' * 70}")
    print("Creating refinement figures...")
    print(f"{'=' * 70}")

    n_grids = len(results)
    fig, axes = plt.subplots(1, n_grids, figsize=(5 * n_grids, 4))

    if n_grids == 1:
        axes = [axes]

    # Common color scale
    vmin = sig_min
    vmax = sig_max

    for idx, res in enumerate(results):
        ax = axes[idx]
        s_grid = res['s_grid']
        x_grid = res['x_grid']
        sigma = res['sigma_star']

        im = ax.imshow(
            sigma.T,
            extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
            aspect='auto',
            origin='lower',
            cmap='RdYlBu_r',
            vmin=vmin,
            vmax=vmax
        )
        ax.set_xlabel('Stock price s', fontsize=11)
        ax.set_ylabel('Wealth x', fontsize=11)
        ax.set_title(f'{res["label"]} ({res["grid"][0]}×{res["grid"][1]})', fontsize=12)
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    # Add colorbar with proper spacing to prevent overlap
    fig.subplots_adjust(right=0.92)
    cbar = fig.colorbar(im, ax=axes, label='σ*', fraction=0.046, pad=0.04)

    plt.savefig('refinement_sigma_switch.png', dpi=200, bbox_inches='tight')
    print("Saved refinement_sigma_switch.png")

    # Alternative: contour overlay plot
    fig2, ax = plt.subplots(1, 1, figsize=(8, 6))

    line_styles = ['-', '--', '-.', ':']
    colors = ['blue', 'red', 'green', 'orange']
    handles = []

    # Consistent tolerance for indicator
    tol_sigma = 0.05 * (sig_max - sig_min)

    for idx, res in enumerate(results):
        s_grid = res['s_grid']
        x_grid = res['x_grid']
        sigma = res['sigma_star']

        # Create switching indicator (same definition as mismatch)
        indicator = (np.abs(sigma - sig_max) < tol_sigma).astype(float)

        # Check if switching boundary exists
        if indicator.min() < 0.5 < indicator.max():
            # Plot contour at 0.5 level (switching boundary)
            S, X = np.meshgrid(s_grid, x_grid, indexing='ij')
            color = colors[idx % len(colors)]
            ls = line_styles[idx % len(line_styles)]

            cs = ax.contour(
                S, X, indicator,
                levels=[0.5],
                colors=color,
                linestyles=ls,
                linewidths=2
            )

            # Create proxy handle for legend
            handles.append(mlines.Line2D(
                [], [],
                color=color,
                linestyle=ls,
                linewidth=2,
                label=f'{res["label"]} ({res["grid"][0]}×{res["grid"][1]})'
            ))
        else:
            print(f"Warning: No switching boundary found on {res['label']} grid")

    if handles:
        ax.set_xlabel('Stock price s', fontsize=12)
        ax.set_ylabel('Wealth x', fontsize=12)
        ax.set_title('σ* Switching Boundary (σ* = σ_max region)', fontsize=13)
        ax.legend(handles=handles, fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('refinement_sigma_boundary.png', dpi=200, bbox_inches='tight')
        print("Saved refinement_sigma_boundary.png")
    else:
        print("Warning: No switching boundaries to plot")
        plt.close(fig2)

    plt.show()

    print(f"\n{'=' * 70}")
    print("Refinement study complete!")
    print("Results saved to:")
    print("  - refinement_summary.txt")
    print("  - refinement_sigma_switch.png")
    if handles:
        print("  - refinement_sigma_boundary.png")
    print("  - refinement_outputs.npz")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    # Run refinement study
    # Set run_fine=False to skip the fine grid if computation is too slow
    run_refinement_study(run_fine=True)