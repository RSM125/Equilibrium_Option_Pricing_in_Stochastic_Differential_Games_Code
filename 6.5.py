import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
import matplotlib.patches as mpatches


def create_publication_plots(t_grid, s_grid, x_grid, V_all, u_star, sigma_star,
                             K=1.0, sig_min=0.1, sig_max=0.4, u_min=-2.0, u_max=2.0):
    """
    Create publication-ready visualizations for HJBI solution.

    Args:
        t_grid: 1D array of time points
        s_grid: 1D array of stock prices
        x_grid: 1D array of wealth values
        V_all: 3D array of value function [time, s, x]
        u_star: 2D array of optimal drift control at t=0 [s, x]
        sigma_star: 2D array of optimal volatility at t=0 [s, x]
        K: strike price (for reference line)
        sig_min, sig_max: volatility bounds
        u_min, u_max: drift control bounds
    """

    # Set publication style
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.titlesize': 14
    })

    # ==========================================================================
    # Figure 1: Value function slices V(t, s, x0) vs s
    # ==========================================================================
    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 6))

    # Fixed x value (middle of domain)
    x0_idx = len(x_grid) // 2
    x0 = x_grid[x0_idx]

    # Time indices to plot
    t_indices = [0, len(t_grid) // 2, len(t_grid) - 1]
    colors = ['blue', 'red', 'green']
    linestyles = ['-', '--', ':']

    for i, t_idx in enumerate(t_indices):
        t = t_grid[t_idx]
        V_slice = V_all[t_idx, :, x0_idx]
        ax1.plot(s_grid, V_slice, color=colors[i], linestyle=linestyles[i],
                 linewidth=2, label=f't = {t:.2f}')

    # Add strike price reference
    ax1.axvline(K, color='gray', linestyle='--', linewidth=1, alpha=0.5, label=f'Strike K={K}')

    ax1.set_xlabel('Stock price s')
    ax1.set_ylabel('Value function V(t, s, x₀)')
    ax1.set_title(f'Value Function Slices at x = {x0:.2f}')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

    plt.tight_layout()
    plt.savefig('fig1_value_slices.png', dpi=300, bbox_inches='tight')
    print("Saved: fig1_value_slices.png")

    # ==========================================================================
    # Figure 2: Heatmap of optimal volatility σ*(0, s, x)
    # ==========================================================================
    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 6))

    im2 = ax2.imshow(sigma_star.T,
                     extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                     aspect='auto', origin='lower', cmap='RdYlBu_r',
                     vmin=sig_min, vmax=sig_max)

    # Add strike price reference
    ax2.axvline(K, color='white', linestyle='--', linewidth=1.5, alpha=0.7, label=f'Strike K={K}')

    ax2.set_xlabel('Stock price s')
    ax2.set_ylabel('Wealth x')
    ax2.set_title('Optimal Volatility σ*(0, s, x)')

    # Colorbar with labels
    cbar2 = plt.colorbar(im2, ax=ax2, label='σ*')
    cbar2.set_ticks([sig_min, (sig_min + sig_max) / 2, sig_max])
    cbar2.set_ticklabels([f'σ_min={sig_min}', f'{(sig_min + sig_max) / 2:.2f}', f'σ_max={sig_max}'])

    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('fig2_sigma_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: fig2_sigma_heatmap.png")

    # ==========================================================================
    # Figure 3: Heatmap of optimal drift control u*(0, s, x)
    # ==========================================================================
    fig3, ax3 = plt.subplots(1, 1, figsize=(8, 6))

    im3 = ax3.imshow(u_star.T,
                     extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                     aspect='auto', origin='lower', cmap='coolwarm',
                     vmin=u_min, vmax=u_max)

    # Add strike price reference
    ax3.axvline(K, color='black', linestyle='--', linewidth=1.5, alpha=0.5, label=f'Strike K={K}')

    # Add zero contour for u*
    S, X = np.meshgrid(s_grid, x_grid, indexing='ij')
    contour = ax3.contour(S, X, u_star, levels=[0], colors='green', linewidths=2, linestyles='-')
    ax3.clabel(contour, inline=True, fontsize=9, fmt='u*=0')

    ax3.set_xlabel('Stock price s')
    ax3.set_ylabel('Wealth x')
    ax3.set_title('Optimal Drift Control u*(0, s, x)')

    # Colorbar with bounds highlighted
    cbar3 = plt.colorbar(im3, ax=ax3, label='u*')
    cbar3.set_ticks([u_min, 0, u_max])
    cbar3.set_ticklabels([f'u_min={u_min}', '0', f'u_max={u_max}'])

    ax3.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig('fig3_u_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: fig3_u_heatmap.png")

    # ==========================================================================
    # Figure 4: Overlay showing σ* regions and |u*| magnitude
    # ==========================================================================
    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: σ* with |u*| contours
    tol_sigma = 0.05 * (sig_max - sig_min)
    sigma_high = (np.abs(sigma_star - sig_max) < tol_sigma).astype(float)

    im4a = ax4a.imshow(sigma_star.T,
                       extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                       aspect='auto', origin='lower', cmap='RdYlBu_r',
                       vmin=sig_min, vmax=sig_max, alpha=0.7)

    # Overlay |u*| contours
    u_abs = np.abs(u_star)
    levels_u = np.linspace(0, u_max, 6)
    contour4a = ax4a.contour(S, X, u_abs, levels=levels_u, colors='black',
                             linewidths=1.5, linestyles='-', alpha=0.8)
    ax4a.clabel(contour4a, inline=True, fontsize=8, fmt='|u*|=%.1f')

    ax4a.axvline(K, color='white', linestyle='--', linewidth=1, alpha=0.7)
    ax4a.set_xlabel('Stock price s')
    ax4a.set_ylabel('Wealth x')
    ax4a.set_title('σ* with |u*| Contours Overlaid')

    cbar4a = plt.colorbar(im4a, ax=ax4a, label='σ*')
    cbar4a.set_ticks([sig_min, sig_max])
    cbar4a.set_ticklabels([f'σ_min', f'σ_max'])

    # Right panel: Binary indicator showing σ*=σ_max regions with |u*| shading
    im4b = ax4b.imshow(u_abs.T,
                       extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                       aspect='auto', origin='lower', cmap='viridis',
                       vmin=0, vmax=u_max, alpha=0.7)

    # Overlay switching boundary
    contour4b = ax4b.contour(S, X, sigma_high, levels=[0.5], colors='red',
                             linewidths=3, linestyles='-')

    ax4b.axvline(K, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax4b.set_xlabel('Stock price s')
    ax4b.set_ylabel('Wealth x')
    ax4b.set_title('|u*| Magnitude with σ*=σ_max Boundary (red)')

    cbar4b = plt.colorbar(im4b, ax=ax4b, label='|u*|')

    # Add legend patch
    red_patch = mpatches.Patch(color='red', label='σ* = σ_max boundary')
    ax4b.legend(handles=[red_patch], loc='upper right')

    plt.tight_layout()
    plt.savefig('fig4_sigma_u_overlay.png', dpi=300, bbox_inches='tight')
    print("Saved: fig4_sigma_u_overlay.png")

    # ==========================================================================
    # Figure 5: Time evolution of σ* switching boundary
    # ==========================================================================
    fig5, axes5 = plt.subplots(2, 3, figsize=(15, 10))
    axes5 = axes5.flatten()

    # Time points to visualize
    n_times = 6
    t_indices_evolution = np.linspace(0, len(t_grid) - 1, n_times, dtype=int)

    for i, t_idx in enumerate(t_indices_evolution):
        ax = axes5[i]
        t = t_grid[t_idx]

        # Extract σ at this time (need to recompute or store it)
        # Since we only have sigma_star at t=0, we'll show V_all evolution instead
        # For proper σ* evolution, you'd need to store it at each time step

        # Show value function as proxy
        V_t = V_all[t_idx]

        im = ax.imshow(V_t.T,
                       extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                       aspect='auto', origin='lower', cmap='viridis')

        ax.axvline(K, color='white', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Stock price s')
        ax.set_ylabel('Wealth x')
        ax.set_title(f't = {t:.3f}')

        plt.colorbar(im, ax=ax, label='V')

    plt.suptitle('Time Evolution of Value Function V(t, s, x)', fontsize=14)
    plt.tight_layout()
    plt.savefig('fig5_time_evolution.png', dpi=300, bbox_inches='tight')
    print("Saved: fig5_time_evolution.png")

    # ==========================================================================
    # Figure 6: Additional - Control switching regions analysis
    # ==========================================================================
    fig6, ((ax6a, ax6b), (ax6c, ax6d)) = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: σ* binary classification
    sigma_classification = np.zeros_like(sigma_star)
    sigma_classification[sigma_star > (sig_min + sig_max) / 2] = 1

    im6a = ax6a.imshow(sigma_classification.T,
                       extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                       aspect='auto', origin='lower', cmap='RdBu_r',
                       vmin=0, vmax=1)
    ax6a.axvline(K, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6a.set_xlabel('Stock price s')
    ax6a.set_ylabel('Wealth x')
    ax6a.set_title('σ* Binary Classification')
    cbar6a = plt.colorbar(im6a, ax=ax6a, ticks=[0, 1])
    cbar6a.set_ticklabels(['σ_min', 'σ_max'])

    # Panel B: u* binary classification (near bounds)
    tol_u = 0.1 * (u_max - u_min)
    u_classification = np.zeros_like(u_star)
    u_classification[u_star > u_max - tol_u] = 1  # Upper bound
    u_classification[u_star < u_min + tol_u] = -1  # Lower bound

    im6b = ax6b.imshow(u_classification.T,
                       extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                       aspect='auto', origin='lower', cmap='RdYlGn',
                       vmin=-1, vmax=1)
    ax6b.axvline(K, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
    ax6b.set_xlabel('Stock price s')
    ax6b.set_ylabel('Wealth x')
    ax6b.set_title('u* Near-Boundary Classification')
    cbar6b = plt.colorbar(im6b, ax=ax6b, ticks=[-1, 0, 1])
    cbar6b.set_ticklabels(['u_min', 'interior', 'u_max'])

    # Panel C: Joint region analysis (σ=σ_max AND |u|≈u_max)
    joint_region = (sigma_classification == 1) & (np.abs(u_star) > u_max - tol_u)

    im6c = ax6c.imshow(joint_region.T.astype(float),
                       extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                       aspect='auto', origin='lower', cmap='Greys',
                       vmin=0, vmax=1)
    ax6c.axvline(K, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    ax6c.set_xlabel('Stock price s')
    ax6c.set_ylabel('Wealth x')
    ax6c.set_title('Joint Region: σ*=σ_max AND |u*|≈u_max')
    cbar6c = plt.colorbar(im6c, ax=ax6c, ticks=[0, 1])
    cbar6c.set_ticklabels(['No', 'Yes'])

    # Panel D: Slice comparison at x=0
    x_zero_idx = np.argmin(np.abs(x_grid))

    ax6d_twin = ax6d.twinx()

    line1 = ax6d.plot(s_grid, sigma_star[:, x_zero_idx], 'b-', linewidth=2, label='σ*')
    line2 = ax6d_twin.plot(s_grid, u_star[:, x_zero_idx], 'r--', linewidth=2, label='u*')

    ax6d.axvline(K, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax6d.axhline(sig_max, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    ax6d.axhline(sig_min, color='blue', linestyle=':', linewidth=1, alpha=0.5)
    ax6d_twin.axhline(u_max, color='red', linestyle=':', linewidth=1, alpha=0.5)
    ax6d_twin.axhline(u_min, color='red', linestyle=':', linewidth=1, alpha=0.5)

    ax6d.set_xlabel('Stock price s')
    ax6d.set_ylabel('σ*', color='b')
    ax6d_twin.set_ylabel('u*', color='r')
    ax6d.set_title(f'Control Profiles at x = {x_grid[x_zero_idx]:.2f}')
    ax6d.tick_params(axis='y', labelcolor='b')
    ax6d_twin.tick_params(axis='y', labelcolor='r')

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax6d.legend(lines, labels, loc='upper left')

    ax6d.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('fig6_control_regions.png', dpi=300, bbox_inches='tight')
    print("Saved: fig6_control_regions.png")

    # Show all figures
    plt.show()

    print("\nAll publication-ready figures generated successfully!")
    print("Summary:")
    print("  - fig1_value_slices.png: V(t,s,x0) evolution")
    print("  - fig2_sigma_heatmap.png: σ*(0,s,x) optimal volatility")
    print("  - fig3_u_heatmap.png: u*(0,s,x) optimal drift control")
    print("  - fig4_sigma_u_overlay.png: Joint σ* and u* analysis")
    print("  - fig5_time_evolution.png: V(t,s,x) time progression")
    print("  - fig6_control_regions.png: Detailed switching region analysis")


# =============================================================================
# Example usage
# =============================================================================
if __name__ == "__main__":
    # Load precomputed solution
    # Assuming you've run the solver and have these arrays

    try:
        # Option 1: Load from solver output
        from hjbi_solver import solve_hjbi

        print("Running HJBI solver to generate visualization data...")
        t_grid, s_grid, x_grid, V_all, u_star, sigma_star = solve_hjbi(
            T=1.0, r=0.02, K=1.0,
            sig_min=0.1, sig_max=0.4,
            u_min=-2.0, u_max=2.0,
            s_min=0.2, s_max=3.0,
            x_min=-2.0, x_max=2.0,
            Ns=80, Nx=80, Nt=80
        )

        print("\nGenerating publication plots...")
        create_publication_plots(t_grid, s_grid, x_grid, V_all, u_star, sigma_star,
                                 K=1.0, sig_min=0.1, sig_max=0.4, u_min=-2.0, u_max=2.0)

    except ImportError:
        print("Could not import solver. Please ensure hjbi_solver.py exists.")
        print("\nAlternatively, load precomputed arrays:")
        print("  data = np.load('solution.npz')")
        print("  t_grid = data['t_grid']")
        print("  s_grid = data['s_grid']")
        print("  x_grid = data['x_grid']")
        print("  V_all = data['V_all']")
        print("  u_star = data['u_star']")
        print("  sigma_star = data['sigma_star']")
        print("  create_publication_plots(t_grid, s_grid, x_grid, V_all, u_star, sigma_star)")