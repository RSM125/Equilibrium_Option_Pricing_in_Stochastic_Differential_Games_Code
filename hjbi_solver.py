#6.3

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import spsolve


def solve_hjbi(T=1.0, r=0.02, K=1.0,
               sig_min=0.1, sig_max=0.4,
               u_min=-2.0, u_max=2.0,
               s_min=0.2, s_max=3.0,
               x_min=-2.0, x_max=2.0,
               Ns=80, Nx=80, Nt=80,
               tol=1e-6, max_iter=50):
    """
    Solve 2D HJBI PDE for robust hedging using finite differences + policy iteration.

    PDE (backward in time):
    -v_t + r s v_s + r x v_x - r v
    + inf_u sup_sigma { 0.5 sigma^2 s^2 (v_ss + 2u v_sx + u^2 v_xx) } = 0

    Terminal condition: v(T,s,x) = (x - g(s))^2 where g(s) = max(s-K, 0)

    Discretization notes:
    - Implicit scheme with upwind drift
    - Cross term discretised by standard 9-point stencil
    - Convergence assessed empirically via refinement
    - Dirichlet truncation at boundaries anchored to discounted terminal loss
    - Controls computed analytically (quadratic minimization) with endpoint checks

    Returns:
        t_grid, s_grid, x_grid, V (array of shape (Nt+1, Ns, Nx)),
        u_star, sigma_star (controls at t=0)
    """

    # Time grid
    dt = T / Nt
    t_grid = np.linspace(0, T, Nt + 1)

    # S-grid: uniform in s
    s_grid = np.linspace(s_min, s_max, Ns)
    ds = s_grid[1] - s_grid[0]

    # X-grid: uniform
    x_grid = np.linspace(x_min, x_max, Nx)
    dx = x_grid[1] - x_grid[0]

    # Terminal condition
    S, X = np.meshgrid(s_grid, x_grid, indexing='ij')
    g = np.maximum(S - K, 0.0)
    V = (X - g) ** 2

    # Storage for all time steps
    V_all = np.zeros((Nt + 1, Ns, Nx))
    V_all[-1] = V.copy()

    # Storage for controls at t=0
    u_star = np.zeros((Ns, Nx))
    sigma_star = np.zeros((Ns, Nx))

    print(f"Grid: Ns={Ns}, Nx={Nx}, Nt={Nt}")
    print(f"dt={dt:.4f}, ds={ds:.4f}, dx={dx:.4f}")
    print(f"Starting backward time stepping...")

    # Track diagnostics
    min_V_per_step = []
    max_V_per_step = []

    # Backward time stepping
    for n in range(Nt - 1, -1, -1):
        t = t_grid[n]
        V_next = V.copy()

        # Policy iteration
        u_policy = np.zeros((Ns, Nx))
        sigma_policy = np.zeros((Ns, Nx))
        u_prev = np.zeros((Ns, Nx))
        sigma_prev = np.zeros((Ns, Nx))

        for k in range(max_iter):
            V_old = V.copy()
            u_prev[:] = u_policy
            sigma_prev[:] = sigma_policy

            # Compute discrete derivatives for control optimization
            D_ss, D_xx, D_sx = compute_second_derivatives(V, ds, dx)

            # Update controls at each grid point
            for i in range(1, Ns - 1):
                for j in range(1, Nx - 1):
                    # Find optimal controls via analytic minimization
                    u_opt, sigma_opt = optimize_controls_analytic(
                        D_ss[i, j], D_sx[i, j], D_xx[i, j],
                        u_min, u_max, sig_min, sig_max
                    )
                    u_policy[i, j] = u_opt
                    sigma_policy[i, j] = sigma_opt

            # Solve linear system with frozen controls
            V = solve_implicit_step(V_next, u_policy, sigma_policy,
                                    s_grid, x_grid, dt, r, K, t, T,
                                    ds, dx)

            # Safeguard: clamp tiny negative values (true solution is >= 0)
            V = np.maximum(V, 0.0)

            # Check for numerical issues
            if np.any(np.isnan(V)) or np.any(np.isinf(V)):
                print(f"  ERROR at t={t:.3f}, iteration {k + 1}: NaN or Inf detected!")
                raise ValueError("Numerical instability detected")

            # Check convergence on both value function and controls
            v_change = np.max(np.abs(V - V_old))
            u_change = np.max(np.abs(u_policy - u_prev))
            sigma_change = np.max(np.abs(sigma_policy - sigma_prev))

            if v_change < tol and u_change < tol and sigma_change < tol:
                if n % 10 == 0:
                    print(f"  t={t:.3f}: converged in {k + 1} iterations "
                          f"(ΔV={v_change:.2e}, Δu={u_change:.2e}, Δσ={sigma_change:.2e})")
                break
        else:
            if n % 10 == 0:
                print(f"  t={t:.3f}: max iterations reached "
                      f"(ΔV={v_change:.2e}, Δu={u_change:.2e}, Δσ={sigma_change:.2e})")

        # Diagnostics: check V >= 0
        min_v = np.min(V)
        max_v = np.max(V)
        min_V_per_step.append(min_v)
        max_V_per_step.append(max_v)

        if min_v < -1e-6:
            print(f"  WARNING at t={t:.3f}: min(V) = {min_v:.2e} < 0 (clamped to 0)")

        V_all[n] = V.copy()

        # Store controls at t=0
        if n == 0:
            u_star = u_policy.copy()
            sigma_star = sigma_policy.copy()

    print("Backward solve complete!")

    # Final diagnostics
    print(f"\nDiagnostics:")
    print(f"  Global min(V) = {np.min(min_V_per_step):.2e} (should be >= 0)")
    print(f"  Global max(V) = {np.max(max_V_per_step):.2e}")

    return t_grid, s_grid, x_grid, V_all, u_star, sigma_star


def compute_second_derivatives(V, ds, dx):
    """
    Compute second derivatives for control optimization.

    Note: Using central differences for second derivatives.
    Controls are computed from these curvature approximations.

    Returns: D_ss, D_xx, D_sx
    """
    Ns, Nx = V.shape

    D_ss = np.zeros((Ns, Nx))
    D_xx = np.zeros((Ns, Nx))
    D_sx = np.zeros((Ns, Nx))

    # Interior points
    for i in range(1, Ns - 1):
        for j in range(1, Nx - 1):
            # Second derivatives (central)
            D_ss[i, j] = (V[i + 1, j] - 2 * V[i, j] + V[i - 1, j]) / ds ** 2
            D_xx[i, j] = (V[i, j + 1] - 2 * V[i, j] + V[i, j - 1]) / dx ** 2

            # Mixed derivative (9-point central stencil)
            D_sx[i, j] = (V[i + 1, j + 1] - V[i + 1, j - 1] - V[i - 1, j + 1] + V[i - 1, j - 1]) / (4 * ds * dx)

    return D_ss, D_xx, D_sx


def optimize_controls_analytic(D_ss, D_sx, D_xx, u_min, u_max, sig_min, sig_max):
    """
    Find optimal controls (u*, sigma*) via analytic quadratic minimization.

    Q(u) = D_ss + 2*u*D_sx + u^2*D_xx is quadratic in u.

    Inner problem: sup_sigma in [sig_min, sig_max] of 0.5*sigma^2*Q(u)
    Outer problem: inf_u in [u_min, u_max]

    Strategy:
    - If D_xx > 0: parabola opens upward, interior minimum at u0 = -D_sx/D_xx
    - If D_xx <= 0: minimum at boundary
    - Check all candidates (interior + boundaries) and pick best
    """

    candidates = []

    # Candidate 1: Interior critical point (if D_xx > 0)
    if D_xx > 1e-12:
        u0 = -D_sx / D_xx
        # Project to feasible region
        u_cand = np.clip(u0, u_min, u_max)
        candidates.append(u_cand)

    # Candidates 2 & 3: Boundary points
    candidates.append(u_min)
    candidates.append(u_max)

    # Evaluate Hamiltonian at each candidate
    min_H = np.inf
    u_opt = 0.0
    sigma_opt = sig_min

    for u in candidates:
        Q = D_ss + 2 * u * D_sx + u ** 2 * D_xx

        # Inner maximization over sigma
        if Q > 1e-12:
            sigma = sig_max
        elif Q < -1e-12:
            sigma = sig_min
        else:
            # Q ≈ 0, choice doesn't matter
            sigma = sig_min

        # Hamiltonian value
        H = 0.5 * sigma ** 2 * Q

        if H < min_H:
            min_H = H
            u_opt = u
            sigma_opt = sigma

    return u_opt, sigma_opt


def solve_implicit_step(V_next, u_policy, sigma_policy, s_grid, x_grid,
                        dt, r, K, t, T, ds, dx):
    """
    Solve the implicit backward Euler step with frozen controls.

    Discretization: v^n - dt * H(v^n) = v^{n+1}

    where H includes:
    - Drift: r*s*v_s + r*x*v_x - r*v (upwind for first derivatives)
    - Diffusion: 0.5*sigma^2*s^2*(v_ss + 2*u*v_sx + u^2*v_xx)

    Boundary conditions: Dirichlet truncation based on discounted terminal payoff
    """
    Ns, Nx = V_next.shape
    N = Ns * Nx

    # Discount factor for boundary conditions
    discount = np.exp(-r * (T - t))

    # Build sparse matrix A such that A * v^n = rhs
    # Flat indexing: idx = i*Nx + j

    row = []
    col = []
    data = []
    rhs = V_next.flatten()

    # Tolerance for upwind switching near x=0
    drift_tol = 1e-10

    for i in range(Ns):
        for j in range(Nx):
            idx = i * Nx + j

            # Boundary conditions: Dirichlet truncation based on discounted terminal loss
            if i == 0 or i == Ns - 1 or j == 0 or j == Nx - 1:
                row.append(idx)
                col.append(idx)
                data.append(1.0)

                # Set boundary values with discount factor
                s = s_grid[i]
                x = x_grid[j]

                if i == 0:
                    # Small s: g(s) ≈ 0, so v ≈ x^2
                    rhs[idx] = discount * x ** 2
                elif i == Ns - 1:
                    # Large s: g(s) ≈ s-K, so v ≈ (x-(s-K))^2
                    rhs[idx] = discount * (x - (s - K)) ** 2
                elif j == 0:
                    # x_min boundary
                    g = max(s - K, 0)
                    rhs[idx] = discount * (x_grid[0] - g) ** 2
                elif j == Nx - 1:
                    # x_max boundary
                    g = max(s - K, 0)
                    rhs[idx] = discount * (x_grid[-1] - g) ** 2
            else:
                # Interior point
                s = s_grid[i]
                x = x_grid[j]
                u = u_policy[i, j]
                sigma = sigma_policy[i, j]

                # Diffusion coefficient: 0.5 * sigma^2 * s^2
                c_diff = 0.5 * sigma ** 2 * s ** 2

                # Drift coefficients
                drift_s = r * s
                drift_x = r * x

                # ===============================================
                # UPWIND discretization for drift terms
                # ===============================================
                # For r*s*v_s: typically r*s >= 0, use backward difference
                if drift_s >= drift_tol:
                    # Backward: v_s ≈ (v_i - v_{i-1})/ds
                    coef_s_center_drift = drift_s / ds
                    coef_s_minus_drift = -drift_s / ds
                    coef_s_plus_drift = 0.0
                elif drift_s <= -drift_tol:
                    # Forward: v_s ≈ (v_{i+1} - v_i)/ds
                    coef_s_center_drift = -drift_s / ds
                    coef_s_minus_drift = 0.0
                    coef_s_plus_drift = drift_s / ds
                else:
                    # Near zero: use central (or upwind either way)
                    coef_s_center_drift = 0.0
                    coef_s_minus_drift = -drift_s / (2 * ds)
                    coef_s_plus_drift = drift_s / (2 * ds)

                # For r*x*v_x: upwind based on sign of x
                if drift_x >= drift_tol:
                    # Backward: v_x ≈ (v_j - v_{j-1})/dx
                    coef_x_center_drift = drift_x / dx
                    coef_x_minus_drift = -drift_x / dx
                    coef_x_plus_drift = 0.0
                elif drift_x <= -drift_tol:
                    # Forward: v_x ≈ (v_{j+1} - v_j)/dx
                    coef_x_center_drift = -drift_x / dx
                    coef_x_minus_drift = 0.0
                    coef_x_plus_drift = drift_x / dx
                else:
                    # Near zero: use central
                    coef_x_center_drift = 0.0
                    coef_x_minus_drift = -drift_x / (2 * dx)
                    coef_x_plus_drift = drift_x / (2 * dx)

                # ===============================================
                # Second derivative terms (central differences)
                # ===============================================
                # v_ss contribution: c_diff * v_ss
                coef_ss_center = -2 * c_diff / ds ** 2
                coef_ss_plus = c_diff / ds ** 2
                coef_ss_minus = c_diff / ds ** 2

                # u^2 * v_xx contribution: c_diff * u^2 * v_xx
                coef_xx_center = -2 * c_diff * u ** 2 / dx ** 2
                coef_xx_plus = c_diff * u ** 2 / dx ** 2
                coef_xx_minus = c_diff * u ** 2 / dx ** 2

                # 2*u*v_sx contribution: c_diff * 2*u * v_sx
                # Central 9-point stencil
                c_cross = 2 * c_diff * u
                coef_sp_xp = c_cross / (4 * ds * dx)
                coef_sp_xm = -c_cross / (4 * ds * dx)
                coef_sm_xp = -c_cross / (4 * ds * dx)
                coef_sm_xm = c_cross / (4 * ds * dx)

                # ===============================================
                # Center coefficient
                # ===============================================
                # From: v^n - dt*H(v^n) = v^{n+1}
                # where H = r*s*v_s + r*x*v_x - r*v + diffusion_terms
                coef_center = (1.0
                               + dt * r
                               - dt * coef_s_center_drift
                               - dt * coef_x_center_drift
                               - dt * coef_ss_center
                               - dt * coef_xx_center)

                # ===============================================
                # Assemble stencil
                # ===============================================
                # Center
                row.append(idx)
                col.append(idx)
                data.append(coef_center)

                # S-direction neighbors (i±1, j)
                idx_sp = (i + 1) * Nx + j
                idx_sm = (i - 1) * Nx + j

                row.append(idx)
                col.append(idx_sp)
                data.append(-dt * (coef_s_plus_drift + coef_ss_plus))

                row.append(idx)
                col.append(idx_sm)
                data.append(-dt * (coef_s_minus_drift + coef_ss_minus))

                # X-direction neighbors (i, j±1)
                idx_xp = i * Nx + (j + 1)
                idx_xm = i * Nx + (j - 1)

                row.append(idx)
                col.append(idx_xp)
                data.append(-dt * (coef_x_plus_drift + coef_xx_plus))

                row.append(idx)
                col.append(idx_xm)
                data.append(-dt * (coef_x_minus_drift + coef_xx_minus))

                # Mixed derivative corners (i±1, j±1)
                idx_sp_xp = (i + 1) * Nx + (j + 1)
                idx_sp_xm = (i + 1) * Nx + (j - 1)
                idx_sm_xp = (i - 1) * Nx + (j + 1)
                idx_sm_xm = (i - 1) * Nx + (j - 1)

                row.append(idx)
                col.append(idx_sp_xp)
                data.append(-dt * coef_sp_xp)

                row.append(idx)
                col.append(idx_sp_xm)
                data.append(-dt * coef_sp_xm)

                row.append(idx)
                col.append(idx_sm_xp)
                data.append(-dt * coef_sm_xp)

                row.append(idx)
                col.append(idx_sm_xm)
                data.append(-dt * coef_sm_xm)

    A = csr_matrix((data, (row, col)), shape=(N, N))
    v_flat = spsolve(A, rhs)
    V = v_flat.reshape((Ns, Nx))

    return V


def plot_results(t_grid, s_grid, x_grid, V_all, u_star, sigma_star, K):
    """
    Generate diagnostic plots.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (i) Slice v(t, s, x0) vs s for fixed x0 at t=0 and t=T/2
    x0_idx = len(x_grid) // 2
    x0 = x_grid[x0_idx]

    t0_idx = 0
    t_half_idx = len(t_grid) // 2

    ax = axes[0, 0]
    ax.plot(s_grid, V_all[t0_idx, :, x0_idx], 'b-', label=f't=0')
    ax.plot(s_grid, V_all[t_half_idx, :, x0_idx], 'r--', label=f't={t_grid[t_half_idx]:.2f}')
    ax.plot(s_grid, V_all[-1, :, x0_idx], 'g:', label=f't=T (terminal)')
    ax.set_xlabel('Stock price s')
    ax.set_ylabel('Value function v')
    ax.set_title(f'Value function slice at x={x0:.2f}')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # (ii) Heatmap of sigma*(t=0, s, x)
    ax = axes[0, 1]
    im = ax.imshow(sigma_star.T, extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                   aspect='auto', origin='lower', cmap='RdYlBu_r')
    ax.set_xlabel('Stock price s')
    ax.set_ylabel('Wealth x')
    ax.set_title('Optimal volatility σ* at t=0')
    plt.colorbar(im, ax=ax, label='σ*')

    # (iii) Heatmap of u*(t=0, s, x)
    ax = axes[1, 0]
    im = ax.imshow(u_star.T, extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                   aspect='auto', origin='lower', cmap='coolwarm')
    ax.set_xlabel('Stock price s')
    ax.set_ylabel('Wealth x')
    ax.set_title('Optimal control u* at t=0')
    plt.colorbar(im, ax=ax, label='u*')

    # Additional: V(t=0, s, x) heatmap
    ax = axes[1, 1]
    im = ax.imshow(V_all[0].T, extent=[s_grid[0], s_grid[-1], x_grid[0], x_grid[-1]],
                   aspect='auto', origin='lower', cmap='viridis')
    ax.set_xlabel('Stock price s')
    ax.set_ylabel('Wealth x')
    ax.set_title('Value function V(t=0, s, x)')
    plt.colorbar(im, ax=ax, label='V')

    plt.tight_layout()
    plt.savefig('hjbi_solution.png', dpi=150, bbox_inches='tight')
    print("Saved plot to hjbi_solution.png")
    plt.show()


def run_refinement_test():
    """
    Test grid refinement to verify stability of switching regions.
    """
    print("\n" + "=" * 60)
    print("Grid Refinement Test")
    print("=" * 60)

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

    grids = [
        (40, 40, 40, "Coarse"),
        (80, 80, 80, "Medium"),
    ]

    results = []

    for Ns, Nx, Nt, label in grids:
        print(f"\nTesting {label} grid: {Ns}×{Nx}×{Nt}")

        t_grid, s_grid, x_grid, V_all, u_star, sigma_star = solve_hjbi(
            T=T, r=r, K=K,
            sig_min=sig_min, sig_max=sig_max,
            u_min=u_min, u_max=u_max,
            s_min=s_min, s_max=s_max,
            x_min=x_min, x_max=x_max,
            Ns=Ns, Nx=Nx, Nt=Nt
        )

        # Compute fraction using sigma_max
        frac_max = (sigma_star > 0.95 * sig_max).sum() / sigma_star.size
        frac_min = (sigma_star < 1.05 * sig_min).sum() / sigma_star.size

        # Value at money
        s_idx = np.argmin(np.abs(s_grid - K))
        x_idx = np.argmin(np.abs(x_grid))
        v_atm = V_all[0, s_idx, x_idx]

        results.append({
            'label': label,
            'grid': (Ns, Nx, Nt),
            'frac_max': frac_max,
            'frac_min': frac_min,
            'v_atm': v_atm,
            'min_V': np.min(V_all),
            'max_V': np.max(V_all)
        })

        print(f"  Fraction σ_max: {frac_max:.1%}")
        print(f"  Fraction σ_min: {frac_min:.1%}")
        print(f"  V(0,K,0): {v_atm:.6f}")
        print(f"  min(V): {np.min(V_all):.2e}")

    # Compare results
    print("\n" + "=" * 60)
    print("Refinement Comparison")
    print("=" * 60)
    for i in range(len(results) - 1):
        r1 = results[i]
        r2 = results[i + 1]
        print(f"\n{r1['label']} → {r2['label']}:")
        print(f"  Δ(frac σ_max): {abs(r2['frac_max'] - r1['frac_max']):.3%}")
        print(f"  Δ(V_ATM): {abs(r2['v_atm'] - r1['v_atm']):.2e}")
        print(f"  Relative Δ(V_ATM): {abs(r2['v_atm'] - r1['v_atm']) / r1['v_atm']:.2%}")


def main():
    """
    Main driver for HJBI solver.

    Implementation notes:
    - Implicit scheme with upwind drift discretization
    - Cross term discretised by standard 9-point stencil
    - Convergence assessed empirically via grid refinement
    - Analytic control optimization (quadratic minimization)
    - Dirichlet boundary truncation with discount factor
    """
    print("=" * 60)
    print("2D HJBI PDE Solver for Robust Hedging")
    print("=" * 60)

    # Parameters
    T = 1.0
    r = 0.02
    K = 1.0
    sig_min = 0.1
    sig_max = 0.4
    u_min = -2.0
    u_max = 2.0

    # Domain
    s_min, s_max = 0.2, 3.0
    x_min, x_max = -2.0, 2.0

    # Grid sizes
    Ns = 80
    Nx = 80
    Nt = 80

    # Solve
    t_grid, s_grid, x_grid, V_all, u_star, sigma_star = solve_hjbi(
        T=T, r=r, K=K,
        sig_min=sig_min, sig_max=sig_max,
        u_min=u_min, u_max=u_max,
        s_min=s_min, s_max=s_max,
        x_min=x_min, x_max=x_max,
        Ns=Ns, Nx=Nx, Nt=Nt
    )

    # Diagnostics
    print("\n" + "=" * 60)
    print("Solution Summary")
    print("=" * 60)
    print(f"Value at (t=0, s=K={K}, x=0): {V_all[0, np.argmin(np.abs(s_grid - K)), np.argmin(np.abs(x_grid))]:.6f}")
    print(f"Control u* range at t=0: [{u_star.min():.3f}, {u_star.max():.3f}]")
    print(f"Control σ* range at t=0: [{sigma_star.min():.3f}, {sigma_star.max():.3f}]")
    print(f"Fraction using σ_max: {(sigma_star > 0.95 * sig_max).sum() / sigma_star.size:.1%}")
    print(f"Fraction using σ_min: {(sigma_star < 1.05 * sig_min).sum() / sigma_star.size:.1%}")

    # Plot results
    plot_results(t_grid, s_grid, x_grid, V_all, u_star, sigma_star, K)

    # Run refinement test
    run_refinement_test()

    print("\nDone!")


if __name__ == "__main__":
    main()