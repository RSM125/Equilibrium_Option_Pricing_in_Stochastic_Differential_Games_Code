import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# Single qualitative visualisation for Section 7.5:
#   Effective volatility σ*(t, x) vs state x at a fixed time t0
#
# You have TWO ways to generate it:
#   A) From your computed control map on a grid (recommended)
#   B) From your toy rule-based σ*(t,x) function (if no grid saved)
# ============================================================


# ---------------------------
# A) RECOMMENDED: from grid
# ---------------------------
def plot_sigma_star_from_grid(t_grid, x_grid, sigma_star_grid, t0, title_prefix="Effective volatility profile"):
    """
    t_grid: 1D array of times, shape (Nt,)
    x_grid: 1D array of states, shape (Nx,)
    sigma_star_grid: 2D array σ*(t_i, x_j), shape (Nt, Nx)
    t0: time at which to take the slice

    Produces one plot: x -> σ*(t0, x)
    """
    t_grid = np.asarray(t_grid)
    x_grid = np.asarray(x_grid)
    sigma_star_grid = np.asarray(sigma_star_grid)

    # nearest time index
    i = int(np.argmin(np.abs(t_grid - t0)))
    t_used = t_grid[i]
    sigma_slice = sigma_star_grid[i, :]

    plt.figure()
    plt.plot(x_grid, sigma_slice)
    plt.xlabel("State x")
    plt.ylabel("Effective volatility σ*(t, x)")
    plt.title(f"{title_prefix}: σ*(t={t_used:.3f}, x)")
    plt.show()


# -----------------------------------------
# B) FALLBACK: from rule-based σ*(t, x)
# -----------------------------------------
def sigma_star_multilevel(t, x, x0, T, levels, thresholds, mode="standard"):
    """
    Discrete multi-level σ*(t,x) used earlier.
    levels: [low, mid, high] increasing
    thresholds: [a1, a2] in |x-x0|
    mode:
      - "standard": low in center, higher outside
      - "adversarial": high by default (or mid in very calm band)
    """
    y = abs(x - x0)
    a1, a2 = thresholds
    sigma_low, sigma_mid, sigma_high = levels

    if mode == "standard":
        if y <= a1:
            return sigma_low
        elif y <= a2:
            return sigma_mid
        else:
            return sigma_high

    if mode == "adversarial":
        if y <= a1:
            return sigma_mid   # very calm inner band
        elif y <= a2:
            return sigma_high
        else:
            return sigma_high

    raise ValueError("mode must be 'standard' or 'adversarial'")


def plot_sigma_star_from_rule(t0, x_min, x_max, Nx, S0=100.0, T=1.0,
                              sigma_low=0.10, sigma_mid=0.20, sigma_high=0.35,
                              thresholds=(0.02, 0.06), mode="standard",
                              title_prefix="Effective volatility profile"):
    """
    One plot: x -> σ*(t0, x)
    Here x is a generic state (e.g. log-price). If your state is log(S), use x0=log(S0).
    """
    x_grid = np.linspace(x_min, x_max, Nx)
    x0 = np.log(S0)

    sigma_vals = np.array([
        sigma_star_multilevel(
            t=t0, x=x, x0=x0, T=T,
            levels=[sigma_low, sigma_mid, sigma_high],
            thresholds=list(thresholds),
            mode=mode
        )
        for x in x_grid
    ])

    plt.figure()
    plt.plot(x_grid, sigma_vals)
    plt.xlabel("State x")
    plt.ylabel("Effective volatility σ*(t, x)")
    plt.title(f"{title_prefix}: σ*(t={t0:.3f}, x)  [{mode}]")
    plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # >>> Choose ONE of the two blocks below <<<

    # ========== A) If you have grid outputs from Section 6 ==========
    # Example placeholders (replace with your actual arrays):
    # t_grid = np.linspace(0, 1.0, 401)
    # x_grid = np.linspace(np.log(60), np.log(140), 301)
    # sigma_star_grid = ...  # shape (len(t_grid), len(x_grid))
    # plot_sigma_star_from_grid(t_grid, x_grid, sigma_star_grid, t0=0.5)

    # ========== B) If you only have rule-based σ*(t,x) ==========
    # Choose a sensible x-range for your state; if x=log(S), center around log(S0)
    S0 = 100.0
    x0 = np.log(S0)

    # Plot at mid-horizon (common choice for a qualitative slice)
    t0 = 0.5

    # Range: +/- 35% in price corresponds to log-range about +/- log(1.35)
    x_min = x0 - np.log(1.35)
    x_max = x0 + np.log(1.35)

    plot_sigma_star_from_rule(
        t0=t0,
        x_min=x_min,
        x_max=x_max,
        Nx=400,
        S0=S0,
        T=1.0,
        sigma_low=0.10,
        sigma_mid=0.20,
        sigma_high=0.35,
        thresholds=(0.02, 0.06),   # tighten/loosen for clearer steps
        mode="standard",           # use "adversarial" if you want mostly-high
        title_prefix="Figure 7.5"
    )
