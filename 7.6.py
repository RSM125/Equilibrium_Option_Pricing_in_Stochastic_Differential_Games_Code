import numpy as np
import matplotlib.pyplot as plt
from math import erf, sqrt, exp, log


# ============================================================
# 7.6 — Black–Scholes hedge vs "Robust" hedge (HJBI-inspired)
# Generates all 3 visuals:
#   (1) Tracking error time series on ONE identical path
#   (2) Terminal hedging error histogram across Monte Carlo paths
#   (3) Mean absolute tracking error vs time across paths
#
# FIXED: Robust hedge now initializes with V^rob(0,S0) using sigma_star
#        at t=0, which centers the error distribution properly.
# ============================================================


# ---------------------------
# Black–Scholes primitives
# ---------------------------
def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bs_call_price(S: float, K: float, r: float, tau: float, sigma: float) -> float:
    if tau <= 0:
        return max(S - K, 0.0)
    if sigma <= 0:
        # deterministic limit
        return max(S - K * exp(-r * tau), 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt(tau))
    d2 = d1 - sigma * sqrt(tau)
    return S * norm_cdf(d1) - K * exp(-r * tau) * norm_cdf(d2)


def bs_call_delta(S: float, K: float, r: float, tau: float, sigma: float) -> float:
    if tau <= 0:
        return 1.0 if S > K else 0.0
    if sigma <= 0:
        return 1.0 if S > K * exp(-r * tau) else 0.0
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * tau) / (sigma * sqrt(tau))
    return norm_cdf(d1)


# ----------------------------------------------------
# State-dependent "robust" effective volatility σ*(t,x)
# (HJBI-inspired switching / saturation)
# ----------------------------------------------------
def sigma_star_multilevel(t: float, x: float, x0: float, T: float,
                          sigma_low: float, sigma_mid: float, sigma_high: float,
                          thr1: float = 0.02, thr2: float = 0.06,
                          mode: str = "standard") -> float:
    """
    x is log-price; x0 = log(S0). Thresholds are in |x-x0|.
    mode:
      - "standard": low in centre, higher in tails
      - "adversarial": high by default (optional)
    """
    y = abs(x - x0)

    if mode == "standard":
        if y <= thr1:
            return sigma_low
        elif y <= thr2:
            return sigma_mid
        else:
            return sigma_high

    if mode == "adversarial":
        # mostly-high (not used by default below, but available)
        if y <= thr1:
            return sigma_mid
        else:
            return sigma_high

    raise ValueError("mode must be 'standard' or 'adversarial'")


# ---------------------------
# True dynamics (misspec setup)
# ---------------------------
def sigma_true(t: float, x: float, x0: float, T: float,
               sigma_calm: float, sigma_stress: float,
               stress_thr: float = 0.05) -> float:
    """
    Simple stress activation: once |x-x0| exceeds threshold, true vol becomes high.
    This creates volatility misspecification for BS (if sigma_model is calm).
    """
    return sigma_stress if abs(x - x0) > stress_thr else sigma_calm


# ---------------------------
# Path simulation
# ---------------------------
def simulate_paths_gbm_state_dependent_sigma(
        M: int, N: int, T: float,
        S0: float, r: float,
        sigma_calm: float, sigma_stress: float, stress_thr: float,
        seed: int = 0
):
    """
    Simulate M paths with state-dependent true sigma.
    Returns:
      t_grid (N+1,), S_paths (M,N+1), X_paths (M,N+1), dW (M,N)
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0.0, T, N + 1)

    S = np.zeros((M, N + 1))
    X = np.zeros((M, N + 1))
    dW = rng.standard_normal((M, N)) * sqrt(dt)

    S[:, 0] = S0
    X[:, 0] = log(S0)
    x0 = X[0, 0]

    for k in range(N):
        tk = t[k]
        xk = X[:, k]

        sig = np.array([sigma_true(tk, x, x0, T, sigma_calm, sigma_stress, stress_thr) for x in xk])

        # risk-neutral drift r
        X[:, k + 1] = xk + (r - 0.5 * sig * sig) * dt + sig * dW[:, k]
        S[:, k + 1] = np.exp(X[:, k + 1])

    return t, S, X


# ---------------------------
# Hedging engine
# ---------------------------
def hedge_along_paths(
        t: np.ndarray, S: np.ndarray, X: np.ndarray,
        K: float, r: float, T: float,
        sigma_model: float,
        sigma_low: float, sigma_mid: float, sigma_high: float,
        thr1: float, thr2: float
):
    """
    Compute BS hedge + robust hedge along identical paths.
    Robust hedge uses delta computed with sigma*(t,logS).

    FIXED: Robust hedge now initializes with V^rob(0,S0) = BS(S0, sigma_star(0, log(S0)))

    Returns tracking errors and terminal errors.
    """
    M, Np1 = S.shape
    N = Np1 - 1
    dt = T / N
    x0 = log(S[0, 0])

    # BS initial capital
    C0_bs = bs_call_price(S[0, 0], K, r, T, sigma_model)

    # Robust initial capital: use actual HJBI value V^rob(0, S0)
    # This is BS price with sigma_star(0, log(S0))
    sig0_rob = sigma_star_multilevel(0.0, x0, x0, T, sigma_low, sigma_mid, sigma_high, thr1, thr2, mode="standard")
    C0_rob = bs_call_price(S[0, 0], K, r, T, sig0_rob)

    # portfolios
    Xp_bs = np.zeros((M, Np1))
    Xp_rb = np.zeros((M, Np1))
    Xp_bs[:, 0] = C0_bs
    Xp_rb[:, 0] = C0_rob

    # tracking error vs each hedger's mark-to-model option price
    track_bs = np.zeros((M, Np1))
    track_rb = np.zeros((M, Np1))

    for k in range(N):
        tk = t[k]
        tau = max(T - tk, 0.0)

        Sk = S[:, k]
        Xk = X[:, k]

        # --- BS hedge ---
        delta_bs = np.array([bs_call_delta(s, K, r, tau, sigma_model) for s in Sk])
        Vk_bs = np.array([bs_call_price(s, K, r, tau, sigma_model) for s in Sk])

        # self-financing update
        cash_bs = Xp_bs[:, k] - delta_bs * Sk
        Xp_bs[:, k + 1] = cash_bs * exp(r * dt) + delta_bs * S[:, k + 1]

        # --- Robust hedge (HJBI-inspired) ---
        sig_eff = np.array([
            sigma_star_multilevel(tk, x, x0, T, sigma_low, sigma_mid, sigma_high, thr1, thr2, mode="standard")
            for x in Xk
        ])
        delta_rb = np.array([bs_call_delta(s, K, r, tau, sig) for s, sig in zip(Sk, sig_eff)])
        Vk_rb = np.array([bs_call_price(s, K, r, tau, sig) for s, sig in zip(Sk, sig_eff)])

        cash_rb = Xp_rb[:, k] - delta_rb * Sk
        Xp_rb[:, k + 1] = cash_rb * exp(r * dt) + delta_rb * S[:, k + 1]

        # tracking errors at time tk (portfolio - option mark-to-model)
        track_bs[:, k] = Xp_bs[:, k] - Vk_bs
        track_rb[:, k] = Xp_rb[:, k] - Vk_rb

    # final tracking error (tau=0, option value = payoff)
    payoff = np.maximum(S[:, -1] - K, 0.0)
    track_bs[:, -1] = Xp_bs[:, -1] - payoff
    track_rb[:, -1] = Xp_rb[:, -1] - payoff

    term_err_bs = Xp_bs[:, -1] - payoff
    term_err_rb = Xp_rb[:, -1] - payoff

    return {
        "Xp_bs": Xp_bs, "Xp_rb": Xp_rb,
        "track_bs": track_bs, "track_rb": track_rb,
        "term_err_bs": term_err_bs, "term_err_rb": term_err_rb,
        "payoff": payoff
    }


# ---------------------------
# Plotting (all 3 visuals)
# ---------------------------
def plot_all_three(t, S, results, path_index: int = 0, bins: int = 60):
    track_bs = results["track_bs"]
    track_rb = results["track_rb"]
    term_err_bs = results["term_err_bs"]
    term_err_rb = results["term_err_rb"]

    # (1) Tracking error time series on one identical path
    plt.figure()
    plt.plot(t, track_bs[path_index, :], label="BS tracking error")
    plt.plot(t, track_rb[path_index, :], label="Robust tracking error")
    plt.xlabel("Time")
    plt.ylabel("Portfolio − option value")
    plt.title("Tracking error along one identical realised path")
    plt.legend()
    plt.show()

    # (2) Terminal hedging error histogram across Monte Carlo
    plt.figure()
    plt.hist(term_err_bs, bins=bins, alpha=0.6, label="BS terminal error")
    plt.hist(term_err_rb, bins=bins, alpha=0.6, label="Robust terminal error")
    plt.xlabel("Terminal hedging error  ε = X_T − payoff")
    plt.ylabel("Frequency")
    plt.title("Terminal hedging error distribution (same true dynamics)")
    plt.legend()
    plt.show()

    # (3) Mean absolute tracking error vs time across paths
    mean_abs_bs = np.mean(np.abs(track_bs), axis=0)
    mean_abs_rb = np.mean(np.abs(track_rb), axis=0)

    plt.figure()
    plt.plot(t, mean_abs_bs, label="BS mean |tracking error|")
    plt.plot(t, mean_abs_rb, label="Robust mean |tracking error|")
    plt.xlabel("Time")
    plt.ylabel("Mean absolute tracking error")
    plt.title("Mean absolute tracking error vs time (Monte Carlo)")
    plt.legend()
    plt.show()


# ---------------------------
# Run (edit parameters here)
# ---------------------------
if __name__ == "__main__":
    # Time/grid
    T = 1.0
    N = 400

    # Market
    S0 = 100.0
    r = 0.02

    # Option
    K = 100.0

    # True dynamics (misspecification is created by stress activation)
    sigma_calm_true = 0.15
    sigma_stress_true = 0.35
    stress_thr = 0.05  # in |logS - logS0|; smaller -> stress more frequent

    # Hedgers
    sigma_model_bs = 0.15  # BS believes calm vol always

    # Robust effective-vol levels (HJBI-like switching)
    sigma_low = 0.10
    sigma_mid = 0.20
    sigma_high = 0.35
    thr1, thr2 = 0.02, 0.06  # thresholds in |logS - logS0|

    # Monte Carlo
    M = 5000
    seed = 7

    # Simulate identical true paths
    t, S, X = simulate_paths_gbm_state_dependent_sigma(
        M=M, N=N, T=T,
        S0=S0, r=r,
        sigma_calm=sigma_calm_true,
        sigma_stress=sigma_stress_true,
        stress_thr=stress_thr,
        seed=seed
    )

    # Hedge along those identical paths
    results = hedge_along_paths(
        t=t, S=S, X=X,
        K=K, r=r, T=T,
        sigma_model=sigma_model_bs,
        sigma_low=sigma_low, sigma_mid=sigma_mid, sigma_high=sigma_high,
        thr1=thr1, thr2=thr2
    )

    # Produce all three visuals
    plot_all_three(t, S, results, path_index=0, bins=70)