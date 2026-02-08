import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Option A: make switching actually happen (state-dependent)
# Key changes:
#  1) Switching rule uses deviation y = X - X0 (not |X|).
#  2) "Rising uncertainty" uses time-varying bounds [low(t), high(t)]
#     and applies the SAME switching rule, so it can actually hit high(t).
#  3) Switching indicator compares to the correct high bound (time-varying).
#  4) FIXED: Rising shows discrete steps > ramp; Adversarial is pure worst-case.
# ============================================================

def sigma_star_rule(t: float, x: float, x0: float, T: float,
                    sigma_low: float, sigma_high: float,
                    base_thresh: float = 0.10, k: float = 0.25,
                    n_levels: int = 3) -> float:
    """
    State-dependent multi-level volatility control.

    Creates n_levels discrete volatility levels based on deviation from x0.
    Thresholds shrink as maturity approaches: boundary(t) = base_thresh + k*sqrt(T-t)
    """
    tau = max(T - t, 0.0)
    y = abs(x - x0)  # deviation from initial log-price

    # Create n_levels evenly spaced volatility levels
    sigma_range = sigma_high - sigma_low
    levels = [sigma_low + (i / (n_levels - 1)) * sigma_range
              for i in range(n_levels)]

    # Create thresholds based on deviation (scaled by time-to-maturity)
    base = base_thresh + k * np.sqrt(tau)
    thresholds = [base * (i + 1) / n_levels for i in range(n_levels - 1)]

    # Determine which level based on deviation
    for i, thresh in enumerate(thresholds):
        if y < thresh:
            return levels[i]
    return levels[-1]  # highest level if all thresholds exceeded


def simulate_path(
        T: float,
        N: int,
        S0: float,
        mu: float,
        scenario: str,
        sigma_calm: float,
        sigma_rising_start: float,
        sigma_rising_end: float,
        rising_spread: float,
        sigma_low: float,
        sigma_high: float,
        seed: int = 0,
        hold_time: int = 15,  # minimum steps to hold a volatility level (hysteresis)
):
    """
    Simulates one path under:
      - 'calm'       : constant sigma_calm
      - 'rising'     : time-varying bounds [low(t), high(t) = low(t)+spread]
                       with discrete state-dependent switching and hysteresis
      - 'adversarial': pure worst-case (sigma_high always)

    Returns: t, S, sigma_used, switch_indicator
    """
    rng = np.random.default_rng(seed)
    dt = T / N
    t = np.linspace(0.0, T, N + 1)

    S = np.zeros(N + 1)
    X = np.zeros(N + 1)
    sigma_used = np.zeros(N + 1)
    switch = np.zeros(N + 1, dtype=int)

    S[0] = S0
    X[0] = np.log(S0)
    x0 = X[0]

    # For hysteresis: track current sigma and how long we've held it
    current_sigma = sigma_calm if scenario == "calm" else sigma_rising_start
    hold_counter = 0

    for n in range(N):
        tn = t[n]
        xn = X[n]

        if scenario == "calm":
            # No robustness here: just show "inactive" behaviour.
            sig = sigma_calm
            high_bound = sigma_high  # irrelevant; indicator should remain 0

        elif scenario == "rising":
            # Rising uncertainty: widening *bounds* over time + state-dependent switching
            low_bound = sigma_rising_start + (sigma_rising_end - sigma_rising_start) * (tn / T)
            high_bound = min(low_bound + rising_spread, sigma_high)

            # Compute candidate sigma from rule
            candidate_sig = sigma_star_rule(
                t=tn, x=xn, x0=x0, T=T,
                sigma_low=low_bound, sigma_high=high_bound,
                base_thresh=0.10, k=0.25, n_levels=3
            )

            # Apply hysteresis: only switch if we've held current level long enough
            if hold_counter >= hold_time:
                if not np.isclose(candidate_sig, current_sigma, rtol=0.0, atol=1e-9):
                    current_sigma = candidate_sig
                    hold_counter = 0
                else:
                    hold_counter += 1
            else:
                hold_counter += 1

            sig = current_sigma

        elif scenario == "adversarial":
            # Pure worst-case: always at upper bound
            low_bound = sigma_low
            high_bound = sigma_high
            sig = sigma_high

        else:
            raise ValueError("scenario must be one of: 'calm', 'rising', 'adversarial'")

        sigma_used[n] = sig
        switch[n] = int(np.isclose(sig, high_bound, rtol=0.0, atol=1e-12))

        # Evolve log-price with chosen sigma
        z = rng.standard_normal()
        X[n + 1] = xn + (mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * z
        S[n + 1] = np.exp(X[n + 1])

    # Repeat last values for plotting alignment
    sigma_used[-1] = sigma_used[-2]
    switch[-1] = switch[-2]

    return t, S, sigma_used, switch


def plot_scenario(t, S, sigma_used, switch, title_prefix: str):
    # 1) Price path
    plt.figure()
    plt.plot(t, S)
    plt.xlabel("Time")
    plt.ylabel("Price S_t")
    plt.title(f"{title_prefix}: Price path S_t")
    plt.show()

    # 2) Active sigma over time
    plt.figure()
    plt.plot(t, sigma_used)
    plt.xlabel("Time")
    plt.ylabel("Active volatility σ*(t, X_t)")
    plt.title(f"{title_prefix}: Active volatility over time")
    plt.show()

    # 3) Switching indicator
    plt.figure()
    plt.step(t, switch, where="post")
    plt.ylim(-0.1, 1.1)
    plt.xlabel("Time")
    plt.ylabel("I_t = 1{σ* = upper bound}")
    plt.title(f"{title_prefix}: Switching indicator (stress regime)")
    plt.show()


if __name__ == "__main__":
    # Common parameters
    T = 1.0
    N = 400
    S0 = 100.0
    mu = 0.02

    # Calm scenario (baseline)
    sigma_calm = 0.15

    # Rising uncertainty: REDUCED ramp to make steps dominant
    # Ramp contributes ~30% of total variation (0.08 out of ~0.26 total range)
    sigma_rising_start = 0.12
    sigma_rising_end = 0.20  # REDUCED from 0.22 to flatten baseline
    rising_spread = 0.14  # INCREASED from 0.10 to make steps bigger

    # Adversarial bounds
    sigma_low = 0.10
    sigma_high = 0.35

    # Calm
    t, S, sig, sw = simulate_path(
        T=T, N=N, S0=S0, mu=mu,
        scenario="calm",
        sigma_calm=sigma_calm,
        sigma_rising_start=sigma_rising_start,
        sigma_rising_end=sigma_rising_end,
        rising_spread=rising_spread,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        seed=1
    )
    plot_scenario(t, S, sig, sw, "Calm volatility")

    # Rising uncertainty (discrete steps should dominate visual)
    t, S, sig, sw = simulate_path(
        T=T, N=N, S0=S0, mu=mu,
        scenario="rising",
        sigma_calm=sigma_calm,
        sigma_rising_start=sigma_rising_start,
        sigma_rising_end=sigma_rising_end,
        rising_spread=rising_spread,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        seed=2,
        hold_time=15  # hysteresis to prevent flickering
    )
    plot_scenario(t, S, sig, sw, "Rising uncertainty")

    # Adversarial volatility choice (pure worst-case: sigma_high always)
    t, S, sig, sw = simulate_path(
        T=T, N=N, S0=S0, mu=mu,
        scenario="adversarial",
        sigma_calm=sigma_calm,
        sigma_rising_start=sigma_rising_start,
        sigma_rising_end=sigma_rising_end,
        rising_spread=rising_spread,
        sigma_low=sigma_low,
        sigma_high=sigma_high,
        seed=3
    )
    plot_scenario(t, S, sig, sw, "Adversarial volatility choice")