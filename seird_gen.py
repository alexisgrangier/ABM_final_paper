"""
seird_abm.py
============
SEIRD Agent-Based Model — standalone, tunable, with diagnostics.

Run modes:
    python seird_abm.py --mode diagnose   # single run, full diagnostics (START HERE)
    python seird_abm.py --mode sweep      # parameter sweep across beta0 × zone × policy
    python seird_abm.py --mode generate   # full 400-run output for PyMC (after tuning)

Primary outcome variable: mean_peak_infectious
    = mean height of all detected epidemic wave peaks across the simulation
    = captures sustained policy effect across multiple waves
    = used as the outcome in the hierarchical PyMC model

Tuning workflow:
    1. Run --mode diagnose and read the printed summary + diagnostics.png
    2. Adjust TUNING PARAMETERS below until all checklist items show ✓
    3. Run --mode sweep to verify all 10 conditions look reasonable
    4. Run --mode generate to produce abm_outputs.csv

Dependencies: run the following line of code:
    pip install numpy pandas scipy matplotlib
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
from scipy.signal import find_peaks

# ═════════════════════════════════════════════════════════════════════════════
# ██  TUNING PARAMETERS — adjust these until all diagnostics show ✓  ██████████
# ═════════════════════════════════════════════════════════════════════════════

# ── Transmission ──────────────────────────────────────────────────────────────
BETA_SCENARIOS       = [0.010, 0.015, 0.020, 0.025, 0.030]
BASE_BETA            = 0.020   # used for --mode diagnose

# ── Population & grid ─────────────────────────────────────────────────────────
POPULATION_SIZE      = 500
GRID_SIZE            = 20

# ── Epidemic timeline ─────────────────────────────────────────────────────────
SIMULATION_DAYS      = 365
INCUBATION_TICKS     = 6       # ~3 days
INFECTIOUS_TICKS     = 14      # ~7 days
ASYMPTOMATIC_FRAC    = 0.40

# ── Seeding ───────────────────────────────────────────────────────────────────
INIT_INFECTED_FRAC   = 0.004
INIT_EXPOSED_FRAC    = 0.002

# ── Spark rate ────────────────────────────────────────────────────────────────
# Probability per susceptible agent per tick of spontaneous exposure.
# Prevents epidemic extinction between waves (models external importation).
# Increase if epidemic keeps dying; decrease if too many micro-outbreaks.
SPARK_RATE           = 0.0002

# ── Immunity ─────────────────────────────────────────────────────────────────
# Drawn per agent from Gamma(shape, scale_days).
# mean = shape × scale     sd = sqrt(shape) × scale
# Current: mean=91d, sd=25d — tight bell, most agents immune 50–130 days
IMMUNITY_GAMMA_SHAPE = 13.0
IMMUNITY_GAMMA_SCALE = 7.0     # days
IMMUNITY_MIN_DAYS    = 30      # hard floor (biologically: min 1 month)
IMMUNITY_MAX_DAYS    = 180     # hard ceiling

# ── Agent personality priors ──────────────────────────────────────────────────
# Compliance: Beta(alpha, beta) → mean = alpha/(alpha+beta)
COMPLIANCE_PRIOR = {
    "dense_periphery":  {"alpha": 1.5, "beta": 4.0},   # mean ≈ 0.27
    "sparse_periphery": {"alpha": 3.0, "beta": 2.0},   # mean ≈ 0.60
}

# Doctor visit probability: Beta(alpha, beta)
DOCTOR_PRIOR = {
    "dense_periphery":  {"alpha": 4.0, "beta": 4.0},   # mean ≈ 0.50
    "sparse_periphery": {"alpha": 6.0, "beta": 3.0},   # mean ≈ 0.67
}

# Fatality multiplier: Beta(a,b) rescaled to mean=1, × age base rate
FATALITY_BETA_A      = 1.2
FATALITY_BETA_B      = 10.0

# Transport: Dirichlet concentration [public_transit, car, walking]
TRANSPORT_DIRICHLET = {
    "dense_periphery":  [5.0, 2.0, 3.0],   # transit-heavy → higher transmission
    "sparse_periphery": [2.0, 5.0, 3.0],   # car-heavy → lower transmission
}
TRANSPORT_MULT_ARR   = np.array([2.0, 0.5, 1.2])

# Age structure
AGE_FRACTIONS        = [0.15, 0.30, 0.35, 0.20]    # child, young, adult, senior
AGE_BASE_FATALITY    = np.array([0.001, 0.003, 0.010, 0.050])

# ── Policy ────────────────────────────────────────────────────────────────────
# Thresholds: fraction of population in 7-day confirmed cases
# Lower → policy triggers earlier. If alert always fires after peak → lower these.
THRESHOLD_ALERT      = [0.0, 0.005, 0.015, 0.030]
MIN_POLICY_TICKS     = 14

# Max β reduction when agent is fully compliant (compliance=1.0)
# effective_β = β₀ × transport_mult × (1 − compliance × POLICY_REDUCTION[alert])
POLICY_REDUCTION     = np.array([0.0, 0.25, 0.55, 0.80])

# ── Wave detection (for mean_peak_infectious) ─────────────────────────────────
# Only peaks above WAVE_MIN_HEIGHT and separated by WAVE_MIN_DIST days are counted.
# Increase WAVE_MIN_HEIGHT to ignore spark-induced micro-outbreaks.
WAVE_MIN_HEIGHT      = 20     # minimum agents to count as a real wave peak
WAVE_MIN_DIST        = 14     # minimum days between two peaks

# ── Simulation settings ───────────────────────────────────────────────────────
N_RUNS               = 20
TICKS_PER_DAY        = 2
TOTAL_TICKS          = TICKS_PER_DAY * SIMULATION_DAYS
CONTACT_RADIUS       = 1


# ═════════════════════════════════════════════════════════════════════════════
# EpiState
# ═════════════════════════════════════════════════════════════════════════════

class S(IntEnum):
    SUSCEPTIBLE             = 0
    EXPOSED                 = 1
    INFECTIOUS_ASYMPTOMATIC = 2
    INFECTIOUS_SYMPTOMATIC  = 3
    RECOVERED               = 4
    DEAD                    = 5


# ═════════════════════════════════════════════════════════════════════════════
# Population arrays
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Population:
    n:               int
    compliance:      np.ndarray   # float [0,1] — individual health plan responsiveness
    doctor_prob:     np.ndarray   # float [0,1] — probability of seeking care
    fatality_rate:   np.ndarray   # float [0,0.5] — individual death probability
    transport_mult:  np.ndarray   # float {0.5, 1.2, 2.0} — contact multiplier
    immunity_ticks:  np.ndarray   # int — ticks of immunity after recovery
    px:              np.ndarray   # int — grid x position
    py:              np.ndarray   # int — grid y position
    state:           np.ndarray   # int — EpiState
    ticks_in_state:  np.ndarray   # int — ticks spent in current state
    remaining_imm:   np.ndarray   # int — immunity countdown
    seen_doctor:     np.ndarray   # bool
    infection_count: np.ndarray   # int — total infections (counts reinfections)


def build_population(n: int, zone: str, rng: np.random.Generator) -> Population:
    # Compliance — Beta(zone)
    cp         = COMPLIANCE_PRIOR[zone]
    compliance = rng.beta(cp["alpha"], cp["beta"], size=n)

    # Doctor probability — Beta(zone)
    dp          = DOCTOR_PRIOR[zone]
    doctor_prob = rng.beta(dp["alpha"], dp["beta"], size=n)

    # Fatality rate — age base × individual Beta multiplier
    raw           = rng.beta(FATALITY_BETA_A, FATALITY_BETA_B, size=n)
    mean_raw      = FATALITY_BETA_A / (FATALITY_BETA_A + FATALITY_BETA_B)
    fatality_mult = raw / mean_raw
    age_idx       = rng.choice(4, size=n, p=AGE_FRACTIONS)
    fatality_rate = np.clip(AGE_BASE_FATALITY[age_idx] * fatality_mult, 0.0, 0.50)

    # Transport — Dirichlet(zone) per agent
    conc              = np.array(TRANSPORT_DIRICHLET[zone])
    transport_fracs   = rng.dirichlet(conc, size=n)
    transport_choices = np.array([rng.choice(3, p=transport_fracs[i]) for i in range(n)])
    transport_mult    = TRANSPORT_MULT_ARR[transport_choices]

    # Immunity duration — Gamma(zone-independent)
    imm_days  = rng.gamma(IMMUNITY_GAMMA_SHAPE, IMMUNITY_GAMMA_SCALE, size=n)
    imm_ticks = np.clip(
        np.round(imm_days * TICKS_PER_DAY).astype(int),
        TICKS_PER_DAY * IMMUNITY_MIN_DAYS,
        TICKS_PER_DAY * IMMUNITY_MAX_DAYS,
    )

    px = rng.integers(0, GRID_SIZE, size=n)
    py = rng.integers(0, GRID_SIZE, size=n)

    return Population(
        n               = n,
        compliance      = compliance,
        doctor_prob     = doctor_prob,
        fatality_rate   = fatality_rate,
        transport_mult  = transport_mult,
        immunity_ticks  = imm_ticks,
        px              = px.copy(),
        py              = py.copy(),
        state           = np.full(n, S.SUSCEPTIBLE, dtype=int),
        ticks_in_state  = np.zeros(n, dtype=int),
        remaining_imm   = np.zeros(n, dtype=int),
        seen_doctor     = np.zeros(n, dtype=bool),
        infection_count = np.zeros(n, dtype=int),
    )


def seed_epidemic(pop: Population, rng: np.random.Generator) -> None:
    n_inf = max(1, int(pop.n * INIT_INFECTED_FRAC))
    n_exp = max(1, int(pop.n * INIT_EXPOSED_FRAC))
    susc  = np.where(pop.state == S.SUSCEPTIBLE)[0]
    rng.shuffle(susc)
    pop.state[susc[:n_inf]]                      = S.INFECTIOUS_SYMPTOMATIC
    pop.ticks_in_state[susc[:n_inf]]             = rng.integers(0, INFECTIOUS_TICKS, n_inf)
    pop.infection_count[susc[:n_inf]]           += 1
    pop.state[susc[n_inf:n_inf + n_exp]]         = S.EXPOSED
    pop.ticks_in_state[susc[n_inf:n_inf + n_exp]]= rng.integers(0, INCUBATION_TICKS, n_exp)


# ═════════════════════════════════════════════════════════════════════════════
# Submodels (vectorised)
# ═════════════════════════════════════════════════════════════════════════════

_OFFSETS = np.array([
    (dx, dy)
    for dx in range(-CONTACT_RADIUS, CONTACT_RADIUS + 1)
    for dy in range(-CONTACT_RADIUS, CONTACT_RADIUS + 1)
])


def transmission_step(
    pop: Population,
    beta0: float,
    alert_level: int,
    policy_active: bool,
    rng: np.random.Generator,
) -> None:
    susc_idx = np.where(pop.state == S.SUSCEPTIBLE)[0]
    cont_idx = np.where(
        (pop.state == S.EXPOSED) |
        (pop.state == S.INFECTIOUS_ASYMPTOMATIC) |
        (pop.state == S.INFECTIOUS_SYMPTOMATIC)
    )[0]
    if len(susc_idx) == 0 or len(cont_idx) == 0:
        return

    pol_red = (POLICY_REDUCTION[alert_level] * pop.compliance[susc_idx]
               if policy_active else np.zeros(len(susc_idx)))

    susc_nx    = pop.px[susc_idx]
    susc_ny    = pop.py[susc_idx]
    cont_cells = pop.px[cont_idx] * GRID_SIZE + pop.py[cont_idx]
    lambda_s   = np.zeros(len(susc_idx))

    for dx, dy in _OFFSETS:
        nx    = (susc_nx + dx) % GRID_SIZE
        ny    = (susc_ny + dy) % GRID_SIZE
        ncell = nx * GRID_SIZE + ny
        for ci, cell in zip(cont_idx, cont_cells):
            mask = ncell == cell
            if mask.any():
                lambda_s[mask] += beta0 * pop.transport_mult[ci] * (1.0 - pol_red[mask])

    newly_exp = susc_idx[rng.random(len(susc_idx)) < (1.0 - np.exp(-lambda_s))]
    pop.state[newly_exp]          = S.EXPOSED
    pop.ticks_in_state[newly_exp] = 0


def progression_step(pop: Population, rng: np.random.Generator) -> None:
    pop.ticks_in_state += 1
    pop.remaining_imm   = np.maximum(0, pop.remaining_imm - 1)

    # EXPOSED → INFECTIOUS
    exp_ready = np.where(
        (pop.state == S.EXPOSED) & (pop.ticks_in_state >= INCUBATION_TICKS)
    )[0]
    if len(exp_ready):
        asymp = rng.random(len(exp_ready)) < ASYMPTOMATIC_FRAC
        pop.state[exp_ready[asymp]]   = S.INFECTIOUS_ASYMPTOMATIC
        pop.state[exp_ready[~asymp]]  = S.INFECTIOUS_SYMPTOMATIC
        pop.ticks_in_state[exp_ready] = 0
        pop.infection_count[exp_ready] += 1

    # INFECTIOUS → RECOVERED / DEAD
    inf_ready = np.where(
        ((pop.state == S.INFECTIOUS_ASYMPTOMATIC) |
         (pop.state == S.INFECTIOUS_SYMPTOMATIC)) &
        (pop.ticks_in_state >= INFECTIOUS_TICKS)
    )[0]
    if len(inf_ready):
        symp  = inf_ready[pop.state[inf_ready] == S.INFECTIOUS_SYMPTOMATIC]
        asymp = inf_ready[pop.state[inf_ready] == S.INFECTIOUS_ASYMPTOMATIC]
        if len(symp):
            draw     = rng.random(len(symp))
            dies     = symp[draw <  pop.fatality_rate[symp]]
            survives = symp[draw >= pop.fatality_rate[symp]]
        else:
            dies, survives = np.array([], dtype=int), np.array([], dtype=int)
        recovers = np.concatenate([survives, asymp])
        pop.state[dies]               = S.DEAD
        pop.state[recovers]           = S.RECOVERED
        pop.ticks_in_state[inf_ready] = 0
        pop.remaining_imm[recovers]   = pop.immunity_ticks[recovers]

    # RECOVERED → SUSCEPTIBLE (immunity waned)
    waned = np.where((pop.state == S.RECOVERED) & (pop.remaining_imm == 0))[0]
    if len(waned):
        pop.state[waned]       = S.SUSCEPTIBLE
        pop.seen_doctor[waned] = False


def medical_step(pop: Population, rng: np.random.Generator) -> int:
    eligible = np.where(
        (pop.state == S.INFECTIOUS_SYMPTOMATIC) & (~pop.seen_doctor)
    )[0]
    if len(eligible) == 0:
        return 0
    sees = rng.random(len(eligible)) < pop.doctor_prob[eligible]
    pop.seen_doctor[eligible[sees]] = True
    return int(sees.sum())


# ═════════════════════════════════════════════════════════════════════════════
# Ministry
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Ministry:
    alert_level:     int   = 0
    ticks_in_policy: int   = 0
    rolling_cases:   deque = field(default_factory=lambda: deque([0] * 7, maxlen=7))
    total_confirmed: int   = 0
    pop_size:        int   = POPULATION_SIZE

    @property
    def prevalence(self) -> float:
        return sum(self.rolling_cases) / self.pop_size

    def update(self, new_cases: int, policy_active: bool) -> None:
        self.rolling_cases.append(new_cases)
        self.total_confirmed += new_cases
        self.ticks_in_policy += 1

        if not policy_active:
            self.alert_level = 0
            return

        prev       = self.alert_level
        can_change = self.ticks_in_policy >= MIN_POLICY_TICKS
        p          = self.prevalence

        if   p >= THRESHOLD_ALERT[3]: target = 3
        elif p >= THRESHOLD_ALERT[2]: target = 2
        elif p >= THRESHOLD_ALERT[1]: target = 1
        else:                         target = 0

        if target > prev or can_change:
            if target != prev:
                self.alert_level     = target
                self.ticks_in_policy = 0


# ═════════════════════════════════════════════════════════════════════════════
# Single run
# ═════════════════════════════════════════════════════════════════════════════

def run_simulation(
    beta0: float,
    zone: str,
    seed: int,
    policy_active: bool,
) -> tuple[list[dict], Population]:

    rng      = np.random.default_rng(seed)
    pop      = build_population(POPULATION_SIZE, zone, rng)
    seed_epidemic(pop, rng)
    ministry = Ministry(pop_size=POPULATION_SIZE)
    daily    = []
    conf_buf = 0

    for tick in range(TOTAL_TICKS):

        # Spark: spontaneous exposure — prevents epidemic extinction between waves
        susc_idx = np.where(pop.state == S.SUSCEPTIBLE)[0]
        if len(susc_idx) > 0:
            sparked = susc_idx[rng.random(len(susc_idx)) < SPARK_RATE]
            if len(sparked):
                pop.state[sparked]          = S.EXPOSED
                pop.ticks_in_state[sparked] = 0

        if tick % 2 == 0:   # morning tick
            transmission_step(pop, beta0, ministry.alert_level, policy_active, rng)
            conf_buf += medical_step(pop, rng)
        else:               # evening tick
            transmission_step(pop, beta0, ministry.alert_level, policy_active, rng)
            ministry.update(conf_buf, policy_active)
            conf_buf = 0

        progression_step(pop, rng)

        if tick % 2 == 1:   # record once per day
            daily.append({
                "day":             tick // 2,
                "susceptible":     int(np.sum(pop.state == S.SUSCEPTIBLE)),
                "exposed":         int(np.sum(pop.state == S.EXPOSED)),
                "infectious":      int(np.sum((pop.state == S.INFECTIOUS_ASYMPTOMATIC) |
                                              (pop.state == S.INFECTIOUS_SYMPTOMATIC))),
                "recovered":       int(np.sum(pop.state == S.RECOVERED)),
                "dead":            int(np.sum(pop.state == S.DEAD)),
                "confirmed_today": ministry.rolling_cases[-1],
                "alert_level":     ministry.alert_level,
                "prevalence_7d":   round(ministry.prevalence, 5),
            })

    return daily, pop


# ═════════════════════════════════════════════════════════════════════════════
# Summary statistics
# ═════════════════════════════════════════════════════════════════════════════

def get_wave_peaks(series: list[int]) -> np.ndarray:
    """Return heights of all detected wave peaks."""
    arr = np.array(series, dtype=float)
    if arr.max() < WAVE_MIN_HEIGHT:
        return np.array([float(arr.max())])   # no real waves; use max as fallback
    peaks, props = find_peaks(arr, height=WAVE_MIN_HEIGHT, distance=WAVE_MIN_DIST)
    if len(peaks) == 0:
        return np.array([float(arr.max())])
    return props["peak_heights"]


def compute_summary(
    daily: list[dict],
    pop: Population,
    zone: str,
    beta0: float,
    run_id: int,
    scenario_id: int,
    policy_active: bool,
) -> dict:
    df = pd.DataFrame(daily)
    if df.empty:
        return {}

    series            = df["infectious"].tolist()
    wave_peaks        = get_wave_peaks(series)

    # ── Primary outcome ───────────────────────────────────────────────────────
    # Mean height across all detected epidemic wave peaks.
    # Lower when policy is active (each wave is suppressed).
    # Zone-dependent: sparse periphery has lower mean_peak due to high compliance.
    mean_peak_infectious = float(wave_peaks.mean())
    max_peak_infectious  = float(wave_peaks.max())
    n_waves              = len(wave_peaks) if df["infectious"].max() >= WAVE_MIN_HEIGHT else 0

    # ── Other outcomes ────────────────────────────────────────────────────────
    total_infections     = int(pop.infection_count.sum())
    total_dead           = int(df["dead"].iloc[-1])
    total_reinfections   = int(np.sum(np.maximum(0, pop.infection_count - 1)))
    max_alert            = int(df["alert_level"].max())
    first_alert_day      = (int(df[df["alert_level"] > 0]["day"].iloc[0])
                            if (df["alert_level"] > 0).any() else -1)
    peak_day             = int(df.loc[df["infectious"].idxmax(), "day"])
    epidemic_duration    = (
        int(df[df["infectious"] > 0]["day"].iloc[-1] -
            df[df["infectious"] > 0]["day"].iloc[0])
        if (df["infectious"] > 0).any() else 0
    )

    alive = pop.state != S.DEAD
    return {
        "scenario_id":            scenario_id,
        "run_id":                 run_id,
        "zone":                   zone,
        "beta0":                  beta0,
        "policy_active":          int(policy_active),
        # ── Primary outcome (use this in PyMC) ───────────────────────────────
        "mean_peak_infectious":   round(mean_peak_infectious, 2),
        "max_peak_infectious":    round(max_peak_infectious, 2),
        "n_waves":                n_waves,
        # ── Secondary outcomes ───────────────────────────────────────────────
        "total_infections":       total_infections,
        "total_dead":             total_dead,
        "total_reinfections":     total_reinfections,
        "epidemic_duration":      epidemic_duration,
        "peak_day":               peak_day,
        # ── Policy tracking ──────────────────────────────────────────────────
        "max_alert":              max_alert,
        "first_alert_day":        first_alert_day,
        "days_alert_0":           int((df["alert_level"] == 0).sum()),
        "days_alert_1":           int((df["alert_level"] == 1).sum()),
        "days_alert_2":           int((df["alert_level"] == 2).sum()),
        "days_alert_3":           int((df["alert_level"] == 3).sum()),
        # ── Population personality (varies across seeds) ──────────────────────
        "mean_compliance":        round(float(pop.compliance[alive].mean()), 4),
        "mean_doctor_prob":       round(float(pop.doctor_prob[alive].mean()), 4),
        "mean_fatality_rate":     round(float(pop.fatality_rate[alive].mean()), 5),
        "mean_immunity_days":     round(float(pop.immunity_ticks[alive].mean() / TICKS_PER_DAY), 1),
        "frac_public_transit":    round(float((pop.transport_mult[alive] == 2.0).mean()), 4),
        "n_agents":               pop.n,
    }


# ═════════════════════════════════════════════════════════════════════════════
# Diagnostics
# ═════════════════════════════════════════════════════════════════════════════

def run_diagnostics():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zones = ["dense_periphery", "sparse_periphery"]
    seeds = [42, 123, 999]

    print("=" * 70)
    print(f"DIAGNOSTICS  |  beta0={BASE_BETA}  |  {SIMULATION_DAYS} days  |  N={POPULATION_SIZE}")
    print(f"Immunity : Gamma({IMMUNITY_GAMMA_SHAPE}, {IMMUNITY_GAMMA_SCALE}d)"
          f" → mean={IMMUNITY_GAMMA_SHAPE*IMMUNITY_GAMMA_SCALE:.0f}d"
          f"  sd={np.sqrt(IMMUNITY_GAMMA_SHAPE)*IMMUNITY_GAMMA_SCALE:.0f}d"
          f"  bounds=[{IMMUNITY_MIN_DAYS}d, {IMMUNITY_MAX_DAYS}d]")
    print(f"Spark    : {SPARK_RATE}  (prob/susceptible/tick)")
    print(f"Thresholds: {THRESHOLD_ALERT}  |  Policy reduction: {POLICY_REDUCTION}")
    print(f"Wave detection: height≥{WAVE_MIN_HEIGHT} agents, distance≥{WAVE_MIN_DIST} days")
    print("=" * 70)

    fig, axes = plt.subplots(len(zones), 3, figsize=(16, 5 * len(zones)))

    # Collect all results for the checklist
    all_results = []

    for zi, zone in enumerate(zones):
        print(f"\n{'─'*70}")
        print(f"ZONE: {zone.upper()}")
        print(f"{'─'*70}")

        daily_on_list, daily_off_list = [], []

        for seed in seeds:
            d_on,  p_on  = run_simulation(BASE_BETA, zone, seed, policy_active=True)
            d_off, p_off = run_simulation(BASE_BETA, zone, seed, policy_active=False)
            daily_on_list.append(d_on)
            daily_off_list.append(d_off)

            df_on   = pd.DataFrame(d_on)
            df_off  = pd.DataFrame(d_off)

            peaks_on   = get_wave_peaks(df_on["infectious"].tolist())
            peaks_off  = get_wave_peaks(df_off["infectious"].tolist())
            mpi_on     = float(peaks_on.mean())
            mpi_off    = float(peaks_off.mean())
            pe         = (mpi_off - mpi_on) / max(mpi_off, 1.0)
            peak_day   = int(df_on.loc[df_on["infectious"].idxmax(), "day"])
            alert_day  = (int(df_on[df_on["alert_level"] > 0]["day"].iloc[0])
                          if (df_on["alert_level"] > 0).any() else -1)
            max_alert  = int(df_on["alert_level"].max())
            n_waves    = len(peaks_on) if df_on["infectious"].max() >= WAVE_MIN_HEIGHT else 0

            print(f"\n  Seed {seed}:")
            print(f"    mean_peak ON={mpi_on:6.1f}  OFF={mpi_off:6.1f}"
                  f"   policy_effect={pe:+.3f}"
                  f"   n_waves={n_waves}")
            print(f"    Alert day={alert_day:3d}  peak_day={peak_day:3d}  "
                  f"→ {'BEFORE peak ✓' if 0 < alert_day < peak_day else 'AFTER/NO peak ✗'}")
            print(f"    Max alert={max_alert}  "
                  f"{'✓' if max_alert >= 2 else '✗ (should reach ≥2)'}")
            print(f"    Mean immunity={p_on.immunity_ticks.mean()/TICKS_PER_DAY:.0f}d  "
                  f"compliance={p_on.compliance.mean():.3f}  "
                  f"doctor_prob={p_on.doctor_prob.mean():.3f}")

            all_results.append({
                "zone":               zone,
                "seed":               seed,
                "mpi_on":             mpi_on,
                "mpi_off":            mpi_off,
                "pe":                 pe,
                "n_waves":            n_waves,
                "max_alert":          max_alert,
                "alert_before_peak":  0 < alert_day < peak_day,
            })

        # ── Panel 1: epidemic curve ───────────────────────────────────────────
        ax1 = axes[zi, 0]
        df_on0  = pd.DataFrame(daily_on_list[0])
        df_off0 = pd.DataFrame(daily_off_list[0])
        ax1.plot(df_on0["day"],  df_on0["infectious"],         "b-",  lw=2,   label="Policy ON")
        ax1.plot(df_off0["day"], df_off0["infectious"],        "r--", lw=2,   label="Policy OFF")
        ax1.plot(df_on0["day"],  df_on0["alert_level"] * 20,  "g:",  lw=1.5, label="Alert ×20")
        ax1.axhline(WAVE_MIN_HEIGHT, color="gray", lw=1, linestyle="--",
                    label=f"Wave threshold ({WAVE_MIN_HEIGHT})")
        ax1.set_title(f"{zone}\nEpidemic curve (seed=42)")
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Infectious agents")
        ax1.legend(fontsize=7)

        # ── Panel 2: daily difference (OFF − ON) across all 3 seeds ─────────
        ax2 = axes[zi, 1]
        for s_i, seed in enumerate(seeds):
            df_on_s  = pd.DataFrame(daily_on_list[s_i])
            df_off_s = pd.DataFrame(daily_off_list[s_i])
            diff = df_off_s["infectious"].values - df_on_s["infectious"].values
            alpha = 0.9 if s_i == 0 else 0.45
            lw    = 2.0 if s_i == 0 else 1.2
            ax2.plot(df_on_s["day"], diff, lw=lw, alpha=alpha, label=f"seed {seed}")
        ax2.axhline(0, color="black", lw=1, linestyle="--")
        ax2.fill_between(
            pd.DataFrame(daily_on_list[0])["day"],
            pd.DataFrame(daily_off_list[0])["infectious"].values -
            pd.DataFrame(daily_on_list[0])["infectious"].values,
            0, alpha=0.12, color="steelblue",
        )
        ax2.set_title("Policy impact: infectious(OFF) minus infectious(ON) -- positive means policy reduces cases")

        ax2.set_xlabel("Day")
        ax2.set_ylabel("Difference in infectious agents (OFF − ON)")
        ax2.legend(fontsize=8)

        # ── Panel 3: policy effect across seeds ──────────────────────────────
        ax3  = axes[zi, 2]
        pes  = [r["pe"] for r in all_results if r["zone"] == zone]
        cols = ["green" if pe > 0 else "red" for pe in pes]
        ax3.bar(range(len(seeds)), pes, color=cols, alpha=0.7)
        ax3.axhline(0, color="black", lw=1)
        ax3.set_xticks(range(len(seeds)))
        ax3.set_xticklabels([f"seed {s}" for s in seeds])
        ax3.set_ylabel("Policy effect\n(mean_peak OFF − ON) / OFF")
        ax3.set_title("Policy effect by seed\n(positive = policy lowers mean peak)")

    # ── Auto-evaluating checklist ─────────────────────────────────────────────
    df_r = pd.DataFrame(all_results)

    dense  = df_r[df_r["zone"] == "dense_periphery"]
    sparse = df_r[df_r["zone"] == "sparse_periphery"]

    checks = [
        ("Alert triggers before peak (>50% of runs)",
         df_r["alert_before_peak"].mean() > 0.5),

        ("Max alert reaches ≥ 2 in all runs",
         (df_r["max_alert"] >= 2).all()),

        ("Policy effect > 0 in >60% of runs",
         (df_r["pe"] > 0).mean() > 0.6),

        ("Mean policy effect > 0.05",
         df_r["pe"].mean() > 0.05),

        ("Policy effect larger in sparse than dense",
         sparse["pe"].mean() > dense["pe"].mean()),

        ("At least 1 wave in all runs",
         (df_r["n_waves"] >= 1).all()),

        ("mean_peak_infectious > 10 in all runs",
         (df_r["mpi_on"] > 10).all()),
    ]

    plt.tight_layout()
    plt.savefig("diagnostics.png", dpi=150, bbox_inches="tight")

    print(f"\n{'='*70}")
    print("CHECKLIST:")
    all_pass = True
    for label, passed in checks:
        icon = "✓" if passed else "✗"
        if not passed:
            all_pass = False
        print(f"  {icon}  {label}")

    if all_pass:
        print("\n  All checks passed → run --mode sweep next")
    else:
        print("\n  Fix failing checks → adjust parameters above → re-run --mode diagnose")

    print(f"\nPlot saved → diagnostics.png")
    print("=" * 70)


# ═════════════════════════════════════════════════════════════════════════════
# Sweep
# ═════════════════════════════════════════════════════════════════════════════

def run_sweep():
    zones  = ["dense_periphery", "sparse_periphery"]
    seeds  = [42, 123, 999]
    rows   = []
    total  = len(BETA_SCENARIOS) * len(zones) * len(seeds) * 2
    done   = 0

    print(f"Quick sweep: {total} runs  ({len(BETA_SCENARIOS)} β₀ × {len(zones)} zones"
          f" × {len(seeds)} seeds × 2 policy)")

    for s_idx, beta0 in enumerate(BETA_SCENARIOS):
        for zone in zones:
            for seed in seeds:
                for policy_active in [True, False]:
                    d, p = run_simulation(beta0, zone, seed, policy_active)
                    rows.append(compute_summary(d, p, zone, beta0, seed, s_idx, policy_active))
                    done += 1
                    if done % 12 == 0:
                        print(f"  {done}/{total}")

    df = pd.DataFrame(rows)

    on   = df[df["policy_active"] == 1][["scenario_id","zone","run_id","mean_peak_infectious"]]\
             .rename(columns={"mean_peak_infectious": "mpi_on"})
    off  = df[df["policy_active"] == 0][["scenario_id","zone","run_id","mean_peak_infectious"]]\
             .rename(columns={"mean_peak_infectious": "mpi_off"})
    pairs= on.merge(off, on=["scenario_id","zone","run_id"])
    pairs["pe"] = (pairs["mpi_off"] - pairs["mpi_on"]) / pairs["mpi_off"].clip(lower=1)

    print(f"\n{'─'*90}")
    print(f"{'beta0':<7} {'zone':<22} {'mpi_on':>8} {'mpi_off':>8} {'pe':>7}"
          f" {'n_waves':>8} {'alert':>7} {'1st_alrt':>9}")
    print("─" * 90)

    on_df = df[df["policy_active"] == 1]
    for (b0, zone), grp in on_df.groupby(["beta0", "zone"]):
        pg = pairs[(pairs["scenario_id"] == grp["scenario_id"].iloc[0]) &
                   (pairs["zone"] == zone)]
        print(f"{b0:<7.3f} {zone:<22}"
              f" {grp['mean_peak_infectious'].mean():8.1f}"
              f" {pg['mpi_off'].mean():8.1f}"
              f" {pg['pe'].mean():+7.3f}"
              f" {grp['n_waves'].mean():8.1f}"
              f" {grp['max_alert'].mean():7.1f}"
              f" {grp['first_alert_day'].median():9.0f}")

    pe_vals = pairs["pe"].values
    print(f"\n  Policy effect > 0 in {(pe_vals > 0).mean():.0%} of pairs  "
          f"{'✓' if (pe_vals > 0).mean() > 0.6 else '✗'}")
    print(f"  Mean policy effect: {pe_vals.mean():+.3f}  "
          f"{'✓' if pe_vals.mean() > 0.05 else '✗ (should be > 0.05)'}")
    dense_pe  = pairs[pairs["zone"] == "dense_periphery"]["pe"].mean()
    sparse_pe = pairs[pairs["zone"] == "sparse_periphery"]["pe"].mean()
    print(f"  Zone effect: dense={dense_pe:+.3f}  sparse={sparse_pe:+.3f}  "
          f"{'✓' if sparse_pe > dense_pe else '✗ (sparse should have larger effect)'}")
    print()
    print("If all ✓ → run --mode generate")


# ═════════════════════════════════════════════════════════════════════════════
# Generate
# ═════════════════════════════════════════════════════════════════════════════

def _worker(args: tuple):
    s_idx, beta0, zone, run_id, seed, policy_active = args
    try:
        d, p = run_simulation(beta0, zone, seed, policy_active)
        return compute_summary(d, p, zone, beta0, run_id, s_idx, policy_active)
    except Exception as e:
        print(f"  ERROR beta0={beta0} zone={zone} run={run_id} policy={policy_active}: {e}")
        return None


def run_generate():
    zones = ["dense_periphery", "sparse_periphery"]
    tasks = [
        (s_idx, beta0, zone, run_id,
         s_idx * 10_000 + (0 if zone == "dense_periphery" else 5_000) + run_id,
         policy_active)
        for s_idx, beta0 in enumerate(BETA_SCENARIOS)
        for zone in zones
        for run_id in range(N_RUNS)
        for policy_active in [True, False]
    ]

    n_cores = mp.cpu_count()
    print(f"Generating {len(tasks)} simulations on {n_cores} cores...")
    t0 = time.time()

    with mp.Pool(n_cores) as pool:
        raw = pool.map(_worker, tasks)

    elapsed = time.time() - t0
    print(f"Done in {elapsed:.1f}s ({elapsed/60:.1f} min)")

    df = pd.DataFrame([r for r in raw if r is not None])

    # Policy effect per matched pair (using mean_peak_infectious)
    on  = df[df["policy_active"] == 1][
        ["scenario_id","zone","run_id","mean_peak_infectious"]
    ].rename(columns={"mean_peak_infectious": "mpi_on"})
    off = df[df["policy_active"] == 0][
        ["scenario_id","zone","run_id","mean_peak_infectious"]
    ].rename(columns={"mean_peak_infectious": "mpi_off"})
    pairs = on.merge(off, on=["scenario_id","zone","run_id"])
    pairs["policy_effect"] = (
        (pairs["mpi_off"] - pairs["mpi_on"]) / pairs["mpi_off"].clip(lower=1)
    )
    df = df.merge(
        pairs[["scenario_id","zone","run_id","policy_effect"]],
        on=["scenario_id","zone","run_id"], how="left",
    )

    # Group IDs for hierarchical model
    df["zone_id"]     = (df["zone"] == "sparse_periphery").astype(int)
    df["group_id"]    = df["scenario_id"] * 2 + df["zone_id"]
    df["group_label"] = (
        df["beta0"].map(lambda b: f"b{b:.3f}") + "_" +
        df["zone"].map(lambda z: "sparse" if z == "sparse_periphery" else "dense")
    )

    cols = [
        "group_id","group_label","scenario_id","zone_id","zone","beta0",
        "run_id","policy_active",
        "mean_peak_infectious","max_peak_infectious","n_waves",
        "total_infections","total_dead","total_reinfections","epidemic_duration","peak_day",
        "policy_effect",
        "max_alert","first_alert_day",
        "days_alert_0","days_alert_1","days_alert_2","days_alert_3",
        "mean_compliance","mean_doctor_prob","mean_fatality_rate",
        "mean_immunity_days","frac_public_transit","n_agents",
    ]
    df = df[[c for c in cols if c in df.columns]]
    df = df.sort_values(["group_id","run_id","policy_active"]).reset_index(drop=True)
    df.to_csv("abm_outputs.csv", index=False)

    print(f"Saved {len(df)} rows → abm_outputs.csv")
    print()
    print("Summary (policy=ON runs, mean across 20 runs per group):")
    on_df = df[df["policy_active"] == 1]
    print(on_df.groupby(["beta0","zone"])[
        ["mean_peak_infectious","n_waves","policy_effect","first_alert_day","mean_compliance"]
    ].mean().round(2).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEIRD ABM")
    parser.add_argument(
        "--mode",
        choices=["diagnose", "sweep", "generate"],
        default="diagnose",
        help=(
            "diagnose : 6 runs + plots + auto checklist  |  "
            "sweep    : 60-run quick validation  |  "
            "generate : full 400-run CSV for PyMC"
        ),
    )
    args = parser.parse_args()

    if args.mode == "diagnose":
        run_diagnostics()
    elif args.mode == "sweep":
        run_sweep()
    elif args.mode == "generate":
        run_generate()
