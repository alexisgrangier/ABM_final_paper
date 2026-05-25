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
import os
import numpy as np
import pandas as pd
import argparse
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from enum import IntEnum
from collections import deque
from scipy.signal import find_peaks

# ── Output directories ────────────────────────────────────────────────────────
DIR_DATA  = os.path.join("data", "processed")
DIR_PLOTS = os.path.join("outputs", "plots")
os.makedirs(DIR_DATA,  exist_ok=True)
os.makedirs(DIR_PLOTS, exist_ok=True)

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
    n_ticks: int | None = None,
) -> tuple[list[dict], Population]:

    rng      = np.random.default_rng(seed)
    pop      = build_population(POPULATION_SIZE, zone, rng)
    seed_epidemic(pop, rng)
    ministry = Ministry(pop_size=POPULATION_SIZE)
    daily    = []
    conf_buf = 0
    _ticks   = n_ticks if n_ticks is not None else TOTAL_TICKS

    for tick in range(_ticks):

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

        # save panel 1 individually
        fig_p1, ax_p1 = plt.subplots(figsize=(7, 4))
        ax_p1.plot(df_on0["day"],  df_on0["infectious"],        "b-",  lw=2,   label="Policy ON")
        ax_p1.plot(df_off0["day"], df_off0["infectious"],       "r--", lw=2,   label="Policy OFF")
        ax_p1.plot(df_on0["day"],  df_on0["alert_level"] * 20, "g:",  lw=1.5, label="Alert ×20")
        ax_p1.axhline(WAVE_MIN_HEIGHT, color="gray", lw=1, linestyle="--",
                      label=f"Wave threshold ({WAVE_MIN_HEIGHT})")
        ax_p1.set_title(f"{zone} — Epidemic curve (seed=42)")
        ax_p1.set_xlabel("Day"); ax_p1.set_ylabel("Infectious agents")
        ax_p1.legend(fontsize=8)
        fig_p1.tight_layout()
        p1_path = os.path.join(DIR_PLOTS, f"diag_{zone}_epidemic_curve.png")
        fig_p1.savefig(p1_path, dpi=150, bbox_inches="tight")
        plt.close(fig_p1)
        print(f"  Plot saved → {p1_path}")

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

        # save panel 2 individually
        fig_p2, ax_p2 = plt.subplots(figsize=(7, 4))
        for s_i, seed in enumerate(seeds):
            df_on_s  = pd.DataFrame(daily_on_list[s_i])
            df_off_s = pd.DataFrame(daily_off_list[s_i])
            diff  = df_off_s["infectious"].values - df_on_s["infectious"].values
            alpha = 0.9 if s_i == 0 else 0.45
            lw    = 2.0 if s_i == 0 else 1.2
            ax_p2.plot(df_on_s["day"], diff, lw=lw, alpha=alpha, label=f"seed {seed}")
        ax_p2.axhline(0, color="black", lw=1, linestyle="--")
        ax_p2.fill_between(
            pd.DataFrame(daily_on_list[0])["day"],
            pd.DataFrame(daily_off_list[0])["infectious"].values -
            pd.DataFrame(daily_on_list[0])["infectious"].values,
            0, alpha=0.12, color="steelblue",
        )
        ax_p2.set_title(f"{zone} — Policy impact (OFF − ON infectious)")
        ax_p2.set_xlabel("Day"); ax_p2.set_ylabel("Difference in infectious agents")
        ax_p2.legend(fontsize=8)
        fig_p2.tight_layout()
        p2_path = os.path.join(DIR_PLOTS, f"diag_{zone}_policy_impact.png")
        fig_p2.savefig(p2_path, dpi=150, bbox_inches="tight")
        plt.close(fig_p2)
        print(f"  Plot saved → {p2_path}")

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

        # save panel 3 individually
        fig_p3, ax_p3 = plt.subplots(figsize=(5, 4))
        ax_p3.bar(range(len(seeds)), pes, color=cols, alpha=0.7)
        ax_p3.axhline(0, color="black", lw=1)
        ax_p3.set_xticks(range(len(seeds)))
        ax_p3.set_xticklabels([f"seed {s}" for s in seeds])
        ax_p3.set_ylabel("Policy effect\n(mean_peak OFF − ON) / OFF")
        ax_p3.set_title(f"{zone} — Policy effect by seed")
        fig_p3.tight_layout()
        p3_path = os.path.join(DIR_PLOTS, f"diag_{zone}_policy_effect.png")
        fig_p3.savefig(p3_path, dpi=150, bbox_inches="tight")
        plt.close(fig_p3)
        print(f"  Plot saved → {p3_path}")

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
    plot_path = os.path.join(DIR_PLOTS, "diagnostics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")

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

    print(f"\nPlot saved → {plot_path}")
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

    # ── Sweep plots ───────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    on_df    = df[df["policy_active"] == 1]
    zones    = ["dense_periphery", "sparse_periphery"]
    betas    = sorted(df["beta0"].unique())
    zone_colors = {"dense_periphery": "steelblue", "sparse_periphery": "darkorange"}

    # Plot 1: mean_peak_infectious ON vs OFF by beta0, per zone
    fig1, axes1 = plt.subplots(1, len(zones), figsize=(6 * len(zones), 4), sharey=True)
    for ax, zone in zip(axes1, zones):
        grp_on  = df[(df["policy_active"] == 1) & (df["zone"] == zone)].groupby("beta0")["mean_peak_infectious"].mean()
        grp_off = df[(df["policy_active"] == 0) & (df["zone"] == zone)].groupby("beta0")["mean_peak_infectious"].mean()
        ax.plot(grp_on.index,  grp_on.values,  "o-", color=zone_colors[zone], lw=2, label="Policy ON")
        ax.plot(grp_off.index, grp_off.values, "s--", color=zone_colors[zone], lw=2, alpha=0.5, label="Policy OFF")
        ax.set_title(zone)
        ax.set_xlabel("β₀")
        ax.set_ylabel("Mean peak infectious")
        ax.legend(fontsize=8)
    fig1.suptitle("Mean peak infectious: policy ON vs OFF by β₀ and zone", fontsize=11)
    fig1.tight_layout()
    p_sweep1 = os.path.join(DIR_PLOTS, "sweep_mean_peak_by_beta.png")
    fig1.savefig(p_sweep1, dpi=150, bbox_inches="tight")
    plt.close(fig1)
    print(f"  Plot saved → {p_sweep1}")

    # Plot 2: policy effect (PE) by beta0, per zone
    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for zone in zones:
        pe_by_beta = pairs.merge(
            df[["scenario_id","zone","beta0"]].drop_duplicates(),
            on=["scenario_id","zone"]
        )
        grp = pe_by_beta[pe_by_beta["zone"] == zone].groupby("beta0")["pe"].mean()
        ax2.plot(grp.index, grp.values, "o-", color=zone_colors[zone], lw=2, label=zone)
    ax2.axhline(0, color="black", lw=1, linestyle="--")
    ax2.set_xlabel("β₀")
    ax2.set_ylabel("Mean policy effect\n(mpi_off − mpi_on) / mpi_off")
    ax2.set_title("Policy effect by β₀ and zone")
    ax2.legend(fontsize=8)
    fig2.tight_layout()
    p_sweep2 = os.path.join(DIR_PLOTS, "sweep_policy_effect_by_beta.png")
    fig2.savefig(p_sweep2, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    print(f"  Plot saved → {p_sweep2}")

    # Plot 3: n_waves heatmap (beta0 × zone)
    fig3, ax3 = plt.subplots(figsize=(6, 3))
    pivot = on_df.groupby(["zone", "beta0"])["n_waves"].mean().unstack("beta0")
    im = ax3.imshow(pivot.values, aspect="auto", cmap="YlOrRd")
    ax3.set_xticks(range(len(betas)))
    ax3.set_xticklabels([f"{b:.3f}" for b in betas], fontsize=8)
    ax3.set_yticks(range(len(pivot.index)))
    ax3.set_yticklabels(pivot.index, fontsize=8)
    ax3.set_xlabel("β₀"); ax3.set_ylabel("Zone")
    ax3.set_title("Mean number of waves (policy ON)")
    plt.colorbar(im, ax=ax3, label="n_waves")
    fig3.tight_layout()
    p_sweep3 = os.path.join(DIR_PLOTS, "sweep_n_waves_heatmap.png")
    fig3.savefig(p_sweep3, dpi=150, bbox_inches="tight")
    plt.close(fig3)
    print(f"  Plot saved → {p_sweep3}")


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
    csv_path = os.path.join(DIR_DATA, "abm_outputs.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved {len(df)} rows → {csv_path}")

    # ── Posterior CSV for Bayesian calibration ────────────────────────────────
    # Keep only policy-ON runs (these are what the app calibrates against).
    # Weight each particle by how well it matches the target behaviour:
    #   - higher policy_effect  → more informative run  (reward)
    #   - lower mean_peak       → epidemic stayed bounded (reward)
    #   - n_waves >= 1          → sustained dynamics, not extinction (reward)
    # Weights are normalised to sum to 1 so they act as a proper discrete
    # probability distribution for the PPC sampler in the app.
    post = df[df["policy_active"] == 1].copy()

    pe_score   = post["policy_effect"].clip(lower=0)
    peak_score = 1.0 / (post["mean_peak_infectious"].clip(lower=1))
    wave_score = (post["n_waves"] >= 1).astype(float)

    raw_weight     = (pe_score + peak_score + wave_score).clip(lower=1e-6)
    post["weight"] = raw_weight / raw_weight.sum()

    posterior_cols = ["beta0", "zone", "weight",
                      "mean_peak_infectious", "n_waves", "policy_effect",
                      "mean_compliance", "mean_immunity_days", "group_label"]
    post = post[[c for c in posterior_cols if c in post.columns]]

    posterior_path = os.path.join(DIR_DATA, "posterior.csv")
    post.to_csv(posterior_path, index=False)
    print(f"Saved {len(post)} posterior particles → {posterior_path}")
    print()
    print("Summary (policy=ON runs, mean across 20 runs per group):")
    on_df = df[df["policy_active"] == 1]
    print(on_df.groupby(["beta0","zone"])[
        ["mean_peak_infectious","n_waves","policy_effect","first_alert_day","mean_compliance"]
    ].mean().round(2).to_string())


# ═════════════════════════════════════════════════════════════════════════════
# ABC-SMC Calibration
# ═════════════════════════════════════════════════════════════════════════════
"""
Proper Approximate Bayesian Computation — Sequential Monte Carlo (ABC-SMC).

Goal
----
Infer a posterior distribution over β₀ (and zone mixture) given a set of
observed summary statistics.  The posterior tells you: given what we observe
about the epidemic, which transmission rates are plausible?

Method
------
ABC-SMC runs T populations of N particles.  Each population tightens an
acceptance threshold ε, so the particles progressively concentrate in the
region of parameter space that reproduces the observed data.

Steps per population t:
  1. Sample a candidate θ* from the previous weighted population
     (or from the prior for t=0).
  2. Perturb θ* with a Gaussian kernel (prevents particle collapse).
  3. Simulate the model with θ*.
  4. Compute distance d(S_obs, S_sim) between observed and simulated
     summary statistics.
  5. Accept if d < ε_t; otherwise reject.
  6. Re-weight accepted particles by prior(θ*) / Σ w_{t-1} K(θ*|θ).
  7. Normalise weights.

Summary statistics (what we match)
-----------------------------------
  - peak_infectious     : height of the largest epidemic wave
  - peak_day            : day on which the peak occurs
  - attack_rate         : fraction of population ever infected
  - n_waves             : number of distinct epidemic waves
  - mean_alert_level    : average alert level (proxy for policy activation)

All statistics are standardised before computing the Euclidean distance so
that no single statistic dominates.

Prior
-----
  β₀  ~ Uniform(0.005, 0.040)   — broad, covers all BETA_SCENARIOS
  zone ~ Categorical([0.5, 0.5]) — equal prior on both zones

Output
------
  data/processed/posterior.csv   — weighted particles ready for app upload
    columns: beta0, zone, weight, [summary stats]
  outputs/plots/abc_smc_posterior.png — marginal + joint posterior plots
"""

# ── ABC targets (set these to match your observed / target epidemic) ──────────
# These are the "observed data" your model needs to reproduce.
# If you have real surveillance data, replace these with your empirical values.
# Default: mid-range synthetic target consistent with BASE_BETA=0.020.
# Targets derived empirically from --mode diagnose at BASE_BETA=0.020,
# averaged across both zones (dense + sparse) and 3 seeds (42, 123, 999),
# policy ON runs. This makes calibration find β₀ values that reproduce
# the reference scenario dynamics rather than arbitrary synthetic values.
ABC_TARGET = {
    "peak_infectious": 57.0,   # mean of ON peaks across 6 runs: (70.8+88.6+73.5+45.5+35.3+30.9)/6
    "peak_day":        25.0,   # first-wave peak day (median across runs, excluding late secondary peaks)
    "attack_rate":      0.70,  # fraction ever infected over 150-tick ABC window (not full 365d)
    "n_waves":          9.0,   # mean wave count: (8+7+8+8+11+14)/6 = 9.3
    "mean_alert_level": 1.8,   # alert fires on day 1–4 in all runs, stays elevated — mean ~1.8
}

# ── Per-statistic weights for the distance function ───────────────────────────
# Increase a weight to make calibration enforce that statistic more strictly.
# n_waves gets triple weight because it was systematically missed at weight=1.
ABC_STAT_WEIGHTS = {
    "peak_infectious":  1.0,
    "peak_day":         0.5,   # less critical — peak timing is noisy
    "attack_rate":      1.0,
    "n_waves":          3.0,   # triple weight — previously not being matched
    "mean_alert_level": 0.5,   # less critical — emerges from other dynamics
}

# ── ABC-SMC hyperparameters ───────────────────────────────────────────────────
ABC_N_PARTICLES    = 200    # particles per population
ABC_N_POPULATIONS  = 5      # number of SMC populations (more → tighter posterior)
ABC_EPSILON_START  = 2.0    # initial (loose) distance threshold
ABC_EPSILON_END    = 0.80   # final threshold — keep achievable (was 0.4)
ABC_PERTURB_SIGMA  = 0.003  # Gaussian perturbation kernel std for β₀
ABC_SIM_TICKS      = TICKS_PER_DAY * 150   # 150 days sufficient to capture peaks (was 365)
ABC_ZONES          = ["dense_periphery", "sparse_periphery"]


def _abc_summary_stats(daily: list[dict], pop: "Population") -> dict:
    """Compute summary statistics from a single simulation run."""
    df = pd.DataFrame(daily)
    if df.empty or df["infectious"].max() == 0:
        return {
            "peak_infectious":  0.0,
            "peak_day":         0.0,
            "attack_rate":      0.0,
            "n_waves":          0.0,
            "mean_alert_level": 0.0,
        }

    series     = df["infectious"].tolist()
    wave_peaks = get_wave_peaks(series)
    n_waves    = len(wave_peaks) if df["infectious"].max() >= WAVE_MIN_HEIGHT else 0

    return {
        "peak_infectious":  float(df["infectious"].max()),
        "peak_day":         float(df.loc[df["infectious"].idxmax(), "day"]),
        "attack_rate":      float((pop.infection_count >= 1).sum()) / pop.n,  # fraction ever infected (not reinfections)
        "n_waves":          float(n_waves),
        "mean_alert_level": float(df["alert_level"].mean()),
    }


def _abc_distance(obs: dict, sim: dict) -> float:
    """
    Weighted standardised Euclidean distance between observed and simulated
    summary statistics.  Each statistic is divided by its target value
    (scale normalisation) then multiplied by ABC_STAT_WEIGHTS so that
    poorly-matched statistics can be penalised more heavily.
    """
    d = 0.0
    for k in obs:
        scale = abs(obs[k]) if abs(obs[k]) > 1e-6 else 1.0
        w = ABC_STAT_WEIGHTS.get(k, 1.0)
        d += w * ((obs[k] - sim[k]) / scale) ** 2
    return float(np.sqrt(d))


def _abc_sample_prior(rng: np.random.Generator) -> tuple[float, str]:
    """Sample (β₀, zone) from the prior."""
    beta0 = rng.uniform(0.005, 0.040)
    zone  = rng.choice(ABC_ZONES)
    return float(beta0), str(zone)


def _abc_perturb(
    beta0: float,
    zone: str,
    rng: np.random.Generator,
) -> tuple[float, str]:
    """Perturb a particle with a Gaussian kernel (β₀) and zone flip."""
    new_beta = -1.0
    while new_beta <= 0.0 or new_beta > 0.060:
        new_beta = beta0 + rng.normal(0, ABC_PERTURB_SIGMA)
    # zone: flip with small probability to allow zone exploration
    new_zone = rng.choice(ABC_ZONES) if rng.random() < 0.10 else zone
    return float(new_beta), str(new_zone)


def _abc_prior_density(beta0: float, zone: str) -> float:
    """Prior density p(θ) — uniform over β₀, equal prob over zones."""
    if 0.005 <= beta0 <= 0.040:
        return (1.0 / (0.040 - 0.005)) * (1.0 / len(ABC_ZONES))
    return 0.0


def _abc_kernel_density(
    beta0_star: float,
    zone_star: str,
    particles: list[tuple[float, str]],
    weights: np.ndarray,
) -> float:
    """
    Transition kernel denominator:
    Σ_i w_i * K(θ* | θ_i)
    Gaussian kernel on β₀, indicator on zone.
    """
    total = 0.0
    for (b, z), w in zip(particles, weights):
        if z == zone_star:
            k = np.exp(-0.5 * ((beta0_star - b) / ABC_PERTURB_SIGMA) ** 2)
            k /= ABC_PERTURB_SIGMA * np.sqrt(2 * np.pi)
            total += w * k
    return max(total, 1e-300)


# ── Parallel ABC worker (module-level so mp.Pool can pickle it) ───────────────

def _abc_worker(args: tuple) -> dict | None:
    """
    Propose one candidate (β₀, zone), simulate, compute distance.
    Returns a result dict if distance < epsilon, else None.
    Called in parallel by mp.Pool — must be a top-level function.
    """
    beta0_star, zone_star, seed, epsilon, obs = args
    rng_w = np.random.default_rng(seed)
    try:
        daily, pop = run_simulation(beta0_star, zone_star,
                                    int(rng_w.integers(0, 2**31)),
                                    policy_active=True,
                                    n_ticks=ABC_SIM_TICKS)
        sim_stats = _abc_summary_stats(daily, pop)
        dist      = _abc_distance(obs, sim_stats)
        if dist < epsilon:
            return {"beta0": beta0_star, "zone": zone_star,
                    "distance": dist, "seed": seed}
    except Exception:
        pass
    return None


def _abc_worker_stats(args: tuple) -> dict:
    """Re-simulate a final particle to attach summary stats."""
    beta0, zone, seed = args
    daily, pop = run_simulation(beta0, zone, seed, policy_active=True)
    return _abc_summary_stats(daily, pop)


def run_calibrate():
    """
    Parallelised ABC-SMC calibration.
    Each population samples candidates in parallel across all CPU cores.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_cores = mp.cpu_count()
    rng     = np.random.default_rng(42)
    obs     = ABC_TARGET

    print("=" * 70)
    print("ABC-SMC CALIBRATION  (parallelised)")
    print(f"  Particles per population : {ABC_N_PARTICLES}")
    print(f"  Populations              : {ABC_N_POPULATIONS}")
    print(f"  ε schedule               : {ABC_EPSILON_START:.2f} → {ABC_EPSILON_END:.2f}")
    print(f"  CPU cores                : {n_cores}")
    print(f"  Prior: β₀ ~ Uniform(0.005, 0.040)  |  zone ~ Cat(0.5, 0.5)")
    print(f"\n  Observed summary statistics (targets):")
    for k, v in obs.items():
        print(f"    {k:<22} = {v}")
    print("=" * 70)

    epsilons = np.linspace(ABC_EPSILON_START, ABC_EPSILON_END, ABC_N_POPULATIONS)

    particles: list[tuple[float, str]] = []
    weights  : np.ndarray              = np.array([])
    distances: list[float]             = []
    all_populations: list[pd.DataFrame] = []

    # Batch size: submit this many candidates at once per round.
    # Large enough to keep all cores busy; small enough to not overshoot target.
    BATCH = n_cores * 8

    for t, epsilon in enumerate(epsilons):
        t0 = time.time()
        print(f"\n── Population {t+1}/{ABC_N_POPULATIONS}  ε={epsilon:.3f}  "
              f"(batch={BATCH}, cores={n_cores}) ──")

        new_particles: list[tuple[float, str]] = []
        new_weights  : list[float]             = []
        new_distances: list[float]             = []
        n_trials = 0

        with mp.Pool(n_cores) as pool:
            while len(new_particles) < ABC_N_PARTICLES:
                # ── Propose a batch of candidates ────────────────────────────
                proposals = []
                for _ in range(BATCH):
                    if t == 0 or len(particles) == 0:
                        b, z = _abc_sample_prior(rng)
                    else:
                        idx  = rng.choice(len(particles), p=weights)
                        b, z = _abc_perturb(particles[idx][0], particles[idx][1], rng)
                        if _abc_prior_density(b, z) == 0.0:
                            b, z = _abc_sample_prior(rng)
                    seed = int(rng.integers(0, 2**31))
                    proposals.append((b, z, seed, epsilon, obs))

                n_trials += BATCH

                # ── Simulate batch in parallel ───────────────────────────────
                results = pool.map(_abc_worker, proposals)

                for res, (b, z, seed, _, _obs) in zip(results, proposals):
                    if res is None:
                        continue
                    if len(new_particles) >= ABC_N_PARTICLES:
                        break

                    # Importance weight
                    if t == 0:
                        w = 1.0
                    else:
                        prior  = _abc_prior_density(b, z)
                        kernel = _abc_kernel_density(b, z, particles, weights)
                        w      = prior / kernel

                    new_particles.append((b, z))
                    new_weights.append(w)
                    new_distances.append(res["distance"])

                accepted = len(new_particles)
                if accepted % 50 < BATCH and accepted > 0:
                    rate = accepted / n_trials
                    eta  = (ABC_N_PARTICLES - accepted) / max(rate * BATCH / (time.time() - t0 + 1e-6), 1e-6)
                    print(f"  accepted {accepted}/{ABC_N_PARTICLES}"
                          f"  trials={n_trials}"
                          f"  rate={rate:.1%}"
                          f"  elapsed={time.time()-t0:.0f}s")

                # Safety cap
                if n_trials >= ABC_N_PARTICLES * 5000:
                    print(f"  WARNING: hit trial cap at ε={epsilon:.3f}.")
                    break

        if not new_particles:
            print(f"  WARNING: no particles accepted. Try raising ABC_EPSILON_START.")
            continue

        weights_arr = np.array(new_weights)
        weights_arr = weights_arr / weights_arr.sum()
        particles   = new_particles
        weights     = weights_arr
        distances   = new_distances

        eff_n = 1.0 / (weights_arr ** 2).sum()
        elapsed = time.time() - t0
        print(f"  Done — accepted={len(particles)}  trials={n_trials}"
              f"  rate={len(particles)/n_trials:.1%}"
              f"  ESS={eff_n:.0f}  time={elapsed:.1f}s")

        pop_df = pd.DataFrame({
            "population": t + 1,
            "epsilon":    epsilon,
            "beta0":      [p[0] for p in particles],
            "zone":       [p[1] for p in particles],
            "weight":     weights_arr,
            "distance":   distances,
        })
        all_populations.append(pop_df)

    if not all_populations:
        print("Calibration failed. Raise ABC_EPSILON_START or adjust ABC_TARGET.")
        return

    # ── Final posterior ───────────────────────────────────────────────────────
    final_df = all_populations[-1].copy()

    # Re-simulate final particles in parallel to attach summary stats
    print("\nComputing summary statistics for final posterior particles...")
    stat_args = [
        (float(row["beta0"]), str(row["zone"]), int(rng.integers(0, 2**31)))
        for _, row in final_df.iterrows()
    ]
    with mp.Pool(n_cores) as pool:
        stats_rows = pool.map(_abc_worker_stats, stat_args)
    stats_df = pd.DataFrame(stats_rows)
    final_df = pd.concat([final_df.reset_index(drop=True), stats_df], axis=1)

    posterior_path = os.path.join(DIR_DATA, "posterior.csv")
    final_df.to_csv(posterior_path, index=False)
    print(f"\nPosterior saved → {posterior_path}  ({len(final_df)} particles)")

    # ── Posterior summary ─────────────────────────────────────────────────────
    w = final_df["weight"].values
    print("\nPosterior summary (weighted):")
    e_beta = np.average(final_df["beta0"], weights=w)
    sd_beta = np.sqrt(np.average((final_df["beta0"] - e_beta)**2, weights=w))
    print(f"  E[β₀]   = {e_beta:.4f}  ± {sd_beta:.4f}")
    for zone in ABC_ZONES:
        frac = w[final_df["zone"] == zone].sum()
        print(f"  P(zone={zone}) = {frac:.3f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    n_pops = len(all_populations)
    fig, axes = plt.subplots(2, n_pops, figsize=(4 * n_pops, 7))
    if n_pops == 1:
        axes = axes.reshape(2, 1)

    colors = {"dense_periphery": "steelblue", "sparse_periphery": "darkorange"}

    for t, pop_df in enumerate(all_populations):
        ax_top = axes[0, t]
        ax_bot = axes[1, t]

        for zone in ABC_ZONES:
            sub = pop_df[pop_df["zone"] == zone]
            ax_top.hist(
                sub["beta0"], bins=20, weights=sub["weight"],
                alpha=0.6, color=colors[zone], label=zone,
                density=True,
            )
        ax_top.axvline(BASE_BETA, color="black", lw=1.5,
                       linestyle="--", label=f"BASE_BETA={BASE_BETA}")
        ax_top.set_title(f"Pop {t+1}  ε={pop_df['epsilon'].iloc[0]:.3f}")
        ax_top.set_xlabel("β₀")
        ax_top.set_ylabel("Weighted density")
        if t == 0:
            ax_top.legend(fontsize=6)

        ax_bot.scatter(
            pop_df["beta0"], pop_df["distance"],
            c=[colors[z] for z in pop_df["zone"]],
            s=10, alpha=0.5,
        )
        ax_bot.axhline(pop_df["epsilon"].iloc[0], color="red",
                       lw=1, linestyle="--", label=f"ε={pop_df['epsilon'].iloc[0]:.3f}")
        ax_bot.set_xlabel("β₀")
        ax_bot.set_ylabel("Distance")
        ax_bot.legend(fontsize=6)

    fig.suptitle(
        "ABC-SMC: β₀ posterior across populations\n"
        f"(blue=dense_periphery, orange=sparse_periphery, dashed=BASE_BETA)",
        fontsize=10,
    )
    fig.tight_layout()
    plot_path = os.path.join(DIR_PLOTS, "abc_smc_posterior.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved → {plot_path}")
    print("=" * 70)
    print("Upload data/processed/posterior.csv to the app's calibration panel.")
    print("=" * 70)
# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SEIRD ABM")
    parser.add_argument(
        "--mode",
        choices=["diagnose", "sweep", "generate", "calibrate"],
        default="diagnose",
        help=(
            "diagnose  : 6 runs + plots + auto checklist  |  "
            "sweep     : 60-run quick validation  |  "
            "generate  : full 400-run CSV for PyMC  |  "
            "calibrate : ABC-SMC posterior over beta0 and zone"
        ),
    )
    parser.add_argument(
        "--target",
        nargs=5,
        metavar=("PEAK_INF", "PEAK_DAY", "ATTACK_RATE", "N_WAVES", "MEAN_ALERT"),
        type=float,
        default=None,
        help=(
            "Override ABC_TARGET with 5 observed summary statistics: "
            "peak_infectious peak_day attack_rate n_waves mean_alert_level"
        ),
    )
    args = parser.parse_args()

    if args.mode == "diagnose":
        run_diagnostics()
    elif args.mode == "sweep":
        run_sweep()
    elif args.mode == "generate":
        run_generate()
    elif args.mode == "calibrate":
        if args.target is not None:
            keys = ["peak_infectious", "peak_day", "attack_rate",
                    "n_waves", "mean_alert_level"]
            ABC_TARGET.update(dict(zip(keys, args.target)))
            print("Overriding ABC_TARGET with command-line values:")
            for k, v in ABC_TARGET.items():
                print(f"  {k} = {v}")
        run_calibrate()