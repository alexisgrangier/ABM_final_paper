# app.py
"""
Streamlit app for the SEIRD ABM.
Updated to reflect new model specifications from seird_gen.py.

Run with:
    streamlit run app.py
"""

from __future__ import annotations
import time
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from scipy.signal import find_peaks

# ─────────────────────────────────────────────────────────────────────────────
# Import the new standalone model from seird_gen.py
# ─────────────────────────────────────────────────────────────────────────────
from seird_gen import (
    # Classes
    Population,
    Ministry,
    S,
    # Functions
    build_population,
    seed_epidemic,
    run_simulation,
    compute_summary,
    get_wave_peaks,
    # Parameters (defaults — overridden via sidebar)
    POPULATION_SIZE         as DEFAULT_POPULATION_SIZE,
    GRID_SIZE               as DEFAULT_GRID_SIZE,
    SIMULATION_DAYS         as DEFAULT_SIMULATION_DAYS,
    BASE_BETA               as DEFAULT_BASE_BETA,
    BETA_SCENARIOS,
    INCUBATION_TICKS        as DEFAULT_INCUBATION_TICKS,
    INFECTIOUS_TICKS        as DEFAULT_INFECTIOUS_TICKS,
    ASYMPTOMATIC_FRAC       as DEFAULT_ASYMPTOMATIC_FRAC,
    INIT_INFECTED_FRAC,
    INIT_EXPOSED_FRAC,
    SPARK_RATE              as DEFAULT_SPARK_RATE,
    IMMUNITY_GAMMA_SHAPE    as DEFAULT_IMMUNITY_GAMMA_SHAPE,
    IMMUNITY_GAMMA_SCALE    as DEFAULT_IMMUNITY_GAMMA_SCALE,
    IMMUNITY_MIN_DAYS       as DEFAULT_IMMUNITY_MIN_DAYS,
    IMMUNITY_MAX_DAYS       as DEFAULT_IMMUNITY_MAX_DAYS,
    COMPLIANCE_PRIOR,
    DOCTOR_PRIOR,
    FATALITY_BETA_A         as DEFAULT_FATALITY_BETA_A,
    FATALITY_BETA_B         as DEFAULT_FATALITY_BETA_B,
    TRANSPORT_DIRICHLET,
    TRANSPORT_MULT_ARR,
    AGE_FRACTIONS,
    AGE_BASE_FATALITY,
    THRESHOLD_ALERT         as DEFAULT_THRESHOLD_ALERT,
    POLICY_REDUCTION        as DEFAULT_POLICY_REDUCTION,
    MIN_POLICY_TICKS        as DEFAULT_MIN_POLICY_TICKS,
    WAVE_MIN_HEIGHT         as DEFAULT_WAVE_MIN_HEIGHT,
    WAVE_MIN_DIST           as DEFAULT_WAVE_MIN_DIST,
    TICKS_PER_DAY,
    TOTAL_TICKS             as DEFAULT_TOTAL_TICKS,
    CONTACT_RADIUS,
)

import seird_gen as _gen   # used to monkey-patch globals for PPC


# ─────────────────────────────────────────────────────────────────────────────
# Thin wrapper: a stateful model object mirroring the old SEIRDModel API
# ─────────────────────────────────────────────────────────────────────────────

class SEIRDModel:
    """
    Thin stateful wrapper around seird_gen's vectorised engine.
    Provides .step(), .run(), .run_until_end(), and .get_results_df()
    so the Streamlit plumbing keeps working unchanged.
    """

    def __init__(
        self,
        population_size: int,
        zone: str,
        beta0: float,
        spark_rate: float,
        policy_active: bool,
        random_seed: int = 42,
        # disease progression
        incubation_ticks: int = DEFAULT_INCUBATION_TICKS,
        infectious_ticks: int = DEFAULT_INFECTIOUS_TICKS,
        asymptomatic_frac: float = DEFAULT_ASYMPTOMATIC_FRAC,
        # immunity
        immunity_gamma_shape: float = DEFAULT_IMMUNITY_GAMMA_SHAPE,
        immunity_gamma_scale: float = DEFAULT_IMMUNITY_GAMMA_SCALE,
        immunity_min_days: int = DEFAULT_IMMUNITY_MIN_DAYS,
        immunity_max_days: int = DEFAULT_IMMUNITY_MAX_DAYS,
        # policy
        threshold_alert: list | None = None,
        policy_reduction: list | None = None,
        min_policy_ticks: int = DEFAULT_MIN_POLICY_TICKS,
        # wave detection
        wave_min_height: int = DEFAULT_WAVE_MIN_HEIGHT,
        wave_min_dist: int = DEFAULT_WAVE_MIN_DIST,
        # simulation length
        total_ticks: int = DEFAULT_TOTAL_TICKS,
    ) -> None:
        self.rng           = np.random.default_rng(random_seed)
        self.beta0         = beta0
        self.zone          = zone
        self.spark_rate    = spark_rate
        self.policy_active = policy_active
        self.total_ticks   = total_ticks
        self.wave_min_height = wave_min_height
        self.wave_min_dist   = wave_min_dist

        # Store overridden parameters back into seird_gen globals so the
        # vectorised submodel functions pick them up.
        _gen.INCUBATION_TICKS      = incubation_ticks
        _gen.INFECTIOUS_TICKS      = infectious_ticks
        _gen.ASYMPTOMATIC_FRAC     = asymptomatic_frac
        _gen.IMMUNITY_GAMMA_SHAPE  = immunity_gamma_shape
        _gen.IMMUNITY_GAMMA_SCALE  = immunity_gamma_scale
        _gen.IMMUNITY_MIN_DAYS     = immunity_min_days
        _gen.IMMUNITY_MAX_DAYS     = immunity_max_days
        _gen.THRESHOLD_ALERT       = threshold_alert or list(DEFAULT_THRESHOLD_ALERT)
        _gen.POLICY_REDUCTION      = np.array(policy_reduction or list(DEFAULT_POLICY_REDUCTION))
        _gen.MIN_POLICY_TICKS      = min_policy_ticks
        _gen.POPULATION_SIZE       = population_size

        self.pop      = build_population(population_size, zone, self.rng)
        seed_epidemic(self.pop, self.rng)
        self.ministry = Ministry(pop_size=population_size)

        self.tick      = 0
        self.conf_buf  = 0
        self._daily: list[dict] = []

        # record tick-0 state
        self._record()

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _record(self) -> None:
        pop = self.pop
        self._daily.append({
            "tick":             self.tick,
            "day":              self.tick / TICKS_PER_DAY,
            "susceptible":      int(np.sum(pop.state == S.SUSCEPTIBLE)),
            "exposed":          int(np.sum(pop.state == S.EXPOSED)),
            "infectious_asymp": int(np.sum(pop.state == S.INFECTIOUS_ASYMPTOMATIC)),
            "infectious_symp":  int(np.sum(pop.state == S.INFECTIOUS_SYMPTOMATIC)),
            "total_infectious": int(np.sum(
                (pop.state == S.INFECTIOUS_ASYMPTOMATIC) |
                (pop.state == S.INFECTIOUS_SYMPTOMATIC)
            )),
            "recovered":        int(np.sum(pop.state == S.RECOVERED)),
            "dead":             int(np.sum(pop.state == S.DEAD)),
            "confirmed_today":  self.ministry.rolling_cases[-1] if self._daily else 0,
            "alert_level":      self.ministry.alert_level,
            "seven_day_prev":   round(self.ministry.prevalence, 5),
            "total_reinfections": int(np.sum(np.maximum(0, self.pop.infection_count - 1))),
        })

    # ── Public API ───────────────────────────────────────────────────────────

    @property
    def current_day(self) -> float:
        return self.tick / TICKS_PER_DAY

    @property
    def individuals(self):
        """Compatibility shim: expose pop for grid visualisation."""
        return self.pop

    def step(self) -> None:
        from seird_gen import transmission_step, progression_step, medical_step

        pop = self.pop

        # Spark: spontaneous exposure (external importation)
        susc_idx = np.where(pop.state == S.SUSCEPTIBLE)[0]
        if len(susc_idx) > 0:
            sparked = susc_idx[self.rng.random(len(susc_idx)) < self.spark_rate]
            if len(sparked):
                pop.state[sparked]          = S.EXPOSED
                pop.ticks_in_state[sparked] = 0

        if self.tick % 2 == 0:   # morning tick
            transmission_step(
                pop, self.beta0, self.ministry.alert_level, self.policy_active, self.rng
            )
            self.conf_buf += medical_step(pop, self.rng)
        else:                     # evening tick
            transmission_step(
                pop, self.beta0, self.ministry.alert_level, self.policy_active, self.rng
            )
            self.ministry.update(self.conf_buf, self.policy_active)
            self.conf_buf = 0

        progression_step(pop, self.rng)
        self.tick += 1
        self._record()

    def run(self, n_ticks: int) -> None:
        for _ in range(n_ticks):
            self.step()

    def run_until_end(self) -> None:
        remaining = self.total_ticks - self.tick
        if remaining > 0:
            self.run(remaining)

    def get_results_df(self) -> pd.DataFrame:
        return pd.DataFrame(self._daily)

    def get_wave_summary(self) -> dict:
        series     = [r["total_infectious"] for r in self._daily]
        peaks      = get_wave_peaks(series)
        return {
            "mean_peak_infectious": round(float(peaks.mean()), 2),
            "max_peak_infectious":  round(float(peaks.max()),  2),
            "n_waves": (
                len(peaks)
                if max(series) >= self.wave_min_height else 0
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SEIRD ABM Simulator", layout="wide")

st.title("SEIRD Agent-Based Model Simulator")
st.caption(
    "Agent-based model of infectious disease transmission · "
    "zone-stratified population · waning immunity · Bayesian calibration"
)

# ─────────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────────

for key, default in [
    ("playing",      False),
    ("model",        None),
    ("results_df",   pd.DataFrame()),
    ("initialized",  False),
    ("posterior_df", None),
    ("ppc_envelope", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation Controls")

    random_seed     = st.number_input("Random seed",      min_value=1,   max_value=999999, value=42,   step=1)
    population_size = st.number_input("Population size",  min_value=100, max_value=10000,  value=DEFAULT_POPULATION_SIZE, step=100)
    run_ticks       = st.slider("Ticks to run",           min_value=1,   max_value=200,    value=10,   step=1)

    st.markdown("---")
    st.subheader("Zone & Transmission")

    zone  = st.selectbox(
        "Population zone",
        options=["dense_periphery", "sparse_periphery"],
        index=0,
        help=(
            "dense_periphery: transit-heavy, lower compliance (mean≈0.27). "
            "sparse_periphery: car-heavy, higher compliance (mean≈0.60)."
        ),
    )

    beta0 = st.select_slider(
        "Base transmission rate (β₀)",
        options=BETA_SCENARIOS,
        value=DEFAULT_BASE_BETA,
        format_func=lambda x: f"{x:.3f}",
    )

    policy_active = st.toggle("Policy active", value=True)

    st.markdown("---")
    st.subheader("Disease Progression")

    incubation_ticks  = st.slider("Incubation ticks",         min_value=2,  max_value=20, value=DEFAULT_INCUBATION_TICKS)
    infectious_ticks  = st.slider("Infectious ticks",         min_value=4,  max_value=30, value=DEFAULT_INFECTIOUS_TICKS)
    asymptomatic_frac = st.slider("Asymptomatic fraction",    min_value=0.0, max_value=1.0, value=DEFAULT_ASYMPTOMATIC_FRAC, step=0.05)
    spark_rate        = st.number_input(
        "Spark rate (prob/susceptible/tick)",
        min_value=0.0, max_value=0.01, value=DEFAULT_SPARK_RATE,
        step=0.0001, format="%.4f",
        help="Spontaneous exposure — models external importation; prevents epidemic extinction between waves.",
    )

    st.markdown("---")
    st.subheader("Waning Immunity")

    imm_shape    = st.slider("Immunity Gamma shape",   min_value=1.0,  max_value=30.0, value=float(DEFAULT_IMMUNITY_GAMMA_SHAPE), step=0.5)
    imm_scale    = st.slider("Immunity Gamma scale (days)", min_value=1.0, max_value=30.0, value=float(DEFAULT_IMMUNITY_GAMMA_SCALE), step=0.5)
    imm_min_days = st.slider("Immunity min days",      min_value=7,   max_value=90,   value=DEFAULT_IMMUNITY_MIN_DAYS)
    imm_max_days = st.slider("Immunity max days",      min_value=60,  max_value=365,  value=DEFAULT_IMMUNITY_MAX_DAYS)

    st.caption(
        f"Gamma mean ≈ {imm_shape * imm_scale:.0f} days  |  "
        f"sd ≈ {np.sqrt(imm_shape) * imm_scale:.0f} days  |  "
        f"bounds [{imm_min_days}d, {imm_max_days}d]"
    )

    st.markdown("---")
    st.subheader("Policy Thresholds & Reduction")

    thr1 = st.number_input("Alert level 1 threshold (prevalence)", min_value=0.001, max_value=0.05,  value=DEFAULT_THRESHOLD_ALERT[1], step=0.001, format="%.3f")
    thr2 = st.number_input("Alert level 2 threshold (prevalence)", min_value=0.001, max_value=0.10,  value=DEFAULT_THRESHOLD_ALERT[2], step=0.001, format="%.3f")
    thr3 = st.number_input("Alert level 3 threshold (prevalence)", min_value=0.005, max_value=0.20,  value=DEFAULT_THRESHOLD_ALERT[3], step=0.005, format="%.3f")
    threshold_alert = [0.0, thr1, thr2, thr3]

    red1 = st.slider("Policy reduction — alert 1", 0.0, 1.0, float(DEFAULT_POLICY_REDUCTION[1]), 0.05)
    red2 = st.slider("Policy reduction — alert 2", 0.0, 1.0, float(DEFAULT_POLICY_REDUCTION[2]), 0.05)
    red3 = st.slider("Policy reduction — alert 3", 0.0, 1.0, float(DEFAULT_POLICY_REDUCTION[3]), 0.05)
    policy_reduction = [0.0, red1, red2, red3]

    min_policy_ticks = st.slider("Min ticks between policy changes", min_value=2, max_value=30, value=DEFAULT_MIN_POLICY_TICKS)

    st.markdown("---")
    st.subheader("Wave Detection")

    wave_min_height = st.slider("Wave min height (agents)", min_value=5,  max_value=100, value=DEFAULT_WAVE_MIN_HEIGHT)
    wave_min_dist   = st.slider("Wave min distance (days)", min_value=5,  max_value=60,  value=DEFAULT_WAVE_MIN_DIST)

    st.markdown("---")

    init_clicked     = st.button("Initialize model",       use_container_width=True)
    step_clicked     = st.button("Run 1 tick",             use_container_width=True)
    run_clicked      = st.button(f"Run {run_ticks} ticks", use_container_width=True)
    run_full_clicked = st.button("Run until end",          use_container_width=True)
    reset_clicked    = st.button("Reset",                  use_container_width=True)

    st.markdown("---")
    st.header("Bayesian Calibration")

    with st.expander("Load posterior particles", expanded=False):
        st.caption(
            "Upload a posterior CSV produced by your calibration pipeline. "
            "Required columns: **beta0**, **zone**, **weight**."
        )

        uploaded = st.file_uploader("Posterior CSV", type="csv", key="posterior_upload")
        if uploaded is not None:
            try:
                df_up    = pd.read_csv(uploaded)
                required = {"beta0", "weight"}
                if required.issubset(df_up.columns):
                    st.session_state.posterior_df = df_up
                    st.session_state.ppc_envelope = None
                    st.success(f"Loaded {len(df_up)} particles ✓")
                else:
                    st.error(f"CSV must contain at minimum columns: {required}")
            except Exception as e:
                st.error(f"Could not parse file: {e}")

    if st.session_state.posterior_df is not None:
        n_p = len(st.session_state.posterior_df)
        st.success(f"Posterior loaded — {n_p} particles")

        n_ppc = st.slider(
            "PPC trajectories to draw",
            min_value=10, max_value=min(200, n_p),
            value=min(20, n_p), step=10, key="n_ppc",
        )
        compute_ppc = st.button("Compute uncertainty bands", use_container_width=True)
    else:
        st.info("No posterior loaded. Upload a CSV above.")
        compute_ppc = False
        n_ppc = 20


# ─────────────────────────────────────────────────────────────────────────────
# Model actions
# ─────────────────────────────────────────────────────────────────────────────

def _build_model_kwargs() -> dict:
    return dict(
        population_size      = int(population_size),
        zone                 = zone,
        beta0                = float(beta0),
        spark_rate           = float(spark_rate),
        policy_active        = bool(policy_active),
        random_seed          = int(random_seed),
        incubation_ticks     = int(incubation_ticks),
        infectious_ticks     = int(infectious_ticks),
        asymptomatic_frac    = float(asymptomatic_frac),
        immunity_gamma_shape = float(imm_shape),
        immunity_gamma_scale = float(imm_scale),
        immunity_min_days    = int(imm_min_days),
        immunity_max_days    = int(imm_max_days),
        threshold_alert      = threshold_alert,
        policy_reduction     = policy_reduction,
        min_policy_ticks     = int(min_policy_ticks),
        wave_min_height      = int(wave_min_height),
        wave_min_dist        = int(wave_min_dist),
    )


if reset_clicked:
    st.session_state.model        = None
    st.session_state.results_df   = pd.DataFrame()
    st.session_state.initialized  = False
    st.session_state.ppc_envelope = None
    st.rerun()

if init_clicked:
    with st.spinner("Initializing model..."):
        st.session_state.model = SEIRDModel(**_build_model_kwargs())
        st.session_state.results_df = st.session_state.model.get_results_df()
        st.session_state.initialized  = True
        st.session_state.ppc_envelope = None

if st.session_state.model is not None:
    if step_clicked:
        st.session_state.model.step()
        st.session_state.results_df = st.session_state.model.get_results_df()

    if run_clicked:
        st.session_state.model.run(run_ticks)
        st.session_state.results_df = st.session_state.model.get_results_df()

    if run_full_clicked:
        st.session_state.model.run_until_end()
        st.session_state.results_df = st.session_state.model.get_results_df()


# ─────────────────────────────────────────────────────────────────────────────
# Posterior predictive check
# ─────────────────────────────────────────────────────────────────────────────

def _run_ppc(
    posterior_df: pd.DataFrame,
    n_draws: int,
    pop_size: int,
    n_ticks: int,
    base_zone: str,
) -> dict[str, pd.DataFrame]:
    weights = posterior_df["weight"].values
    weights = weights / weights.sum()

    rng     = np.random.default_rng(0)
    indices = rng.choice(len(posterior_df), size=n_draws, replace=True, p=weights)
    sampled = posterior_df.iloc[indices].reset_index(drop=True)

    track_cols = ["total_infectious", "susceptible", "exposed", "dead"]
    all_runs: dict[str, list] = {c: [] for c in track_cols}
    tick_index = None

    for _, row in sampled.iterrows():
        b0   = float(row["beta0"])
        z    = str(row["zone"]) if "zone" in row else base_zone
        seed = int(rng.integers(0, 2**31))

        m = SEIRDModel(
            population_size = pop_size,
            zone            = z,
            beta0           = b0,
            spark_rate      = DEFAULT_SPARK_RATE,
            policy_active   = True,
            random_seed     = seed,
        )
        m.run(n_ticks)
        run_df = m.get_results_df()

        if tick_index is None:
            tick_index = run_df["tick"].values

        for col in track_cols:
            if col in run_df.columns:
                series = run_df.set_index("tick")[col].reindex(tick_index).ffill().values
                all_runs[col].append(series)

    envelopes: dict[str, pd.DataFrame] = {}
    for col in track_cols:
        mat = np.array(all_runs[col])
        if mat.size == 0:
            continue
        envelopes[col] = pd.DataFrame({
            "tick":   tick_index,
            "median": np.median(mat, axis=0),
            "q025":   np.percentile(mat, 2.5,  axis=0),
            "q975":   np.percentile(mat, 97.5, axis=0),
            "q25":    np.percentile(mat, 25,   axis=0),
            "q75":    np.percentile(mat, 75,   axis=0),
        })

    return envelopes


if compute_ppc and st.session_state.posterior_df is not None:
    n_sim_ticks = (
        st.session_state.model.tick
        if st.session_state.model is not None
        else DEFAULT_TOTAL_TICKS
    )
    with st.spinner("Computing posterior predictive bands ..."):
        result = _run_ppc(
            posterior_df = st.session_state.posterior_df,
            n_draws      = int(n_ppc),
            pop_size     = int(population_size),
            n_ticks      = n_sim_ticks,
            base_zone    = zone,
        )
    st.session_state.ppc_envelope = result


# ─────────────────────────────────────────────────────────────────────────────
# Main display
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.model is None:
    st.info("Initialize the model from the sidebar to start the simulation.")
else:
    df     = st.session_state.results_df
    latest = df.iloc[-1]

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Tick",              int(latest["tick"]))
    c2.metric("Day",               f"{latest['day']:.1f}")
    c3.metric("Zone",              zone)
    c4.metric("Susceptible",       int(latest["susceptible"]))
    c5.metric("Exposed",           int(latest["exposed"]))
    c6.metric("Infectious",        int(latest["total_infectious"]))
    c7.metric("Dead",              int(latest["dead"]))
    c8.metric("Alert level",       int(latest["alert_level"]))

    st.markdown("---")

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["Epidemic curves", "Policy", "Reinfections & waves", "Grid snapshot", "Raw data"]
    )

    # ── Tab 1: Epidemic curves ────────────────────────────────────────────────
    with tab1:
        st.subheader("SEIRD epidemic curves")

        epi_long = df.melt(
            id_vars    = ["tick", "day"],
            value_vars = ["susceptible", "exposed", "total_infectious",
                          "infectious_asymp", "infectious_symp", "recovered", "dead"],
            var_name   = "state",
            value_name = "count",
        )
        chart = (
            alt.Chart(epi_long)
            .mark_line(interpolate="step-after")
            .encode(
                x       = alt.X("tick:Q", title="Tick"),
                y       = alt.Y("count:Q", title="Agents"),
                color   = alt.Color("state:N", title="State"),
                tooltip = ["tick", "day", "state", "count"],
            )
            .properties(height=420)
        )
        st.altair_chart(chart, use_container_width=True)

        # Incidence panel
        incidence_long = df.melt(
            id_vars    = ["tick", "day"],
            value_vars = ["confirmed_today", "dead"],
            var_name   = "indicator",
            value_name = "value",
        )
        inc_chart = (
            alt.Chart(incidence_long)
            .mark_line()
            .encode(
                x       = alt.X("tick:Q", title="Tick"),
                y       = alt.Y("value:Q", title="Count"),
                color   = alt.Color("indicator:N", title="Indicator"),
                tooltip = ["tick", "day", "indicator", "value"],
            )
            .properties(height=280, title="Daily confirmed cases & deaths")
        )
        st.altair_chart(inc_chart, use_container_width=True)

        # ── Posterior predictive bands ────────────────────────────────────────
        st.markdown("---")
        st.subheader("Posterior predictive uncertainty")

        envelope = st.session_state.ppc_envelope

        if envelope is None and st.session_state.posterior_df is None:
            st.info("Load a posterior in the sidebar to see Bayesian uncertainty bands.")
        elif envelope is None:
            st.info("Posterior loaded ✓ — press **Compute uncertainty bands** in the sidebar.")
        else:
            state_cfg = {
                "total_infectious": {"label": "Infectious",  "color": "#e45756"},
                "exposed":          {"label": "Exposed",     "color": "#f58518"},
                "susceptible":      {"label": "Susceptible", "color": "#4c78a8"},
                "dead":             {"label": "Dead",        "color": "#54a24b"},
            }

            for state_key, cfg in state_cfg.items():
                if state_key not in envelope:
                    continue
                env   = envelope[state_key].copy()
                color = cfg["color"]
                label = cfg["label"]

                band       = (alt.Chart(env).mark_area(opacity=0.15, color=color)
                              .encode(x=alt.X("tick:Q"), y=alt.Y("q025:Q"), y2=alt.Y2("q975:Q")))
                iqr        = (alt.Chart(env).mark_area(opacity=0.25, color=color)
                              .encode(x="tick:Q", y=alt.Y("q25:Q"), y2=alt.Y2("q75:Q")))
                med_line   = (alt.Chart(env).mark_line(color=color, strokeDash=[4,2], strokeWidth=1.5)
                              .encode(x="tick:Q", y=alt.Y("median:Q"),
                                      tooltip=["tick:Q","median:Q","q025:Q","q975:Q"]))

                if state_key in df.columns:
                    obs_line = (
                        alt.Chart(df[["tick", state_key]].rename(columns={state_key: "value"}))
                        .mark_line(color=color, strokeWidth=2.5)
                        .encode(x="tick:Q", y="value:Q", tooltip=["tick:Q","value:Q"])
                    )
                    combined = (band + iqr + med_line + obs_line).properties(
                        height=220,
                        title=f"{label}: single run (solid) vs posterior predictive (bands)",
                    )
                else:
                    combined = (band + iqr + med_line).properties(
                        height=220,
                        title=f"{label}: posterior predictive envelope",
                    )
                st.altair_chart(combined, use_container_width=True)

            st.caption(
                "**Bands**: outer = 95% CI, inner = IQR.  "
                "**Dashed**: posterior median.  **Solid**: current single run."
            )

    # ── Tab 2: Policy ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Health Ministry policy state")

        alert_chart = (
            alt.Chart(df)
            .mark_line(interpolate="step-after", color="steelblue")
            .encode(
                x       = alt.X("tick:Q", title="Tick"),
                y       = alt.Y("alert_level:Q", title="Alert level", scale=alt.Scale(domain=[0, 3])),
                tooltip = ["tick", "day", "alert_level", "seven_day_prev"],
            )
            .properties(height=280, title="Alert level over time")
        )
        st.altair_chart(alert_chart, use_container_width=True)

        prev_chart = (
            alt.Chart(df.dropna(subset=["seven_day_prev"]))
            .mark_line(color="firebrick")
            .encode(
                x       = alt.X("tick:Q", title="Tick"),
                y       = alt.Y("seven_day_prev:Q", title="7-day prevalence"),
                tooltip = ["tick", "day", "seven_day_prev", "alert_level"],
            )
            .properties(height=280, title="7-day rolling prevalence (confirmed / population)")
        )
        # overlay threshold lines
        thr_df = pd.DataFrame({
            "threshold": [thr1, thr2, thr3],
            "label":     ["Alert 1", "Alert 2", "Alert 3"],
        })
        thr_rules = (
            alt.Chart(thr_df)
            .mark_rule(strokeDash=[6,3])
            .encode(
                y     = alt.Y("threshold:Q"),
                color = alt.Color("label:N"),
            )
        )
        st.altair_chart((prev_chart + thr_rules), use_container_width=True)

        # Summary table
        alert_days = {
            f"Days at alert {i}": int((df["alert_level"] == i).sum())
            for i in range(4)
        }
        alert_days["First alert tick"] = (
            int(df[df["alert_level"] > 0]["tick"].iloc[0])
            if (df["alert_level"] > 0).any() else "—"
        )
        st.table(pd.DataFrame(alert_days, index=["Value"]).T)

    # ── Tab 3: Reinfections & waves ───────────────────────────────────────────
    with tab3:
        st.subheader("Reinfections & wave detection")

        col_ri, col_wv = st.columns(2)

        with col_ri:
            ri_chart = (
                alt.Chart(df)
                .mark_line(color="purple")
                .encode(
                    x       = alt.X("tick:Q", title="Tick"),
                    y       = alt.Y("total_reinfections:Q", title="Cumulative reinfections"),
                    tooltip = ["tick", "day", "total_reinfections"],
                )
                .properties(height=300, title="Cumulative reinfections over time")
            )
            st.altair_chart(ri_chart, use_container_width=True)

        with col_wv:
            wave_info = st.session_state.model.get_wave_summary()
            st.metric("Detected waves",       wave_info["n_waves"])
            st.metric("Mean peak infectious", f"{wave_info['mean_peak_infectious']:.1f}")
            st.metric("Max peak infectious",  f"{wave_info['max_peak_infectious']:.1f}")
            st.caption(
                f"Wave counted when peak ≥ {wave_min_height} agents "
                f"and separated by ≥ {wave_min_dist} days."
            )

        # Immunity profile
        st.markdown("---")
        st.subheader("Population personality snapshot")

        pop = st.session_state.model.individuals
        alive_mask = pop.state != S.DEAD

        snap = pd.DataFrame({
            "compliance":    pop.compliance[alive_mask],
            "doctor_prob":   pop.doctor_prob[alive_mask],
            "fatality_rate": pop.fatality_rate[alive_mask],
            "immunity_days": pop.immunity_ticks[alive_mask] / TICKS_PER_DAY,
        })

        col_a, col_b = st.columns(2)
        with col_a:
            comp_chart = (
                alt.Chart(snap)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X("compliance:Q", bin=alt.Bin(maxbins=30), title="Compliance"),
                    y=alt.Y("count():Q", title="Agents"),
                )
                .properties(height=220, title=f"Compliance distribution ({zone})")
            )
            st.altair_chart(comp_chart, use_container_width=True)

        with col_b:
            imm_chart = (
                alt.Chart(snap)
                .mark_bar(opacity=0.7)
                .encode(
                    x=alt.X("immunity_days:Q", bin=alt.Bin(maxbins=30), title="Immunity duration (days)"),
                    y=alt.Y("count():Q", title="Agents"),
                )
                .properties(height=220, title="Immunity duration distribution (Gamma)")
            )
            st.altair_chart(imm_chart, use_container_width=True)

        st.dataframe(snap.describe().round(4), use_container_width=True)

    # ── Tab 4: Grid snapshot ──────────────────────────────────────────────────
    with tab4:
        st.subheader("Live spatial animation")

        col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 2])
        with col_ctrl1:
            if st.button("▶ Play / ⏸ Pause", use_container_width=True):
                st.session_state.playing = not st.session_state.get("playing", False)
        with col_ctrl2:
            tick_delay = st.slider("Speed (sec/tick)", 0.1, 2.0, 0.5, 0.1)

        st.caption(f"Status: {'▶ Playing' if st.session_state.get('playing') else '⏸ Paused'}")

        chart_placeholder = st.empty()
        info_placeholder  = st.empty()

        def render_grid():
            pop = st.session_state.model.individuals
            state_names = {
                S.SUSCEPTIBLE:             "Susceptible",
                S.EXPOSED:                 "Exposed",
                S.INFECTIOUS_ASYMPTOMATIC: "Infectious (A)",
                S.INFECTIOUS_SYMPTOMATIC:  "Infectious (S)",
                S.RECOVERED:               "Recovered",
                S.DEAD:                    "Dead",
            }
            pos_df = pd.DataFrame({
                "x":     pop.px,
                "y":     pop.py,
                "state": [state_names.get(s, str(s)) for s in pop.state],
            })
            sample_size = min(2000, len(pos_df))
            pos_df = pos_df.sample(sample_size, random_state=42)

            scatter = (
                alt.Chart(pos_df)
                .mark_circle(size=20, opacity=0.65)
                .encode(
                    x       = alt.X("x:Q", title="X", scale=alt.Scale(domain=[0, DEFAULT_GRID_SIZE])),
                    y       = alt.Y("y:Q", title="Y", scale=alt.Scale(domain=[0, DEFAULT_GRID_SIZE])),
                    color   = alt.Color("state:N", title="State"),
                    tooltip = ["x", "y", "state"],
                )
                .properties(height=600)
            )
            chart_placeholder.altair_chart(scatter, use_container_width=True)
            info_placeholder.caption(
                f"Tick {st.session_state.model.tick} | "
                f"Day {st.session_state.model.current_day:.1f} | "
                f"Zone: {zone} | Showing {sample_size} agents"
            )

        render_grid()

        if st.session_state.get("playing", False):
            while st.session_state.get("playing", False):
                if st.session_state.model.tick >= st.session_state.model.total_ticks:
                    st.session_state.playing = False
                    st.info("Simulation ended.")
                    break
                st.session_state.model.step()
                st.session_state.results_df = st.session_state.model.get_results_df()
                render_grid()
                time.sleep(tick_delay)

    # ── Tab 5: Raw data ───────────────────────────────────────────────────────
    with tab5:
        st.subheader("Raw simulation output")
        st.dataframe(df, use_container_width=True)

        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label     = "Download CSV results",
            data      = csv,
            file_name = "simulation_results.csv",
            mime      = "text/csv",
        )

        if st.session_state.model is not None:
            wave_summary = st.session_state.model.get_wave_summary()
            st.markdown("**Wave summary**")
            st.json(wave_summary)
