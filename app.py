# app.py
"""
Streamlit app for the SEIRD ABM.
Run with:  streamlit run app.py
"""

from __future__ import annotations
import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from seird_gen import (
    Population, Ministry, S,
    build_population, seed_epidemic, run_simulation,
    compute_summary, get_wave_peaks,
    POPULATION_SIZE   as DEFAULT_POPULATION_SIZE,
    GRID_SIZE         as DEFAULT_GRID_SIZE,
    BASE_BETA         as DEFAULT_BASE_BETA,
    BETA_SCENARIOS,
    INCUBATION_TICKS  as DEFAULT_INCUBATION_TICKS,
    INFECTIOUS_TICKS  as DEFAULT_INFECTIOUS_TICKS,
    ASYMPTOMATIC_FRAC as DEFAULT_ASYMPTOMATIC_FRAC,
    SPARK_RATE        as DEFAULT_SPARK_RATE,
    IMMUNITY_GAMMA_SHAPE as DEFAULT_IMMUNITY_GAMMA_SHAPE,
    IMMUNITY_GAMMA_SCALE as DEFAULT_IMMUNITY_GAMMA_SCALE,
    IMMUNITY_MIN_DAYS as DEFAULT_IMMUNITY_MIN_DAYS,
    IMMUNITY_MAX_DAYS as DEFAULT_IMMUNITY_MAX_DAYS,
    THRESHOLD_ALERT   as DEFAULT_THRESHOLD_ALERT,
    POLICY_REDUCTION  as DEFAULT_POLICY_REDUCTION,
    MIN_POLICY_TICKS  as DEFAULT_MIN_POLICY_TICKS,
    WAVE_MIN_HEIGHT   as DEFAULT_WAVE_MIN_HEIGHT,
    WAVE_MIN_DIST     as DEFAULT_WAVE_MIN_DIST,
    TICKS_PER_DAY,
    TOTAL_TICKS       as DEFAULT_TOTAL_TICKS,
    ABC_TARGET,
)
import seird_gen as _gen


# ─────────────────────────────────────────────────────────────────────────────
# SEIRDModel wrapper
# ─────────────────────────────────────────────────────────────────────────────

class SEIRDModel:
    def __init__(
        self,
        population_size: int,
        zone: str,
        beta0: float,
        spark_rate: float,
        policy_active: bool,
        random_seed: int = 42,
        total_ticks: int = DEFAULT_TOTAL_TICKS,
    ) -> None:
        self.rng           = np.random.default_rng(random_seed)
        self.beta0         = beta0
        self.zone          = zone
        self.spark_rate    = spark_rate
        self.policy_active = policy_active
        self.total_ticks   = total_ticks
        _gen.POPULATION_SIZE = population_size

        self.pop      = build_population(population_size, zone, self.rng)
        seed_epidemic(self.pop, self.rng)
        self.ministry = Ministry(pop_size=population_size)
        self.tick     = 0
        self.conf_buf = 0
        self._daily: list[dict] = []
        self._record()

    def _record(self) -> None:
        pop = self.pop
        self._daily.append({
            "tick":               self.tick,
            "day":                self.tick / TICKS_PER_DAY,
            "susceptible":        int(np.sum(pop.state == S.SUSCEPTIBLE)),
            "exposed":            int(np.sum(pop.state == S.EXPOSED)),
            "infectious_asymp":   int(np.sum(pop.state == S.INFECTIOUS_ASYMPTOMATIC)),
            "infectious_symp":    int(np.sum(pop.state == S.INFECTIOUS_SYMPTOMATIC)),
            "total_infectious":   int(np.sum(
                (pop.state == S.INFECTIOUS_ASYMPTOMATIC) |
                (pop.state == S.INFECTIOUS_SYMPTOMATIC)
            )),
            "recovered":          int(np.sum(pop.state == S.RECOVERED)),
            "dead":               int(np.sum(pop.state == S.DEAD)),
            "confirmed_today":    self.ministry.rolling_cases[-1] if self._daily else 0,
            "alert_level":        self.ministry.alert_level,
            "seven_day_prev":     round(self.ministry.prevalence, 5),
            "total_reinfections": int(np.sum(np.maximum(0, pop.infection_count - 1))),
        })

    @property
    def current_day(self) -> float:
        return self.tick / TICKS_PER_DAY

    @property
    def individuals(self):
        return self.pop

    def step(self) -> None:
        from seird_gen import transmission_step, progression_step, medical_step
        pop = self.pop
        susc_idx = np.where(pop.state == S.SUSCEPTIBLE)[0]
        if len(susc_idx) > 0:
            sparked = susc_idx[self.rng.random(len(susc_idx)) < self.spark_rate]
            if len(sparked):
                pop.state[sparked]          = S.EXPOSED
                pop.ticks_in_state[sparked] = 0
        if self.tick % 2 == 0:
            transmission_step(pop, self.beta0, self.ministry.alert_level, self.policy_active, self.rng)
            self.conf_buf += medical_step(pop, self.rng)
        else:
            transmission_step(pop, self.beta0, self.ministry.alert_level, self.policy_active, self.rng)
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
        series = [r["total_infectious"] for r in self._daily]
        peaks  = get_wave_peaks(series)
        return {
            "mean_peak_infectious": round(float(peaks.mean()), 2),
            "max_peak_infectious":  round(float(peaks.max()),  2),
            "n_waves": len(peaks) if max(series) >= DEFAULT_WAVE_MIN_HEIGHT else 0,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Page config & session state
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="SEIRD ABM Simulator", layout="wide")
st.title("SEIRD Agent-Based Model Simulator")
st.caption(
    "Agent-based model of infectious disease transmission · "
    "zone-stratified population · waning immunity · ABC-SMC Bayesian calibration"
)

for key, default in [
    ("playing",       False),
    ("model",         None),
    ("results_df",    pd.DataFrame()),
    ("initialized",   False),
    ("posterior_df",  None),
    ("ppc_envelope",  None),
    ("bayes_summary", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — slim: only runtime controls
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Simulation Controls")

    random_seed     = st.number_input("Random seed",     min_value=1,   max_value=999999, value=42,  step=1)
    population_size = st.number_input("Population size", min_value=100, max_value=10000,
                                      value=DEFAULT_POPULATION_SIZE, step=100)
    run_ticks       = st.slider("Ticks to run", min_value=1, max_value=200, value=10, step=1)

    st.markdown("---")
    st.subheader("Zone & Transmission")

    # If a posterior is loaded, show the posterior-weighted β₀ as suggestion
    _post = st.session_state.posterior_df
    _suggested_beta = DEFAULT_BASE_BETA
    if _post is not None:
        _w  = _post["weight"].values / _post["weight"].values.sum()
        _suggested_beta = float(np.average(_post["beta0"].values, weights=_w))
        # snap to nearest scenario
        _suggested_beta = min(BETA_SCENARIOS, key=lambda x: abs(x - _suggested_beta))

    zone = st.selectbox(
        "Population zone",
        options=["dense_periphery", "sparse_periphery"],
        help="dense: transit-heavy, lower compliance. sparse: car-heavy, higher compliance.",
    )
    beta0 = st.select_slider(
        "Base transmission rate (β₀)",
        options=BETA_SCENARIOS,
        value=_suggested_beta,
        format_func=lambda x: f"{x:.3f}",
        help="If a posterior is loaded, this is pre-set to the posterior mean β₀.",
    )
    if _post is not None:
        st.caption(f"Posterior mean β₀ ≈ {np.average(_post['beta0'].values, weights=_post['weight'].values / _post['weight'].values.sum()):.4f}")

    policy_active = st.toggle("Policy active", value=True)

    st.markdown("---")
    init_clicked     = st.button("Initialize model",       use_container_width=True)
    step_clicked     = st.button("Run 1 tick",             use_container_width=True)
    run_clicked      = st.button(f"Run {run_ticks} ticks", use_container_width=True)
    run_full_clicked = st.button("Run until end",          use_container_width=True)
    reset_clicked    = st.button("Reset",                  use_container_width=True)

    st.markdown("---")
    st.header("Bayesian Calibration")
    st.caption("Upload the `posterior.csv` produced by `python seird_gen.py --mode calibrate`.")

    uploaded = st.file_uploader("posterior.csv", type="csv", key="posterior_upload")
    if uploaded is not None:
        try:
            df_up = pd.read_csv(uploaded)
            if {"beta0", "weight"}.issubset(df_up.columns):
                df_up["weight"] = df_up["weight"] / df_up["weight"].sum()
                st.session_state.posterior_df  = df_up
                st.session_state.ppc_envelope  = None
                st.session_state.bayes_summary = None
                st.success(f"Loaded {len(df_up)} particles ✓")
            else:
                st.error("CSV must contain columns: beta0, weight")
        except Exception as e:
            st.error(f"Could not parse file: {e}")

    if st.session_state.posterior_df is not None:
        n_p = len(st.session_state.posterior_df)
        st.success(f"Posterior loaded — {n_p} particles")
        n_ppc = st.slider("PPC trajectories", min_value=10,
                          max_value=min(200, n_p), value=min(50, n_p), step=10)
        compute_ppc = st.button("Compute uncertainty bands", use_container_width=True)
    else:
        st.info("No posterior loaded.")
        compute_ppc = False
        n_ppc = 50


# ─────────────────────────────────────────────────────────────────────────────
# Model actions
# ─────────────────────────────────────────────────────────────────────────────

def _new_model() -> SEIRDModel:
    return SEIRDModel(
        population_size = int(population_size),
        zone            = zone,
        beta0           = float(beta0),
        spark_rate      = DEFAULT_SPARK_RATE,
        policy_active   = bool(policy_active),
        random_seed     = int(random_seed),
    )


if reset_clicked:
    st.session_state.model        = None
    st.session_state.results_df   = pd.DataFrame()
    st.session_state.initialized  = False
    st.session_state.ppc_envelope = None
    st.rerun()

if init_clicked:
    with st.spinner("Initializing model..."):
        st.session_state.model       = _new_model()
        st.session_state.results_df  = st.session_state.model.get_results_df()
        st.session_state.initialized = True
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
# PPC computation
# ─────────────────────────────────────────────────────────────────────────────

def _run_ppc(posterior_df, n_draws, pop_size, n_ticks, base_zone):
    w       = posterior_df["weight"].values / posterior_df["weight"].values.sum()
    rng     = np.random.default_rng(0)
    indices = rng.choice(len(posterior_df), size=n_draws, replace=True, p=w)
    sampled = posterior_df.iloc[indices].reset_index(drop=True)

    track_cols = ["total_infectious", "susceptible", "exposed", "dead"]
    all_runs   = {c: [] for c in track_cols}
    tick_index = None

    for _, row in sampled.iterrows():
        b0   = float(row["beta0"])
        z    = str(row["zone"]) if "zone" in row.index else base_zone
        seed = int(rng.integers(0, 2**31))
        m    = SEIRDModel(pop_size, z, b0, DEFAULT_SPARK_RATE, True, seed)
        m.run(n_ticks)
        run_df = m.get_results_df()
        if tick_index is None:
            tick_index = run_df["tick"].values
        for col in track_cols:
            if col in run_df.columns:
                all_runs[col].append(
                    run_df.set_index("tick")[col].reindex(tick_index).ffill().values
                )

    envelopes = {}
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
    n_sim_ticks = st.session_state.model.tick if st.session_state.model else DEFAULT_TOTAL_TICKS
    with st.spinner("Running posterior predictive simulations..."):
        st.session_state.ppc_envelope = _run_ppc(
            st.session_state.posterior_df, int(n_ppc),
            int(population_size), n_sim_ticks, zone,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Bayesian summary statistics helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_bayes_summary(posterior_df: pd.DataFrame) -> dict:
    """
    Compute weighted posterior moments and credible intervals for β₀,
    zone probabilities, and any available summary statistics.
    """
    w    = posterior_df["weight"].values / posterior_df["weight"].values.sum()
    beta = posterior_df["beta0"].values

    e_beta  = float(np.average(beta, weights=w))
    var_beta = float(np.average((beta - e_beta) ** 2, weights=w))
    sd_beta = float(np.sqrt(var_beta))

    # Weighted quantiles via sorted cumulative weights
    order   = np.argsort(beta)
    beta_s  = beta[order]
    w_s     = w[order]
    cdf     = np.cumsum(w_s)
    ci_lo   = float(beta_s[np.searchsorted(cdf, 0.025)])
    ci_hi   = float(beta_s[np.searchsorted(cdf, 0.975)])
    median  = float(beta_s[np.searchsorted(cdf, 0.50)])

    # Zone probabilities
    zone_probs = {}
    for z in ["dense_periphery", "sparse_periphery"]:
        mask = posterior_df["zone"] == z if "zone" in posterior_df.columns else np.zeros(len(w), dtype=bool)
        zone_probs[z] = float(w[mask].sum()) if "zone" in posterior_df.columns else 0.5

    # Effective sample size (measure of posterior degeneracy)
    ess = float(1.0 / (w ** 2).sum())

    # Weighted summary stats — use ABC summary stat column names
    # (peak_infectious, peak_day, attack_rate, n_waves, mean_alert_level)
    extra = {}
    for col in ["peak_infectious", "peak_day", "attack_rate", "n_waves", "mean_alert_level"]:
        if col in posterior_df.columns:
            extra[col] = {
                "mean": float(np.average(posterior_df[col].values, weights=w)),
                "sd":   float(np.sqrt(np.average(
                    (posterior_df[col].values - np.average(posterior_df[col].values, weights=w))**2,
                    weights=w
                ))),
            }

    return {
        "e_beta":     e_beta,
        "sd_beta":    sd_beta,
        "median_beta": median,
        "ci_lo":      ci_lo,
        "ci_hi":      ci_hi,
        "zone_probs": zone_probs,
        "ess":        ess,
        "n_particles": len(posterior_df),
        "extra":      extra,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main display
# ─────────────────────────────────────────────────────────────────────────────

if st.session_state.model is None and st.session_state.posterior_df is None:
    st.info("Initialize the model from the sidebar, or load a posterior CSV to explore Bayesian inference.")

# ── Always show the Bayesian Inference tab if a posterior is loaded ───────────
tabs_labels = ["Epidemic curves", "Policy", "Reinfections & waves",
               "Bayesian Inference", "Grid snapshot", "Raw data"]
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(tabs_labels)


# ── Tab 4: Bayesian Inference ─────────────────────────────────────────────────
with tab4:
    st.subheader("Bayesian Inference")

    if st.session_state.posterior_df is None:
        st.info(
            "No posterior loaded. "
            "Run `python seird_gen.py --mode calibrate` "
            "then upload `data/processed/posterior.csv` in the sidebar."
        )
    else:
        post = st.session_state.posterior_df
        summ = _compute_bayes_summary(post)

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 1 — What did calibration find?
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 1 · What did calibration find?")
        st.markdown(
            "The ABC-SMC algorithm searched for values of **β₀** (the per-contact "
            "transmission rate) and **zone** that produce an epidemic matching your "
            "target observations. The results below are the **posterior** — the "
            "probability distribution over plausible parameter values given the data."
        )

        # Key numbers in plain language
        col_a, col_b, col_c = st.columns(3)
        col_a.metric(
            "Most likely β₀",
            f"{summ['median_beta']:.4f}",
            help="Posterior median — the single most representative transmission rate."
        )
        col_b.metric(
            "Plausible range (95%)",
            f"{summ['ci_lo']:.4f} – {summ['ci_hi']:.4f}",
            help="95% credible interval — β₀ is almost certainly somewhere in this range."
        )
        col_c.metric(
            "Calibration quality",
            f"ESS {summ['ess']:.0f} / {summ['n_particles']}",
            delta="Good" if summ["ess"] / summ["n_particles"] > 0.5 else "Re-run with more populations",
            delta_color="normal" if summ["ess"] / summ["n_particles"] > 0.5 else "inverse",
            help="Effective sample size. Above 50% of N means weights are well spread — no degeneracy."
        )

        st.markdown(" ")

        # β₀ histogram — two columns: chart left, plain-language reading right
        col_hist, col_read = st.columns([3, 1])

        with col_hist:
            hist_df = post[["beta0", "weight"]].copy()
            hist_df["weight_norm"] = hist_df["weight"] / hist_df["weight"].sum()
            ci_df   = pd.DataFrame({"x": [summ["ci_lo"], summ["ci_hi"]], "label": ["2.5 %", "97.5 %"]})
            mean_df = pd.DataFrame({"x": [summ["e_beta"]], "label": ["mean"]})

            beta_hist = (
                alt.Chart(hist_df).mark_bar(opacity=0.75, color="#4c78a8")
                .encode(
                    x=alt.X("beta0:Q", bin=alt.Bin(maxbins=35), title="β₀"),
                    y=alt.Y("sum(weight_norm):Q", title="Probability"),
                    tooltip=[alt.Tooltip("beta0:Q", format=".4f"),
                             alt.Tooltip("sum(weight_norm):Q", format=".3f", title="prob. mass")],
                ).properties(height=260, title="Posterior distribution of β₀")
            )
            ci_rules = (
                alt.Chart(ci_df).mark_rule(strokeDash=[5,3], color="firebrick", strokeWidth=1.5)
                .encode(x="x:Q", tooltip=["label:N", alt.Tooltip("x:Q", format=".4f")])
            )
            mean_rule = (
                alt.Chart(mean_df).mark_rule(color="white", strokeWidth=2)
                .encode(x="x:Q", tooltip=["label:N", alt.Tooltip("x:Q", format=".4f")])
            )
            st.altair_chart(
                alt.layer(beta_hist, ci_rules, mean_rule).resolve_scale(y="shared"),
                use_container_width=True
            )
            st.caption("White line = posterior mean · red dashed = 95% credible interval")

        with col_read:
            st.markdown("**How to read this**")
            _med  = f"{summ['median_beta']:.4f}"
            _lo   = f"{summ['ci_lo']:.4f}"
            _hi   = f"{summ['ci_hi']:.4f}"
            st.markdown(
                "Each bar is a range of β₀ values. "
                "Taller bars are more plausible given your epidemic targets.  \n\n"
                f"The model is most confident that β₀ is around **{_med}**, "
                f"and almost rules out anything outside **{_lo} – {_hi}**.  \n\n"
                "A wide histogram means the data doesn't strongly constrain β₀. "
                "A narrow one means the epidemic targets pin it down precisely."
            )

        # Zone probabilities — only if zone column present
        if "zone" in post.columns:
            st.markdown(" ")
            col_z1, col_z2 = st.columns([2, 1])
            with col_z1:
                zone_df = pd.DataFrame({
                    "zone":        list(summ["zone_probs"].keys()),
                    "probability": list(summ["zone_probs"].values()),
                    "label": [
                        f"{v:.1%}" for v in summ["zone_probs"].values()
                    ],
                })
                zone_chart = (
                    alt.Chart(zone_df).mark_bar()
                    .encode(
                        x=alt.X("probability:Q", title="Posterior probability",
                                scale=alt.Scale(domain=[0, 1])),
                        y=alt.Y("zone:N", title="Zone", sort="-x"),
                        color=alt.Color("zone:N", scale=alt.Scale(
                            domain=["dense_periphery", "sparse_periphery"],
                            range=["steelblue", "darkorange"],
                        ), legend=None),
                        text=alt.Text("label:N"),
                        tooltip=["zone:N", alt.Tooltip("probability:Q", format=".3f")],
                    ).properties(height=120, title="Which zone is more consistent with the data?")
                )
                st.altair_chart(
                    zone_chart + zone_chart.mark_text(align="left", dx=4, color="white"),
                    use_container_width=True
                )
            with col_z2:
                dense_p  = summ["zone_probs"].get("dense_periphery", 0.5)
                sparse_p = summ["zone_probs"].get("sparse_periphery", 0.5)
                winner   = "dense periphery" if dense_p > sparse_p else "sparse periphery"
                st.markdown("**How to read this**")
                st.markdown(
                    f"The data slightly favours **{winner}** "
                    f"({max(dense_p, sparse_p):.1%} vs {min(dense_p, sparse_p):.1%}). "
                    f"Values near 50/50 mean zone cannot be identified from the "
                    f"summary statistics alone — both produce similar epidemics."
                )

        st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 2 — Did the model fit the targets?
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 2 · Did the model reproduce the targets?")
        st.markdown(
            "Calibration works by matching five **summary statistics** to target "
            "values. The table below compares what you asked for (target) against "
            "what the posterior particles actually produce on average (fitted). "
            "Close agreement means the calibration succeeded."
        )
        st.info(
            "**Target justification** — targets are derived empirically from the "
            "model's own reference scenario: `--mode diagnose` at β₀=0.020, "
            "averaged across both zones and 3 random seeds (policy ON). "
            "Calibration therefore finds which β₀ values reproduce the dynamics "
            "of the reference scenario, not arbitrary synthetic values."
        )

        if summ["extra"]:
            stat_labels = {
                "peak_infectious":  "Peak infectious agents",
                "peak_day":         "Day of peak",
                "attack_rate":      "Attack rate (fraction infected)",
                "n_waves":          "Number of waves",
                "mean_alert_level": "Mean alert level",
            }
            fit_rows = []
            for k, v in ABC_TARGET.items():
                fitted = summ["extra"].get(k, {}).get("mean", None)
                fitted_sd = summ["extra"].get(k, {}).get("sd", None)
                gap = abs(fitted - v) / max(abs(v), 1e-6) if fitted is not None else None
                fit_rows.append({
                    "Statistic":   stat_labels.get(k, k),
                    "Target":      f"{v:.3g}",
                    "Fitted (mean ± SD)": (
                        f"{fitted:.3g} ± {fitted_sd:.3g}" if fitted is not None else "—"
                    ),
                    "Relative gap": (
                        f"{gap:.1%}" if gap is not None else "—"
                    ),
                    "_gap": gap if gap is not None else 999,
                })

            fit_df = pd.DataFrame(fit_rows).drop(columns=["_gap"])
            st.dataframe(fit_df.set_index("Statistic"), use_container_width=True)
            st.caption(
                "Relative gap = |target − fitted| / target. "
                "Below 10% is good; above 30% suggests the target is hard to hit "
                "with the current model — consider adjusting ABC_TARGET in seird_gen.py."
            )
        else:
            st.info(
                "Summary statistics not available in this posterior file. "
                "Re-run `--mode calibrate` to generate a posterior with fitted statistics."
            )

        st.divider()

        # ═══════════════════════════════════════════════════════════════════
        # SECTION 3 — Posterior predictive check
        # ═══════════════════════════════════════════════════════════════════
        st.markdown("## 3 · Does the model behave consistently with the posterior?")
        st.markdown(
            "A **posterior predictive check (PPC)** re-simulates the model 50 times, "
            "each time drawing β₀ and zone from the posterior. The shaded bands show "
            "the range of outcomes across those simulations. "
            "Your single run (solid line) should sit inside the bands — if it does, "
            "your chosen β₀ is consistent with what calibration found."
        )

        envelope = st.session_state.ppc_envelope

        if envelope is None:
            st.info("Press **Compute uncertainty bands** in the sidebar to generate the PPC.")
        else:
            df_cur = st.session_state.results_df if st.session_state.model else pd.DataFrame()

            # Show only infectious (the most interpretable) at full width,
            # then exposed and dead side by side, susceptible collapsed below
            def _ppc_chart(state_key, label, color, height=240):
                if state_key not in envelope:
                    return None
                env = envelope[state_key][["tick","q025","q25","median","q75","q975"]].copy()
                band = (alt.Chart(env).mark_area(opacity=0.12, color=color)
                        .encode(x=alt.X("tick:Q", title="Tick"),
                                y=alt.Y("q025:Q", title="Agents"),
                                y2=alt.Y2("q975:Q")))
                iqr  = (alt.Chart(env).mark_area(opacity=0.22, color=color)
                        .encode(x="tick:Q", y=alt.Y("q25:Q"), y2=alt.Y2("q75:Q")))
                med  = (alt.Chart(env).mark_line(color=color, strokeDash=[4,2], strokeWidth=1.5)
                        .encode(x="tick:Q", y=alt.Y("median:Q", title="Agents"),
                                tooltip=[alt.Tooltip("tick:Q", title="Tick"),
                                         alt.Tooltip("median:Q", title="Median"),
                                         alt.Tooltip("q025:Q",   title="2.5%"),
                                         alt.Tooltip("q975:Q",   title="97.5%")]))
                layers = [band, iqr, med]
                if not df_cur.empty and state_key in df_cur.columns:
                    obs_df = df_cur[["tick", state_key]].rename(columns={state_key: "obs"})
                    obs = (alt.Chart(obs_df).mark_line(color=color, strokeWidth=2.5)
                           .encode(x="tick:Q", y=alt.Y("obs:Q", title="Agents"),
                                   tooltip=[alt.Tooltip("tick:Q", title="Tick"),
                                            alt.Tooltip("obs:Q",  title="Your run")]))
                    layers.append(obs)
                return (alt.layer(*layers).resolve_scale(y="shared")
                        .properties(height=height, title=label))

            # Infectious — full width, most important
            c = _ppc_chart("total_infectious", "Infectious agents over time", "#e45756", 280)
            if c:
                st.altair_chart(c, use_container_width=True)
                in_band = None
                if not df_cur.empty and "total_infectious" in df_cur.columns:
                    env_inf = envelope["total_infectious"]
                    obs_vals = df_cur.set_index("tick")["total_infectious"].reindex(env_inf["tick"]).ffill()
                    pct_in = ((obs_vals.values >= env_inf["q025"].values) &
                              (obs_vals.values <= env_inf["q975"].values)).mean()
                    colour = "🟢" if pct_in > 0.8 else "🟡" if pct_in > 0.5 else "🔴"
                    st.caption(
                        f"{colour} Your run is inside the 95% band for "
                        f"**{pct_in:.0%}** of ticks. "
                        + ("Good fit — β₀ is consistent with the posterior."
                           if pct_in > 0.8 else
                           "Partial fit — your β₀ may differ from the posterior mean."
                           if pct_in > 0.5 else
                           "Poor fit — consider initialising with the posterior mean β₀ shown in the sidebar.")
                    )

            # Exposed and Dead — side by side
            col_e, col_d = st.columns(2)
            with col_e:
                c = _ppc_chart("exposed", "Exposed agents", "#f58518", 220)
                if c: st.altair_chart(c, use_container_width=True)
            with col_d:
                c = _ppc_chart("dead", "Cumulative deaths", "#54a24b", 220)
                if c: st.altair_chart(c, use_container_width=True)

            # Susceptible — collapsed, less critical
            with st.expander("Susceptible agents (expand to view)"):
                c = _ppc_chart("susceptible", "Susceptible agents", "#4c78a8", 200)
                if c: st.altair_chart(c, use_container_width=True)

            st.caption(
                "**Bands**: light = 95% credible interval · dark = 50% interval · "
                "dashed = posterior median · solid = your current simulation run."
            )


# ── Tabs that require a running model ─────────────────────────────────────────

if st.session_state.model is None:
    with tab1: st.info("Initialize the model from the sidebar.")
    with tab2: st.info("Initialize the model from the sidebar.")
    with tab3: st.info("Initialize the model from the sidebar.")
    with tab5: st.info("Initialize the model from the sidebar.")
    with tab6: st.info("Initialize the model from the sidebar.")
else:
    df     = st.session_state.results_df
    latest = df.iloc[-1]

    c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
    c1.metric("Tick",         int(latest["tick"]))
    c2.metric("Day",          f"{latest['day']:.1f}")
    c3.metric("Zone",         zone)
    c4.metric("β₀",           f"{beta0:.3f}")
    c5.metric("Susceptible",  int(latest["susceptible"]))
    c6.metric("Infectious",   int(latest["total_infectious"]))
    c7.metric("Dead",         int(latest["dead"]))
    c8.metric("Alert level",  int(latest["alert_level"]))

    # ── Tab 1: Epidemic curves ────────────────────────────────────────────────
    with tab1:
        st.subheader("SEIRD epidemic curves")
        epi_long = df.melt(
            id_vars=["tick", "day"],
            value_vars=["susceptible", "exposed", "total_infectious",
                        "infectious_asymp", "infectious_symp", "recovered", "dead"],
            var_name="state", value_name="count",
        )
        st.altair_chart(
            alt.Chart(epi_long).mark_line(interpolate="step-after")
            .encode(
                x=alt.X("tick:Q", title="Tick"),
                y=alt.Y("count:Q", title="Agents"),
                color=alt.Color("state:N"),
                tooltip=["tick", "day", "state", "count"],
            ).properties(height=400),
            use_container_width=True,
        )
        inc_long = df.melt(
            id_vars=["tick", "day"], value_vars=["confirmed_today", "dead"],
            var_name="indicator", value_name="value",
        )
        st.altair_chart(
            alt.Chart(inc_long).mark_line()
            .encode(
                x=alt.X("tick:Q"), y=alt.Y("value:Q"),
                color=alt.Color("indicator:N"),
                tooltip=["tick", "day", "indicator", "value"],
            ).properties(height=250, title="Daily confirmed cases & deaths"),
            use_container_width=True,
        )

    # ── Tab 2: Policy ─────────────────────────────────────────────────────────
    with tab2:
        st.subheader("Health Ministry policy state")
        st.altair_chart(
            alt.Chart(df).mark_line(interpolate="step-after", color="steelblue")
            .encode(
                x=alt.X("tick:Q"), y=alt.Y("alert_level:Q", scale=alt.Scale(domain=[0, 3])),
                tooltip=["tick", "day", "alert_level", "seven_day_prev"],
            ).properties(height=260, title="Alert level over time"),
            use_container_width=True,
        )
        thr_df = pd.DataFrame({
            "threshold": list(DEFAULT_THRESHOLD_ALERT[1:]),
            "label": ["Alert 1", "Alert 2", "Alert 3"],
        })
        st.altair_chart(
            (alt.Chart(df.dropna(subset=["seven_day_prev"])).mark_line(color="firebrick")
             .encode(x="tick:Q", y=alt.Y("seven_day_prev:Q", title="7-day prevalence"),
                     tooltip=["tick", "day", "seven_day_prev"])
             .properties(height=260, title="7-day prevalence & alert thresholds")
             + alt.Chart(thr_df).mark_rule(strokeDash=[6, 3])
             .encode(y="threshold:Q", color=alt.Color("label:N"))),
            use_container_width=True,
        )
        alert_days = {f"Days at alert {i}": int((df["alert_level"] == i).sum()) for i in range(4)}
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
            st.altair_chart(
                alt.Chart(df).mark_line(color="purple")
                .encode(x="tick:Q", y=alt.Y("total_reinfections:Q", title="Cumulative reinfections"),
                        tooltip=["tick", "day", "total_reinfections"])
                .properties(height=300, title="Cumulative reinfections"),
                use_container_width=True,
            )
        with col_wv:
            wv = st.session_state.model.get_wave_summary()
            st.metric("Detected waves",       wv["n_waves"])
            st.metric("Mean peak infectious", f"{wv['mean_peak_infectious']:.1f}")
            st.metric("Max peak infectious",  f"{wv['max_peak_infectious']:.1f}")

        st.markdown("---")
        st.subheader("Population personality snapshot")
        pop        = st.session_state.model.individuals
        alive_mask = pop.state != S.DEAD
        snap = pd.DataFrame({
            "compliance":    pop.compliance[alive_mask],
            "doctor_prob":   pop.doctor_prob[alive_mask],
            "immunity_days": pop.immunity_ticks[alive_mask] / TICKS_PER_DAY,
        })
        col_a, col_b = st.columns(2)
        with col_a:
            st.altair_chart(
                alt.Chart(snap).mark_bar(opacity=0.7)
                .encode(x=alt.X("compliance:Q", bin=alt.Bin(maxbins=30)),
                        y=alt.Y("count():Q"))
                .properties(height=220, title=f"Compliance ({zone})"),
                use_container_width=True,
            )
        with col_b:
            st.altair_chart(
                alt.Chart(snap).mark_bar(opacity=0.7)
                .encode(x=alt.X("immunity_days:Q", bin=alt.Bin(maxbins=30)),
                        y=alt.Y("count():Q"))
                .properties(height=220, title="Immunity duration (days)"),
                use_container_width=True,
            )

    # ── Tab 5: Grid snapshot ──────────────────────────────────────────────────
    with tab5:
        st.subheader("Live spatial animation")
        col_ctrl1, col_ctrl2, _ = st.columns([1, 1, 2])
        with col_ctrl1:
            if st.button("▶ Play / ⏸ Pause", use_container_width=True):
                st.session_state.playing = not st.session_state.get("playing", False)
        with col_ctrl2:
            tick_delay = st.slider("Speed (sec/tick)", 0.1, 2.0, 0.5, 0.1)

        chart_ph = st.empty()
        info_ph  = st.empty()

        def render_grid():
            p = st.session_state.model.individuals
            snames = {
                S.SUSCEPTIBLE: "Susceptible", S.EXPOSED: "Exposed",
                S.INFECTIOUS_ASYMPTOMATIC: "Infectious (A)",
                S.INFECTIOUS_SYMPTOMATIC:  "Infectious (S)",
                S.RECOVERED: "Recovered", S.DEAD: "Dead",
            }
            pos_df = pd.DataFrame({"x": p.px, "y": p.py,
                                   "state": [snames.get(s, str(s)) for s in p.state]})
            n = min(2000, len(pos_df))
            pos_df = pos_df.sample(n, random_state=42)
            chart_ph.altair_chart(
                alt.Chart(pos_df).mark_circle(size=20, opacity=0.65)
                .encode(
                    x=alt.X("x:Q", scale=alt.Scale(domain=[0, DEFAULT_GRID_SIZE])),
                    y=alt.Y("y:Q", scale=alt.Scale(domain=[0, DEFAULT_GRID_SIZE])),
                    color=alt.Color("state:N"),
                    tooltip=["x", "y", "state"],
                ).properties(height=560),
                use_container_width=True,
            )
            info_ph.caption(f"Tick {st.session_state.model.tick} | Day {st.session_state.model.current_day:.1f} | {n} agents shown")

        render_grid()
        if st.session_state.get("playing", False):
            while st.session_state.get("playing", False):
                if st.session_state.model.tick >= st.session_state.model.total_ticks:
                    st.session_state.playing = False
                    break
                st.session_state.model.step()
                st.session_state.results_df = st.session_state.model.get_results_df()
                render_grid()
                time.sleep(tick_delay)

    # ── Tab 6: Raw data ───────────────────────────────────────────────────────
    with tab6:
        st.subheader("Raw simulation output")
        st.dataframe(df, use_container_width=True)
        st.download_button(
            "Download CSV", df.to_csv(index=False).encode("utf-8"),
            "simulation_results.csv", "text/csv",
        )
        st.json(st.session_state.model.get_wave_summary())