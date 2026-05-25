# ABM_final_paper
This repository contains the code, the data and the final jupyter notebook for the final reearch paper of the course Bayesian Modelling of Complexity. I passed this course as part of the requirements for th Master of Data Science for Public Policy, Hertie School, Spring 2026.

The python script `seird_gen.py` generates the data and saves the outputs as .png files. A notebook `hierarchical_bayes_abm.ipynb`accompanies the analysis.

# Does Government Policy Save Lives?
### A Hierarchical Bayesian Analysis of Epidemic Wave Severity Across Compliance Zones

**Alexis Grangier · Student ID 229735 · Bayesian Modeling of Complexity · Hertie School MDS**

---

## Overview

This repository contains the full computational pipeline for a research paper that asks: *does a government health alert system reduce epidemic wave severity, and is this effect moderated by zone-level compliance?*

The answer is derived in two stages:

1. **Agent-Based Model (ABM)** — 500 agents simulate one year of infectious disease transmission in a city divided into two residential zones. The model is run 400 times under a factorial design (5 transmission rates × 2 zones × 20 seeds × 2 policy conditions) to produce a dataset of epidemic outcomes.

2. **Hierarchical Bayesian Model (PyMC)** — a hierarchical Normal regression is fit to the 400-run dataset. It estimates the policy effect, the zone compliance effect, and their interaction, with partial pooling across the 10 (β₀ × zone) groups.

A third component — **ABC-SMC calibration** — finds which transmission rates are consistent with the reference epidemic dynamics, and is visualised in the Streamlit app.

---

## Model Description

### Agent-Based Model

Each agent is initialised with individual personality parameters drawn from zone-specific priors:

| Parameter | Dense periphery | Sparse periphery |
|---|---|---|
| Compliance | Beta(1.5, 4.0) · mean ≈ 0.27 | Beta(3.0, 2.0) · mean ≈ 0.60 |
| Doctor visit prob. | Beta(4.0, 4.0) · mean ≈ 0.50 | Beta(6.0, 3.0) · mean ≈ 0.67 |
| Immunity duration | Gamma(13, 7 days) · mean ≈ 91 days | same |
| Transport mode | Transit-heavy (Dirichlet [5,2,3]) | Car-heavy (Dirichlet [2,5,3]) |

**Disease states**: Susceptible → Exposed → Infectious (asymptomatic or symptomatic) → Recovered → Susceptible (waning immunity). Dead is absorbing.

**Transmission**: at each 12-hour tick, susceptible agents within contact radius 1 of infectious agents face force of infection `λ = β₀ × transport_mult × (1 − compliance × policy_reduction[alert])`.

**Spark rate**: `λ_spark = 0.0002` per susceptible per tick models imported cases, preventing epidemic extinction between waves.

**Policy**: a Health Ministry tracks 7-day confirmed case prevalence and activates four alert levels (0–3) at prevalence thresholds [0, 0.005, 0.015, 0.030]. Each level reduces effective β by [0%, 25%, 55%, 80%] × agent compliance.

**Primary outcome**: `mean_peak_infectious` — mean height of detected epidemic wave peaks across 365 days (scipy.find_peaks, height ≥ 20 agents, distance ≥ 14 days).

### Hierarchical Model

```
y_ig ~ Normal(μ_ig, σ_obs)
μ_ig = α_g + β_policy·p_i + β_zone·z_g + β_β0·β̃₀_g + β_int·z_g·β̃₀_g
α_g  = μ_α + σ_α·δ_g,   δ_g ~ Normal(0, 1)   [non-centred]
```

Groups `g` are the 10 (β₀ × zone) combinations. Estimated with NUTS via PyMC, 4 chains × 2,000 draws, target acceptance 0.95.

### ABC-SMC Calibration

Approximate Bayesian Computation — Sequential Monte Carlo identifies which values of β₀ (and zone) reproduce five summary statistics of the reference scenario (β₀=0.020, policy ON, averaged across both zones and 3 seeds):

| Statistic | Target |
|---|---|
| Peak infectious agents | 57 |
| Day of first peak | 25 |
| Attack rate | 0.70 |
| Number of waves | 9 |
| Mean alert level | 1.8 |

Prior: `β₀ ~ Uniform(0.005, 0.040)`, `zone ~ Categorical(0.5, 0.5)`. 5 populations of 200 particles, ε from 2.0 → 0.80. Parallelised across all CPU cores via `mp.Pool`.

---

## Repository Structure

```
ABM_PAPER_REPO/
│
├── seird_gen.py                  # ABM engine + ABC-SMC calibration
│                                 # Run modes:
│                                 #   --mode diagnose   single run + diagnostic plots
│                                 #   --mode sweep      60-run parameter sweep
│                                 #   --mode generate   full 400-run dataset for PyMC
│                                 #   --mode calibrate  ABC-SMC posterior over β₀ and zone
│
├── app.py                        # Streamlit interactive app
│                                 # Tabs: epidemic curves, policy, reinfections,
│                                 #       Bayesian inference (PPC + posterior), grid, raw data
│
├── hierarchical_bayes_abm.ipynb  # Fully executed PyMC notebook
│                                 # EDA → model specification → MCMC → diagnostics
│                                 # → posterior analysis → PPC → sensitivity analysis
│
├── data/
│   └── processed/
│       ├── abm_outputs.csv       # 400-run factorial dataset (generated by --mode generate)
│       └── posterior.csv         # ABC-SMC posterior particles (generated by --mode calibrate)
│
├── outputs/
│   └── plots/                    # All saved figures
│       ├── abc_smc_posterior.png             # Posterior evolution across SMC populations
│       ├── diagnostics.png                   # Combined diagnostic grid (--mode diagnose)
│       ├── diag_dense_periphery_*.png        # Per-zone diagnostic panels
│       ├── diag_sparse_periphery_*.png
│       ├── sweep_mean_peak_by_beta.png       # Sweep summary plots
│       ├── sweep_policy_effect_by_beta.png
│       ├── sweep_n_waves_heatmap.png
│       └── fig1_eda.png … fig8_ppc.png       # PyMC notebook figures
│
├── pyproject.toml                # uv project configuration
├── uv.lock                       # Locked dependencies
└── README.md                     # This file
```

---

## Quickstart

```bash
# 1. Install dependencies
uv sync

# 2. Check model diagnostics (always start here after changing parameters)
uv run seird_gen.py --mode diagnose

# 3. Generate the 400-run dataset
uv run seird_gen.py --mode generate

# 4. Calibrate — produces data/processed/posterior.csv
uv run seird_gen.py --mode calibrate

# 5. Launch the interactive app
uv run streamlit run app.py
```

To run the PyMC notebook, open `hierarchical_bayes_abm.ipynb` in Jupyter and run all cells. Dependencies: `pymc`, `arviz`, `pandas`, `numpy`, `matplotlib`.

---

## Key Results

| Parameter | Posterior mean | 94% HDI | P(< 0) |
|---|---|---|---|
| β_policy (policy effect) | −23.5 agents | [−26.0, −20.7] | 1.000 |
| β_zone (sparse vs dense) | −28.1 agents | [−33.2, −22.7] | 1.000 |
| β_β₀ (transmission slope) | +23.0 agents | [+19.6, +26.5] | 0.000 |
| β_int (zone × β₀) | −11.4 agents | [−16.8, −6.7] | 0.999 |

The health alert system reduces mean wave peak by ~28%. This effect is larger in sparse (high-compliance) periphery and grows with transmission rate — the policy works best precisely where populations are already disposed to comply.

---

## Dependencies

```
python  ≥ 3.11
numpy   pandas  scipy  matplotlib
pymc    arviz
streamlit  altair
```

---

## Citation

Grangier, A. (2026). *Does Government Policy Save Lives? A Hierarchical Bayesian Analysis of Epidemic Wave Severity Across Compliance Zones.* Bayesian Modeling of Complexity, Hertie School MDS.

ABM originally designed with Anna Jurek based on a shared ODD protocol.

