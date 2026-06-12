# Smart Spare Parts Demand Forecasting & JIT Inventory System

End-to-end demand forecasting for automotive spare parts, turning a raw service-invoice log into actionable **reorder points** so service centres can move toward a Just-In-Time (JIT) inventory standard.

**Live demo:** https://spare-parts-demand-forecasting-z2ttznq93vpklednappv8md.streamlit.app/
**Notebook:** [`analysis.ipynb`](analysis.ipynb)

---

## The problem

A service centre's spare-parts inventory is a constant balancing act. Too much stock locks up capital and risks obsolescence; too little causes stockouts and delayed repairs. The client (NewX Services) wanted a predictive model to forecast spare-parts demand and support JIT stocking.

The catch: **the dataset has no demand or quantity column.** It is a transactional log with one row per part used during a service visit. The core data-science task was to *engineer* a demand signal from this log, then forecast it, then convert the forecast into operational stocking decisions.

## What this project does

1. **Engineers a weekly demand time series** for each spare part from ~28k raw service-invoice rows.
2. **Classifies each part's demand pattern** using the Syntetos-Boylan ADI / CV-squared framework (Smooth, Erratic, Intermittent, Lumpy) - because different patterns need different forecasting methods.
3. **Runs a 9-algorithm bake-off per part** and selects the best model for each.
4. **Explains the ML model with SHAP** so forecasts are interpretable, not a black box.
5. **Validates with a walk-forward backtest** (rolling-origin) to prove the model choices generalise, not just fit one lucky test window.
6. **Converts forecasts into reorder points and safety stock** at a target service level - the operational JIT deliverable.
7. **Organises parts with an ABC-XYZ matrix** so each part gets the right stocking policy.
8. **Ships a live Streamlit dashboard** for interactive exploration.

## What makes it different

- **Per-part model selection across 9 algorithms**, not one global model forced on every part:
  - Classical: Holt-Winters, SARIMA
  - Intermittent demand: Croston's, SBA (Syntetos-Boylan Approximation), TSB (Teunter-Syntetos-Babai)
  - Modern ML: a single **global LightGBM** trained across all parts on engineered lag / rolling / calendar features
  - Plus Prophet and the Theta method as strong independent benchmarks
- **SHAP applied to a forecasting model** - uncommon, and it shows that recent demand (lag-1, rolling mean) drives next-week demand, with calendar features adding seasonal nudges.
- **Walk-forward backtest + calibration check** - the time-series analogue of cross-validation and calibration curves.
- **An honest engineering decision to exclude deep learning** - with only ~49 weekly points per part, an LSTM would overfit; this is documented in the notebook's Future Work.

## Tech stack

`Python` `pandas` `numpy` `statsmodels` `LightGBM` `Prophet` `SHAP` `scikit-learn` `Plotly` `Streamlit`

## The Streamlit app

Four tabs:
- **Data Insights** - top parts, vehicle models, monthly trend, and the demand-pattern quadrant
- **Forecasts** - per-part 8-week forecast with the selected model and the algorithm leaderboard
- **Reorder Calculator** - interactive sliders for lead time and service level that recompute reorder points and safety stock live
- **ABC-XYZ Strategy** - the inventory matrix with a recommended policy per quadrant

The app loads precomputed artifacts (exported from the notebook), so it is fast, self-contained, and does not need a live database connection.

## Repo structure

```
.
├── app.py                  # Streamlit dashboard
├── requirements.txt        # pinned dependencies
├── .streamlit/
│   └── config.toml         # app theme
├── artifacts/              # precomputed CSV/JSON the app loads
├── analysis.ipynb          # full analysis notebook (EDA -> models -> inventory)
└── README.md
```

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Reproduce the analysis

1. Open `analysis.ipynb` and run it top to bottom (database credentials are provided separately and are not committed to this repo).
2. Run the export cell at the end to regenerate the `artifacts/` folder.

## Deploy to Streamlit Cloud

1. Push this repo to GitHub.
2. On [share.streamlit.io](https://share.streamlit.io), create a new app pointing at `app.py`.
3. In Advanced settings, set **Python version to 3.11** before deploying.

## Results summary

- A weekly demand series was successfully engineered for 12 stockable spare parts spanning the full demand spectrum (high-volume consumables to lumpy slow movers).
- The best-performing models cluster around the steady, high-throughput parts that matter most for JIT - exactly where accurate forecasts deliver the most inventory savings.
- The forecast-driven reorder policy holds less working stock than a naive blanket policy while protecting a 95% service level.

## Limitations

- Roughly 11 months of history, so annual seasonality cannot be modelled.
- One invoice row is treated as one unit of demand (no explicit quantity field).
- Lead time and service level are configurable assumptions, not client-confirmed values.

See the notebook's Limitations and Future Work section for the full discussion and a roadmap (more history for seasonal models, cost-aware optimisation, demand-distribution simulation for intermittent parts).

---

*Built as a final client data-science project. Dataset is proprietary and not included in this repository.*
