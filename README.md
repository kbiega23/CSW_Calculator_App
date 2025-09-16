
# CSW Streamlit Scaffold (Prototype)

This folder contains a working scaffold for a Streamlit app that mirrors your Excel-based CSW Savings Calculator.

## What's included
- `data/`
  - `weather_information.csv` — exported from **Weather Information** sheet
  - `savings_lookup.csv` — exported from **Savings Lookup** sheet
  - `regresson_list_*.csv` — exported from the **Regression List_* ** sheets
  - `lists.csv` — exported from **Lists**
- `engine/engine.py` — data loaders + placeholders for regression selection & savings math
- `app/streamlit_app.py` — step-by-step wizard UI (State → City → Building → CSW → Rates → Results)
- (Add a `requirements.txt` with: `streamlit`, `pandas`)

## Run locally
```bash
pip install streamlit pandas
streamlit run app/streamlit_app.py
```

## Next steps
- Implement `engine.regression_value(...)` to select the correct `a,b,c` based on keys (Base, CSW Type, HVAC Type, Fuel, Size, Building Type).
- Map **Savings Lookup** joins to build the composite keys (equivalent to Excel `VLOOKUP(R24, ...)`).
- Compute **kWh/therms/$ savings** and EUI % savings, and display them in the Results section.
- Add lead capture (Google Sheets or HubSpot) and download buttons (CSV/PDF).

> NOTE: Column names come directly from the exported sheets and may need one-time normalization.
