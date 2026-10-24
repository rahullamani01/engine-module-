# Adaptive Distribution Intelligence Engine (Byte Works, Triwizathon 2026)

Decision-support prototype. All data is SYNTHETIC (seed 42); results are not evidence of real-world performance.

## Status: gate G1 (data + MA4 baseline)
Implemented (PRD v5.0 schemas: locations.csv, products.csv, demand_history.csv, hidden_truth.csv, config.json): config, synthetic data, validation, MA4 residual-quantile forecaster, pinball/coverage/WAPE, tests.
Not yet: gradient-boosting model (G2), optimizer/checker (G3), Streamlit app (G4).

## Run
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m pytest
python -m src.baseline normal          # or scarce_supply / expiry_trap
```
Outputs go to `data/<scenario>/` and `outputs/<scenario>/` (git-ignored, regenerated deterministically).
