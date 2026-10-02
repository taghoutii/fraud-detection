# Fraud Risk Console

Read-only React + Vite dashboard over the pipeline's saved results. It trains nothing and
computes no new metrics: it reads JSON from `public/data/` and draws it.

```bash
npm install
npm run dev        # http://localhost:5173
npm run build      # static site in dist/
```

`public/data/` is committed, so a clean clone runs with no Python step. To regenerate it after
the results in `reports/results/` change (uses the project's Python environment):

```bash
python scripts/prepare_data.py     # from dashboard/, or: npm run prepare-data
```

The script reformats the Parquet/JSON results into frontend-friendly JSON (all 118,108 test
predictions, the full 426-feature SHAP sample split into 100-row chunks) and checks the result:
PR-AUC/ROC-AUC rebuilt from the converted predictions must match Stage 6, and every SHAP row
must still add up to its saved probability.

**Stack:** React 18, Recharts (all charts except the SHAP summary, which is drawn on canvas
because 40,000 points is too many SVG nodes), plain `fetch` + `useState` for data. Colors are CSS
custom properties in `src/styles/tokens.css`; red is reserved for fraud and risk.
