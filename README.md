# SME Invoice Finance Risk & Exposure Monitor

Turns real transaction data into invoice-finance risk metrics, fraud flags, and an
early-warning default model — mirroring the core work of a data analyst at an
invoice finance lender.

## Business context
An invoice finance lender advances cash against clients' unpaid invoices. The
central question is always "what is our exposure and where is risk building?"
This project answers it across four analyst functions:

| Function            | Metric / output                                            |
|---------------------|------------------------------------------------------------|
| Exposure monitoring | funding-in-use, net ledger value                           |
| Credit risk         | dilution rate, debtor concentration, arrears/dormancy      |
| Early warning       | transparent weighted risk scorecard (client watchlist)     |
| Fraud detection     | round-number & duplicate-invoice anomaly flags             |

## Real data
- **Invoice ledger:** UCI Online Retail II — archive.ics.uci.edu/dataset/502
  (customers = clients, credit-note invoices `C...` = dilution)
- **Default outcomes (optional upgrade):** Lending Club Loan Data (Kaggle) —
  join real `default` labels to fit a *predictive* model on the scorecard's
  features, with `score_clients()` as the explainable baseline to beat.

Framing is adapted (customers->clients, returns->dilution); the numbers are real.

## Run
```
pip install pandas numpy openpyxl
python src/pipeline.py
```
Drop `online_retail_II.xlsx` into `data/` to run on the full ~1M-row dataset;
otherwise a bundled sample runs automatically. Each run also builds a
self-contained `outputs/dashboard.html` — open it in any browser, no server needed.

## Outputs
- `outputs/client_risk_metrics.csv` — per-client exposure & risk metrics
- `outputs/fraud_flags.csv` — flagged suspicious invoices
- `outputs/client_watchlist.csv` — clients ranked by scorecard risk score
  (with per-signal component columns for auditability)
- `outputs/ledger_trend.csv` — weekly invoicing advanced & dilution rate (flow view)
- `outputs/dashboard.html` — interactive risk console over all of the above
  (KPIs, trend, exposure, dilution×concentration, watchlist, fraud), generated
  from `src/dashboard_template.html` with the run's own data embedded

## Next steps
- Add real Lending Club default outcomes and fit a predictive model on the
  scorecard's features — using the transparent scorecard as the baseline to beat
- Enrich clients with real company data via the Companies House API
- Extend the trend from a flow view to true outstanding-balance ageing once
  settlement/payment dates are available
