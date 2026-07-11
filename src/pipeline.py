"""
SME Invoice Finance Risk & Exposure Monitor
-------------------------------------------
Pipeline that turns real transaction data (UCI Online Retail II) into
invoice-finance risk metrics, then trains an early-warning default model.

Real data sources this runs on:
  - UCI Online Retail II  -> archive.ics.uci.edu/dataset/502  (invoice ledger)
  - Lending Club Loan Data -> Kaggle                          (real default outcomes)

Framing note: customers are treated as "clients", returns/credit-note invoices
(prefix 'C') are treated as "dilution". The numbers are real; only the lending
framing is adapted.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path

DATA = Path(__file__).resolve().parent.parent / "data"
OUT = Path(__file__).resolve().parent.parent / "outputs"
OUT.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------------
def load_online_retail(path=None):
    """Load real UCI Online Retail II. Falls back to bundled sample if absent."""
    path = path or (DATA / "online_retail_II.xlsx")
    if Path(path).exists():
        # real file has two sheets; concat them
        xls = pd.ExcelFile(path)
        df = pd.concat([xls.parse(s) for s in xls.sheet_names], ignore_index=True)
    else:
        print(f"[warn] {path} not found - using bundled sample.")
        df = pd.read_csv(DATA / "sample_retail.csv")

    df.columns = [c.strip() for c in df.columns]
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df.dropna(subset=["Customer ID"])
    df["Customer ID"] = df["Customer ID"].astype(int)
    df["LineValue"] = df["Quantity"] * df["Price"]
    df["IsCreditNote"] = df["Invoice"].astype(str).str.startswith("C")
    return df


# ---------------------------------------------------------------------------
# 2. RISK METRICS PER CLIENT  (the SQL-equivalent logic in pandas)
# ---------------------------------------------------------------------------
def client_risk_metrics(df):
    """One row per client with the core invoice-finance risk metrics."""
    gross = (df[~df.IsCreditNote].groupby("Customer ID")["LineValue"]
             .sum().rename("gross_invoicing"))
    credits = (df[df.IsCreditNote].groupby("Customer ID")["LineValue"]
               .sum().abs().rename("credit_notes"))
    n_invoices = df.groupby("Customer ID")["Invoice"].nunique().rename("n_invoices")
    last_activity = df.groupby("Customer ID")["InvoiceDate"].max().rename("last_activity")

    # debtor concentration proxy: largest single stock line as % of client ledger.
    # Vectorised (max/sum over the per-client stock-line totals) so it scales to
    # the full ~1M-row ledger instead of a per-group Python apply.
    line = (df[~df.IsCreditNote]
            .groupby(["Customer ID", "StockCode"])["LineValue"].sum())
    grp = line.groupby(level=0)
    gsum = grp.sum()
    line_share = (grp.max() / gsum.where(gsum > 0)).rename("top_line_concentration")

    m = pd.concat([gross, credits, n_invoices, last_activity, line_share], axis=1)
    m["credit_notes"] = m["credit_notes"].fillna(0)
    m["gross_invoicing"] = m["gross_invoicing"].fillna(0)

    # funding-in-use: cash advanced = net ledger value * advance rate (typ. 80%)
    ADVANCE_RATE = 0.80
    m["net_ledger"] = m["gross_invoicing"] - m["credit_notes"]
    m["funding_in_use"] = (m["net_ledger"] * ADVANCE_RATE).clip(lower=0)

    # dilution rate: credit notes as % of gross invoicing (key risk metric)
    m["dilution_rate"] = np.where(m["gross_invoicing"] > 0,
                                  m["credit_notes"] / m["gross_invoicing"], 0)

    # recency of activity (days since last invoice) - slowing/dormant clients
    asof = df["InvoiceDate"].max()
    m["days_since_activity"] = (asof - m["last_activity"]).dt.days

    return m.reset_index()


# ---------------------------------------------------------------------------
# 3. FRAUD / ANOMALY FLAGS  (rules-based, run on real ledger)
# ---------------------------------------------------------------------------
def fraud_flags(df):
    inv = df.groupby("Invoice").agg(
        client=("Customer ID", "first"),
        value=("LineValue", "sum"),
        n_lines=("StockCode", "nunique"),
    ).reset_index()

    inv["flag_round_number"] = (inv["value"] % 100 == 0) & (inv["value"] > 0)
    # duplicate value+client on same day is a classic duplicate-invoice signature
    dmap = df.groupby("Invoice")["InvoiceDate"].first()
    inv["date"] = inv["Invoice"].map(dmap).dt.date
    inv["flag_duplicate"] = inv.duplicated(subset=["client", "value", "date"], keep=False) & (inv["value"] > 0)

    flagged = inv[(inv.flag_round_number) | (inv.flag_duplicate)]
    return flagged.sort_values("value", ascending=False)


# ---------------------------------------------------------------------------
# 4. EARLY-WARNING RISK SCORECARD
# ---------------------------------------------------------------------------
# Weights and normalisation caps for the scorecard. Kept here so they're the
# single, auditable place risk appetite is expressed.
SCORE_WEIGHTS = {"dilution": 0.50, "concentration": 0.25, "dormancy": 0.25}
DILUTION_CAP = 0.30    # dilution >= 30% saturates the dilution signal
DORMANCY_CAP = 90.0    # 90+ days since last invoice saturates the dormancy signal


def score_clients(metrics):
    """
    Transparent, auditable risk scorecard: a weighted blend of the three core
    invoice-finance risk signals (dilution, debtor concentration, dormancy),
    each normalised to 0..1. This is a documented *rule*, not a trained model,
    so there is no target leakage - which is exactly how credit scorecards work.

    To make this *predictive* rather than descriptive, join real default
    outcomes (e.g. Lending Club) and fit a classifier on these same features;
    the scorecard then becomes the explainable baseline to beat. See README.
    """
    d = metrics.copy()

    dil = (d["dilution_rate"] / DILUTION_CAP).clip(0, 1)
    conc = d["top_line_concentration"].fillna(0).clip(0, 1)
    dorm = (d["days_since_activity"] / DORMANCY_CAP).clip(0, 1)

    d["dilution_component"] = (dil * SCORE_WEIGHTS["dilution"]).round(4)
    d["concentration_component"] = (conc * SCORE_WEIGHTS["concentration"]).round(4)
    d["dormancy_component"] = (dorm * SCORE_WEIGHTS["dormancy"]).round(4)
    d["risk_score"] = (d["dilution_component"]
                       + d["concentration_component"]
                       + d["dormancy_component"]).round(4)

    # priority = risk-weighted exposure (expected-loss style). This is what a
    # lender acts on: a maxed-out score on a client with £0 advanced is a note,
    # not an action; the money at risk sits where score AND funding are high.
    d["priority"] = (d["risk_score"] * d["funding_in_use"]).round(2)
    return d


# ---------------------------------------------------------------------------
# 4b. PREDICTIVE EARLY-WARNING MODEL  (out-of-time, leakage-free)
# ---------------------------------------------------------------------------
# The scorecard above is descriptive. This is the *predictive* version, done
# honestly with the ledger's own forward outcomes rather than an external label
# that has no join key to these clients:
#   features  = client behaviour strictly BEFORE a cutoff date
#   label     = client goes into run-off (no invoicing) in the window AFTER it
# A random hold-out measures generalisation, and the scorecard is scored on the
# same forward label as the baseline to beat.
MODEL_FEATURES = ["gross_invoicing", "credit_notes", "dilution_rate",
                  "top_line_concentration", "days_since_activity",
                  "funding_in_use", "n_invoices", "tenure_days",
                  "avg_invoice_value"]


def train_default_model(df, outcome_days=182, min_invoices=2):
    """Out-of-time run-off (default-proxy) model. Returns (report, scored_pop)."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score

    asof = df["InvoiceDate"].max()
    cutoff = asof - pd.Timedelta(days=outcome_days)
    pre = df[df["InvoiceDate"] < cutoff]
    post = df[df["InvoiceDate"] >= cutoff]

    # features as of the cutoff (client_risk_metrics keys recency off pre's max)
    feat = client_risk_metrics(pre).set_index("Customer ID")
    first = pre.groupby("Customer ID")["InvoiceDate"].min()
    feat["tenure_days"] = (cutoff - first).dt.days
    feat["avg_invoice_value"] = feat["gross_invoicing"] / feat["n_invoices"].clip(lower=1)

    # only clients with a real trading relationship pre-cutoff
    feat = feat[(feat["gross_invoicing"] > 0) & (feat["n_invoices"] >= min_invoices)]

    # forward label: no gross invoicing in the outcome window => run-off
    post_gross = post[~post.IsCreditNote].groupby("Customer ID")["LineValue"].sum()
    feat["post_invoicing"] = feat.index.to_series().map(post_gross).fillna(0.0)
    feat["ran_off"] = (feat["post_invoicing"] <= 0).astype(int)

    X = feat[MODEL_FEATURES].fillna(0.0)
    y = feat["ran_off"]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25,
                                          random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=300, random_state=42,
                                   class_weight="balanced_subsample").fit(Xtr, ytr)
    model_auc = roc_auc_score(yte, model.predict_proba(Xte)[:, 1])

    # scorecard baseline on the SAME forward label / hold-out
    base = score_clients(feat.reset_index()).set_index("Customer ID")["risk_score"]
    base_auc = roc_auc_score(yte, base.loc[yte.index])

    feat["runoff_prob"] = model.predict_proba(X)[:, 1]
    feat["in_test"] = feat.index.isin(Xte.index)
    importances = sorted(zip(MODEL_FEATURES, model.feature_importances_),
                         key=lambda t: -t[1])
    report = {
        "cutoff": cutoff, "outcome_days": outcome_days,
        "n_clients": int(len(feat)), "n_runoff": int(y.sum()),
        "runoff_rate": float(y.mean()), "test_n": int(len(yte)),
        "baseline_auc": float(base_auc), "model_auc": float(model_auc),
        "importances": [(f, float(i)) for f, i in importances],
    }
    return report, feat.reset_index()


# ---------------------------------------------------------------------------
# 5. LEDGER TREND  (weekly flow: invoicing advanced & dilution over time)
# ---------------------------------------------------------------------------
def ledger_trend(df, freq="W"):
    """Per-period origination flow: gross invoiced, credit notes, net advanced,
    and the dilution rate for the period. This is a flow view (new invoicing
    each week), not outstanding balance."""
    d = df.copy()
    d["period"] = d["InvoiceDate"].dt.to_period(freq).dt.start_time
    gross = d[~d.IsCreditNote].groupby("period")["LineValue"].sum().rename("gross")
    credits = d[d.IsCreditNote].groupby("period")["LineValue"].sum().abs().rename("credits")
    t = pd.concat([gross, credits], axis=1).fillna(0.0).sort_index()
    t["net"] = t["gross"] - t["credits"]
    t["advanced"] = (t["net"] * 0.80).clip(lower=0)
    t["dilution_rate"] = np.where(t["gross"] > 0, t["credits"] / t["gross"], 0.0)
    return t.reset_index()


# ---------------------------------------------------------------------------
# 6. DASHBOARD  (self-contained HTML built from the run's own outputs)
# ---------------------------------------------------------------------------
def build_dashboard(scored, fraud, trend, n_rows, asof, dataset_label,
                    model_report=None):
    """Inject this run's data into dashboard_template.html and write
    outputs/dashboard.html. Open it in any browser - no server needed."""
    tmpl_path = Path(__file__).resolve().parent / "dashboard_template.html"
    if not tmpl_path.exists():
        print("[warn] dashboard_template.html not found - skipping dashboard build.")
        return
    tmpl = tmpl_path.read_text(encoding="utf-8")

    def num(v, dp):
        return round(float(v), dp) if pd.notna(v) else 0.0

    clients = [[
        int(r["Customer ID"]), num(r["gross_invoicing"], 2), num(r["credit_notes"], 2),
        int(r["n_invoices"]), num(r["top_line_concentration"], 6), num(r["net_ledger"], 2),
        num(r["funding_in_use"], 2), num(r["dilution_rate"], 6),
        int(r["days_since_activity"]), num(r["risk_score"], 4), num(r["priority"], 2),
    ] for _, r in scored.iterrows()]

    fraud_recs = [{
        "inv": str(r["Invoice"]), "client": int(r["client"]), "value": num(r["value"], 2),
        "lines": int(r["n_lines"]), "round": bool(r["flag_round_number"]),
        "dup": bool(r["flag_duplicate"]), "date": str(r["date"]),
    } for _, r in fraud.iterrows()]

    trend_recs = [[
        r["period"].strftime("%Y-%m-%d"), num(r["gross"], 2), num(r["credits"], 2),
        num(r["advanced"], 2), num(r["dilution_rate"], 6),
    ] for _, r in trend.iterrows()]

    model = None
    if model_report:
        model = {
            "cutoff": model_report["cutoff"].strftime("%Y-%m-%d"),
            "outcome_days": model_report["outcome_days"],
            "n_clients": model_report["n_clients"],
            "runoff_rate": round(model_report["runoff_rate"], 4),
            "baseline_auc": round(model_report["baseline_auc"], 3),
            "model_auc": round(model_report["model_auc"], 3),
            "importances": [[f, round(i, 4)] for f, i in model_report["importances"]],
        }

    payload = {
        "clients": clients, "fraud": fraud_recs, "trend": trend_recs,
        "tx_rows": int(n_rows), "asof": asof.strftime("%Y-%m-%d"),
        "dataset": dataset_label, "model": model,
    }
    html = tmpl.replace("__DATA__", json.dumps(payload))
    (OUT / "dashboard.html").write_text(html, encoding="utf-8")
    print(f"Dashboard written -> {OUT / 'dashboard.html'}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    real_file = DATA / "online_retail_II.xlsx"
    dataset_label = "Full dataset · UCI Online Retail II" if real_file.exists() else "Sample run · UCI Online Retail II"

    df = load_online_retail()
    print(f"Loaded {len(df):,} transaction rows, {df['Customer ID'].nunique()} clients\n")

    metrics = client_risk_metrics(df)
    metrics.to_csv(OUT / "client_risk_metrics.csv", index=False)
    print("=== Client risk metrics (top exposure) ===")
    print(metrics.sort_values("funding_in_use", ascending=False)
          .head(10).to_string(index=False), "\n")

    fraud = fraud_flags(df)
    fraud.to_csv(OUT / "fraud_flags.csv", index=False)
    print(f"=== Fraud/anomaly flags: {len(fraud)} invoices flagged ===")
    print(fraud.head(5).to_string(index=False), "\n")

    scored = score_clients(metrics)
    scored.to_csv(OUT / "client_watchlist.csv", index=False)
    print("=== Early-warning risk scorecard ===")
    print("Weights: " + ", ".join(f"{k} {v:.0%}" for k, v in SCORE_WEIGHTS.items()))
    print("\nTop 10 by priority (risk-weighted exposure):")
    print(scored.sort_values("priority", ascending=False)
          [["Customer ID", "funding_in_use", "dilution_rate",
            "top_line_concentration", "days_since_activity",
            "risk_score", "priority"]]
          .head(10).to_string(index=False))

    trend = ledger_trend(df)
    trend.to_csv(OUT / "ledger_trend.csv", index=False)
    print(f"\n=== Ledger trend: {len(trend)} weekly periods ===")

    try:
        report, model_pop = train_default_model(df)
        model_pop.to_csv(OUT / "default_model.csv", index=False)
        print("\n=== Predictive early-warning model (out-of-time run-off) ===")
        print(f"Cutoff {report['cutoff'].date()} | {report['n_clients']} clients | "
              f"run-off rate {report['runoff_rate']:.1%} ({report['n_runoff']} of {report['n_clients']})")
        print(f"Scorecard baseline AUC: {report['baseline_auc']:.3f}   "
              f"Trained model AUC: {report['model_auc']:.3f}   "
              f"(hold-out n={report['test_n']})")
        print("Top features: " + ", ".join(f"{f} {i:.0%}" for f, i in report["importances"][:4]))
    except Exception as e:
        report = None
        print(f"\n[warn] predictive model skipped: {e}")

    build_dashboard(scored, fraud, trend, len(df), df["InvoiceDate"].max(),
                    dataset_label, model_report=report)
