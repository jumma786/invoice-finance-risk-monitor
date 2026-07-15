"""Unit tests for the invoice-finance risk pipeline.

Run with:  pytest -q
Uses small, hand-built synthetic ledgers so the expected numbers are checkable
by hand - no dependency on the large real dataset.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pipeline  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def make_df(records):
    """Build a processed ledger frame with the derived columns the pipeline
    functions expect (mirrors load_online_retail's transforms)."""
    df = pd.DataFrame(records)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["LineValue"] = df["Quantity"] * df["Price"]
    df["IsCreditNote"] = df["Invoice"].astype(str).str.startswith("C")
    return df


def row(invoice, client, stock, qty, price, date):
    return {"Invoice": invoice, "Customer ID": client, "StockCode": stock,
            "Quantity": qty, "Price": price, "InvoiceDate": date}


# ---------------------------------------------------------------------------
# client_risk_metrics
# ---------------------------------------------------------------------------
def test_client_risk_metrics_core_numbers():
    # Client 100: invoice A has two stock lines (60 + 40 = 100 gross),
    # credit note C1 returns 20. Net 80, funding 80*0.8=64, dilution 20/100=0.2.
    df = make_df([
        row("A", 100, "X", 6, 10, "2024-01-10"),
        row("A", 100, "Y", 4, 10, "2024-01-10"),
        row("C1", 100, "X", -2, 10, "2024-01-15"),
    ])
    m = pipeline.client_risk_metrics(df).set_index("Customer ID")

    assert m.loc[100, "gross_invoicing"] == pytest.approx(100.0)
    assert m.loc[100, "credit_notes"] == pytest.approx(20.0)
    assert m.loc[100, "net_ledger"] == pytest.approx(80.0)
    assert m.loc[100, "funding_in_use"] == pytest.approx(64.0)
    assert m.loc[100, "dilution_rate"] == pytest.approx(0.2)
    assert m.loc[100, "n_invoices"] == 2               # A and C1
    # concentration = biggest stock line (60) / gross (100)
    assert m.loc[100, "top_line_concentration"] == pytest.approx(0.6)


def test_funding_in_use_never_negative():
    # A client whose credit notes exceed gross must not show negative funding.
    df = make_df([
        row("A", 200, "X", 1, 10, "2024-01-10"),      # +10 gross
        row("C1", 200, "X", -5, 10, "2024-01-12"),     # -50 credit note
    ])
    m = pipeline.client_risk_metrics(df).set_index("Customer ID")
    assert m.loc[200, "net_ledger"] < 0
    assert m.loc[200, "funding_in_use"] == 0.0


def test_days_since_activity_measured_from_latest_date():
    df = make_df([
        row("A", 1, "X", 1, 10, "2024-01-01"),
        row("B", 2, "X", 1, 10, "2024-01-31"),         # latest -> asof
    ])
    m = pipeline.client_risk_metrics(df).set_index("Customer ID")
    assert m.loc[2, "days_since_activity"] == 0
    assert m.loc[1, "days_since_activity"] == 30


# ---------------------------------------------------------------------------
# fraud_flags
# ---------------------------------------------------------------------------
def test_fraud_round_number_flag():
    df = make_df([
        row("R1", 1, "X", 10, 10, "2024-02-01"),       # value 100 -> round
        row("N1", 2, "X", 3, 33, "2024-02-01"),        # value 99  -> not round
    ])
    flags = pipeline.fraud_flags(df).set_index("Invoice")
    assert bool(flags.loc["R1", "flag_round_number"]) is True
    assert "N1" not in flags.index                     # nothing flagged it


def test_fraud_duplicate_flag():
    # Same client, same value, same day, two different invoices -> both flagged.
    df = make_df([
        row("D1", 5, "X", 1, 6958.17, "2024-03-19"),
        row("D2", 5, "X", 1, 6958.17, "2024-03-19"),
    ])
    flags = pipeline.fraud_flags(df).set_index("Invoice")
    assert bool(flags.loc["D1", "flag_duplicate"]) is True
    assert bool(flags.loc["D2", "flag_duplicate"]) is True


# ---------------------------------------------------------------------------
# score_clients
# ---------------------------------------------------------------------------
def test_scorecard_components_sum_and_priority():
    metrics = pd.DataFrame([{
        "Customer ID": 1, "dilution_rate": 0.15, "top_line_concentration": 0.4,
        "days_since_activity": 45, "funding_in_use": 1000.0,
    }])
    scored = pipeline.score_clients(metrics).iloc[0]

    # dilution: (0.15/0.30)=0.5 * 0.50 weight = 0.25
    assert scored["dilution_component"] == pytest.approx(0.25)
    # concentration: 0.4 * 0.25 weight = 0.10
    assert scored["concentration_component"] == pytest.approx(0.10)
    # dormancy: (45/90)=0.5 * 0.25 weight = 0.125
    assert scored["dormancy_component"] == pytest.approx(0.125)
    # components sum to risk_score
    assert scored["risk_score"] == pytest.approx(0.475)
    # priority = risk_score * funding_in_use
    assert scored["priority"] == pytest.approx(0.475 * 1000.0, rel=1e-4)


def test_scorecard_caps_saturate():
    # dilution and dormancy well past their caps must clamp to full weight.
    metrics = pd.DataFrame([{
        "Customer ID": 1, "dilution_rate": 0.90, "top_line_concentration": 2.0,
        "days_since_activity": 500, "funding_in_use": 0.0,
    }])
    scored = pipeline.score_clients(metrics).iloc[0]
    assert scored["dilution_component"] == pytest.approx(pipeline.SCORE_WEIGHTS["dilution"])
    assert scored["concentration_component"] == pytest.approx(pipeline.SCORE_WEIGHTS["concentration"])
    assert scored["dormancy_component"] == pytest.approx(pipeline.SCORE_WEIGHTS["dormancy"])
    assert scored["risk_score"] == pytest.approx(1.0)
    assert scored["priority"] == 0.0                   # no money advanced


# ---------------------------------------------------------------------------
# ledger_trend
# ---------------------------------------------------------------------------
def test_ledger_trend_weekly_flow():
    df = make_df([
        row("A", 1, "X", 10, 10, "2024-01-01"),        # +100 gross, week of Jan 1
        row("C1", 1, "X", -2, 10, "2024-01-02"),        # -20 credit, same week
    ])
    t = pipeline.ledger_trend(df)
    assert len(t) == 1
    r = t.iloc[0]
    assert r["gross"] == pytest.approx(100.0)
    assert r["credits"] == pytest.approx(20.0)
    assert r["net"] == pytest.approx(80.0)
    assert r["advanced"] == pytest.approx(64.0)
    assert r["dilution_rate"] == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# train_default_model  (+ calibration)
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synthetic_ledger():
    """80 clients: 40 that keep trading after the cutoff, 40 that run off.
    Each has >=2 pre-cutoff invoices so it qualifies for the model."""
    rng = np.random.default_rng(0)
    recs = []
    for cid in range(1, 81):
        runs_off = cid <= 40
        # two pre-cutoff invoices (well before the Jan-31 cutoff)
        for k, d in enumerate(["2024-01-05", "2024-01-20"]):
            qty = int(rng.integers(1, 20))
            price = float(rng.integers(5, 50))
            recs.append(row(f"P{cid}_{k}", cid, f"S{cid % 5}", qty, price, d))
        # active clients also invoice after the cutoff -> ran_off = 0
        if not runs_off:
            recs.append(row(f"Q{cid}", cid, f"S{cid % 5}", 5, 20, "2024-03-01"))
    return make_df(recs)


def test_train_default_model_report_shape(synthetic_ledger):
    report, hist, current = pipeline.train_default_model(
        synthetic_ledger, outcome_days=30, min_invoices=2)

    for key in ("baseline_auc", "model_auc", "calibration",
                "brier_raw", "brier_cal", "importances", "fwd_n"):
        assert key in report

    assert 0.0 <= report["model_auc"] <= 1.0
    assert 0.0 <= report["baseline_auc"] <= 1.0
    # ~half the population ran off by construction
    assert report["runoff_rate"] == pytest.approx(0.5, abs=0.15)
    # importances cover every model feature and sum to ~1
    assert {f for f, _ in report["importances"]} == set(pipeline.MODEL_FEATURES)
    assert sum(i for _, i in report["importances"]) == pytest.approx(1.0, abs=1e-6)


def test_predicted_probabilities_are_valid(synthetic_ledger):
    _, hist, current = pipeline.train_default_model(
        synthetic_ledger, outcome_days=30, min_invoices=2)

    for frame in (hist, current):
        assert frame["runoff_prob"].between(0.0, 1.0).all()
    # expected exposure = prob * funding, so it can't exceed funding
    assert (current["exp_exposure"] <= current["funding_in_use"] + 1e-6).all()


def test_calibration_reports_finite_brier(synthetic_ledger):
    report, _, _ = pipeline.train_default_model(
        synthetic_ledger, outcome_days=30, min_invoices=2, calibration="sigmoid")
    assert report["calibration"] == "sigmoid"
    assert np.isfinite(report["brier_raw"])
    assert np.isfinite(report["brier_cal"])
    # Brier score is a probability-squared error, bounded to [0, 1]
    assert 0.0 <= report["brier_cal"] <= 1.0
