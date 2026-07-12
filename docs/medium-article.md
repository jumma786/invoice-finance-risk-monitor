# I Built an Invoice-Finance Risk Monitor From a Public Retail Dataset — Here's What I Learned About Exposure, Dilution, and Honest Prediction

*How I turned 824,364 rows of retail transactions into a lender's early-warning system — and why the hardest part wasn't the model, it was refusing to cheat.*

---

Most analyst portfolios I see fall into one of two traps. Either they're a Titanic notebook dressed up with nicer charts, or they're a "real" business project built on data that was quietly engineered to make the model look brilliant. I wanted to build something that would survive a conversation with an actual credit risk manager — where the numbers are real, the framing maps onto a real lending product, and the model's performance is believable *because* it isn't perfect.

So I built an **SME Invoice Finance Risk & Exposure Monitor**: a pipeline that turns a raw transaction ledger into the exact set of metrics, fraud flags, and early-warning scores that a data analyst at an invoice finance lender produces every week. It runs end-to-end on real data, writes a set of CSVs an operations team could actually work from, and generates a self-contained HTML dashboard you can open in any browser with no server.

This is the story of how I built it, the decisions that mattered, and the one principle I refused to compromise on.

---

## First, what is invoice finance?

If you haven't worked in this corner of lending, here's the whole business in two sentences. An invoice finance lender advances cash to a business against its unpaid invoices — typically around 80% of the invoice value up front, with the remainder (less fees) paid when the invoice settles. The lender's entire risk lives in one question: **"What is our exposure, and where is risk quietly building before it becomes a loss?"**

That question breaks down into four jobs an analyst does constantly:

| Function | What you're actually measuring |
|---|---|
| **Exposure monitoring** | How much cash is advanced and outstanding right now |
| **Credit risk** | Dilution, debtor concentration, arrears, dormancy |
| **Early warning** | Which clients are heading for trouble *next* |
| **Fraud detection** | Invoices that don't look like honest trading |

I wanted every one of these four functions represented by a real output, computed from real data. That constraint drove everything else.

---

## The data problem — and the framing that solved it

There is no public, labelled invoice-finance ledger. Lenders don't publish their books. So the honest options are: fabricate data (worthless), or find real transactional data whose *shape* matches an invoice ledger and adapt the framing without touching the numbers.

I used the **UCI Online Retail II** dataset — 824,364 real transactions from a UK online retailer, covering 5,942 customers over two years. The mapping is clean and defensible:

- **Customers → clients.** Each customer is a business the lender funds.
- **Invoices → invoices.** They already are invoices.
- **Credit notes → dilution.** In the raw data, returns and cancellations appear as invoices with a `C` prefix. In invoice finance, exactly this — credit notes that reduce the value of a ledger the lender has already advanced against — is called *dilution*, and it's one of the single most important risk signals there is.

That last point is the whole trick. Dilution isn't a metaphor I bolted on; the credit-note structure in retail data *is* structurally the same event as dilution in a funded ledger. The framing is adapted. The numbers are real. I put that disclaimer directly in the README and the code, because the moment you hide an assumption like that, you've built the second kind of dishonest portfolio.

---

## Building the metrics: the SQL an analyst lives in, written in pandas

The core of the pipeline is one function that collapses 824k transaction rows into one row per client, carrying the metrics a lender acts on. A few of them are worth explaining because they *are* the job.

**Funding-in-use** — the actual cash at risk:

```python
ADVANCE_RATE = 0.80
m["net_ledger"] = m["gross_invoicing"] - m["credit_notes"]
m["funding_in_use"] = (m["net_ledger"] * ADVANCE_RATE).clip(lower=0)
```

Net ledger is gross invoicing minus credit notes; funding-in-use is 80% of that, floored at zero. Across the whole portfolio this comes to **£13.4M of funding-in-use** — the headline exposure number.

**Dilution rate** — credit notes as a share of gross invoicing:

```python
m["dilution_rate"] = np.where(
    m["gross_invoicing"] > 0,
    m["credit_notes"] / m["gross_invoicing"],
    0,
)
```

Portfolio-wide dilution lands at **6.2%**. On a single client, a creeping dilution rate is often the first visible sign that something is wrong — goods coming back, disputes rising, or worse.

**Debtor concentration** — how lopsided a client's ledger is. I proxy it with the largest single stock line as a share of the client's total. The naive way to compute this is a per-client Python `apply`, which is fine on a sample and unusable on a million rows. Vectorising it was the difference between a toy and something that scales:

```python
line = (df[~df.IsCreditNote]
        .groupby(["Customer ID", "StockCode"])["LineValue"].sum())
grp = line.groupby(level=0)
line_share = (grp.max() / grp.sum().where(grp.sum() > 0))
```

**Dormancy** — days since the client's last invoice. A funded client who stops invoicing is a client whose ledger is running off, and a running-off ledger is where a lender's advance can get stranded.

Put together, the top of the exposure table looks like this — a handful of clients carrying hundreds of thousands of pounds each:

| Client | Funding-in-use | Dilution | Days since activity |
|---|---|---|---|
| 18102 | £478,572 | 1.7% | 0 |
| 14646 | £418,674 | 1.0% | 1 |
| 14156 | £237,252 | 5.5% | 9 |

This is already a useful artefact. But a table of metrics is descriptive. The interesting question — and the one that separates an analyst from a report generator — is *which clients are about to go wrong.*

---

## Fraud flags: cheap rules that catch real signatures

Before the modelling, a rules-based pass over every invoice. Two classic signatures:

```python
inv["flag_round_number"] = (inv["value"] % 100 == 0) & (inv["value"] > 0)
inv["flag_duplicate"] = inv.duplicated(
    subset=["client", "value", "date"], keep=False
) & (inv["value"] > 0)
```

Round-number invoices (a £10,000.00 invoice is far rarer in honest trading than fraud) and duplicate invoices (same client, same value, same day — the fingerprint of an invoice submitted twice for funding). On the full ledger this flags **253 invoices**. Rules like these will never be your whole fraud strategy, but they're transparent, they're instant, and every flag is explainable to a human who has to action it. Sometimes the boring tool is the right tool.

---

## Early warning, part 1: a scorecard you can defend in an audit

My first early-warning output is deliberately *not* a machine-learning model. It's a transparent, weighted scorecard — the kind of thing a credit committee can read, challenge, and sign off:

```python
SCORE_WEIGHTS = {"dilution": 0.50, "concentration": 0.25, "dormancy": 0.25}
DILUTION_CAP = 0.30   # dilution >= 30% saturates the signal
DORMANCY_CAP = 90.0   # 90+ days dormant saturates the signal
```

Each signal is normalised to 0–1, capped so one extreme value can't dominate, and blended by weights that live in one place at the top of the file — the single, auditable spot where the lender's risk appetite is expressed. There's no trained label here, so there's nothing to leak. That's a feature, not a limitation: this is exactly how real credit scorecards work.

But a raw risk score has a failure mode: it happily gives a maxed-out score to a dormant client with £0 advanced. That's a note, not an emergency. So the number the watchlist actually ranks on is **priority — risk score × funding-in-use**:

```python
d["priority"] = (d["risk_score"] * d["funding_in_use"]).round(2)
```

This is an expected-loss-style ranking. It floats the clients where *real money* meets *real risk* to the top, and pushes the dramatic-but-harmless cases down. Across the portfolio, **£1.58M of exposure sits in critical-risk clients** — and now I know exactly which ones.

---

## Early warning, part 2: predicting run-off without cheating

Here's where most portfolio projects quietly fall apart, and where I spent most of my thinking time.

The tempting move is to grab a labelled dataset — say, Lending Club's loan defaults — and train a classifier to predict "default." It would produce a big impressive AUC. It would also be **completely meaningless**, because Lending Club borrowers share no join key with these retail clients. Attaching those labels wouldn't be modelling; it would be *inventing* labels and hoping nobody checks. I refused to do it.

Instead, I let the ledger label itself, using time. This is an **out-of-time** design:

- **Pick a cutoff date.** I used 2011-06-10, six months before the data ends.
- **Features come strictly from behaviour *before* the cutoff.**
- **The label comes from behaviour *after* it:** did the client stop invoicing entirely in the following 182 days? If so, their ledger ran off.

```python
asof   = df["InvoiceDate"].max()
cutoff = asof - pd.Timedelta(days=outcome_days)   # 182 days
pre    = df[df["InvoiceDate"] <  cutoff]           # features
post   = df[df["InvoiceDate"] >= cutoff]           # outcome window

train = _model_features(pre, cutoff, min_invoices)
post_gross = post[~post.IsCreditNote].groupby("Customer ID")["LineValue"].sum()
train["ran_off"] = (train.index.to_series()
                    .map(post_gross).fillna(0.0) <= 0).astype(int)
```

Because the label is *strictly in the future relative to the features*, there is no way for outcome information to leak backwards into a feature. This is the single most important design decision in the entire project. Leakage is the reason so many models look brilliant in a notebook and fall over in production — the model was secretly reading the answer. An out-of-time split makes that structurally impossible.

I train a `RandomForestClassifier` (300 trees, `class_weight="balanced_subsample"` because run-off is the minority outcome), evaluate it on a held-out 25% split, and — crucially — score the transparent scorecard on the *exact same forward label* as the baseline to beat.

The result:

```
Cutoff 2011-06-10 | 3,657 clients | run-off rate 38.5%
Scorecard baseline AUC: 0.730   Trained model AUC: 0.782   (hold-out n=915)
Top features: days_since_activity 22%, funding_in_use 14%,
              gross_invoicing 13%, top_line_concentration 11%
```

**0.782 AUC.** Not 0.98. And that's the point. A believable, leakage-free model that adds a real, measurable **+0.05 AUC over a strong transparent baseline** is worth a hundred suspiciously-perfect notebook models. The feature importances also pass the sniff test: *days since activity* dominates, which is exactly what a human analyst would tell you predicts a client going quiet.

---

## From prediction to action: deploying the model forward

A hold-out AUC is a validation number, not a business output. The last step is the one that makes this a *monitor* rather than an *experiment*: refit the model on **all** available history, then point it at every *current* client as of the latest date to predict who runs off in their next ~6 months.

```python
deploy = rf().fit(X, y)                       # refit on all history
current = _model_features(df, asof, min_invoices)
current["runoff_prob"]  = deploy.predict_proba(current[MODEL_FEATURES])[:, 1]
current["exp_exposure"] = (current["runoff_prob"]
                           * current["funding_in_use"]).round(2)
```

Deployed forward, the model scores **4,477 current clients**, flags **1,629** at a run-off probability of 0.5 or higher, and — the number a lender actually cares about — surfaces **£1.9M of expected exposure at risk** (probability × funding) over the next six months.

And here's the payoff that justified the whole out-of-time detour: this forward ranking **genuinely differs** from the scorecard. The scorecard tells you who looks risky *today*. The model tells you who will *run off next* — and those aren't the same clients. A client can look pristine on today's dilution and concentration while the model, reading subtler behavioural patterns, still flags them as likely to go quiet. That difference is the entire value of a predictive early-warning system over a descriptive one.

---

## Wrapping it in something a human can use

Every run writes a set of CSVs — per-client metrics, fraud flags, the scorecard watchlist, the forward watchlist, the weekly ledger trend, and the model's training population with its predictions. But CSVs don't get looked at in a Monday risk meeting. A dashboard does.

So the pipeline injects each run's own data into an HTML template and writes a **single self-contained `dashboard.html`** — KPIs, ledger trend, an exposure vs dilution×concentration view, the priority-ranked watchlist with each client's run-off probability, the model validation panel, and the fraud flags. No server, no build step, no dependencies. Open it in a browser and it's all there.

![Portfolio Risk Monitor dashboard](dashboard.png)

*(Upload `docs/dashboard.png` here when publishing.)*

The whole thing runs with:

```bash
pip install pandas numpy scikit-learn openpyxl
python src/pipeline.py
```

Drop the full `online_retail_II.xlsx` into `data/` and it runs on all ~1M rows; leave it out and a bundled sample runs automatically so anyone can reproduce it in seconds.

---

## What I'd actually build next

I'm treating this as a living project, and the honest limitations point straight at the roadmap:

- **Calibrate the probabilities.** A random forest's `predict_proba` is a ranking, not a true probability. Platt scaling or isotonic regression would make `P(run-off)` mean what it says — important the moment anyone multiplies it by pounds.
- **Temporal cross-validation.** One cutoff gives one estimate. Rolling the cutoff across several dates would test whether 0.782 holds up across time or got lucky on one window.
- **True ageing.** The trend view is currently a *flow* (new invoicing per week). With settlement dates, I could build genuine outstanding-balance ageing — the real-world version of what a lender watches.
- **Enrich with real company data** via the Companies House API, turning anonymous client IDs into businesses with sectors and filing histories.

---

## The one thing I want you to take from this

The model was never the hard part. `RandomForestClassifier` is one import. The hard part — the part that took real discipline — was **refusing to let the project lie to me.**

It would have been easy to bolt on an external default label and post a 0.97 AUC. It would have been easy to compute dilution in a way that leaked. It would have been easy to rank the watchlist on a pure risk score and quietly ignore that half the top of the list had no money at risk. Every one of those shortcuts would have made the project *look* better and made it *worth* less.

Instead I built something where every number is real, every assumption is stated out loud, and the model's 0.782 AUC is exactly as good as it honestly is. If you're building an analytics portfolio, that's the bar I'd aim for. A believable project you can defend beats an impressive one you can't — every single time.

---

*The full code and dashboard are on GitHub. Thanks for reading — I'm always happy to talk about invoice finance, leakage-free modelling, or turning messy transaction data into something a business can act on.*
