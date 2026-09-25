"""
Burgernomics: does the Big Mac Index predict exchange rates, and who does the adjusting,
the currency or the burger?

1. Mean reversion: does today's burger misvaluation predict the next h years?
2. Decomposition: misvaluation closes through (a) the exchange rate or (b) relative burger inflation.
3. Out-of-sample forecasting horse race vs the random walk (Meese-Rogoff), incl. gradient boosting.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from utils import DATA, HIGHLIGHT, MUTED, PALETTE, get, md_table, savefig, setup, write_results

URL = "https://raw.githubusercontent.com/TheEconomist/big-mac-data/master/output-data/big-mac-full-index.csv"
SRC = "The Economist, Big Mac Index (github.com/TheEconomist/big-mac-data)"
CRISIS = {"ARG", "VEN", "LBN"}                                   # hyperinflation / collapse: excluded
PEGS = {"HKG", "SAU", "ARE", "BHR", "KWT", "OMN", "QAT", "JOR", "DNK"}  # hard or near-hard USD/EUR pegs
HORIZONS = [1, 2, 3, 5]
MAIN_H = 2


def load() -> pd.DataFrame:
    d = pd.read_csv(io.StringIO(get(URL).text), parse_dates=["date"])
    d.to_csv(DATA / "big-mac-full-index.csv", index=False)
    us = d[d["iso_a3"] == "USA"][["date", "local_price"]].rename(columns={"local_price": "us_price"})
    d = d.merge(us, on="date")
    d = d[(d["iso_a3"] != "USA") & ~d["iso_a3"].isin(CRISIS)].copy()
    d["v"] = np.log(d["dollar_price"] / d["us_price"])            # log misvaluation vs USD (<0 = undervalued)
    d["v_adj"] = np.log1p(d["USD_adjusted"])                      # GDP-adjusted (Balassa-Samuelson) version
    d["ln_gdp"] = np.log(d["GDP_bigmac"].where(d["GDP_bigmac"] > 100))
    d["peg"] = d["iso_a3"].isin(PEGS)
    return d.sort_values(["iso_a3", "date"]).reset_index(drop=True)


def forward(d: pd.DataFrame, h: int) -> pd.DataFrame:
    """Attach values h years ahead (nearest release within 75 days)."""
    f = d[["iso_a3", "date", "dollar_ex", "local_price", "us_price", "v"]].copy()
    f["date"] = f["date"] - pd.DateOffset(years=h)
    out = pd.merge_asof(d.sort_values("date"), f.sort_values("date"), on="date", by="iso_a3",
                        suffixes=("", "_f"), direction="nearest", tolerance=pd.Timedelta(days=75))
    out = out.dropna(subset=["dollar_ex_f"])
    out["fx"] = np.log(out["dollar_ex_f"] / out["dollar_ex"])       # + = local currency depreciates vs USD
    out["rel_p"] = np.log(out["local_price_f"] / out["local_price"]) - np.log(out["us_price_f"] / out["us_price"])
    out["dv"] = out["v_f"] - out["v"]                               # = rel_p - fx
    # drop currency redenominations (e.g. Turkey 2005: 1,000,000 old lira = 1 new lira)
    return out[(out["fx"].abs() < 2.5) & (out["rel_p"].abs() < 2.5)]


def clustered(formula: str, d: pd.DataFrame):
    return smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(d["iso_a3"])[0]})


def lagged_features(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    back = d[["iso_a3", "date", "dollar_ex", "local_price", "us_price"]].copy()
    back["date"] = back["date"] + pd.DateOffset(years=1)
    d = pd.merge_asof(d.sort_values("date"), back.sort_values("date"), on="date", by="iso_a3",
                      suffixes=("", "_b"), direction="nearest", tolerance=pd.Timedelta(days=75))
    d["fx_mom"] = np.log(d["dollar_ex"] / d["dollar_ex_b"])
    d["burger_infl"] = np.log(d["local_price"] / d["local_price_b"]) - np.log(d["us_price"] / d["us_price_b"])
    ok = (d["fx_mom"].abs() < 2.5) & (d["burger_infl"].abs() < 2.5)          # redenominations
    d.loc[~ok, ["fx_mom", "burger_infl"]] = np.nan
    return d


def main() -> None:
    setup()
    d = load()
    md = []

    # ---- 1 + 2. Mean reversion and who adjusts, by horizon ----------------------------------------
    rows = []
    for h in HORIZONS:
        f = forward(d, h)
        fl = f[~f["peg"]]
        for name, data in [("Floaters", fl), ("Pegs", f[f["peg"]])]:
            if data["iso_a3"].nunique() < 3:
                continue
            m_dv = clustered("dv ~ v + C(date)", data)
            m_fx = clustered("fx ~ v + C(date)", data)
            m_p = clustered("rel_p ~ v + C(date)", data)
            rows.append({"Group": name, "h (yrs)": h, "Closure of gap (−β on Δv)": -m_dv.params["v"],
                         "via exchange rate": m_fx.params["v"], "via burger prices": -m_p.params["v"],
                         "p (Δv)": m_dv.pvalues["v"], "Obs": int(m_dv.nobs)})
    rev = pd.DataFrame(rows)
    fl_rev = rev[rev["Group"] == "Floaters"].set_index("h (yrs)")
    lam = fl_rev.loc[1, "Closure of gap (−β on Δv)"] if 1 in fl_rev.index else np.nan
    half_life = np.log(0.5) / np.log(1 - lam) if 0 < lam < 1 else np.nan

    # Fig 1: scatter v vs subsequent 2y FX change (floaters)
    f2 = forward(d, MAIN_H)
    fl2 = f2[~f2["peg"]]
    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    ax.scatter(100 * fl2["v"], 100 * fl2["fx"], s=12, color=PALETTE[0], alpha=0.45, lw=0)
    m = smf.ols("fx ~ v", data=fl2).fit()
    xs = np.linspace(fl2["v"].min(), fl2["v"].max(), 50)
    ax.plot(100 * xs, 100 * (m.params["Intercept"] + m.params["v"] * xs), color="#222", lw=2)
    ind = fl2[fl2["iso_a3"] == "IND"]
    ax.scatter(100 * ind["v"], 100 * ind["fx"], s=40, color=HIGHLIGHT, label="India", zorder=4)
    ax.axhline(0, color="#999", lw=.8); ax.axvline(0, color="#999", lw=.8)
    ax.set_xlabel("Burger misvaluation vs USD today (log %, <0 = undervalued)")
    ax.set_ylabel(f"Change in exchange rate over next {MAIN_H} years (log %, + = local currency weakens)")
    verdict = "Undervalued currencies tend to keep weakening" if m.params["v"] < 0 else "Undervalued currencies tend to strengthen"
    ax.set_title(f"{verdict} (slope {m.params['v']:.2f})")
    ax.legend()
    fig1 = savefig(fig, "01_misvaluation_vs_fx", SRC)

    # Fig 2: who closes the gap, by horizon
    fig, ax = plt.subplots(figsize=(9.5, 5))
    x = np.arange(len(fl_rev))
    ax.bar(x - 0.18, 100 * fl_rev["via exchange rate"], width=0.36, color=PALETTE[0], label="Exchange rate")
    ax.bar(x + 0.18, 100 * fl_rev["via burger prices"], width=0.36, color=PALETTE[1], label="Relative burger inflation")
    ax.plot(x, 100 * fl_rev["Closure of gap (−β on Δv)"], color="#111", marker="o", lw=2, label="Total gap closed")
    ax.axhline(0, color="#333", lw=.8)
    ax.set_xticks(x, [f"{h} yr" for h in fl_rev.index])
    ax.set_ylabel("% of today's misvaluation closed")
    ax.set_title("Who does the adjusting: the currency or the burger? (floating currencies)")
    ax.legend()
    fig2 = savefig(fig, "02_who_adjusts", SRC)

    # ---- 3. Out-of-sample horse race --------------------------------------------------------------
    feats = ["v", "v_adj", "ln_gdp", "fx_mom", "burger_infl"]
    data = lagged_features(d)
    data = forward(data, MAIN_H)
    data = data[~data["peg"]].dropna(subset=feats + ["fx"]).sort_values("date")
    data["target_date"] = data["date"] + pd.DateOffset(years=MAIN_H)
    origins = sorted(data["date"].unique())
    preds = []
    for o in origins:
        train = data[data["target_date"] <= o]                    # only outcomes already observed at o
        test = data[data["date"] == o]
        if len(train) < 150 or test.empty:
            continue
        X, y = train[feats], train["fx"]
        ridge = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))).fit(X, y)
        ppp = smf.ols("fx ~ v", data=train).fit()
        gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03,
                                        subsample=0.8, random_state=0).fit(X, y)
        out = test[["iso_a3", "date", "fx"]].copy()
        out["Random walk"] = 0.0
        out["PPP reversion (OLS)"] = ppp.predict(test)
        out["Ridge (5 features)"] = ridge.predict(test[feats])
        out["Gradient boosting"] = gbm.predict(test[feats])
        preds.append(out)
    P = pd.concat(preds)
    models = ["Random walk", "PPP reversion (OLS)", "Ridge (5 features)", "Gradient boosting"]
    score = []
    rmse_rw = np.sqrt(np.mean(P["fx"] ** 2))
    for mdl in models:
        e = P["fx"] - P[mdl]
        score.append({"Model": mdl, "RMSE (log pts)": np.sqrt(np.mean(e ** 2)),
                      "RMSE vs random walk": np.sqrt(np.mean(e ** 2)) / rmse_rw,
                      "Direction hit rate": np.nan if mdl == "Random walk" else float(np.mean(np.sign(P[mdl]) == np.sign(P["fx"])))})
    score = pd.DataFrame(score)
    final_gbm = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.03, subsample=0.8,
                                          random_state=0).fit(data[feats], data["fx"])
    imp = permutation_importance(final_gbm, data[feats], data["fx"], n_repeats=20, random_state=0)
    imp = pd.Series(imp.importances_mean, index=feats).sort_values()
    labels = {"v": "Raw burger misvaluation", "v_adj": "GDP-adjusted misvaluation", "ln_gdp": "log GDP per capita",
              "fx_mom": "Past-year FX change", "burger_infl": "Past-year relative burger inflation"}

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    cols = [MUTED, PALETTE[0], PALETTE[2], PALETTE[1]]
    ax.barh(score["Model"], score["RMSE vs random walk"], color=cols)
    ax.axvline(1, color="#333", ls="--", lw=1)
    ax.set_xlabel("Out-of-sample RMSE relative to random walk (<1 beats it)")
    ax.set_title(f"{MAIN_H}-year-ahead exchange-rate forecasts")
    ax = axes[1]
    ax.barh([labels[i] for i in imp.index], imp.values, color=PALETTE[1])
    ax.set_xlabel("Permutation importance (gradient boosting)")
    ax.set_title("What the ML model leans on")
    fig3 = savefig(fig, "03_horse_race", SRC)

    # Fig 4: India
    ind = d[d["iso_a3"] == "IND"].sort_values("date")
    fig, ax1 = plt.subplots(figsize=(10, 4.8))
    ax1.plot(ind["date"], 100 * ind["USD_raw"], color=HIGHLIGHT, lw=2.4, marker="o", ms=3, label="Raw Big Mac valuation")
    ax1.plot(ind["date"], 100 * ind["USD_adjusted"], color=HIGHLIGHT, lw=1.6, ls="--", label="GDP-adjusted valuation")
    ax1.set_ylabel("Rupee valuation vs USD (%)", color=HIGHLIGHT)
    ax2 = ax1.twinx()
    ax2.plot(ind["date"], ind["dollar_ex"], color=PALETTE[0], lw=2, label="₹ per $ (rhs)")
    ax2.set_ylabel("₹ per US$", color=PALETTE[0]); ax2.grid(False)
    ax1.set_title("The rupee in burgers: valuation vs the exchange rate")
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", frameon=True, framealpha=0.9, ncol=3, fontsize=9)
    fig4 = savefig(fig, "04_india", SRC)

    # ---- Write-up ---------------------------------------------------------------------------------
    r2 = fl_rev.loc[MAIN_H]
    best = score.iloc[1:].sort_values("RMSE vs random walk").iloc[0]
    li = ind.iloc[-1]
    md.append("### Headline numbers\n")
    md.append(f"- Over {MAIN_H} years, floating currencies close **{100 * r2['Closure of gap (−β on Δv)']:.0f}%** of their burger misvaluation. "
              f"Exchange-rate moves contribute **{100 * r2['via exchange rate']:+.0f} pp** and relative burger inflation **{100 * r2['via burger prices']:+.0f} pp** "
              "(negative = that channel widens the gap).")
    if np.isfinite(half_life):
        md.append(f"- Implied half-life of a misvaluation: **{half_life:.1f} years**.")
    md.append(f"- Best out-of-sample model: **{best['Model']}**, RMSE **{best['RMSE vs random walk']:.2f}×** the random walk "
              f"(direction right {100 * best['Direction hit rate']:.0f}% of the time). {len(P):,} forecasts from {P['date'].nunique()} origins.")
    md.append(f"- India ({li['date']:%b %Y}): the Big Mac puts the rupee at **{100 * li['USD_raw']:+.0f}%** vs the dollar "
              f"(**{100 * li['USD_adjusted']:+.0f}%** after adjusting for income).\n")
    md.append("### Mean reversion by horizon (time fixed effects, SEs clustered by country)\n")
    md.append(md_table(rev, "{:.2f}"))
    md.append("\n### Forecast horse race\n")
    md.append(md_table(score, "{:.3f}"))
    md.append("\n### Figures\n")
    for f, cap in [(fig1, "Misvaluation vs subsequent FX change"), (fig2, "Who adjusts"),
                   (fig3, "Forecast horse race"), (fig4, "India")]:
        md.append(f"**{cap}**\n\n![{cap}]({f})\n")
    write_results("\n".join(md))
    print("done")


if __name__ == "__main__":
    main()
