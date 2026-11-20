"""Q3. Under what conditions do late or near-deadline shipments occur?

slack_days      = RequiredDate - ShippedDate   (negative = late)
processing_days = ShippedDate  - OrderDate
Late            = slack < 0
Near-deadline   = 0 <= slack <= 2
At risk         = late or near-deadline (slack <= 2)
Unshipped orders (no ShippedDate) are excluded and counted.

With only ~37 late orders, every comparison reports counts, rates with
Wilson 95% intervals, and a permutation chi-square p-value (valid even when
expected cell counts are tiny). Nothing here is a causal estimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter
from scipy import stats

import plot_style as ps
from load_data import OUT_TAB, ensure_output_dirs, load_clean

NEAR_MAX = 2              # near-deadline = 0..NEAR_MAX days of slack
MIN_COUNTRY_ORDERS = 20   # countries need this many shipped orders to be shown
SLOW_DAYS = 14            # "slow processing" threshold for the mechanism check
N_PERM = 5000
SEED = 561
LATE_C, NEAR_C = ps.SERIES[1], ps.SERIES[3]   # orange, yellow


def wilson(k, n, z=1.96):
    """Compute Wilson score confidence bounds for proportions.

    Args:
        k (array-like): Number of successes.
        n (array-like): Number of trials.
        z (float): Normal quantile. Default 1.96 (95%).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Lower and upper bounds.
    """
    k, n = np.asarray(k, float), np.asarray(n, float)
    p = np.divide(k, n, out=np.zeros_like(k), where=n > 0)
    den = 1 + z ** 2 / n
    mid = (p + z ** 2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return mid - half, mid + half


def perm_chi2(groups: pd.Series, outcome: pd.Series, rng) -> tuple[float, float]:
    """Test outcome versus group by a permutation chi-square.

    Args:
        groups (pandas.Series): Group label for each order.
        outcome (pandas.Series): Boolean outcome for each order.
        rng (numpy.random.Generator): Source of shuffles.

    Returns:
        tuple[float, float]: Observed chi-square statistic and the
            permutation p-value from ``N_PERM`` shuffles.
    """
    g = pd.Categorical(groups).codes
    y = outcome.astype(int).values
    k = g.max() + 1

    def chi(yv):
        """Compute the chi-square statistic of a group-outcome table.

        Args:
            yv (numpy.ndarray): 0/1 outcome per order, aligned with the
                enclosing group codes.

        Returns:
            float: Chi-square statistic.
        """
        tab = np.zeros((k, 2))
        np.add.at(tab, (g, yv), 1)
        exp = tab.sum(1, keepdims=True) * tab.sum(0, keepdims=True) / tab.sum()
        return float(((tab - exp) ** 2 / np.where(exp > 0, exp, 1)).sum())

    obs = chi(y)
    null = np.array([chi(rng.permutation(y)) for _ in range(N_PERM)])
    return obs, (1 + (null >= obs).sum()) / (N_PERM + 1)


def factor_table(s: pd.DataFrame, col: str, rng) -> tuple[pd.DataFrame, dict]:
    """Summarise lateness by one factor and test it.

    Args:
        s (pandas.DataFrame): Shipped orders with ``late``, ``near``,
            ``at_risk``, ``slow``, ``slack_days``, ``processing_days``
            and ``OrderID``.
        col (str): Factor column to group by.
        rng (numpy.random.Generator): Source of shuffles.

    Returns:
        tuple[pandas.DataFrame, dict]: Per-group counts, rates and
            Wilson intervals; and test results (permutation p-values
            for late and at-risk rates, Kruskal-Wallis p for processing
            days, group count and smallest group size).
    """
    t = s.groupby(col).agg(
        shipped=("OrderID", "size"), late=("late", "sum"),
        near_deadline=("near", "sum"), at_risk=("at_risk", "sum"),
        median_slack=("slack_days", "median"),
        median_processing=("processing_days", "median"),
        mean_processing=("processing_days", "mean"),
        slow_processing=("slow", "sum"))
    t["late_rate"] = t["late"] / t["shipped"]
    t["at_risk_rate"] = t["at_risk"] / t["shipped"]
    t["late_ci_low"], t["late_ci_high"] = wilson(t["late"], t["shipped"])
    t["slow_rate"] = t["slow_processing"] / t["shipped"]
    chi_late, p_late = perm_chi2(s[col], s["late"], rng)
    _, p_risk = perm_chi2(s[col], s["at_risk"], rng)
    kw = stats.kruskal(*[g["processing_days"].values
                         for _, g in s.groupby(col)])
    test = {"factor": col, "groups": len(t), "chi2_late": chi_late,
            "perm_p_late": p_late, "perm_p_at_risk": p_risk,
            "kruskal_p_processing": kw.pvalue,
            "min_group_n": int(t["shipped"].min())}
    return t, test


def plot_box(s: pd.DataFrame, test: dict) -> str:
    """Plot slack days by shipper as box plots with order dots.

    Args:
        s (pandas.DataFrame): Shipped orders with ``ShipperName`` and
            ``slack_days``.
        test (dict): Shipper test results, including
            ``kruskal_p_slack``, ``perm_p_late`` and ``excluded``.

    Returns:
        str: Path of the saved q3a PNG.
    """
    order = s.groupby("ShipperName")["slack_days"].median().sort_values().index
    fig, ax = ps.plt.subplots(figsize=(9, 5.4))
    data = [s.loc[s["ShipperName"] == k, "slack_days"].values for k in order]
    ax.axvspan(0, NEAR_MAX, color=NEAR_C, alpha=0.18, lw=0, zorder=0)
    ax.axvspan(s["slack_days"].min() - 3, 0, color=LATE_C, alpha=0.08, lw=0,
               zorder=0)
    ax.axvline(0, color=ps.INK_2, lw=0.8)
    bp = ax.boxplot(data, vert=False, widths=0.45, patch_artist=True,
                    showfliers=False, medianprops=dict(color=ps.INK, lw=2),
                    boxprops=dict(facecolor=ps.BLUE_RAMP[2], edgecolor=ps.SERIES[0]),
                    whiskerprops=dict(color=ps.SERIES[0]),
                    capprops=dict(color=ps.SERIES[0]))
    rng = np.random.default_rng(SEED)
    for i, d in enumerate(data, start=1):
        y = i + rng.uniform(-0.17, 0.17, len(d))
        col = np.where(d < 0, LATE_C, np.where(d <= NEAR_MAX, ps.INK_2,
                                               ps.SERIES[0]))
        ax.scatter(d, y, s=7, c=col, alpha=0.55, lw=0, zorder=3)
    labels = []
    for k, d in zip(order, data):
        labels.append(f"{k}\n{(d < 0).sum()} late of {len(d)} "
                      f"({(d < 0).mean():.1%})")
    ax.set_yticks(range(1, len(order) + 1))
    ax.set_yticklabels(labels)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(s["slack_days"].min() - 2, s["slack_days"].max() + 2)
    ax.set_xlabel("Slack: RequiredDate − ShippedDate (days; negative = late)")
    ax.text(-0.5, len(order) + 0.42, "late ←", ha="right", fontsize=8.5,
            color=LATE_C)
    ax.text(NEAR_MAX + 0.5, len(order) + 0.42,
            f"near-deadline (0–{NEAR_MAX} days)", ha="left", fontsize=8.5,
            color=ps.INK_2)
    verdict = ("differs" if test["kruskal_p_slack"] < 0.05 else
               "is similar")
    ax.set_title(f"Slack {verdict} across shippers; late orders are the "
                 "long tail for all three")
    ps.subtitle(ax, f"Each dot is a shipped order (n={len(s)}). Kruskal–Wallis "
                f"on slack p={test['kruskal_p_slack']:.2f}; late-rate "
                f"permutation χ² p={test['perm_p_late']:.2f}.")
    ps.footnote(fig, f"{test['excluded']} unshipped orders excluded. Box = "
                "middle 50%, line = median, whiskers = 1.5×IQR.")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return ps.save(fig, "q3a_slack_by_shipper")


def plot_country(ct: pd.DataFrame, overall_late: float, overall_risk: float,
                 test: dict) -> str:
    """Plot late and near-deadline rates by ship country.

    Only countries with at least ``MIN_COUNTRY_ORDERS`` shipped orders
    are shown.

    Args:
        ct (pandas.DataFrame): Per-country table from ``factor_table``.
        overall_late (float): Overall late rate; not used in the body.
        overall_risk (float): Overall at-risk rate, drawn as a line.
        test (dict): Country test results with ``perm_p_at_risk``.

    Returns:
        str: Path of the saved q3b PNG.
    """
    c = ct[ct["shipped"] >= MIN_COUNTRY_ORDERS].sort_values(
        ["at_risk_rate", "shipped"])
    fig, ax = ps.plt.subplots(figsize=(9, 6))
    y = np.arange(len(c))
    late_r = c["late"] / c["shipped"]
    near_r = c["near_deadline"] / c["shipped"]
    ax.barh(y, late_r, color=LATE_C, height=0.62, label="Late (slack < 0)",
            edgecolor=ps.SURFACE, lw=1.5)
    ax.barh(y, near_r, left=late_r, color=NEAR_C, height=0.62,
            label=f"Near-deadline (0–{NEAR_MAX} days)", edgecolor=ps.SURFACE,
            lw=1.5)
    lo, hi = wilson(c["at_risk"], c["shipped"])
    err = np.clip([c["at_risk_rate"] - lo, hi - c["at_risk_rate"]], 0, None)
    ax.errorbar(c["at_risk_rate"], y, xerr=err,
                fmt="none", ecolor=ps.MUTED, elinewidth=1, capsize=2.5)
    for yi, (_, r) in zip(y, c.iterrows()):
        ax.text(hi[yi] + 0.006, yi,
                f"{int(r.late)} late + {int(r.near_deadline)} near / "
                f"{int(r.shipped)}", va="center", fontsize=8, color=ps.INK_2)
    ax.axvline(overall_risk, color=ps.INK_2, lw=1)
    ax.text(overall_risk, len(c) - 0.35, f"  all countries {overall_risk:.1%}",
            fontsize=8, color=ps.INK_2, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels(c.index)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(hi.max() + 0.12, 0.35))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Share of shipped orders late or near-deadline (%), "
                  "whisker = 95% Wilson interval")
    verdict = ("differ" if test["perm_p_at_risk"] < 0.05 else
               "do not differ significantly")
    ax.set_title(f"At-risk shipment rates {verdict} by country "
                 f"(permutation p={test['perm_p_at_risk']:.2f})")
    ps.subtitle(ax, f"Countries with ≥{MIN_COUNTRY_ORDERS} shipped orders "
                f"({len(c)} of {len(ct)}). Wide intervals: most countries "
                "have under 5 late orders.")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return ps.save(fig, "q3b_late_rate_by_country")


def plot_lead(lt: pd.DataFrame, odds: float, fisher_p: float) -> str:
    """Plot late rate by requested lead time.

    Args:
        lt (pandas.DataFrame): Per-lead-time table indexed by labels
            such as "14 days".
        odds (float): Odds ratio, 14 days versus longer.
        fisher_p (float): Fisher exact p-value for that comparison.

    Returns:
        str: Path of the saved q3d PNG.
    """
    t = lt.loc[sorted(lt.index, key=lambda k: int(k.split()[0]))]
    fig, ax = ps.plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(t))
    late_r = t["late"] / t["shipped"]
    near_r = t["near_deadline"] / t["shipped"]
    ax.bar(x, late_r, color=LATE_C, width=0.55, label="Late (slack < 0)",
           edgecolor=ps.SURFACE, lw=1.5)
    ax.bar(x, near_r, bottom=late_r, color=NEAR_C, width=0.55,
           label=f"Near-deadline (0–{NEAR_MAX} days)", edgecolor=ps.SURFACE,
           lw=1.5)
    lo, hi = wilson(t["late"], t["shipped"])
    err = np.clip([late_r - lo, hi - late_r], 0, None)
    ax.errorbar(x, late_r, yerr=err, fmt="none", ecolor=ps.INK_2,
                elinewidth=1, capsize=3)
    for xi, (_, r) in zip(x, t.iterrows()):
        ax.text(xi + 0.08, hi[xi] + 0.008,
                f"{int(r.late)} late, {int(r.near_deadline)} near\n"
                f"of {int(r.shipped)} shipped", ha="left", fontsize=8.5,
                color=ps.INK_2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k} to deliver" for k in t.index])
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_ylim(0, max(hi) + 0.08)
    ax.set_ylabel("Share of shipped orders (%)")
    ax.set_xlabel("Lead time the customer gave: RequiredDate − OrderDate")
    r14 = late_r.iloc[0]
    r_rest = t["late"].iloc[1:].sum() / t["shipped"].iloc[1:].sum()
    ax.set_title(f"Orders due in 14 days are late {r14:.0%} of the time vs "
                 f"{r_rest:.1%} for longer lead times")
    ps.subtitle(ax, f"Fisher exact test, 14 days vs longer: odds ratio "
                f"{odds:.1f}, p={fisher_p:.1g}. Whisker = 95% Wilson interval "
                "on the late rate.")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return ps.save(fig, "q3d_late_rate_by_lead_time")


def plot_monthly(mt: pd.DataFrame, partial_months: list, p_year: float) -> str:
    """Plot monthly late-order counts and rates.

    Args:
        mt (pandas.DataFrame): Monthly table indexed by month, with
            ``shipped``, ``late`` and ``near_deadline``.
        partial_months (list): Months drawn faded because they contain
            unshipped orders.
        p_year (float): Permutation p-value for late rate by order year.

    Returns:
        str: Path of the saved q3c PNG.
    """
    fig, (a1, a2) = ps.plt.subplots(2, 1, figsize=(11, 6.6), sharex=True,
                                    gridspec_kw={"height_ratios": [1.15, 1]})
    x = np.arange(len(mt))
    ontime = mt["shipped"] - mt["late"] - mt["near_deadline"]
    alpha = np.where(mt.index.isin(partial_months), 0.4, 1.0)
    for xi, a in zip(x, alpha):
        a1.bar(xi, ontime.iloc[xi], color=ps.BLUE_RAMP[3], width=0.75,
               alpha=a, edgecolor=ps.SURFACE, lw=1)
        a1.bar(xi, mt["near_deadline"].iloc[xi], bottom=ontime.iloc[xi],
               color=NEAR_C, width=0.75, alpha=a, edgecolor=ps.SURFACE, lw=1)
        a1.bar(xi, mt["late"].iloc[xi],
               bottom=ontime.iloc[xi] + mt["near_deadline"].iloc[xi],
               color=LATE_C, width=0.75, alpha=a, edgecolor=ps.SURFACE, lw=1)
    a1.set_ylabel("Shipped orders (count)")
    a1.legend(handles=[Patch(color=ps.BLUE_RAMP[3], label="On time (>2 days slack)"),
                       Patch(color=NEAR_C, label=f"Near-deadline (0–{NEAR_MAX} days)"),
                       Patch(color=LATE_C, label="Late")],
              loc="upper left", ncol=3)
    full = ~mt.index.isin(partial_months)
    a2.plot(x[full], mt.loc[full, "late_rate"], color=LATE_C, marker="o",
            ms=4, lw=1.2, label="Late rate, monthly")
    a2.plot(x[full], mt.loc[full, "late_rate_3m"], color=ps.INK_2, lw=2,
            label="Late rate, 3-month rolling")
    a2.plot(x[~full], mt.loc[~full, "late_rate"], ls="", marker="o", ms=5,
            mfc=ps.SURFACE, mec=LATE_C, label="Incomplete month")
    a2.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    a2.set_ylabel("Late shipments (% of shipped)")
    a2.legend(loc="upper left", ncol=3)
    a2.set_ylim(0, max(0.15, mt["late_rate"].max() * 1.25))
    a2.set_xticks(x)
    a2.set_xticklabels([f"{p.strftime('%b')}\n{p.year}" if p.month in (1, 7)
                        or i == 0 else p.strftime("%b")
                        for i, p in enumerate(mt.index)], fontsize=8)
    a2.set_xlabel("Month the order was placed")
    yr = mt[full].groupby(mt[full].index.year)[["late", "shipped"]].sum()
    rates = ", ".join(f"{y}: {r.late / r.shipped:.1%}" for y, r in yr.iterrows())
    a1.set_title("Late shipments stay rare while volume grows; no "
                 "significant change in the late rate" if p_year >= 0.05
                 else "The late-shipment rate changes significantly over time")
    ps.subtitle(a1, f"Late rate by order year — {rates} (permutation "
                f"p={p_year:.2f}). "
                f"{', '.join(str(p) for p in partial_months)} faded: many "
                "orders still unshipped at data end.")
    fig.tight_layout()
    return ps.save(fig, "q3c_monthly_late_trend")


def main() -> dict:
    """Run the Q3 shipping analysis and write its tables and figures.

    Args:
        None.

    Returns:
        dict: ``late`` (int, number of late orders) and ``excluded``
            (int, unshipped orders left out). Also writes CSVs and PNGs
            and prints the results.
    """
    ensure_output_dirs()
    ps.apply_style()
    nw = load_clean()
    o = nw.orders
    rng = np.random.default_rng(SEED)

    excluded = int((~o["is_shipped"]).sum())
    s = o[o["is_shipped"]].copy()
    s["late"] = s["slack_days"] < 0
    s["near"] = s["slack_days"].between(0, NEAR_MAX)
    s["at_risk"] = s["late"] | s["near"]
    s["slow"] = s["processing_days"] > SLOW_DAYS
    s["order_month"] = s["OrderDate"].dt.month
    s["season"] = "Q" + s["OrderDate"].dt.quarter.astype(str)
    s["lines_bucket"] = pd.cut(s["n_lines"], [0, 1, 2, 3, 99],
                               labels=["1", "2", "3", "4+"])
    s["revenue_quartile"] = pd.qcut(s["order_revenue"], 4,
                                    labels=["Q1 (smallest)", "Q2", "Q3",
                                            "Q4 (largest)"])
    s["lead_time"] = s["lead_days"].astype(str) + " days"

    factors = {"ShipperName": "shipper", "ShipCountry": "country",
               "EmployeeName": "employee", "order_month": "order month",
               "season": "season (calendar quarter)",
               "lead_time": "lead time (RequiredDate − OrderDate)",
               "lines_bucket": "lines per order",
               "revenue_quartile": "order value quartile"}
    tables, tests = {}, []
    for col, name in factors.items():
        t, test = factor_table(s, col, rng)
        test["factor"] = name
        tables[col] = t
        tests.append(test)
        t.to_csv(OUT_TAB / f"q3_by_{col.lower()}.csv", float_format="%.4f")
    tests = pd.DataFrame(tests)

    kw_slack = stats.kruskal(*[g["slack_days"].values
                               for _, g in s.groupby("ShipperName")])
    ship_test = tests.iloc[0].to_dict()
    ship_test.update({"kruskal_p_slack": kw_slack.pvalue, "excluded": excluded})

    # Lead time: 14-day vs longer, exact test.
    a = s["lead_days"] == 14
    tab = [[int((a & s.late).sum()), int((a & ~s.late).sum())],
           [int((~a & s.late).sum()), int((~a & ~s.late).sum())]]
    odds, fisher_p = stats.fisher_exact(tab)
    # Mechanism: an order is late exactly when processing exceeds lead time.
    mech = int(((s["processing_days"] > s["lead_days"]) == s["late"]).sum())
    late_proc = s.loc[s["late"], "processing_days"]
    ok_proc = s.loc[~s["late"], "processing_days"]
    mw_value = stats.mannwhitneyu(s.loc[s.late, "order_revenue"],
                                  s.loc[~s.late, "order_revenue"])
    mw_freight = stats.mannwhitneyu(s.loc[s.late, "Freight"],
                                    s.loc[~s.late, "Freight"])

    # Monthly trend (by month the order was placed).
    mt = s.groupby("OrderMonth").agg(shipped=("OrderID", "size"),
                                     late=("late", "sum"),
                                     near_deadline=("near", "sum"))
    mt["placed"] = o.groupby("OrderMonth").size()
    mt["late_rate"] = mt["late"] / mt["shipped"]
    mt["late_rate_3m"] = (mt["late"].rolling(3).sum()
                          / mt["shipped"].rolling(3).sum())
    partial = mt.index[mt["placed"] > mt["shipped"]].tolist()
    mt.to_csv(OUT_TAB / "q3_monthly_late.csv", float_format="%.4f")
    tests.to_csv(OUT_TAB / "q3_tests.csv", index=False, float_format="%.4g")

    late_orders = s[s["late"]].sort_values("slack_days")[
        ["OrderID", "OrderDate", "RequiredDate", "ShippedDate", "lead_days",
         "processing_days", "slack_days", "ShipperName", "ShipCountry",
         "EmployeeName", "n_lines", "order_revenue"]]
    late_orders.to_csv(OUT_TAB / "q3_late_orders.csv", index=False,
                       float_format="%.2f", date_format="%Y-%m-%d")

    f1 = plot_box(s, ship_test)
    f2 = plot_country(tables["ShipCountry"], s["late"].mean(),
                      s["at_risk"].mean(), tests.iloc[1].to_dict())
    _, p_year = perm_chi2(s["OrderYear"], s["late"], rng)
    f3 = plot_monthly(mt, partial, p_year)
    f4 = plot_lead(tables["lead_time"], odds, fisher_p)

    # --------------------------------------------------------- printed check
    print("\n" + "=" * 78 + "\nQ3  LATE / NEAR-DEADLINE SHIPMENTS\n" + "=" * 78)
    print(f"Orders: {len(o)}; excluded unshipped: {excluded}; analysed: {len(s)}")
    print(f"Late (slack<0): {s.late.sum()} ({s.late.mean():.1%}); "
          f"near-deadline (0-{NEAR_MAX}d): {s.near.sum()} ({s.near.mean():.1%}); "
          f"at risk: {s.at_risk.sum()} ({s.at_risk.mean():.1%})")
    print("slack_days: " + ", ".join(
        f"p{int(q * 100)}={v:.0f}" for q, v in
        s["slack_days"].quantile([0, .05, .25, .5, .75, 1]).items()))
    print(f"processing_days: median {s.processing_days.median():.0f}; late "
          f"orders median {late_proc.median():.0f} (min {late_proc.min()}), "
          f"others median {ok_proc.median():.0f}")
    print(f"Identity: slack = lead_days - processing_days, so late <=> "
          f"processing_days > lead_days (holds for {mech}/{len(s)} orders)")
    print(f"Slow processing (>{SLOW_DAYS} days): {s.slow.sum()} orders; "
          f"{(s.slow & s.late).sum()} of them late")
    print(f"Lead time 14 days vs longer: late {tab[0][0]}/{sum(tab[0])} "
          f"({tab[0][0] / sum(tab[0]):.1%}) vs {tab[1][0]}/{sum(tab[1])} "
          f"({tab[1][0] / sum(tab[1]):.1%}); odds ratio {odds:.1f}, "
          f"Fisher exact p={fisher_p:.2g}")
    print(f"Order value, late vs not: Mann-Whitney p={mw_value.pvalue:.2f}; "
          f"freight: p={mw_freight.pvalue:.2f}")
    print(f"Slack by shipper: Kruskal-Wallis p={kw_slack.pvalue:.3f}")
    print("\nPer-factor tests (permutation chi-square on late / at-risk; "
          "Kruskal-Wallis on processing days):")
    print(tests[["factor", "groups", "min_group_n", "perm_p_late",
                 "perm_p_at_risk", "kruskal_p_processing"]]
          .round(4).to_string(index=False))
    for col in ["ShipperName", "lead_time", "EmployeeName", "season"]:
        t = tables[col]
        print(f"\nBy {factors[col]}:")
        print(t[["shipped", "late", "near_deadline", "late_rate",
                 "late_ci_low", "late_ci_high", "median_processing",
                 "slow_processing"]].round(3).to_string())
    ct = tables["ShipCountry"]
    big = ct[ct["shipped"] >= MIN_COUNTRY_ORDERS]
    print(f"\nCountries shown (>= {MIN_COUNTRY_ORDERS} shipped): {len(big)}; "
          f"they hold {big.late.sum()} of {s.late.sum()} late orders")
    print(big.sort_values("at_risk_rate", ascending=False)[
        ["shipped", "late", "near_deadline", "at_risk_rate"]].round(3).to_string())
    print(f"Late rate by order year: permutation p={p_year:.2f}")
    print(f"\nMonths with unshipped orders (rates biased low): "
          f"{[str(p) for p in partial]}")
    alpha = 0.05 / len(tests)
    survivors = tests.loc[tests["perm_p_late"] < alpha, "factor"].tolist()
    print(f"Multiple comparisons: {len(tests)} factors tested; Bonferroni "
          f"threshold p<{alpha:.4f}. Factors that pass on late rate: {survivors}")
    print("CAVEAT: 37 late orders is a small sample - differences that do not "
          "pass the threshold are reported as counts only, not as findings.")
    print(f"Wrote {f1}\nWrote {f2}\nWrote {f3}\nWrote {f4}")
    return {"late": int(s.late.sum()), "excluded": excluded}


if __name__ == "__main__":
    main()
