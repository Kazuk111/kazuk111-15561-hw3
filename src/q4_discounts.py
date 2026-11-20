"""Q4. Do discounts increase quantity and revenue? Where to apply them?

There is no cost data, so this evaluates discounted REVENUE, never profit.
Only the standard levels (0/5/10/15/20/25%) are compared; the 8 stray lines
(0.01-0.06, all in order 11077) are excluded and counted.

Discounts are not randomly assigned, so raw comparisons can be confounded by
product/price level, category, customer and order size. Besides raw tests
(Welch t, Mann-Whitney, effect sizes) we compare each discounted line with
undiscounted lines of the SAME product, and fit log-quantity regressions with
product and customer fixed effects (standard errors clustered by order).
Even then this is association, not a causal effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, PercentFormatter
from scipy import stats

import plot_style as ps
from load_data import OUT_TAB, STANDARD_DISCOUNTS, ensure_output_dirs, load_clean

LEVELS = list(STANDARD_DISCOUNTS)


# ------------------------------------------------------------------ helpers
def hedges_g(a, b) -> float:
    """Compute Hedges' g, a bias-corrected standardised difference.

    Args:
        a (array-like): Values for the first group.
        b (array-like): Values for the second group.

    Returns:
        float: (mean(a) - mean(b)) / pooled SD, small-sample corrected.
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1))
                 / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float((a.mean() - b.mean()) / sp * j)


def compare(a, b) -> dict:
    """Compare discounted and undiscounted values.

    Args:
        a (array-like): Values for discounted lines.
        b (array-like): Values for undiscounted lines.

    Returns:
        dict: Group sizes and means, ``diff_pct`` (mean(a)/mean(b) - 1),
            Welch and Mann-Whitney p-values, ``hedges_g`` and
            ``rank_biserial``.
    """
    w = stats.ttest_ind(a, b, equal_var=False)
    u = stats.mannwhitneyu(a, b, alternative="two-sided")
    return {"n_disc": len(a), "n_none": len(b),
            "mean_disc": float(np.mean(a)), "mean_none": float(np.mean(b)),
            "diff_pct": float(np.mean(a) / np.mean(b) - 1),
            "welch_p": float(w.pvalue), "mw_p": float(u.pvalue),
            "hedges_g": hedges_g(a, b),
            # rank-biserial: P(a > b) - P(a < b)
            "rank_biserial": float(2 * u.statistic / (len(a) * len(b)) - 1)}


def mean_ci(x, z=1.96):
    """Compute a normal-approximation confidence interval for a mean.

    Args:
        x (array-like): Sample values.
        z (float): Normal quantile. Default 1.96 (95%).

    Returns:
        tuple[float, float, float]: Mean, lower bound, upper bound.
    """
    x = np.asarray(x, float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, m - z * se, m + z * se


def holm(p: pd.Series) -> pd.Series:
    """Adjust p-values with the Holm step-down method.

    Args:
        p (pandas.Series): Raw p-values (no missing values).

    Returns:
        pandas.Series: Adjusted p-values capped at 1, same index.
    """
    order = p.sort_values()
    m = len(p)
    adj = (order * (m - np.arange(m))).cummax().clip(upper=1)
    return adj.reindex(p.index)


def ols_cluster(y, X, groups):
    """Fit OLS with cluster-robust (CR1) standard errors.

    Args:
        y (array-like): Outcome, length n.
        X (array-like): Design matrix of shape (n, k), with intercept.
        groups (array-like): Cluster label for each of the n rows.

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]: Coefficients and their
            cluster-robust standard errors, each of length k.
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    e = y - X @ beta
    g = pd.factorize(groups)[0]
    G = g.max() + 1
    scores = np.zeros((G, X.shape[1]))
    np.add.at(scores, g, X * e[:, None])
    meat = scores.T @ scores
    n, k = X.shape
    c = G / (G - 1) * (n - 1) / (n - k)
    V = c * XtX_inv @ meat @ XtX_inv
    return beta, np.sqrt(np.diag(V))


FULL_FE = ["ProductID", "CustomerID"]


def fe_effect(df: pd.DataFrame, outcome: str, fes: list[str]) -> dict:
    """Estimate the percent effect of discounting with fixed effects.

    Regresses log(outcome) on an any-discount indicator plus dummies
    for each fixed effect, clustering errors by order.

    Args:
        df (pandas.DataFrame): Lines with ``is_disc``, ``OrderID``,
            the positive ``outcome`` column and the ``fes`` columns.
        outcome (str): Column to model on the log scale.
        fes (list[str]): Columns to include as fixed effects.

    Returns:
        dict: ``pct`` (effect as a fraction), ``lo`` and ``hi`` (95%
            interval) and ``p`` (two-sided p-value).
    """
    parts = [pd.Series(1.0, index=df.index, name="const"),
             df["is_disc"].astype(float)]
    for fe in fes:
        parts.append(pd.get_dummies(df[fe].astype(str), prefix=fe,
                                    drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1).values
    b, se = ols_cluster(np.log(df[outcome].values), X, df["OrderID"].values)
    return {"pct": np.exp(b[1]) - 1, "lo": np.exp(b[1] - 1.96 * se[1]) - 1,
            "hi": np.exp(b[1] + 1.96 * se[1]) - 1,
            "p": 2 * stats.norm.sf(abs(b[1] / se[1]))}


def category_fe(l: pd.DataFrame) -> pd.DataFrame:
    """Estimate discount effects within each product category.

    Uses product and customer fixed effects (``FULL_FE``).

    Args:
        l (pandas.DataFrame): Standard-discount lines with ``is_disc``,
            ``CategoryName``, ``Quantity``, ``revenue`` and ids.

    Returns:
        pandas.DataFrame: Indexed by category, with line counts,
            quantity and revenue effects and intervals, and the net
            revenue difference of discounted lines in USD.
    """
    rows = []
    for cat, g in l.groupby("CategoryName"):
        q = fe_effect(g, "Quantity", FULL_FE)
        r = fe_effect(g, "revenue", FULL_FE)
        disc_rev = g.loc[g["is_disc"], "revenue"].sum()
        rows.append({"category": cat, "lines": len(g),
                     "lines_disc": int(g["is_disc"].sum()),
                     "customers": g["CustomerID"].nunique(),
                     "fe_qty_pct": q["pct"], "fe_qty_lo": q["lo"],
                     "fe_qty_hi": q["hi"], "fe_qty_p": q["p"],
                     "fe_rev_pct": r["pct"], "fe_rev_lo": r["lo"],
                     "fe_rev_hi": r["hi"], "fe_rev_p": r["p"],
                     # discounted revenue minus what the same lines would
                     # have earned at the fitted no-discount level
                     "fe_net_revenue_diff_total":
                         disc_rev - disc_rev / (1 + r["pct"])})
    return pd.DataFrame(rows).set_index("category")


def fe_models(l: pd.DataFrame) -> pd.DataFrame:
    """Fit log-outcome models with progressively more controls.

    Covers Quantity and revenue; no controls, then product, customer
    and year fixed effects; each for any discount and by level.

    Args:
        l (pandas.DataFrame): Standard-discount lines with ``is_disc``,
            ``Discount``, ``Quantity``, ``revenue``, ids and
            ``OrderYear``.

    Returns:
        pandas.DataFrame: One row per outcome, model and term with
            coefficient, standard error, percent effect, interval,
            p-value, n and parameter count.
    """
    rows = []
    specs = {
        "raw (no controls)": [],
        "+ product FE": ["ProductID"],
        "+ product & customer FE": ["ProductID", "CustomerID"],
        "+ product, customer & year FE": ["ProductID", "CustomerID",
                                          "OrderYear"],
    }
    for outcome in ["Quantity", "revenue"]:
        y = np.log(l[outcome].values)
        for name, fes in specs.items():
            for kind in ["any discount", "by level"]:
                if kind == "any discount":
                    main = l[["is_disc"]].astype(float)
                else:
                    main = pd.get_dummies(l["Discount"], prefix="d",
                                          dtype=float).drop(columns="d_0.0")
                parts = [pd.Series(1.0, index=l.index, name="const"), main]
                for fe in fes:
                    parts.append(pd.get_dummies(l[fe].astype(str), prefix=fe,
                                                drop_first=True, dtype=float))
                X = pd.concat(parts, axis=1)
                b, se = ols_cluster(y, X.values, l["OrderID"].values)
                for j, col in enumerate(X.columns[1:1 + main.shape[1]], start=1):
                    rows.append({
                        "outcome": f"log({outcome})", "model": name,
                        "term": col.replace("d_", "discount ")
                        if kind == "by level" else "any discount",
                        "coef": b[j], "se": se[j],
                        "pct_effect": np.exp(b[j]) - 1,
                        "ci_low": np.exp(b[j] - 1.96 * se[j]) - 1,
                        "ci_high": np.exp(b[j] + 1.96 * se[j]) - 1,
                        "p": 2 * stats.norm.sf(abs(b[j] / se[j])),
                        "n": len(l), "params": X.shape[1]})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- tables
def level_table(l: pd.DataFrame) -> pd.DataFrame:
    """Summarise quantity and revenue by discount level.

    Args:
        l (pandas.DataFrame): Standard-discount lines with ``Discount``,
            ``Quantity``, ``revenue`` and ``UnitPrice``.

    Returns:
        pandas.DataFrame: Indexed by discount level, with means,
            intervals, uplift versus 0%, break-even uplift, tests
            against 0% and Holm-adjusted Mann-Whitney p-values.
    """
    base = l[l["Discount"] == 0]
    rows = []
    for d in LEVELS:
        g = l[l["Discount"] == d]
        q, qlo, qhi = mean_ci(g["Quantity"])
        r, rlo, rhi = mean_ci(g["revenue"])
        row = {"discount": d, "lines": len(g), "mean_qty": q,
               "qty_ci_low": qlo, "qty_ci_high": qhi,
               "median_qty": g["Quantity"].median(),
               "mean_revenue": r, "rev_ci_low": rlo, "rev_ci_high": rhi,
               "mean_unit_price": g["UnitPrice"].mean(),
               "qty_uplift_vs_0": q / base["Quantity"].mean() - 1,
               # quantity must rise by d/(1-d) for revenue per line to hold
               "breakeven_qty_uplift": d / (1 - d),
               "revenue_diff_vs_0": r - base["revenue"].mean()}
        if d > 0:
            cq = compare(g["Quantity"], base["Quantity"])
            cr = compare(g["revenue"], base["revenue"])
            row.update({"qty_welch_p": cq["welch_p"], "qty_mw_p": cq["mw_p"],
                        "qty_hedges_g": cq["hedges_g"],
                        "rev_welch_p": cr["welch_p"], "rev_mw_p": cr["mw_p"],
                        "rev_hedges_g": cr["hedges_g"]})
        rows.append(row)
    t = pd.DataFrame(rows).set_index("discount")
    for c in ["qty_mw_p", "rev_mw_p"]:
        t[c + "_holm"] = holm(t[c].dropna())
    return t


def within_product(g: pd.DataFrame) -> dict:
    """Compare discounted and undiscounted lines of the same product.

    Only products sold both ways count; each product is weighted by
    its number of discounted lines.

    Args:
        g (pandas.DataFrame): Lines with ``ProductID``, ``is_disc``,
            ``Quantity`` and ``revenue``.

    Returns:
        dict: ``products_compared``, ``adj_qty_diff_pct``,
            ``adj_rev_diff_pct``, ``adj_rev_diff_per_line`` and
            ``adj_net_revenue_diff_total``.
    """
    p = g.groupby(["ProductID", "is_disc"]).agg(
        n=("Quantity", "size"), q=("Quantity", "mean"),
        r=("revenue", "mean")).unstack("is_disc")
    p = p.dropna()
    w = p[("n", True)]
    dq = (w * (p[("q", True)] - p[("q", False)])).sum()
    dr_line = (w * (p[("r", True)] - p[("r", False)])).sum() / w.sum()
    return {"products_compared": len(p),
            "adj_qty_diff_pct": dq / (w * p[("q", False)]).sum(),
            "adj_rev_diff_pct": (w * (p[("r", True)] - p[("r", False)])).sum()
            / (w * p[("r", False)]).sum(),
            "adj_rev_diff_per_line": dr_line,
            "adj_net_revenue_diff_total": dr_line * w.sum()}


def category_table(l: pd.DataFrame) -> pd.DataFrame:
    """Compare discounted and undiscounted lines by category.

    Args:
        l (pandas.DataFrame): Standard-discount lines with ``is_disc``,
            ``CategoryName``, ``Quantity``, ``revenue``,
            ``discount_amount`` and ``ProductID``.

    Returns:
        pandas.DataFrame: Indexed by category and sorted by
            ``adj_rev_diff_pct`` descending, with raw and within-product
            differences and Holm-adjusted p-values.
    """
    rows = []
    for cat, g in l.groupby("CategoryName"):
        a, b = g[g["is_disc"]], g[~g["is_disc"]]
        cq = compare(a["Quantity"], b["Quantity"])
        cr = compare(a["revenue"], b["revenue"])
        wp = within_product(g)
        rows.append({
            "category": cat, "lines_disc": len(a), "lines_none": len(b),
            "share_lines_discounted": len(a) / len(g),
            "mean_qty_none": cq["mean_none"], "mean_qty_disc": cq["mean_disc"],
            "qty_diff_pct": cq["diff_pct"], "qty_mw_p": cq["mw_p"],
            "qty_rank_biserial": cq["rank_biserial"],
            "mean_rev_none": cr["mean_none"], "mean_rev_disc": cr["mean_disc"],
            "rev_diff_per_line": cr["mean_disc"] - cr["mean_none"],
            "rev_diff_pct": cr["diff_pct"], "rev_welch_p": cr["welch_p"],
            "rev_mw_p": cr["mw_p"], "rev_hedges_g": cr["hedges_g"],
            "discount_given_away": a["discount_amount"].sum(),
            **wp})
    t = pd.DataFrame(rows).set_index("category")
    t["qty_mw_p_holm"] = holm(t["qty_mw_p"])
    t["rev_mw_p_holm"] = holm(t["rev_mw_p"])
    return t.sort_values("adj_rev_diff_pct", ascending=False)


# ----------------------------------------------------------------- charts
def plot_qty(lt: pd.DataFrame, kw_p: float, fe_q: dict) -> str:
    """Plot mean units per line by discount level.

    Args:
        lt (pandas.DataFrame): Table from ``level_table``.
        kw_p (float): Kruskal-Wallis p-value for quantity across levels.
        fe_q (dict): Fixed-effects quantity effect from ``fe_effect``
            (``pct``, ``lo``, ``hi``).

    Returns:
        str: Path of the saved q4a PNG.
    """
    fig, ax = ps.plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(lt))
    colors = [ps.NEUTRAL] + [ps.SERIES[0]] * (len(lt) - 1)
    ax.bar(x, lt["mean_qty"], color=colors, width=0.6, edgecolor=ps.SURFACE,
           lw=1.5)
    ax.errorbar(x, lt["mean_qty"],
                yerr=[lt["mean_qty"] - lt["qty_ci_low"],
                      lt["qty_ci_high"] - lt["mean_qty"]],
                fmt="none", ecolor=ps.INK_2, elinewidth=1, capsize=3)
    base = lt["mean_qty"].iloc[0]
    be = base * (1 + lt["breakeven_qty_uplift"])
    ax.scatter(x[1:], be.iloc[1:], marker="_", s=900, color=ps.SERIES[1],
               lw=2.5, zorder=4, label="Units needed per line to keep revenue "
               "equal to 0% lines")
    for xi, (d, r) in zip(x, lt.iterrows()):
        txt = (f"{r.mean_qty:.1f}" if d == 0 else
               f"{r.mean_qty:.1f}\n({r.qty_uplift_vs_0:+.0%})")
        ax.text(xi, 1.5, txt, ha="center", va="bottom", fontsize=8.5,
                color=ps.SURFACE if d > 0 else ps.INK, fontweight="bold")
        ax.text(xi, r.qty_ci_high + 0.6, f"n={int(r.lines)}", ha="center",
                fontsize=8, color=ps.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0%}" for d in lt.index])
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Discount level")
    ax.set_ylabel("Average quantity per order line (units)")
    ax.set_ylim(0, max(lt["qty_ci_high"].max(), be.max()) * 1.18)
    above = (lt["mean_qty"].iloc[1:] > be.iloc[1:]).sum()
    ax.set_title(f"Discounted lines carry {lt['qty_uplift_vs_0'].iloc[1:].min():.0%}–"
                 f"{lt['qty_uplift_vs_0'].iloc[1:].max():.0%} more units; "
                 f"{above} of {len(lt) - 1} levels clear the revenue break-even",
                 pad=36)
    ps.subtitle(ax, f"Mean units per line, 95% CI (n={int(lt['lines'].sum()):,} "
                f"lines; Kruskal–Wallis p={kw_p:.1g}). Raw comparison.\nWithin "
                f"the same customer and product the uplift is only "
                f"{fe_q['pct']:+.0%} (95% CI {fe_q['lo']:+.0%} to "
                f"{fe_q['hi']:+.0%}).")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return ps.save(fig, "q4a_quantity_by_discount")


def plot_rev(lt: pd.DataFrame, kw_p: float) -> str:
    """Plot mean revenue per line by discount level.

    Args:
        lt (pandas.DataFrame): Table from ``level_table``.
        kw_p (float): Kruskal-Wallis p-value for revenue across levels.

    Returns:
        str: Path of the saved q4b PNG.
    """
    fig, ax = ps.plt.subplots(figsize=(9, 5.2))
    x = np.arange(len(lt))
    colors = [ps.NEUTRAL] + [ps.SERIES[0]] * (len(lt) - 1)
    ax.bar(x, lt["mean_revenue"], color=colors, width=0.6,
           edgecolor=ps.SURFACE, lw=1.5)
    ax.errorbar(x, lt["mean_revenue"],
                yerr=[lt["mean_revenue"] - lt["rev_ci_low"],
                      lt["rev_ci_high"] - lt["mean_revenue"]],
                fmt="none", ecolor=ps.INK_2, elinewidth=1, capsize=3)
    ax.axhline(lt["mean_revenue"].iloc[0], color=ps.INK_2, lw=1)
    for xi, (d, r) in zip(x, lt.iterrows()):
        txt = (f"${r.mean_revenue:,.0f}" if d == 0 else
               f"${r.mean_revenue:,.0f}\n({r.revenue_diff_vs_0:+,.0f})")
        ax.text(xi, 25, txt, ha="center", va="bottom", fontsize=8.5,
                color=ps.SURFACE if d > 0 else ps.INK, fontweight="bold")
        ax.text(xi, r.rev_ci_high + 15, f"n={int(r.lines)}", ha="center",
                fontsize=8, color=ps.MUTED)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{d:.0%}" for d in lt.index])
    ax.grid(axis="x", visible=False)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlabel("Discount level")
    ax.set_ylabel("Average revenue per order line (USD, after discount)")
    ax.set_ylim(0, lt["rev_ci_high"].max() * 1.15)
    adj = lt["rev_mw_p_holm"].dropna()
    sig_up = int(((adj < 0.05) & (lt["revenue_diff_vs_0"].iloc[1:] > 0)).sum())
    sig_dn = int(((adj < 0.05) & (lt["revenue_diff_vs_0"].iloc[1:] < 0)).sum())
    ax.set_title(f"Revenue per line: {sig_up} of {len(adj)} discount levels "
                 f"significantly above 0%, {sig_dn} below (after correction)")
    ps.subtitle(ax, f"Mean revenue per line with 95% CI. Kruskal–Wallis across "
                f"levels p={kw_p:.2f}; level vs 0% uses Mann–Whitney with "
                "Holm correction.")
    fig.tight_layout()
    return ps.save(fig, "q4b_revenue_by_discount")


def plot_category(cf: pd.DataFrame, pooled_q: dict, pooled_r: dict,
                  never: int, n_cust: int) -> str:
    """Plot fixed-effects discount effects by category.

    Args:
        cf (pandas.DataFrame): Table from ``category_fe``.
        pooled_q (dict): Pooled quantity effect from ``fe_effect``.
        pooled_r (dict): Pooled revenue effect from ``fe_effect``.
        never (int): Customers never given a discount.
        n_cust (int): Customers in the analysed lines.

    Returns:
        str: Path of the saved q4c PNG.
    """
    t = cf.sort_values("fe_rev_pct")
    fig, ax = ps.plt.subplots(figsize=(10, 6.2))
    y = np.arange(len(t))
    off = 0.17
    for col, color, dy, name in [
            ("fe_qty", ps.SERIES[0], off, "Units per line"),
            ("fe_rev", ps.SERIES[1], -off, "Revenue per line (after discount)")]:
        err = np.clip([t[f"{col}_pct"] - t[f"{col}_lo"],
                       t[f"{col}_hi"] - t[f"{col}_pct"]], 0, None)
        ax.errorbar(t[f"{col}_pct"], y + dy, xerr=err, fmt="o", ms=7,
                    color=color, ecolor=color, elinewidth=1.6, capsize=0,
                    label=name, mec=ps.SURFACE, mew=1.5)
    ax.axvline(0, color=ps.INK_2, lw=0.9)
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.text(1.02, yi, f"net {r.fe_net_revenue_diff_total:+,.0f} USD",
                transform=ax.get_yaxis_transform(), va="center", fontsize=8,
                color=ps.INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{c}\n{int(r.lines_disc)} of {int(r.lines)} lines "
                        "discounted" for c, r in t.iterrows()], fontsize=8.5)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Effect of any discount vs none, same product and same "
                  "customer (%), whisker = 95% CI")
    pos = int((t["fe_rev_lo"] > 0).sum())
    neg = int((t["fe_rev_hi"] < 0).sum())
    ax.set_title(f"Same customer, same product: a discount raises revenue per "
                 f"line in {pos} of {len(t)} categories and lowers it in {neg}")
    ps.subtitle(ax, f"All categories pooled: units {pooled_q['pct']:+.0%} "
                f"(95% CI {pooled_q['lo']:+.0%} to {pooled_q['hi']:+.0%}), "
                f"revenue {pooled_r['pct']:+.0%} ({pooled_r['lo']:+.0%} to "
                f"{pooled_r['hi']:+.0%}). Net = discounted revenue vs fitted "
                "no-discount level.")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.22), ncol=2)
    ps.footnote(fig, "Log-linear regressions with product and customer fixed "
                "effects, SE clustered by order. Association only: discounts "
                f"are not randomly assigned ({never} of {n_cust} customers "
                "never get one). No cost data, so revenue, not profit.")
    fig.tight_layout(rect=(0, 0.04, 0.9, 1))
    return ps.save(fig, "q4c_discount_effect_by_category")


def main() -> dict:
    """Run the Q4 discount analysis and write its tables and figures.

    Args:
        None.

    Returns:
        dict: ``overall_q`` and ``overall_r``, the raw discounted versus
            undiscounted comparisons for quantity and revenue (from
            ``compare``). Also writes CSVs and PNGs and prints results.
    """
    ensure_output_dirs()
    ps.apply_style()
    nw = load_clean()
    all_lines = nw.lines
    stray = all_lines[~all_lines["is_standard_discount"]]
    l = all_lines[all_lines["is_standard_discount"]].copy()
    l["is_disc"] = l["Discount"] > 0

    lt = level_table(l)
    # Confounding checks.
    price = compare(l.loc[l.is_disc, "UnitPrice"], l.loc[~l.is_disc, "UnitPrice"])
    cust_rate = l.groupby("CustomerID")["is_disc"].mean()
    od = l.groupby("OrderID")["Discount"].agg(["min", "max", "nunique"])
    lines_per_order = l["OrderID"].map(l.groupby("OrderID").size())
    size_cmp = compare(lines_per_order[l.is_disc], lines_per_order[~l.is_disc])
    overall_q = compare(l.loc[l.is_disc, "Quantity"], l.loc[~l.is_disc, "Quantity"])
    overall_r = compare(l.loc[l.is_disc, "revenue"], l.loc[~l.is_disc, "revenue"])
    kw_q = stats.kruskal(*[g["Quantity"] for _, g in l.groupby("Discount")])
    kw_r = stats.kruskal(*[g["revenue"] for _, g in l.groupby("Discount")])
    wp_all = within_product(l)
    ct = category_table(l)
    cf = category_fe(l)
    ct = ct.join(cf.drop(columns=["lines", "lines_disc"]))
    fe = fe_models(l)
    pooled_q = fe_effect(l, "Quantity", FULL_FE)
    pooled_r = fe_effect(l, "revenue", FULL_FE)


    lt.to_csv(OUT_TAB / "q4_by_discount_level.csv", float_format="%.4f")
    ct.to_csv(OUT_TAB / "q4_by_category.csv", float_format="%.4f")
    fe.to_csv(OUT_TAB / "q4_fixed_effects_models.csv", index=False,
              float_format="%.4f")
    pd.DataFrame([{"outcome": "Quantity", **overall_q},
                  {"outcome": "revenue", **overall_r}]).to_csv(
        OUT_TAB / "q4_discounted_vs_not_tests.csv", index=False,
        float_format="%.4g")

    f1 = plot_qty(lt, kw_q.pvalue, pooled_q)
    f2 = plot_rev(lt, kw_r.pvalue)
    f3 = plot_category(cf, pooled_q, pooled_r, int((cust_rate == 0).sum()),
                       len(cust_rate))

    # --------------------------------------------------------- printed check
    print("\n" + "=" * 78 + "\nQ4  DISCOUNTS (revenue, not profit: no cost data)\n"
          + "=" * 78)
    print(f"Lines: {len(all_lines)}; excluded non-standard discounts: "
          f"{len(stray)} (order(s) {sorted(stray.OrderID.unique())}); "
          f"analysed: {len(l)} ({l.is_disc.sum()} discounted, "
          f"{(~l.is_disc).sum()} not)")
    print(f"Discount given away on analysed lines: "
          f"{l['discount_amount'].sum():,.2f} "
          f"(all lines: {all_lines['discount_amount'].sum():,.2f})")
    print("\nBy level:")
    print(lt[["lines", "mean_qty", "qty_ci_low", "qty_ci_high", "median_qty",
              "qty_uplift_vs_0", "breakeven_qty_uplift", "mean_revenue",
              "revenue_diff_vs_0", "qty_mw_p_holm", "rev_mw_p_holm"]]
          .round(3).to_string())
    print(f"Kruskal-Wallis across levels: quantity p={kw_q.pvalue:.2g}, "
          f"revenue p={kw_r.pvalue:.2g}")
    for name, c in [("Quantity", overall_q), ("Revenue", overall_r)]:
        print(f"{name}, discounted vs not: {c['mean_disc']:.2f} vs "
              f"{c['mean_none']:.2f} ({c['diff_pct']:+.1%}); Welch p="
              f"{c['welch_p']:.2g}, Mann-Whitney p={c['mw_p']:.2g}, Hedges g="
              f"{c['hedges_g']:.2f}, rank-biserial={c['rank_biserial']:.2f}")
    print(f"Within-product ({wp_all['products_compared']} products sold both "
          f"ways): units {wp_all['adj_qty_diff_pct']:+.1%}, revenue per line "
          f"{wp_all['adj_rev_diff_pct']:+.1%} "
          f"({wp_all['adj_rev_diff_per_line']:+,.0f} USD/line, "
          f"{wp_all['adj_net_revenue_diff_total']:+,.0f} USD in total)")
    print("\nFixed-effects regressions (any discount; SE clustered by order):")
    show = fe[fe["term"] == "any discount"][["outcome", "model", "pct_effect",
                                             "ci_low", "ci_high", "p"]]
    print(show.round(3).to_string(index=False))
    lv = fe[(fe["model"] == "+ product & customer FE")
            & (fe["term"] != "any discount")]
    print("\nBy level, product & customer FE:")
    print(lv[["outcome", "term", "pct_effect", "ci_low", "ci_high", "p"]]
          .round(3).to_string(index=False))
    print("\nBy category - raw and within-product (same product only):")
    print(ct[["lines_disc", "lines_none", "qty_diff_pct", "adj_qty_diff_pct",
              "rev_diff_pct", "adj_rev_diff_pct", "adj_net_revenue_diff_total",
              "qty_mw_p_holm", "rev_mw_p_holm"]].round(3).to_string())
    print("\nBy category - product & customer FE (used in the chart):")
    print(ct[["customers", "fe_qty_pct", "fe_qty_lo", "fe_qty_hi",
              "fe_rev_pct", "fe_rev_lo", "fe_rev_hi", "fe_rev_p",
              "fe_net_revenue_diff_total"]]
          .sort_values("fe_rev_pct", ascending=False).round(3).to_string())
    print("\nConfounding checks:")
    print(f"  Unit price, discounted vs not: {price['mean_disc']:.2f} vs "
          f"{price['mean_none']:.2f} (Mann-Whitney p={price['mw_p']:.2f})")
    print(f"  Lines per order, discounted vs not: {size_cmp['mean_disc']:.2f} "
          f"vs {size_cmp['mean_none']:.2f} (p={size_cmp['mw_p']:.2f})")
    print(f"  Customers never given a discount: {(cust_rate == 0).sum()} of "
          f"{len(cust_rate)}; share of lines discounted by category: "
          + ", ".join(f"{k} {v:.0%}" for k, v in
                      l.groupby('CategoryName')['is_disc'].mean().items()))
    print(f"  Orders with a discount: {(od['max'] > 0).sum()}; same rate on "
          f"every line: {((od['max'] > 0) & (od['nunique'] == 1)).sum()}; "
          f"mix of 0% and >0%: {((od['max'] > 0) & (od['min'] == 0)).sum()}")
    print("NOTE: discounts may be granted BECAUSE an order is large (reverse "
          "causality); fixed effects cannot rule that out.")
    print(f"Wrote {f1}\nWrote {f2}\nWrote {f3}")
    return {"overall_q": overall_q, "overall_r": overall_r}


if __name__ == "__main__":
    main()
