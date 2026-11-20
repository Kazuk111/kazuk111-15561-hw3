"""Q2. Do purchasing patterns vary by region? Consolidated shipping and
under-served customers.

Region = grouping of countries in load_data.REGIONS. Order-level analyses use
ShipCountry; the customer roster (incl. zero-order customers) uses the
customer's own Country.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.ticker import PercentFormatter
from scipy import stats

import plot_style as ps
from load_data import OUT_TAB, REGIONS, ensure_output_dirs, load_clean

WINDOW_DAYS = 7                    # headline consolidation window
WINDOW_GRID = list(range(1, 15))   # sensitivity range for the chart
REGION_ORDER = list(REGIONS)


# ---------------------------------------------------------------- (a) mix
N_PERM = 2000
SEED = 561


def mix_permutation_test(lines: pd.DataFrame, unit: str) -> dict:
    """Test whether regional revenue mix beats chance.

    The statistic is a revenue-weighted chi-square-style distance
    between each region's category shares and the overall shares.
    Region labels are shuffled across whole units, so large orders and
    customer tastes stay intact under the null. A unit's region is the
    most common region among its lines.

    Args:
        lines (pandas.DataFrame): Order lines with ``revenue``,
            ``CategoryName``, ``Region`` and the ``unit`` column.
        unit (str): Column to shuffle, e.g. "OrderID" or "CustomerID".

    Returns:
        dict: ``unit``, ``n_units``, observed ``stat``, ``null_median``
            and permutation ``p`` (``N_PERM`` shuffles).
    """
    m = lines.pivot_table(index=unit, columns="CategoryName", values="revenue",
                          aggfunc="sum", fill_value=0.0)
    region = lines.groupby(unit)["Region"].agg(lambda s: s.mode().iloc[0])
    codes = pd.Categorical(region.loc[m.index], categories=REGION_ORDER).codes
    R = m.values
    overall = R.sum(axis=0) / R.sum()

    def stat(c):
        """Compute the mix statistic for one region assignment.

        Args:
            c (numpy.ndarray): Integer region code per unit, aligned
                with the rows of the enclosing revenue matrix.

        Returns:
            float: Revenue-weighted divergence from the overall mix.
        """
        g = np.zeros((len(REGION_ORDER), R.shape[1]))
        np.add.at(g, c, R)
        tot = g.sum(axis=1, keepdims=True)
        share = np.divide(g, tot, out=np.zeros_like(g), where=tot > 0)
        return float((tot[:, 0] * ((share - overall) ** 2 / overall).sum(1)).sum()
                     / R.sum())

    obs = stat(codes)
    rng = np.random.default_rng(SEED)
    null = np.array([stat(rng.permutation(codes)) for _ in range(N_PERM)])
    return {"unit": unit, "n_units": len(codes), "stat": obs,
            "null_median": float(np.median(null)),
            "p": (1 + (null >= obs).sum()) / (N_PERM + 1)}


def category_mix(lines: pd.DataFrame):
    """Compute regional category revenue, shares and tests.

    Args:
        lines (pandas.DataFrame): Order lines with ``Region``,
            ``CategoryName``, ``revenue``, ``OrderID``, ``CustomerID``.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, dict, pandas.Series]:
            Revenue by region and category (USD); row-normalised shares
            with an "All regions" row; test results (chi-square on line
            counts, Cramer's V and its label, and both permutation
            tests); and the largest single customer's share of each
            region's revenue.
    """
    rev = lines.pivot_table(index="Region", columns="CategoryName",
                            values="revenue", aggfunc="sum", fill_value=0.0)
    col_order = rev.sum().sort_values(ascending=False).index
    rev = rev.loc[REGION_ORDER, col_order]
    share = rev.div(rev.sum(axis=1), axis=0)
    overall = rev.sum() / rev.values.sum()
    share.loc["All regions"] = overall

    # Chi-square on line counts (revenue is not a count). Cramér's V = effect.
    counts = pd.crosstab(lines["Region"], lines["CategoryName"])
    chi2, p, dof, exp = stats.chi2_contingency(counts)
    v = np.sqrt(chi2 / (counts.values.sum() * (min(counts.shape) - 1)))
    k = min(counts.shape) - 1
    # Cohen's benchmarks for Cramér's V scale with 1/sqrt(min(r, c) - 1).
    size = ("negligible" if v < 0.1 / np.sqrt(k) else "small" if v < 0.3 / np.sqrt(k)
            else "medium" if v < 0.5 / np.sqrt(k) else "large")
    test = {"chi2": chi2, "p": p, "dof": dof, "cramers_v": v, "effect": size,
            "min_expected": exp.min(), "n_lines": int(counts.values.sum()),
            "perm_order": mix_permutation_test(lines, "OrderID"),
            "perm_customer": mix_permutation_test(lines, "CustomerID")}

    # How much one customer drives each region's revenue (mix caveat).
    by_cust = lines.groupby(["Region", "CustomerID"])["revenue"].sum()
    top_share = (by_cust.groupby(level=0).max()
                 / by_cust.groupby(level=0).sum()).reindex(REGION_ORDER)
    return rev, share, test, top_share


def plot_heatmap(share: pd.DataFrame, rev: pd.DataFrame, test: dict) -> str:
    """Plot the region-by-category revenue share heatmap.

    Args:
        share (pandas.DataFrame): Row-normalised shares including the
            "All regions" row.
        rev (pandas.DataFrame): Revenue by region and category (USD).
        test (dict): Results from ``category_mix``.

    Returns:
        str: Path of the saved q2a PNG.
    """
    fig, ax = ps.plt.subplots(figsize=(11, 6))
    data = share.values
    im = ax.imshow(data, cmap=ps.SEQ_CMAP, vmin=0, vmax=data.max(),
                   aspect="auto")
    n_reg = len(REGION_ORDER)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            v = data[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if v > data.max() * 0.55 else ps.INK)
    ax.axhline(n_reg - 0.5, color=ps.SURFACE, lw=4)
    totals = rev.sum(axis=1)
    ylabels = [f"{r}  (${totals[r] / 1e3:,.0f}k)" for r in REGION_ORDER]
    ylabels.append(f"All regions  (${totals.sum() / 1e3:,.0f}k)")
    ax.set_yticks(range(len(ylabels)))
    ax.set_yticklabels(ylabels)
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(share.columns, rotation=0, fontsize=9)
    ax.tick_params(length=0)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xlabel("Product category (ordered by total revenue)")
    ax.set_ylabel("Region of ship country (total revenue)")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02,
                      format=PercentFormatter(1.0, decimals=0))
    cb.set_label("Share of the region's revenue (%)", color=ps.INK_2)
    cb.outline.set_visible(False)
    top_cat = share.columns[0]
    spread = share.loc[REGION_ORDER, top_cat]
    pc = test["perm_customer"]["p"]
    if pc < 0.05:
        ax.set_title(f"Category mix differs by region: {top_cat} ranges from "
                     f"{spread.min():.0%} to {spread.max():.0%} of revenue")
    else:
        ax.set_title(f"{top_cat} ranges from {spread.min():.0%} to "
                     f"{spread.max():.0%} of regional revenue, but the gaps are "
                     f"within customer-to-customer noise (p={pc:.2f})")
    ps.subtitle(ax, "Row-normalised: each row sums to 100% of that region's "
                "discounted revenue, Jul 1996–May 1998.")
    po, pcu = test["perm_order"], test["perm_customer"]
    ps.footnote(fig, f"Tests: permutation test on revenue mix, shuffling region "
                f"across {pcu['n_units']} customers p={pcu['p']:.3f}, across "
                f"{po['n_units']} orders p={po['p']:.3f} ({N_PERM} shuffles); "
                f"chi-square on order-line counts χ²={test['chi2']:.0f}, "
                f"df={test['dof']}, p={test['p']:.2g}, Cramér's "
                f"V={test['cramers_v']:.2f} ({test['effect']}). "
                "Regions: " + "; ".join(f"{r} = {', '.join(c)}"
                                        for r, c in REGIONS.items()) + ".")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return ps.save(fig, "q2a_region_category_heatmap")


# -------------------------------------------------------- (b) consolidation
def combinable(orders: pd.DataFrame, key: list[str], days: int,
               date_col: str = "OrderDate") -> pd.Series:
    """Flag orders with another order to the same key nearby.

    Args:
        orders (pandas.DataFrame): Orders with the ``key`` columns and
            ``date_col``.
        key (list[str]): Columns defining the destination.
        days (int): Half-width of the window in days.
        date_col (str): Date column to compare. Default "OrderDate".

    Returns:
        pandas.Series: Boolean, True if another order with the same key
            lies within ``days`` days; indexed like ``orders``.
    """
    o = orders.sort_values(key + [date_col])
    out = pd.Series(False, index=o.index)
    for _, g in o.groupby(key):
        d = g[date_col].values.astype("datetime64[D]").astype(np.int64)
        if len(d) < 2:
            continue
        gap_prev = np.r_[np.inf, np.diff(d)]
        gap_next = np.r_[np.diff(d), np.inf]
        out.loc[g.index] = np.minimum(gap_prev, gap_next) <= days
    return out.reindex(orders.index)


def greedy_batches(orders: pd.DataFrame, key: list[str], days: int):
    """Group each destination's orders into batches by date.

    A batch opens at an order and takes every later order placed
    within ``days`` of the opener.

    Args:
        orders (pandas.DataFrame): Orders with the ``key`` columns and
            ``OrderDate``.
        key (list[str]): Columns defining the destination.
        days (int): Maximum days from the batch opener.

    Returns:
        pandas.Series: Integer batch id for each order, unique across
            destinations; indexed like ``orders``.
    """
    o = orders.sort_values(key + ["OrderDate"])
    batch = pd.Series(-1, index=o.index)
    bid = 0
    for _, g in o.groupby(key):
        start = None
        for idx, dt in g["OrderDate"].items():
            if start is None or (dt - start).days > days:
                bid += 1
                start = dt
            batch[idx] = bid
    return batch.reindex(orders.index)


def consolidation(orders: pd.DataFrame):
    """Estimate how many shipments same-city batching could save.

    Args:
        orders (pandas.DataFrame): One row per order with destination,
            dates, ``Region``, ``Freight`` and ``is_shipped``.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame,
        dict]: Share of combinable orders for each window of 1-14
            days; per-region stats; per-city stats; and a summary dict
            of counts (combinable orders, batches, feasibility of
            batches against RequiredDate, freight totals).
    """
    city = ["ShipCountry", "ShipCity"]
    cust = ["CustomerID", "ShipCountry", "ShipCity"]
    grid = pd.DataFrame({
        "window_days": WINDOW_GRID,
        "same_city_share": [combinable(orders, city, n).mean()
                            for n in WINDOW_GRID],
        "same_customer_share": [combinable(orders, cust, n).mean()
                                for n in WINDOW_GRID],
    })

    o = orders.copy()
    o["combinable_city"] = combinable(o, city, WINDOW_DAYS)
    o["combinable_customer"] = combinable(o, cust, WINDOW_DAYS)
    o["batch"] = greedy_batches(o, city, WINDOW_DAYS)
    bsize = o.groupby("batch")["OrderID"].transform("size")

    # Feasibility check on shipped orders: if a batch went out on the date
    # its LAST member actually shipped, would every member still meet its
    # RequiredDate?
    sh = o[o["is_shipped"]].copy()
    sh["batch_ship"] = sh.groupby("batch")["ShippedDate"].transform("max")
    sh["batch_req"] = sh.groupby("batch")["RequiredDate"].transform("min")
    multi = sh[sh.groupby("batch")["OrderID"].transform("size") > 1]
    feas = multi.groupby("batch").apply(
        lambda g: (g["batch_ship"] <= g["batch_req"]).all())

    shipped_check = combinable(o[o["is_shipped"]], city, WINDOW_DAYS,
                               "ShippedDate").mean()

    by_region = o.groupby("Region").agg(
        orders=("OrderID", "size"),
        combinable_same_city=("combinable_city", "mean"),
        combinable_same_customer=("combinable_customer", "mean"),
        shipments_after_batching=("batch", "nunique"),
        freight_of_combinable=("Freight",
                               lambda f: f[o.loc[f.index, "combinable_city"]].sum()),
    ).reindex(REGION_ORDER)
    by_region["shipment_reduction"] = 1 - (by_region["shipments_after_batching"]
                                           / by_region["orders"])
    by_city = (o.groupby(city).agg(orders=("OrderID", "size"),
                                   customers=("CustomerID", "nunique"),
                                   combinable=("combinable_city", "sum"),
                                   batches=("batch", "nunique"))
               .sort_values("combinable", ascending=False))
    summary = {
        "orders": len(o),
        "destinations": o.groupby(city).ngroups,
        "combinable_city": int(o["combinable_city"].sum()),
        "combinable_customer": int(o["combinable_customer"].sum()),
        "batches": int(o["batch"].nunique()),
        "orders_in_multi_batches": int((bsize > 1).sum()),
        "multi_batches_shipped": int(len(feas)),
        "multi_batches_feasible": int(feas.sum()),
        "shipped_date_check": shipped_check,
        "freight_combinable": o.loc[o["combinable_city"], "Freight"].sum(),
        "freight_total": o["Freight"].sum(),
    }
    return grid, by_region, by_city, summary


def plot_consolidation(grid: pd.DataFrame, summary: dict) -> str:
    """Plot combinable-order share against window length.

    Args:
        grid (pandas.DataFrame): Window grid from ``consolidation``.
        summary (dict): Summary counts from ``consolidation``.

    Returns:
        str: Path of the saved q2b PNG.
    """
    fig, ax = ps.plt.subplots(figsize=(9, 5.2))
    x = grid["window_days"]
    for col, color, name in [("same_city_share", ps.SERIES[0],
                              "Same destination city (any customer)"),
                             ("same_customer_share", ps.SERIES[1],
                              "Same customer and city")]:
        ax.plot(x, grid[col], color=color, marker="o", ms=4, label=name)
        y_end = grid[col].iloc[-1]
        ax.annotate(f"{y_end:.0%}", (x.iloc[-1], y_end), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9,
                    color=ps.INK_2)
    ax.axvline(WINDOW_DAYS, color=ps.AXIS, lw=1)
    r7 = grid.set_index("window_days").loc[WINDOW_DAYS]
    for col in ["same_city_share", "same_customer_share"]:
        ax.annotate(f"{r7[col]:.0%}", (WINDOW_DAYS, r7[col]), xytext=(-8, 6),
                    textcoords="offset points", ha="right", fontsize=9,
                    color=ps.INK, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xlim(0.5, 15)
    ax.set_ylim(0, max(0.5, grid["same_city_share"].max() * 1.15))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Window: another order placed within ± N days (days)")
    ax.set_ylabel("Share of all 830 orders that could be combined (%)")
    ax.set_title(f"Within {WINDOW_DAYS} days, {r7['same_city_share']:.0%} of "
                 "orders share a destination city with another order")
    ps.subtitle(ax, "An order counts as combinable if a second order to the "
                "same ship city was placed within the window (by OrderDate).")
    ax.legend(loc="upper left")
    ps.footnote(fig, f"Greedy {WINDOW_DAYS}-day batching would cut "
                f"{summary['orders']} shipments to {summary['batches']}. "
                "Theoretical upper bound: ignores warehouse timing, "
                "carrier capacity and exact street address.")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return ps.save(fig, "q2b_consolidation_window")


# ---------------------------------------------------- (c) under-served
def underserved(nw, share: pd.DataFrame):
    """Identify customers who order rarely or never.

    Under-served means zero orders, or at most ``cutoff`` orders, where
    ``cutoff`` is the 25th percentile (truncated) of order counts among
    customers with orders.

    Args:
        nw (Northwind): Cleaned data from ``load_clean``.
        share (pandas.DataFrame): Category revenue shares by region,
            used to suggest categories.

    Returns:
        tuple[DataFrame, DataFrame, DataFrame, int]:
            All customers with order stats and status; under-served
            customers with suggested categories; per-region summary;
            and the order-count cutoff.
    """
    o, ln, cu = nw.orders, nw.lines, nw.customers
    end = o["OrderDate"].max()
    per = o.groupby("CustomerID").agg(
        n_orders=("OrderID", "size"), revenue=("order_revenue", "sum"),
        first_order=("OrderDate", "min"), last_order=("OrderDate", "max"))
    c = cu[["CustomerID", "CompanyName", "City", "Country", "Region"]].merge(
        per, on="CustomerID", how="left")
    c["n_orders"] = c["n_orders"].fillna(0).astype(int)
    c["revenue"] = c["revenue"].fillna(0.0)
    c["days_since_last_order"] = (end - c["last_order"]).dt.days
    active = c[c["n_orders"] > 0]
    cutoff = int(active["n_orders"].quantile(0.25))
    c["status"] = np.select([c["n_orders"] == 0, c["n_orders"] <= cutoff],
                            ["zero orders", f"low (1–{cutoff} orders)"], "ok")

    # Categories each customer already buys.
    bought = ln.groupby("CustomerID")["CategoryName"].agg(set)
    # Best-selling product per (region, category), by revenue.
    best = (ln.groupby(["Region", "CategoryName", "ProductName"])["revenue"]
              .sum().reset_index()
              .sort_values("revenue", ascending=False)
              .drop_duplicates(["Region", "CategoryName"])
              .set_index(["Region", "CategoryName"])["ProductName"])

    def suggest(r):
        """Suggest categories for one under-served customer.

        Args:
            r (pandas.Series): Customer row with ``Region`` and
                ``CustomerID``.

        Returns:
            str: Up to three categories ranked by regional share that
                the customer has not bought (the region's top three if
                none), each with the region's best-selling product.
        """
        ranked = share.loc[r.Region].sort_values(ascending=False).index
        have = bought.get(r.CustomerID, set())
        picks = [k for k in ranked if k not in have][:3] or list(ranked[:3])
        return "; ".join(f"{k} (e.g. {best.get((r.Region, k), 'n/a')})"
                         for k in picks)

    c["region_top3_categories"] = c["Region"].map(
        lambda r: ", ".join(share.loc[r].sort_values(ascending=False)
                            .index[:3]))
    c["categories_bought"] = c["CustomerID"].map(
        lambda k: len(bought.get(k, set())))
    us = c[c["status"] != "ok"].copy()
    us["suggested_categories"] = us.apply(suggest, axis=1)
    us = us.sort_values(["Region", "n_orders", "revenue"])

    reg = c.groupby("Region").agg(
        customers=("CustomerID", "size"),
        zero_orders=("status", lambda s: (s == "zero orders").sum()),
        low_orders=("status", lambda s: s.str.startswith("low").sum()),
        region_revenue=("revenue", "sum"),
    ).reindex(REGION_ORDER)
    reg["underserved"] = reg["zero_orders"] + reg["low_orders"]
    reg["underserved_share"] = reg["underserved"] / reg["customers"]
    reg["underserved_revenue_share"] = (
        us.groupby("Region")["revenue"].sum().reindex(REGION_ORDER).fillna(0)
        / reg["region_revenue"])
    reg["top3_categories"] = [", ".join(share.loc[r].sort_values(
        ascending=False).index[:3]) for r in REGION_ORDER]
    return c, us, reg, cutoff


def plot_underserved(reg: pd.DataFrame, cutoff: int) -> str:
    """Plot under-served customer counts by region.

    Args:
        reg (pandas.DataFrame): Per-region summary from ``underserved``.
        cutoff (int): Maximum orders counted as low, used in labels.

    Returns:
        str: Path of the saved q2c PNG.
    """
    fig, ax = ps.plt.subplots(figsize=(9, 4.8))
    r = reg.sort_values(["underserved_share", "underserved"])
    y = np.arange(len(r))
    ax.barh(y, r["zero_orders"], color=ps.SERIES[1], height=0.6,
            label="Zero orders", edgecolor=ps.SURFACE, linewidth=1.5)
    ax.barh(y, r["low_orders"], left=r["zero_orders"], color=ps.SERIES[0],
            height=0.6, label=f"1–{cutoff} orders (bottom quartile)",
            edgecolor=ps.SURFACE, linewidth=1.5)
    for yi, (_, row) in zip(y, r.iterrows()):
        ax.text(row["underserved"] + 0.15, yi,
                f"{int(row['underserved'])} of {int(row['customers'])} "
                f"customers ({row['underserved_share']:.0%})",
                va="center", fontsize=9, color=ps.INK_2)
    ax.set_yticks(y)
    ax.set_yticklabels(r.index)
    ax.set_xlim(0, r["underserved"].max() + 4)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Under-served customers (count)")
    top = r.index[-1]
    ax.set_title(f"{top} has the highest share of under-served customers "
                 f"({int(r.loc[top, 'underserved'])} of "
                 f"{int(r.loc[top, 'customers'])})")
    ps.subtitle(ax, f"Under-served = zero orders, or {cutoff} or fewer orders "
                "in Jul 1996–May 1998 (bottom quartile of active customers).")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return ps.save(fig, "q2c_underserved_by_region")


def main() -> dict:
    """Run the Q2 regional analysis and write its tables and figures.

    Args:
        None.

    Returns:
        dict: ``summary`` (consolidation counts) and ``test`` (category
            mix tests). Also writes CSVs and PNGs and prints checks.
    """
    ensure_output_dirs()
    ps.apply_style()
    nw = load_clean()

    rev, share, test, top_share = category_mix(nw.lines)
    grid, by_region, by_city, summ = consolidation(nw.orders)
    cust, us, reg, cutoff = underserved(nw, share.loc[REGION_ORDER])

    share.to_csv(OUT_TAB / "q2a_region_category_share.csv", float_format="%.4f")
    rev.to_csv(OUT_TAB / "q2a_region_category_revenue.csv", float_format="%.2f")
    grid.to_csv(OUT_TAB / "q2b_consolidation_by_window.csv", index=False,
                float_format="%.4f")
    by_region.to_csv(OUT_TAB / "q2b_consolidation_by_region.csv",
                     float_format="%.4f")
    by_city.to_csv(OUT_TAB / "q2b_consolidation_by_city.csv")
    us_cols = ["Region", "Country", "CustomerID", "CompanyName", "City",
               "status", "n_orders", "revenue", "last_order",
               "days_since_last_order", "categories_bought",
               "region_top3_categories", "suggested_categories"]
    us[us_cols].to_csv(OUT_TAB / "q2c_underserved_customers.csv", index=False,
                       float_format="%.2f", date_format="%Y-%m-%d")
    reg.to_csv(OUT_TAB / "q2c_underserved_by_region.csv", float_format="%.4f")

    f1 = plot_heatmap(share, rev, test)
    f2 = plot_consolidation(grid, summ)
    f3 = plot_underserved(reg, cutoff)

    # --------------------------------------------------------- printed check
    print("\n" + "=" * 78 + "\nQ2  REGIONS\n" + "=" * 78)
    for r, cs in REGIONS.items():
        print(f"  {r:16s}: {', '.join(cs)}")
    print(f"\n(a) Revenue by region sums to {rev.values.sum():,.2f} "
          f"(matches total: {np.isclose(rev.values.sum(), nw.lines['revenue'].sum())})")
    print("Revenue share by category (row %):")
    print((share * 100).round(1).to_string())
    dev = (share.loc[REGION_ORDER] - share.loc["All regions"]) * 100
    big = dev.stack().sort_values()
    print("Largest deviations from overall mix (percentage points):")
    print("  below: " + ", ".join(f"{r}/{k} {v:+.1f}" for (r, k), v in big.head(4).items()))
    print("  above: " + ", ".join(f"{r}/{k} {v:+.1f}" for (r, k), v in big.tail(4).items()))
    print(f"Chi-square (line counts): chi2={test['chi2']:.1f}, df={test['dof']}, "
          f"p={test['p']:.3g}, Cramer's V={test['cramers_v']:.3f} ({test['effect']}), "
          f"min expected count={test['min_expected']:.1f}, n={test['n_lines']}")
    for k in ["perm_order", "perm_customer"]:
        t = test[k]
        print(f"Permutation test on revenue mix, shuffling region across "
              f"{t['n_units']} {t['unit']}s: stat={t['stat']:.4f} "
              f"(null median {t['null_median']:.4f}), p={t['p']:.4f}")
    print("Largest single customer's share of region revenue: "
          + ", ".join(f"{r} {v:.0%}" for r, v in top_share.items()))

    print(f"\n(b) Consolidation ({WINDOW_DAYS}-day window, by OrderDate, "
          f"{summ['destinations']} destination cities)")
    print(f"  Orders with another order to the same city within ±{WINDOW_DAYS}d: "
          f"{summ['combinable_city']} / {summ['orders']} "
          f"({summ['combinable_city'] / summ['orders']:.1%})")
    print(f"  ...same customer and city: {summ['combinable_customer']} "
          f"({summ['combinable_customer'] / summ['orders']:.1%})")
    print(f"  Greedy batching: {summ['orders']} orders -> {summ['batches']} "
          f"shipments ({1 - summ['batches'] / summ['orders']:.1%} fewer); "
          f"{summ['orders_in_multi_batches']} orders sit in multi-order batches")
    print(f"  Feasibility (shipped orders): {summ['multi_batches_feasible']} of "
          f"{summ['multi_batches_shipped']} multi-order batches would still "
          "meet every member's RequiredDate if shipped on the last member's "
          "actual ShippedDate")
    print(f"  Check using ShippedDate instead (809 shipped orders): "
          f"{summ['shipped_date_check']:.1%} combinable")
    print(f"  Freight on combinable orders: {summ['freight_combinable']:,.0f} of "
          f"{summ['freight_total']:,.0f} total (no cost model, so no savings "
          "estimate)")
    print(by_region.round(3).to_string())
    print("Top cities:", ", ".join(
        f"{c[1]} ({int(r.combinable)}/{int(r.orders)})"
        for c, r in by_city.head(6).iterrows()))

    print(f"\n(c) Under-served customers (customer Country -> region); "
          f"bottom-quartile cut-off = {cutoff} orders")
    print(f"  Customers: {len(cust)}; zero orders: "
          f"{(cust['status'] == 'zero orders').sum()} "
          f"({', '.join(cust.loc[cust['status'] == 'zero orders', 'CustomerID'])}); "
          f"under-served total: {len(us)}")
    print(reg.drop(columns=["region_revenue"]).round(3).to_string())
    print(us[["Region", "CustomerID", "Country", "n_orders", "revenue",
              "days_since_last_order", "suggested_categories"]]
          .to_string(index=False, max_colwidth=70))
    print("CAVEAT: order counts are not adjusted for when a customer started "
          "buying, and customer size/potential is unknown - 'under-served' "
          "means 'buys rarely', not 'has unmet demand'.")
    print(f"Wrote {f1}\nWrote {f2}\nWrote {f3}")
    return {"summary": summ, "test": test}


if __name__ == "__main__":
    main()
