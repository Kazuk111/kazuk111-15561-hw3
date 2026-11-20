"""Q1. Which markets are growing or declining, and where should sales focus?

Market = ShipCountry (see load_data.MARKET_COL). Growth is year-over-year on
matching calendar months so seasonality cancels out and the partial 1998 Q2 is
never used:
  primary   : Jul 1996-Mar 1997  vs  Jul 1997-Mar 1998   (9 months each)
  check (1) : Oct 1996-Mar 1997  vs  Oct 1997-Mar 1998   (6 months each)
  check (2) : 1997 Q1            vs  1998 Q1             (3 months each)
  check (3) : primary windows, but market = the customer's own Country
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter,
                               NullFormatter, PercentFormatter)

import plot_style as ps
from load_data import OUT_TAB, PARTIAL_QUARTER, ensure_output_dirs, load_clean

WINDOWS = {
    "9m": (("1996-07-01", "1997-03-31"), ("1997-07-01", "1998-03-31")),
    "6m": (("1996-10-01", "1997-03-31"), ("1997-10-01", "1998-03-31")),
    "q1": (("1997-01-01", "1997-03-31"), ("1998-01-01", "1998-03-31")),
}
MIN_ORDERS = 5        # fewer orders than this in either window -> low confidence
TOP_N_LINES = 8
ACTIONS = ["Invest", "Maintain", "Review"]
ACTION_COLOR = dict(zip(ACTIONS, ps.SERIES[:3]))


def window_stats(orders: pd.DataFrame, market: str, key: str) -> pd.DataFrame:
    """Compare per-market revenue between two matching windows.

    Growth is +inf when a market has revenue only in the later window
    and NaN when it has none in either.

    Args:
        orders (pandas.DataFrame): One row per order, with
            ``OrderDate``, ``order_revenue``, ``CustomerID`` and
            the ``market`` column.
        market (str): Column to group by (e.g. "ShipCountry").
        key (str): Window pair in ``WINDOWS``: "9m", "6m" or "q1".

    Returns:
        tuple[pandas.DataFrame, float]: Per-market table (rev_before,
            rev_after, orders_before, orders_after, customers_after,
            growth) and the company-wide growth rate.
    """
    (a0, a1), (b0, b1) = WINDOWS[key]
    before = orders[orders["OrderDate"].between(a0, a1)]
    after = orders[orders["OrderDate"].between(b0, b1)]
    out = pd.DataFrame({
        "rev_before": before.groupby(market)["order_revenue"].sum(),
        "rev_after": after.groupby(market)["order_revenue"].sum(),
        "orders_before": before.groupby(market).size(),
        "orders_after": after.groupby(market).size(),
        "customers_after": after.groupby(market)["CustomerID"].nunique(),
    }).fillna(0)
    # No revenue in the earlier window -> +inf (new market) or NaN (none).
    ratio = out["rev_after"] / out["rev_before"].replace(0, np.nan) - 1
    out["growth"] = np.where(out["rev_before"] > 0, ratio,
                             np.where(out["rev_after"] > 0, np.inf, np.nan))
    company = after["order_revenue"].sum() / before["order_revenue"].sum() - 1
    return out, company


def classify(size: pd.Series, growth: pd.Series, company_growth: float,
             size_cut: float) -> pd.DataFrame:
    """Assign each market a quadrant and an action.

    Action is Invest if growth >= company growth, else Maintain if
    the market is large and growth >= 0, else Review.

    Args:
        size (pandas.Series): Total revenue per market.
        growth (pandas.Series): Growth rate per market, same index.
        company_growth (float): Company-wide growth rate.
        size_cut (float): Revenue at or above which a market is large.

    Returns:
        pandas.DataFrame: ``quadrant`` and ``action`` columns, indexed
            like ``size``.
    """
    large = size >= size_cut
    outgrowing = growth >= company_growth
    quadrant = np.select(
        [large & outgrowing, ~large & outgrowing, large & ~outgrowing],
        ["Large & outgrowing", "Small & outgrowing", "Large & lagging"],
        "Small & lagging")
    action = np.select([outgrowing, large & (growth >= 0)],
                       ["Invest", "Maintain"], "Review")
    return pd.DataFrame({"quadrant": quadrant, "action": action},
                        index=size.index)


def build_tables(nw):
    """Build the Q1 quarterly and market-classification tables.

    Args:
        nw (Northwind): Cleaned data from ``load_clean``.

    Returns:
        tuple[pandas.DataFrame, pandas.DataFrame, float, float]:
            Quarterly revenue by ship country (1998Q2 labelled
            partial); per-country table with size, 9-month growth,
            action, confidence flags, sensitivity actions and caveats,
            sorted by revenue; company-wide 9-month growth; and the
            median market revenue used as the size cut.
    """
    o = nw.orders
    mk = "ShipCountry"

    # --- quarterly revenue by country (1998Q2 kept but labelled partial) ----
    q = o.pivot_table(index=mk, columns="OrderQuarter", values="order_revenue",
                      aggfunc="sum", fill_value=0.0)
    q.columns = [f"{c} (partial)" if c == PARTIAL_QUARTER else str(c)
                 for c in q.columns]

    # --- size + primary growth --------------------------------------------
    size = o.groupby(mk)["order_revenue"].sum().rename("total_revenue")
    stats, company = window_stats(o, mk, "9m")
    size_cut = size.median()
    cls = classify(size, stats["growth"], company, size_cut)

    t = pd.concat([size, stats, cls], axis=1)
    t["revenue_share"] = t["total_revenue"] / t["total_revenue"].sum()
    t["orders_total"] = o.groupby(mk).size()
    t["customers_total"] = o.groupby(mk)["CustomerID"].nunique()
    t["low_confidence"] = (t[["orders_before", "orders_after"]].min(axis=1)
                           < MIN_ORDERS)
    t["concentrated"] = t["customers_after"] <= 2

    # --- sensitivity checks -----------------------------------------------
    for key in ["6m", "q1"]:
        s, c = window_stats(o, mk, key)
        t[f"growth_{key}"] = s["growth"]
        t[f"action_{key}"] = classify(size, s["growth"].fillna(-np.inf), c,
                                      size_cut)["action"]
    cust_size = o.groupby("CustomerCountry")["order_revenue"].sum()
    s, c = window_stats(o, "CustomerCountry", "9m")
    t["action_customer_country"] = classify(cust_size, s["growth"], c,
                                            cust_size.median())["action"]

    def note(r):
        """Describe the caveats that apply to one country.

        Args:
            r (pandas.Series): Row of the country table with the
                confidence flags, order and customer counts, ``action``
                and the sensitivity ``action_*`` columns.

        Returns:
            str: Caveats joined by "; ", or "" if there are none.
        """
        bits = []
        if r.low_confidence:
            bits.append(f"only {int(min(r.orders_before, r.orders_after))} "
                        f"orders in one window: growth % is unreliable")
        if r.concentrated:
            bits.append(f"{int(r.customers_after)} active customer(s): "
                        f"growth hinges on very few accounts")
        changed = [k for k in ["action_6m", "action_q1", "action_customer_country"]
                   if r[k] != r.action]
        if changed:
            bits.append("action changes under " + ", ".join(
                k.replace("action_", "") for k in changed))
        return "; ".join(bits)

    t["caveats"] = t.apply(note, axis=1)
    t = t.sort_values("total_revenue", ascending=False)
    t.index.name = "country"
    return q, t, company, size_cut


def plot_lines(q: pd.DataFrame, t: pd.DataFrame) -> str:
    """Plot quarterly revenue for the top markets as small multiples.

    Args:
        q (pandas.DataFrame): Quarterly revenue by country, from
            ``build_tables``; partial quarters are omitted.
        t (pandas.DataFrame): Country table sorted by revenue, with
            a ``growth`` column.

    Returns:
        str: Path of the saved q1a PNG.
    """
    full = [c for c in q.columns if "partial" not in c]
    top = t.index[:TOP_N_LINES]
    data = q.loc[:, full]
    x = np.arange(len(full))
    fig, axes = ps.plt.subplots(2, 4, figsize=(13, 6.2), sharex=True,
                                sharey=True)
    for ax, country in zip(axes.flat, top):
        for other in top:
            if other != country:
                ax.plot(x, data.loc[other], color=ps.NEUTRAL, lw=1, alpha=0.6)
        ax.plot(x, data.loc[country], color=ps.SERIES[0], lw=2.2,
                marker="o", ms=4)
        g = t.loc[country, "growth"]
        ax.set_title(country, fontsize=11, pad=4)
        ax.text(0.03, 0.93, f"YoY {g:+.0%}", transform=ax.transAxes,
                fontsize=8.5, color=ps.INK_2, va="top")
        ax.yaxis.set_major_formatter(FuncFormatter(ps.money))
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("Q", "\nQ") for c in full], fontsize=7.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("Revenue per quarter (USD)")
    n_peak = int((data.loc[top].idxmax(axis=1) == full[-1]).sum())
    fig.suptitle(f"{n_peak} of the top {len(top)} markets had their best "
                 f"quarter in {full[-1]}",
                 x=0.01, ha="left", fontsize=14, fontweight="bold", y=1.015)
    fig.text(0.01, 0.948,
             "Quarterly discounted revenue by ship country, top 8 by total "
             "revenue. Blue = named country, grey = the other seven. "
             "YoY = Jul 97–Mar 98 vs Jul 96–Mar 97.",
             fontsize=9.5, color=ps.INK_2, ha="left")
    ps.footnote(fig, "1998 Q2 omitted: data ends 1998-05-06 (36 of 91 days). "
                "Revenue = Quantity × UnitPrice × (1 − Discount).")
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    return ps.save(fig, "q1a_quarterly_revenue_top8")


def plot_quadrants(t: pd.DataFrame, company: float, size_cut: float) -> str:
    """Plot market size against growth with action colors.

    Args:
        t (pandas.DataFrame): Country table with ``total_revenue``,
            ``growth``, ``action`` and ``low_confidence``.
        company (float): Company-wide growth, drawn as a line.
        size_cut (float): Median market revenue, drawn as a line.

    Returns:
        str: Path of the saved q1b PNG.
    """
    fig, ax = ps.plt.subplots(figsize=(10, 7))
    y_cap = 3.2
    ys = t["growth"].clip(upper=y_cap)
    for action in ACTIONS:
        sub = t[t["action"] == action]
        for low, face in [(False, ACTION_COLOR[action]), (True, ps.SURFACE)]:
            s = sub[sub["low_confidence"] == low]
            capped = s["growth"] > y_cap
            ax.scatter(s.loc[~capped, "total_revenue"], ys[s.index][~capped],
                       s=70, facecolor=face, edgecolor=ACTION_COLOR[action],
                       linewidth=1.8, zorder=3)
            ax.scatter(s.loc[capped, "total_revenue"], ys[s.index][capped],
                       s=90, marker="^", facecolor=face,
                       edgecolor=ACTION_COLOR[action], linewidth=1.8, zorder=3)
    ax.set_xscale("log")
    ax.axvline(size_cut, color=ps.AXIS, lw=1, zorder=1)
    ax.axhline(company, color=ps.AXIS, lw=1, zorder=1)
    ax.axhline(0, color=ps.INK_2, lw=0.8, zorder=1)
    ax.set_ylim(-0.95, y_cap + 0.45)
    ax.set_xlim(2500, 400000)
    kw = dict(fontsize=9, color=ps.MUTED, fontweight="bold")
    quads = [
        ax.text(0.985, 0.985, "LARGE & OUTGROWING → invest",
                transform=ax.transAxes, ha="right", va="top", **kw),
        ax.text(0.015, 0.985, "SMALL & OUTGROWING → invest",
                transform=ax.transAxes, ha="left", va="top", **kw),
        ax.text(0.985, 0.015, "LARGE & LAGGING → maintain\n(review if declining)",
                transform=ax.transAxes, ha="right", va="bottom", **kw),
        ax.text(0.015, 0.015, "SMALL & LAGGING → review",
                transform=ax.transAxes, ha="left", va="bottom", **kw),
        ax.text(3.6e5, company + 0.04, f"company-wide growth {company:+.0%}",
                fontsize=8, color=ps.MUTED, ha="right"),
        ax.text(size_cut * 1.04, -0.62, f"median market\n{ps.money(size_cut)}",
                fontsize=8, color=ps.MUTED, ha="left"),
    ]
    labels = [f"{c} ({g:+.0%})" if g > y_cap else c
              for c, g in zip(t.index, t["growth"])]
    ps.place_labels(ax, t["total_revenue"].values, ys.values, labels,
                    fontsize=8.5, obstacles=quads)
    ax.xaxis.set_major_locator(FixedLocator([5e3, 1e4, 2e4, 5e4, 1e5, 2e5]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(FuncFormatter(ps.money))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Market size: total revenue Jul 1996–May 1998 (USD, log scale)")
    ax.set_ylabel("Revenue growth, Jul 97–Mar 98 vs Jul 96–Mar 97 (%)")
    large = t["total_revenue"] >= size_cut
    n_lo = int((large & (t["growth"] >= company)).sum())
    n_dec = int((t["growth"] < 0).sum())
    ax.set_title(f"{n_lo} of {int(large.sum())} large markets outgrew the "
                 f"company; {n_dec} markets shrank year over year")
    ps.subtitle(ax, "Each dot is a ship country. Quadrant lines: median market "
                "size and company-wide year-over-year growth.")
    handles = [Line2D([], [], marker="o", ls="", ms=8, mfc=ACTION_COLOR[a],
                      mec=ACTION_COLOR[a], label=a) for a in ACTIONS]
    handles += [Line2D([], [], marker="o", ls="", ms=8, mfc=ps.SURFACE,
                       mec=ps.INK_2, label=f"fewer than {MIN_ORDERS} orders in a "
                       "window (low confidence)"),
                Line2D([], [], marker="^", ls="", ms=8, mfc=ps.SURFACE,
                       mec=ps.INK_2, label=f"growth above {y_cap:.0%}, "
                       "drawn at the top edge")]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.1),
              ncol=3, fontsize=8.5)
    return ps.save(fig, "q1b_market_size_vs_growth")


def main() -> dict:
    """Run the Q1 market analysis and write its tables and figures.

    Args:
        None.

    Returns:
        dict: ``company_growth`` (float) and ``table`` (the country
            table). Also writes two CSVs and two PNGs and prints checks.
    """
    ensure_output_dirs()
    ps.apply_style()
    nw = load_clean()
    q, t, company, size_cut = build_tables(nw)

    q.to_csv(OUT_TAB / "q1_quarterly_revenue_by_country.csv",
             float_format="%.2f")
    cols = ["action", "quadrant", "total_revenue", "revenue_share",
            "orders_total", "customers_total", "rev_before", "rev_after",
            "orders_before", "orders_after", "customers_after", "growth",
            "growth_6m", "growth_q1", "action_6m", "action_q1",
            "action_customer_country", "low_confidence", "concentrated",
            "caveats"]
    out = t[cols].rename(columns={
        "rev_before": "rev_jul96_mar97", "rev_after": "rev_jul97_mar98",
        "orders_before": "orders_jul96_mar97",
        "orders_after": "orders_jul97_mar98",
        "customers_after": "customers_jul97_mar98",
        "growth": "growth_yoy_9m", "growth_6m": "growth_yoy_6m_octmar",
        "growth_q1": "growth_yoy_q1"})
    out.to_csv(OUT_TAB / "q1_market_classification.csv", float_format="%.4f")

    f1 = plot_lines(q, t)
    f2 = plot_quadrants(t, company, size_cut)

    # --------------------------------------------------------- printed check
    print("\n" + "=" * 78 + "\nQ1  MARKETS (market = ShipCountry)\n" + "=" * 78)
    total = t["total_revenue"].sum()
    print(f"Total revenue across countries: {total:,.2f} "
          f"(matches line-level total: "
          f"{np.isclose(total, nw.lines['revenue'].sum())})")
    print(f"Orders across countries: {t['orders_total'].sum()} (expect 830)")
    q_tot = q.sum()
    print("Quarterly revenue, all countries: "
          + ", ".join(f"{k} {v:,.0f}" for k, v in q_tot.items()))
    print(f"Company-wide YoY growth (Jul97-Mar98 vs Jul96-Mar97): {company:+.1%}")
    print(f"Median market size (quadrant split): {size_cut:,.0f}")
    show = t[["total_revenue", "revenue_share", "orders_before",
              "orders_after", "growth", "growth_6m", "growth_q1", "action",
              "low_confidence", "concentrated"]].copy()
    show["total_revenue"] = show["total_revenue"].map("{:,.0f}".format)
    for c in ["revenue_share", "growth", "growth_6m", "growth_q1"]:
        show[c] = show[c].map(lambda v: "n/a" if pd.isna(v) else f"{v:+.0%}"
                              if c != "revenue_share" else f"{v:.1%}")
    print(show.to_string())
    print("\nActions:", t["action"].value_counts().to_dict())
    for k in ["action_6m", "action_q1", "action_customer_country"]:
        diff = t.index[t[k] != t["action"]].tolist()
        print(f"Sensitivity {k:26s}: {len(diff)} change(s) {diff}")
    print("Low-confidence countries:", t.index[t["low_confidence"]].tolist())
    print("Concentrated (<=2 active customers):",
          t.index[t["concentrated"]].tolist())
    print("NOTE: revenue only - there is no cost data, so this ranks markets "
          "by sales, not profitability.")
    print(f"Wrote {f1}\nWrote {f2}")
    return {"company_growth": company, "table": t}


if __name__ == "__main__":
    main()
