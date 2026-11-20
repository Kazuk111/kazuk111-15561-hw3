"""Regenerate every number, table, figure and the webpage from the workbook.

Usage (from the project folder):  python src/run_all.py
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd

import build_site
import inspect_data
import q1_markets
import q2_regions
import q3_shipping
import q4_discounts
from load_data import OUT_FIG, OUT_TAB, PARTIAL_QUARTER, ensure_output_dirs, load_clean


def clean_outputs() -> None:
    """Delete previously generated PNG and CSV outputs.

    Args:
        None.

    Returns:
        None. Removes ``*.png`` from figures and ``*.csv`` from tables.
    """
    ensure_output_dirs()
    for f in list(OUT_FIG.glob("*.png")) + list(OUT_TAB.glob("*.csv")):
        f.unlink()


def verify() -> None:
    """Cross-check revenue and order totals and print a summary.

    Reads the saved Q1 and Q2 tables and compares them with totals
    recomputed from the raw data.

    Args:
        None.

    Returns:
        None. Prints each check and the excluded rows.

    Raises:
        SystemExit: If any check fails.
    """
    nw = load_clean()
    o, ln = nw.orders, nw.lines
    q1 = pd.read_csv(OUT_TAB / "q1_market_classification.csv", index_col=0)
    q2 = pd.read_csv(OUT_TAB / "q2a_region_category_revenue.csv", index_col=0)

    rev_lines = ln["revenue"].sum()
    rev_manual = (ln["Quantity"] * ln["UnitPrice"] * (1 - ln["Discount"])).sum()
    rev_orders = o["order_revenue"].sum()
    checks = [
        ("Revenue: line-level vs order-level", rev_lines, rev_orders),
        ("Revenue: stored column vs Qty*Price*(1-Disc)", rev_lines, rev_manual),
        ("Revenue: Q1 country table vs lines", q1["total_revenue"].sum(), rev_lines),
        ("Revenue: Q2 region x category vs lines", q2.values.sum(), rev_lines),
        ("Orders: Q1 country table sums to 830", q1["orders_total"].sum(), 830),
        ("Orders: ShipCountry groups sum to 830",
         o.groupby("ShipCountry").size().sum(), 830),
        ("Lines: 2155 order lines", len(ln), 2155),
    ]
    print("\n" + "=" * 78 + "\nVERIFICATION SUMMARY\n" + "=" * 78)
    ok_all = True
    for name, a, b in checks:
        ok = bool(np.isclose(a, b))
        ok_all &= ok
        print(f"  [{'OK' if ok else 'FAIL':4s}] {name}: {a:,.2f} vs {b:,.2f}")
    print("\n  Excluded / flagged rows:")
    print(f"    Q1: {int(o['is_partial_quarter'].sum())} orders in {PARTIAL_QUARTER} "
          "(partial quarter) left out of growth windows; still in market size")
    print(f"    Q2: {int((~nw.customers['CustomerID'].isin(o['CustomerID'])).sum())} "
          "customers with zero orders kept in the under-served list")
    print(f"    Q3: {int((~o['is_shipped']).sum())} unshipped orders excluded "
          "from slack/late analysis")
    print(f"    Q4: {int((~ln['is_standard_discount']).sum())} order lines with "
          "non-standard discounts excluded from discount-level comparisons")
    print(f"\n  Figures written: {len(list(OUT_FIG.glob('*.png')))}   "
          f"Tables written: {len(list(OUT_TAB.glob('*.csv')))}")
    if not ok_all:
        raise SystemExit("A verification check failed - see above.")


def main() -> None:
    """Run every analysis step, the verification, then build the site.

    Args:
        None.

    Returns:
        None. Prints elapsed time when finished.
    """
    t0 = time.time()
    clean_outputs()
    for step in [inspect_data, q1_markets, q2_regions, q3_shipping,
                 q4_discounts]:
        step.main()
    verify()
    build_site.main()
    print(f"\nDone in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
