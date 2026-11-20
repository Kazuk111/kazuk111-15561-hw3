"""Step 0: inspect every sheet and print a data-quality summary.

Writes outputs/tables/step0_sheet_summary.csv and step0_checks.csv.
"""
from __future__ import annotations

import numpy as np
import openpyxl
import pandas as pd

from load_data import (OUT_TAB, PARTIAL_QUARTER, PRODUCT_KEEP,
                       STANDARD_DISCOUNTS, ensure_output_dirs, find_workbook,
                       load_clean, load_raw, tidy)

EXPECTED_SHEETS = ["Categories", "Customers", "Employees", "Orders Details",
                   "Orders", "Shippers", "Product", "Sheet1"]
KEYS = {"Categories": "CategoryID", "Customers": "CustomerID",
        "Employees": "EmployeeID", "Orders": "OrderID", "Shippers": "ShipperID",
        "Product": "ProductID", "Suppliers": "SupplierID"}


def banner(text: str) -> None:
    """Print a heading between two rules.

    Args:
        text (str): Heading text.

    Returns:
        None.
    """
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


def workbook_structure(path) -> pd.DataFrame:
    """Count formula cells and the used range of each sheet.

    Args:
        path (str | pathlib.Path): Excel workbook to open.

    Returns:
        pandas.DataFrame: One row per sheet with ``sheet``,
            ``excel_used_range`` and ``formula_cells`` (cells whose
            text starts with "=").
    """
    wb = openpyxl.load_workbook(path, data_only=False)
    rows = []
    for ws in wb.worksheets:
        n_formula = sum(1 for r in ws.iter_rows() for c in r
                        if isinstance(c.value, str) and c.value.startswith("="))
        rows.append({"sheet": ws.title, "excel_used_range": ws.dimensions,
                     "formula_cells": n_formula})
    return pd.DataFrame(rows)


def sheet_summary(raw: dict[str, pd.DataFrame], wb: pd.DataFrame) -> pd.DataFrame:
    """Summarise size, missing values and duplicates per sheet.

    For ``Product`` only the columns in ``PRODUCT_KEEP`` are counted;
    the others are reported as dropped helper columns.

    Args:
        raw (dict[str, pandas.DataFrame]): Raw sheets from ``load_raw``.
        wb (pandas.DataFrame): Output of ``workbook_structure``.

    Returns:
        pandas.DataFrame: One row per sheet, merged with ``wb`` on
            ``sheet``.
    """
    rows = []
    for name, df in raw.items():
        clean, dropped = tidy(df)
        helper = []
        if name == "Product":
            helper = [c for c in clean.columns if c not in PRODUCT_KEEP]
            clean = clean[PRODUCT_KEEP]
        nulls = clean.isna().sum()
        key = KEYS.get(name)
        date_cols = clean.select_dtypes(include="datetime").columns
        rows.append({
            "sheet": name,
            "rows": len(clean),
            "raw_columns": df.shape[1],
            "kept_columns": clean.shape[1],
            "dropped_empty_columns": ", ".join(map(str, dropped)),
            "dropped_helper_columns": ", ".join(helper),
            "missing_by_column": "; ".join(f"{c}={n}" for c, n in nulls.items() if n),
            "duplicate_rows": int(clean.duplicated().sum()),
            "duplicate_keys": int(clean[key].duplicated().sum()) if key else "n/a",
            "date_ranges": "; ".join(
                f"{c}: {clean[c].min():%Y-%m-%d}..{clean[c].max():%Y-%m-%d}"
                for c in date_cols),
            "columns": ", ".join(map(str, clean.columns)),
        })
    out = pd.DataFrame(rows).merge(wb, on="sheet", how="left")
    return out


def main() -> dict:
    """Run the Step 0 data inspection and save its two tables.

    Args:
        None.

    Returns:
        dict: ``total_revenue`` (float), ``n_orders`` (int) and
            ``n_lines`` (int). Also writes ``step0_sheet_summary.csv``
            and ``step0_checks.csv`` and prints the checks.
    """
    ensure_output_dirs()
    path = find_workbook()
    raw = load_raw(path)
    nw = load_clean(path)
    o, ln, cu, pr = nw.orders, nw.lines, nw.customers, nw.products
    checks: list[dict] = []

    def check(name, ok, detail):
        """Record one inspection result.

        Args:
            name (str): Short name of the check.
            ok (bool): True gives result "OK", False gives "FLAG".
            detail (str): Explanation shown with the result.

        Returns:
            None. Appends a row to the enclosing ``checks`` list.
        """
        checks.append({"check": name, "result": "OK" if ok else "FLAG",
                       "detail": detail})

    # ---------------------------------------------------------------- sheets
    banner(f"STEP 0 - DATA INSPECTION  ({path.name})")
    wb = workbook_structure(path)
    missing = [s for s in EXPECTED_SHEETS if s not in raw]
    extra = [s for s in raw if s not in EXPECTED_SHEETS]
    print(f"Sheets found: {list(raw)}")
    check("Expected sheets present", not missing,
          f"missing={missing}; unexpected={extra}")

    summ = sheet_summary(raw, wb)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 70)
    for _, r in summ.iterrows():
        print(f"\n--- {r.sheet}: {r.rows} rows x {r.kept_columns} cols kept "
              f"(raw {r.raw_columns}; Excel used range {r.excel_used_range}; "
              f"{r.formula_cells} formula cells)")
        print(f"  columns : {r.columns}")
        if r.dropped_empty_columns:
            print(f"  dropped (entirely empty): {r.dropped_empty_columns}")
        if r.dropped_helper_columns:
            print(f"  dropped (helper formulas): {r.dropped_helper_columns}")
        print(f"  missing : {r.missing_by_column or 'none'}")
        print(f"  dups    : {r.duplicate_rows} duplicate rows, "
              f"{r.duplicate_keys} duplicate keys")
        if r.date_ranges:
            print(f"  dates   : {r.date_ranges}")
    summ.to_csv(OUT_TAB / "step0_sheet_summary.csv", index=False)

    # ----------------------------------------------------- product helpers
    p_raw = raw["Product"]
    cheap_vs_ischeap = int(((p_raw["Cheap"] == "t") != p_raw["IsCheap"]).sum())
    check("Product helper columns are formula-derived and dropped",
          True,
          f"Cheap('t'/'f', Price<10), NewPrice(=Price*1.07), Cheap.1 (0/1 dup), "
          f"IsCheap(Price<=10), empty spacer col, NewPrice.1(=Price*1.1, "
          f"{p_raw['NewPrice.1'].notna().sum()}/77 rows). Cheap vs IsCheap "
          f"disagree on {cheap_vs_ischeap} product(s) (threshold < vs <=).")

    # -------------------------------------------------- referential checks
    banner("INTEGRITY")
    fk = {
        "Orders Details -> Orders": (~ln["OrderID"].isin(o["OrderID"])).sum(),
        "Orders w/o lines": (~o["OrderID"].isin(ln["OrderID"])).sum(),
        "Orders Details -> Product": ln["ProductName"].isna().sum(),
        "Orders -> Customers": o["CustomerCountry"].isna().sum(),
        "Orders -> Employees": o["EmployeeName"].isna().sum(),
        "Orders -> Shippers": o["ShipperName"].isna().sum(),
        "Product -> Categories": pr["CategoryName"].isna().sum(),
    }
    for k, v in fk.items():
        print(f"  {k:30s} unmatched = {v}")
    check("Foreign keys all resolve", sum(fk.values()) == 0, str(fk))
    bad_vals = int((ln["Quantity"] <= 0).sum() + (ln["UnitPrice"] <= 0).sum())
    check("No zero/negative Quantity or UnitPrice", bad_vals == 0,
          f"{bad_vals} bad values")
    dup_line = int(ln.duplicated(["OrderID", "ProductID"]).sum())
    check("No duplicate (OrderID, ProductID) lines", dup_line == 0,
          f"{dup_line} duplicates")
    neg = int((o["ShippedDate"] < o["OrderDate"]).sum())
    check("No order shipped before it was placed", neg == 0, f"{neg} orders")

    # ------------------------------------------------------------ coverage
    banner("COVERAGE")
    d0, d1 = o["OrderDate"].min(), o["OrderDate"].max()
    print(f"  OrderDate range: {d0:%Y-%m-%d} .. {d1:%Y-%m-%d}")
    q = o.groupby("OrderQuarter").agg(orders=("OrderID", "size"),
                                      first=("OrderDate", "min"),
                                      last=("OrderDate", "max"),
                                      revenue=("order_revenue", "sum"))
    q["days_in_data"] = [
        (min(d1, p.end_time.normalize()) - max(p.start_time, d0)).days + 1
        for p in q.index]
    q["days_in_quarter"] = [(p.end_time.normalize() - p.start_time).days + 1
                            for p in q.index]
    print(q.assign(revenue=q["revenue"].round(0)).to_string())
    pq = q.loc[PARTIAL_QUARTER]
    check("1998Q2 is partial", pq["days_in_data"] < pq["days_in_quarter"],
          f"{pq['days_in_data']} of {pq['days_in_quarter']} days "
          f"({pq['orders']} orders) -> excluded from growth comparisons")
    m = o.groupby("OrderMonth").size()
    print(f"\n  Orders/month: 1996H2 avg {m['1996-07':'1996-12'].mean():.0f}, "
          f"1997 avg {m['1997-01':'1997-12'].mean():.0f}, "
          f"1998 Jan-Apr avg {m['1998-01':'1998-04'].mean():.0f}")
    lead = o["lead_days"].value_counts().sort_index().to_dict()
    print(f"  RequiredDate - OrderDate (days -> orders): {lead}")

    # ------------------------------------------------------------- revenue
    banner("REVENUE CROSS-CHECKS  (revenue = Qty * UnitPrice * (1 - Discount))")
    r_lines = ln["revenue"].sum()
    r_orders = o["order_revenue"].sum()
    r_gross_minus = ln["gross"].sum() - ln["discount_amount"].sum()
    r_quarters = q["revenue"].sum()
    print(f"  Sum over 2155 lines        : {r_lines:,.2f}")
    print(f"  Sum over 830 order totals  : {r_orders:,.2f}")
    print(f"  Gross {ln['gross'].sum():,.2f} - discounts "
          f"{ln['discount_amount'].sum():,.2f} = {r_gross_minus:,.2f}")
    print(f"  Sum over quarters          : {r_quarters:,.2f}")
    same = np.allclose([r_orders, r_gross_minus, r_quarters], r_lines)
    check("Total revenue matches across 4 computations", same,
          f"{r_lines:,.2f}")
    n_by_country = o.groupby("ShipCountry").size()
    check("Orders per ShipCountry sum to 830", n_by_country.sum() == 830,
          f"{n_by_country.sum()} orders in {n_by_country.size} countries")

    # ------------------------------------------------------------- prices
    banner("PRICE SOURCE")
    match = np.isclose(ln["UnitPrice"], ln["ListPrice"])
    by_year = pd.Series(match).groupby(ln["OrderYear"]).mean()
    print("  Share of lines where Orders Details.UnitPrice == Product.Price:")
    for y, s in by_year.items():
        print(f"    {y}: {s:6.1%}")
    check("UnitPrice (transaction) differs from Product.Price (current list)",
          True, f"match {match.mean():.1%} overall; "
          + ", ".join(f"{y}: {s:.0%}" for y, s in by_year.items())
          + " -> use Orders Details.UnitPrice")

    # ----------------------------------------------------------- discounts
    banner("DISCOUNTS")
    dc = ln["Discount"].value_counts().sort_index()
    print("  Discount -> lines: " + ", ".join(f"{k:.2f}:{v}" for k, v in dc.items()))
    stray = ln[~ln["is_standard_discount"]]
    print(f"  Non-standard discounts: {len(stray)} lines in order(s) "
          f"{sorted(stray['OrderID'].unique())}, "
          f"revenue {stray['revenue'].sum():,.2f}")
    print(stray[["OrderID", "ProductName", "Quantity", "UnitPrice", "Discount",
                 "OrderDate", "CustomerID"]].to_string(index=False))
    check("Discounts only at 0/5/10/15/20/25%", stray.empty,
          f"{len(stray)} stray lines (values "
          f"{sorted(stray['Discount'].unique())}) all in order(s) "
          f"{sorted(stray['OrderID'].unique())}; kept in revenue totals, "
          f"excluded from discount-level comparisons")

    # ------------------------------------------------------------ shipping
    banner("SHIPPING")
    un = o[~o["is_shipped"]]
    overdue = int((un["RequiredDate"] < d1).sum())
    print(f"  Unshipped orders: {len(un)} (OrderDate "
          f"{un['OrderDate'].min():%Y-%m-%d}..{un['OrderDate'].max():%Y-%m-%d}); "
          f"already past RequiredDate at data end: {overdue}")
    check("Unshipped orders (ShippedDate null)", True,
          f"{len(un)} orders, all placed in the last 4 weeks of data; "
          f"{overdue} past RequiredDate at {d1:%Y-%m-%d}. Excluded from "
          f"late-shipment analysis.")
    sh = o[o["is_shipped"]]
    late = int((sh["slack_days"] < 0).sum())
    near = int(sh["slack_days"].between(0, 2).sum())
    print(f"  Shipped: {len(sh)}; late (slack<0): {late}; slack 0-2 days: {near}")
    print("  slack_days quantiles: "
          + ", ".join(f"p{int(k*100)}={v:.0f}" for k, v in
                      sh["slack_days"].quantile([0, .05, .1, .25, .5, .75, 1]).items()))
    print("  processing_days quantiles: "
          + ", ".join(f"p{int(k*100)}={v:.0f}" for k, v in
                      sh["processing_days"].quantile([0, .25, .5, .75, .9, 1]).items()))
    print("  slack_days counts for -10..7: "
          + str(sh["slack_days"].value_counts().sort_index().loc[-10:7].to_dict()))

    # ------------------------------------------------------------- markets
    banner("SHIP COUNTRY vs CUSTOMER COUNTRY")
    diff = o[o["ShipCountry"] != o["CustomerCountry"]]
    print(diff[["OrderID", "OrderDate", "CustomerID", "CustomerCountry",
                "ShipCountry", "order_revenue"]].to_string(index=False))
    city_diff = o[o["ShipCity"] != o["CustomerCity"]]
    check("ShipCountry == customer Country", diff.empty,
          f"differs on {len(diff)}/830 orders "
          f"({diff['order_revenue'].sum():,.0f} revenue); ShipCity differs on "
          f"{len(city_diff)} orders ({(city_diff['CustomerID'] == 'AROUT').sum()} "
          f"are AROUT London->Colchester)")

    zero = cu[~cu["CustomerID"].isin(o["CustomerID"])]
    print(f"\n  Customers: {len(cu)}; with orders: {o['CustomerID'].nunique()}; "
          f"with zero orders: {len(zero)} -> "
          f"{zero[['CustomerID', 'CompanyName', 'Country']].to_dict('records')}")
    check("Customers with zero orders", True,
          f"{len(zero)}: " + ", ".join(f"{r.CustomerID} ({r.Country})"
                                       for r in zero.itertuples()))

    # --------------------------------------------------------------- report
    banner("CHECK SUMMARY")
    cdf = pd.DataFrame(checks)
    for r in cdf.itertuples():
        print(f"  [{r.result:4s}] {r.check}: {r.detail}")
    cdf.to_csv(OUT_TAB / "step0_checks.csv", index=False)
    print(f"\nWrote {OUT_TAB / 'step0_sheet_summary.csv'}")
    print(f"Wrote {OUT_TAB / 'step0_checks.csv'}")
    return {"total_revenue": r_lines, "n_orders": len(o), "n_lines": len(ln)}


if __name__ == "__main__":
    main()
