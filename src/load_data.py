"""Shared loading + cleaning for the Northwind HW3 analyses.

Everything is read straight from the raw workbook; nothing is hand-edited.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_FIG = ROOT / "outputs" / "figures"
OUT_TAB = ROOT / "outputs" / "tables"

# --- Shared definitions (used by every analysis) ---------------------------
# Discount levels Northwind actually uses. Anything else is a "stray" value.
STANDARD_DISCOUNTS = (0.00, 0.05, 0.10, 0.15, 0.20, 0.25)
# Data ends 1998-05-06, so 1998 Q2 holds only ~5 weeks of orders.
PARTIAL_QUARTER = pd.Period("1998Q2", freq="Q")
# Market = destination country on the order (see inspect_data.py for the
# check that it differs from the customer's own country on only 4/830 orders).
MARKET_COL = "ShipCountry"

# Region = a grouping of the 21 countries (state it wherever it is used).
REGIONS = {
    "North America": ["USA", "Canada"],
    "Latin America": ["Mexico", "Brazil", "Venezuela", "Argentina"],
    "British Isles": ["UK", "Ireland"],
    "Western Europe": ["France", "Belgium"],
    "Central Europe": ["Germany", "Austria", "Switzerland", "Poland"],
    "Nordics": ["Sweden", "Denmark", "Finland", "Norway"],
    "Southern Europe": ["Spain", "Portugal", "Italy"],
}
REGION_OF = {c: r for r, cs in REGIONS.items() for c in cs}

# Product sheet: keep only the clean source columns. The rest are Excel
# helper formulas (Cheap, NewPrice, a second "Cheap", IsCheap, a second
# "NewPrice") plus an empty spacer column.
PRODUCT_KEEP = [
    "ProductID", "ProductName", "SupplierID", "CategoryID", "QuantityPerUnit",
    "Price", "UnitsInStock", "UnitsOnOrder", "ReorderLevel", "Discontinued",
]


def find_workbook() -> Path:
    """Locate the Northwind workbook in the project root.

    Args:
        None.

    Returns:
        pathlib.Path: First ``Northwind*Database*.xlsx`` in sorted
            order, ignoring Excel lock files (names starting ``~$``).

    Raises:
        FileNotFoundError: If no matching workbook exists.
    """
    hits = [p for p in sorted(ROOT.glob("Northwind*Database*.xlsx"))
            if not p.name.startswith("~$")]
    if not hits:
        raise FileNotFoundError(f"No Northwind*Database*.xlsx found in {ROOT}")
    return hits[0]


def ensure_output_dirs() -> None:
    """Create the figure and table output folders if missing.

    Args:
        None.

    Returns:
        None. Creates ``outputs/figures`` and ``outputs/tables``.
    """
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    OUT_TAB.mkdir(parents=True, exist_ok=True)


def load_raw(path: Path | None = None) -> dict[str, pd.DataFrame]:
    """Read every sheet of the workbook exactly as pandas parses it.

    Args:
        path (pathlib.Path | None): Workbook to read. None means the
            file found by ``find_workbook()``.

    Returns:
        dict[str, pandas.DataFrame]: Raw frame for each sheet name.
    """
    return pd.read_excel(path or find_workbook(), sheet_name=None)


def tidy(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop empty columns and rows and strip text whitespace.

    Args:
        df (pandas.DataFrame): Raw sheet.

    Returns:
        tuple[pandas.DataFrame, list[str]]: The cleaned copy and the
            names of the entirely empty columns that were dropped.

    Raises:
        ValueError: If an "Unnamed" column contains data.
    """
    empty = [c for c in df.columns if df[c].isna().all()]
    unnamed_with_data = [c for c in df.columns
                         if str(c).startswith("Unnamed") and c not in empty]
    if unnamed_with_data:
        raise ValueError(f"Unnamed columns contain data: {unnamed_with_data}")
    out = df.drop(columns=empty).dropna(how="all").copy()
    for c in out.select_dtypes(include="object"):
        out[c] = out[c].map(lambda v: v.strip() if isinstance(v, str) else v)
    return out, empty


@dataclass
class Northwind:
    """Cleaned and joined Northwind tables.

    Args:
        categories (pandas.DataFrame): Category id, name, description.
        customers (pandas.DataFrame): Customers, with ``Region``.
        employees (pandas.DataFrame): Employees, with ``EmployeeName``.
        shippers (pandas.DataFrame): Shippers, with ``ShipperName``.
        suppliers (pandas.DataFrame): Suppliers (empty if no sheet).
        products (pandas.DataFrame): Products with ``CategoryName``.
        orders (pandas.DataFrame): One row per order (830) with derived
            dates, slack/processing/lead days and ``order_revenue``.
        lines (pandas.DataFrame): One row per order line (2155), joined
            to product, order and region, with ``revenue`` columns.

    Attributes:
        Same names as the constructor arguments.
    """
    categories: pd.DataFrame
    customers: pd.DataFrame
    employees: pd.DataFrame
    shippers: pd.DataFrame
    suppliers: pd.DataFrame
    products: pd.DataFrame
    orders: pd.DataFrame       # one row per order (830), with derived fields
    lines: pd.DataFrame        # one row per order line (2155), fully joined


def load_clean(path: Path | None = None) -> Northwind:
    """Load, clean and join all sheets into one object.

    Args:
        path (pathlib.Path | None): Workbook to read. None means the
            file found by ``find_workbook()``.

    Returns:
        Northwind: Cleaned tables. Revenue is Quantity * UnitPrice *
            (1 - Discount) using the transaction UnitPrice.

    Raises:
        ValueError: If an order or customer country has no region.
    """
    raw = load_raw(path)
    t = {name: tidy(df)[0] for name, df in raw.items()}

    categories = t["Categories"][["CategoryID", "CategoryName", "Description"]]
    customers = t["Customers"].assign(
        Region=lambda d: d["Country"].map(REGION_OF))
    shippers = t["Shippers"].rename(columns={"CompanyName": "ShipperName"})
    suppliers = t.get("Suppliers", pd.DataFrame())
    employees = t["Employees"].assign(
        EmployeeName=lambda d: d["FirstName"] + " " + d["LastName"])

    products = t["Product"][PRODUCT_KEEP].merge(
        categories[["CategoryID", "CategoryName"]], on="CategoryID", how="left")

    # ---- Orders: one row per order --------------------------------------
    orders = t["Orders"].copy()
    for c in ["OrderDate", "RequiredDate", "ShippedDate"]:
        orders[c] = pd.to_datetime(orders[c])
    orders = (orders
              .merge(shippers[["ShipperID", "ShipperName"]],
                     left_on="ShipVia", right_on="ShipperID", how="left")
              .drop(columns="ShipperID")
              .merge(employees[["EmployeeID", "EmployeeName"]],
                     on="EmployeeID", how="left")
              .merge(customers[["CustomerID", "Country", "City"]]
                     .rename(columns={"Country": "CustomerCountry",
                                      "City": "CustomerCity"}),
                     on="CustomerID", how="left"))
    orders["Region"] = orders["ShipCountry"].map(REGION_OF)
    orders["OrderYear"] = orders["OrderDate"].dt.year
    orders["OrderQuarter"] = orders["OrderDate"].dt.to_period("Q")
    orders["OrderMonth"] = orders["OrderDate"].dt.to_period("M")
    orders["is_partial_quarter"] = orders["OrderQuarter"] == PARTIAL_QUARTER
    orders["is_shipped"] = orders["ShippedDate"].notna()
    # slack < 0 means the order shipped after RequiredDate (late).
    orders["slack_days"] = (orders["RequiredDate"] - orders["ShippedDate"]).dt.days
    orders["processing_days"] = (orders["ShippedDate"] - orders["OrderDate"]).dt.days
    orders["lead_days"] = (orders["RequiredDate"] - orders["OrderDate"]).dt.days

    # ---- Order lines -----------------------------------------------------
    lines = t["Orders Details"].copy()
    # Discounts are stored as float32 (0.05 -> 0.0500000007...); round to 2 dp.
    lines["Discount"] = lines["Discount"].round(2)
    lines["is_standard_discount"] = lines["Discount"].isin(STANDARD_DISCOUNTS)
    # Revenue uses the transaction UnitPrice from Orders Details, not the
    # Product sheet's current list price. No cost data exists, so this is
    # discounted REVENUE, never profit.
    lines["gross"] = lines["Quantity"] * lines["UnitPrice"]
    lines["revenue"] = lines["gross"] * (1 - lines["Discount"])
    lines["discount_amount"] = lines["gross"] - lines["revenue"]

    lines = (lines
             .merge(products[["ProductID", "ProductName", "CategoryID",
                              "CategoryName", "Price"]]
                    .rename(columns={"Price": "ListPrice"}),
                    on="ProductID", how="left")
             .merge(orders[["OrderID", "CustomerID", "EmployeeID", "OrderDate",
                            "OrderYear", "OrderQuarter", "OrderMonth",
                            "is_partial_quarter", "ShipVia", "ShipperName",
                            "ShipCity", "ShipCountry", "Region",
                            "CustomerCountry"]],
                    on="OrderID", how="left"))

    order_rev = lines.groupby("OrderID")["revenue"].sum()
    orders["order_revenue"] = orders["OrderID"].map(order_rev).fillna(0.0)
    orders["n_lines"] = orders["OrderID"].map(lines.groupby("OrderID").size())

    unmapped = set(orders["ShipCountry"]) | set(customers["Country"])
    unmapped -= set(REGION_OF)
    if unmapped:
        raise ValueError(f"Countries without a region: {sorted(unmapped)}")

    return Northwind(categories=categories, customers=customers,
                     employees=employees, shippers=shippers,
                     suppliers=suppliers, products=products,
                     orders=orders, lines=lines)
