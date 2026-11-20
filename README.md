# HW3 – Northwind Traders: Market, Shipping and Discount Analysis

15.561 Digital Revolution, Fall 2026. Python code that rebuilds every number,
table, figure and the webpage of this analysis from the raw workbook
`Northwind Database.xlsx`.

## Quick start

Requires Python 3.10+ (tested with 3.11.5).

```bash
pip install -r requirements.txt
python src/run_all.py
```

The full run takes under a minute. It reads the workbook directly. No
intermediate files are needed, and none are kept from an earlier run: the
script deletes the old PNG/CSV outputs before it starts.

## Repository layout

| Path | What it is |
|---|---|
| `Northwind Database.xlsx` | Raw data. The only input. |
| `requirements.txt` | Python packages (pandas, numpy, matplotlib, seaborn, scipy, openpyxl). |
| `src/run_all.py` | Single entry point: runs every step below in order, then checks the results. |
| `src/load_data.py` | Loads the workbook, joins sheets, and adds derived columns (revenue, slack days, region, …). |
| `src/inspect_data.py` | Step 0: sheet sizes, missing values, and data-quality checks. |
| `src/q1_markets.py` | Q1: which markets to invest in. |
| `src/q2_regions.py` | Q2: regional category mix, shipment consolidation, under-served customers. |
| `src/q3_shipping.py` | Q3: what drives late shipments. |
| `src/q4_discounts.py` | Q4: do discounts increase sales. |
| `src/plot_style.py` | Shared chart style: colours, fonts, label placement, PNG saving. |
| `src/build_site.py` | Builds the webpage `outputs/site/index.html` and copies the figures next to it. |
| `outputs/figures/` | All figures (PNG, 200 dpi). |
| `outputs/tables/` | All tables (CSV). |
| `outputs/site/` | The webpage: `index.html` + `figures/`. |

## What `run_all.py` does

1. **Clean**: deletes old `outputs/figures/*.png` and `outputs/tables/*.csv`.
2. **Step 0** (`inspect_data`): prints the data-quality report and writes `step0_*.csv`.
3. **Q1–Q4**: each script prints its key numbers to the terminal and writes its figures and tables.
4. **Verify**: prints a `VERIFICATION SUMMARY`. The run stops with an error if any check fails.
5. **Build site** (`build_site`): copies the 12 figures into `outputs/site/figures/` and writes `outputs/site/index.html`.

Any single step can also be run alone, after the full run has created its inputs:

```bash
python src/q3_shipping.py
```

## How to check that you reproduced the results

The end of the terminal output should match these lines:

```
[OK  ] Revenue: line-level vs order-level: 1,265,793.04 vs 1,265,793.04
[OK  ] Orders: Q1 country table sums to 830: 830.00 vs 830.00
[OK  ] Lines: 2155 order lines: 2,155.00 vs 2,155.00
Figures written: 12   Tables written: 26
```

Revenue is cross-checked four ways: line level, order level, the Q1 country
table, and the Q2 region × category table. The summary also lists every
excluded row: 88 orders in the partial 1998 Q2, 21 unshipped orders, and 8
non-standard-discount lines.

Permutation tests use a fixed random seed (`SEED = 561` in `q2_regions.py` and
`q3_shipping.py`), so p-values are identical on every run.

## Where each webpage element comes from

| Webpage | Figure (`outputs/figures/`) | Table (`outputs/tables/`) | Key numbers |
|---|---|---|---|
| Q1 Markets | `q1b_market_size_vs_growth.png`, `q1a_quarterly_revenue_top8.png` | `q1_market_classification.csv` (the table on the page), `q1_quarterly_revenue_by_country.csv` | Company growth +83%; USA +107%, Germany +150%, Brazil +120% |
| Q2a Category mix | `q2a_region_category_heatmap.png` | `q2a_region_category_share.csv`, `q2a_region_category_revenue.csv` | Permutation p = 0.40; chi-square p = 0.11 (printed) |
| Q2b Consolidation | `q2b_consolidation_window.png` | `q2b_consolidation_by_window.csv`, `_by_region.csv`, `_by_city.csv` | 830 → 714 shipments (−14%); 93 of 102 batches meet the deadline |
| Q2c Under-served | `q2c_underserved_by_region.png` | `q2c_underserved_customers.csv`, `q2c_underserved_by_region.csv` | 28 of 91 customers |
| Q3 Late shipments | `q3d_…`, `q3a_…`, `q3b_…`, `q3c_…` | `q3_tests.csv`, `q3_by_lead_time.csv`, `q3_late_orders.csv`, other `q3_by_*.csv` | 37 of 809 late; 14-day lead time 17.6% vs 3.4% (Fisher p = 1.6e-5) |
| Q4 Discounts | `q4a_…`, `q4b_…`, `q4c_…` | `q4_discounted_vs_not_tests.csv`, `q4_fixed_effects_models.csv`, `q4_by_discount_level.csv`, `q4_by_category.csv` | Raw +26% units; same product & customer: units +2% (−5%, +10%), revenue −13% (−19%, −6%) |

The Q1 table on the page is generated from the CSV each time. The narrative text
and the numbers quoted in it were written from the printed output. They live in
`src/build_site.py`, and re-running does not rewrite them.

## Definitions used throughout

- **Revenue** = `UnitPrice × Quantity × (1 − Discount)` from the Order Details sheet, using the price on the order line. The data has no cost, so this is revenue, not profit.
- **Market** = `ShipCountry` (21 countries). **Region** = one of seven groups defined in `load_data.REGIONS`.
- **Growth** = Jul 1997–Mar 1998 vs Jul 1996–Mar 1997 (the same 9 months a year apart). 1998 Q2 is partial (data end on 6 May 1998), so it is left out of growth comparisons.
- **Late** = `ShippedDate > RequiredDate`. **Near-deadline** = shipped with 0–2 days to spare. **Lead time** = `RequiredDate − OrderDate`.
- **Small sample** = fewer than 5 orders in a growth window. These countries are flagged, not dropped.

## Viewing the webpage locally

Open `outputs/site/index.html` in a browser. It is a single static page with
relative image paths, so it can be hosted as-is (e.g. on GitHub Pages).

## Documentation

All Python files have Google-style docstrings, written with the
`python-docstrings` Claude Code skill (`.claude/skills/python-docstrings/SKILL.md`,
invoked as `/python-docstrings`).
