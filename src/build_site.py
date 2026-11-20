"""Build the static webpage ``outputs/site/index.html``.

Copies every figure from ``outputs/figures/`` into
``outputs/site/figures/`` and fills the Q1 table from
``outputs/tables/q1_market_classification.csv``. The narrative text
and the numbers quoted in it were written by hand from the results
the q1-q4 scripts print; this script does not rewrite them.

Usage (from the project folder):  python src/build_site.py
(also run automatically at the end of ``python src/run_all.py``)
"""
from __future__ import annotations

import csv
import re
import shutil
from pathlib import Path

from load_data import OUT_FIG, OUT_TAB, ROOT

SITE = ROOT / "outputs" / "site"
SITE_FIG = SITE / "figures"


def pct(x: str) -> str:
    """Format a growth ratio as a signed whole percentage.

    Args:
        x (str): Ratio as read from the CSV, e.g. ``"1.07"``.

    Returns:
        str: Text such as ``"+107%"``, or ``"n/a"`` if ``x`` is empty or
        not numeric.
    """
    try:
        return f"{float(x)*100:+.0f}%"
    except (TypeError, ValueError):
        return "n/a"


def q1_table(csv_path: Path) -> str:
    """Render the Q1 market classification CSV as an HTML table.

    Args:
        csv_path (pathlib.Path): ``q1_market_classification.csv``.

    Returns:
        str: ``<table>`` HTML, one row per country in file order;
        rows get an ``invest``/``maintain``/``review`` class.
    """
    with open(csv_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    trs = ""
    for r in rows:
        flag = "Small sample" if r['low_confidence']=='True' else ("Few customers" if r['concentrated']=='True' else "")
        trs += f"<tr class='{r['action'].lower()}'><td>{r['country']}</td><td><b>{r['action']}</b></td><td>{float(r['revenue_share'])*100:.1f}%</td><td>{pct(r['growth_yoy_9m'])}</td><td>{r['orders_jul97_mar98']}</td><td>{r['customers_jul97_mar98']}</td><td>{flag}</td></tr>"
    return f"<table><thead><tr><th>Country</th><th>Action</th><th>Share of revenue</th><th>Growth (9-mo YoY)</th><th>Orders (latest 9 mo)</th><th>Active customers</th><th>Confidence flag</th></tr></thead><tbody>{trs}</tbody></table>"


def fig(f: str, alt: str, cap: str) -> str:
    """Return a ``<figure>`` for one PNG in ``figures/``.

    Args:
        f (str): PNG file name inside ``outputs/site/figures/``.
        alt (str): Alt text for the image.
        cap (str): Caption HTML.

    Returns:
        str: ``<figure>`` HTML with image and caption.
    """
    return f"<figure><img src='figures/{f}' alt='{alt}'><figcaption>{cap}</figcaption></figure>"


def sec(id: str, n: int, title: str, question: str, why: str,
        analysis: str, finding: str, interp: str) -> str:
    """Assemble one question section with its six labelled blocks.

    ``analysis`` is split at its first ``<h4>`` or ``<figure>``: the
    text before it becomes "Analytical approach", the rest
    "Visualization".

    Args:
        id (str): HTML id of the section, used by the nav links.
        n (int): Question number shown in the badge.
        title (str): Section heading.
        question (str): HTML for the "Question" block.
        why (str): HTML for the "Why it matters" block.
        analysis (str): Approach HTML followed by figures and tables;
            must contain ``<h4>`` or ``<figure``.
        finding (str): HTML for the "Main finding" callout.
        interp (str): HTML for the "Business interpretation" block.

    Returns:
        str: ``<section>`` HTML.
    """
    m = re.search(r"<h4>|<figure", analysis)
    approach, viz = analysis[:m.start()], analysis[m.start():]
    return f"""<section id="{id}"><h2><span class="n">Q{n}</span>{title}</h2>
<div class="block"><h3>Question</h3>{question}</div>
<div class="block"><h3>Why it matters</h3>{why}</div>
<div class="block"><h3>Analytical approach</h3>{approach}</div>
<div class="block"><h3>Visualization</h3>{viz}</div>
<div class="callout"><h3>Main finding</h3>{finding}</div>
<div class="block"><h3>Business interpretation</h3>{interp}</div></section>"""


def q1_section(table_html: str) -> str:
    """Build the Q1 (markets) section.

    Args:
        table_html (str): HTML table from :func:`q1_table`.

    Returns:
        str: ``<section id="q1">`` HTML.
    """
    return sec("q1",1,"Which markets deserve investment?",
    "<p>Which countries should Northwind <b>Invest</b> in, <b>Maintain</b>, or <b>Review</b>, based on market size and revenue growth?</p>",
    "<p>Sales force, marketing and inventory are limited. A country that is both large and growing faster than the company is where an extra dollar is most likely to pay back; a shrinking market may need a different approach or less attention.</p>",
    "<p>Markets are defined by shipping country (21 countries). Growth compares the same nine calendar months a year apart (Jul 1996&ndash;Mar 1997 vs Jul 1997&ndash;Mar 1998), so seasonality cancels out. Whole-company revenue grew <b>+83%</b> over the same windows, so a country is &ldquo;outgrowing&rdquo; only if it beats that. Countries with fewer than 5 orders in a window are drawn hollow and flagged as small samples.</p>"
    + fig("q1b_market_size_vs_growth.png","Market size vs growth","Market size (total revenue Jul 1996&ndash;May 1998, USD, log scale) against 9-month year-over-year revenue growth, classified into Invest / Maintain / Review. Hollow markers = small samples.")
    + fig("q1a_quarterly_revenue_top8.png","Quarterly revenue, top 8 countries","Quarterly revenue (USD) for the eight largest markets. 1998 Q2 is a partial quarter (36 of 91 days).")
    + "<details open><summary>Full classification table</summary>"+table_html+"</details>",
    "<ul><li><b>Invest (large &amp; outgrowing):</b> USA (+107%), Germany (+150%), Brazil (+120%) are big and clearly beat the company's +83%.</li><li><b>Maintain (large, slower growth):</b> Austria, France, UK, Venezuela, Canada, Belgium.</li><li><b>Review (shrinking):</b> Denmark, Mexico, Finland, Portugal.</li><li>Several small markets look like &ldquo;Invest&rdquo; on paper (e.g. Argentina +771%, 2 orders growing to 10) but rest on a handful of orders and are flagged low-confidence.</li></ul>",
    "<p>Concentrate growth spending on USA, Germany and Brazil, which together are about 46% of revenue and growing fastest. Keep serving the Maintain group efficiently. Review the four shrinking markets before committing resources. Treat the small-market &ldquo;Invest&rdquo; labels (and Sweden / Ireland, which depend on 1&ndash;2 customers) as pilots, not commitments. Using the customer's own country instead of the ship-to country changes no classification; shorter 6-month or Q1-only windows change 3&ndash;4 borderline countries (Canada, Belgium, USA, Switzerland, Italy), as noted in the caveats.</p>")


def q2_section() -> str:
    """Build the Q2 (regions, consolidation, under-served) section.

    Args:
        None.

    Returns:
        str: ``<section id="q2">`` HTML.
    """
    return sec("q2",2,"Regional mix, shipping consolidation and under-served customers",
    "<p>(a) Do regions differ in which product categories they buy, so we could suggest each region's best-sellers? (b) Can orders to the same destination be consolidated to save shipments? (c) Which customers buy too rarely and what could we offer them?</p>",
    "<p>Region-specific recommendations only help if regional tastes truly differ. Consolidation cuts shipping effort, and dormant customers are the cheapest revenue to win back.</p>",
    "<p>Countries were grouped into seven regions (North America, Latin America, British Isles, Western Europe, Central Europe, Nordics, Southern Europe).</p><ul><li><b>(a)</b> Compare each region's revenue share by category, then test whether the differences exceed chance with a permutation test that shuffles regions across whole customers and a chi-square on order-line counts.</li><li><b>(b)</b> Count orders to the same city within a &plusmn;7-day window and greedily batch them; check that every batched order would still meet its RequiredDate.</li><li><b>(c)</b> Flag customers with zero orders or at most 5 orders (bottom quartile), by region.</li></ul>"
    + "<h4>(a) Category mix by region</h4>"+fig("q2a_region_category_heatmap.png","Region by category heatmap","Share of each region's revenue by product category.")
    + "<p>The heatmap looks varied (Beverages ranges 8%&ndash;28%), so we tested it. Shuffling regions across whole customers gives <b>p = 0.40</b>; the line-count chi-square gives p = 0.11 (Cram&eacute;r's V = 0.064, small). One customer makes up 46% of British Isles revenue.</p>"
    + "<h4>(b) Consolidation</h4>"+fig("q2b_consolidation_window.png","Consolidation window","Share of orders that could be combined with another order to the same city, by time window.")
    + "<p>28% of orders have another order to the same city within &plusmn;7 days (21% for the same customer). Batching would cut <b>830 shipments to 714 (14% fewer)</b>; in 93 of 102 multi-order batches every order would still meet its deadline.</p>"
    + "<h4>(c) Under-served customers</h4>"+fig("q2c_underserved_by_region.png","Under-served customers by region","Customers with zero orders or 5 or fewer orders (bottom quartile), by region.")
    + "<p>28 of 91 customers are under-served: the 2 with no orders (FISSA, PARIS) plus everyone with 5 or fewer orders. Western Europe has the highest share (7 of 13).</p>",
    "<ul><li><b>(a)</b> Regional category differences are statistically indistinguishable from chance, so &ldquo;recommend each region's best-sellers&rdquo; has weak evidence behind it.</li><li><b>(b)</b> Up to ~14% fewer shipments is possible, highest in Latin America (21%) and British Isles (20%), lowest in Southern Europe (5%).</li><li><b>(c)</b> Rarely-buying customers are concentrated in Western Europe and North America (7 each).</li></ul>",
    "<p>Do not build region-specific product campaigns on this data; use company-wide best-sellers instead. Pilot a weekly consolidation window where the same city is served often (London, Boise, Rio de Janeiro, S&atilde;o Paulo). The 14% is an upper bound: it ignores warehouse timing and exact street addresses. For under-served customers, &ldquo;buys rarely&rdquo; does not prove unmet demand (counts are not adjusted for when a customer started), so start with outreach to the two zero-order customers and the larger dormant accounts (e.g. FOLIG, 5 orders, $11.7k).</p>")


def q3_section() -> str:
    """Build the Q3 (late shipments) section.

    Args:
        None.

    Returns:
        str: ``<section id="q3">`` HTML.
    """
    return sec("q3",3,"What drives late shipments?",
    "<p>Which orders ship late or near their deadline, and which factors (shipper, country, employee, season, lead time, order size) explain it?</p>",
    "<p>Late delivery damages customer trust. Knowing the real driver avoids blaming the wrong shipper or region.</p>",
    "<p>An order is <b>late</b> if shipped after its RequiredDate and <b>near-deadline</b> if it shipped with 0&ndash;2 days to spare. The 21 unshipped orders are excluded (809 analysed). Eight factors were tested with a Bonferroni-corrected threshold (p &lt; 0.0063).</p>"
    + fig("q3d_late_rate_by_lead_time.png","Late rate by lead time","Late rate by requested lead time (RequiredDate &minus; OrderDate), with 95% intervals.")
    + fig("q3a_slack_by_shipper.png","Slack by shipper","Days of slack (RequiredDate &minus; ShippedDate) by shipper.")
    + fig("q3b_late_rate_by_country.png","Late rate by country","At-risk rate by country (countries with 20+ shipped orders).")
    + fig("q3c_monthly_late_trend.png","Monthly late trend","Monthly late-order trend. The latest months are biased low because of unshipped orders.")
    ,
    "<ul><li>Only <b>37 of 809 orders (4.6%)</b> were late, plus 9 near-deadline.</li><li><b>Lead time is the only factor that holds up:</b> orders due in 14 days are late <b>17.6%</b> of the time vs 3.4% for 28 or 42 days (Fisher p = 1.6e-5, odds ratio 6.1).</li><li>Shipper, country, employee, month, season and order size show no significant difference.</li><li>Late orders are very slow orders (median 30 days to process vs 7 for the rest), not slightly slow ones.</li></ul>",
    "<p>Switching shippers or targeting countries would not help. The practical fix is to <b>flag 14-day-deadline orders at entry and fast-track them</b>, and to watch for orders stuck in processing beyond about 14 days. With only 37 late orders, differences that do not pass the threshold are reported as counts, not findings.</p>")


def q4_section() -> str:
    """Build the Q4 (discounts) section.

    Args:
        None.

    Returns:
        str: ``<section id="q4">`` HTML.
    """
    return sec("q4",4,"Do discounts increase sales?",
    "<p>Do discounted order lines sell more units and more revenue than undiscounted ones, and does this vary by discount level and category?</p>",
    "<p>Discounts give away margin. They are only worth it if they lift volume enough to offset the lower price.</p>",
    "<p>2,147 order lines at the six standard discount levels (8 stray lines from one order were excluded). Revenue is used because the data has no cost. We compare raw averages, then compare the <b>same product and same customer</b> using fixed-effects regressions, so that customers who buy in bulk anyway do not distort the result.</p>"
    + fig("q4a_quantity_by_discount.png","Quantity by discount level","Units per line by discount level, with 95% intervals.")
    + fig("q4b_revenue_by_discount.png","Revenue by discount level","Revenue per line by discount level.")
    + fig("q4c_discount_effect_by_category.png","Discount effect by category","Estimated effect of discounts by category, same product and customer.")
    ,
    "<ul><li><b>At face value</b>, discounted lines carry +26% units (p &lt; 1e-10), but revenue per line is only +9% and not robust.</li><li><b>Same customer, same product:</b> the unit lift shrinks to <b>+2%</b> (95% CI &minus;5% to +10%, not significant), and revenue per line is <b>&minus;13%</b> (CI &minus;19% to &minus;6%), roughly the discount itself.</li><li>5% discounts look revenue-neutral; 10%, 20% and 25% significantly reduce revenue per line. Confections is the only category with a significant drop.</li></ul>",
    "<p>The apparent volume gain comes from <b>who gets discounts</b> (large, frequent buyers; 35 of 89 customers never receive one), not from the discount itself. Discounts mostly give away revenue. Recommendation: stop broad discounting, keep small (5%) discounts if needed, and test deeper discounts only with a controlled experiment. Limits: revenue, not profit; association, not causation (discounts may be granted because an order is large).</p>")


def render_page(q1: str, q2: str, q3: str, q4: str) -> str:
    """Wrap the four sections in the full HTML page with inline CSS.

    Args:
        q1 (str): Q1 section HTML.
        q2 (str): Q2 section HTML.
        q3 (str): Q3 section HTML.
        q4 (str): Q4 section HTML.

    Returns:
        str: Complete HTML document.
    """
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Northwind Business Analysis</title>
<style>
:root{{--bg:#fafaf7;--fg:#1d2733;--mut:#5c6a78;--acc:#1f5f8b;--card:#fff;--line:#dde2e8;--call:#eaf3fa}}
@media (prefers-color-scheme:dark){{:root{{--bg:#14181d;--fg:#e6ebf0;--mut:#9aa7b4;--acc:#7db8e3;--card:#1c2229;--line:#2e3742;--call:#1a2b38}}}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--fg);font:16px/1.6 -apple-system,"Segoe UI",Helvetica,Arial,sans-serif}}
header,main,footer{{max-width:980px;margin:0 auto;padding:0 16px}}
header{{padding-top:40px}}h1{{margin:0 0 4px;font-size:2rem}}.sub{{color:var(--mut);margin:0 0 16px}}
nav{{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:8px 0;margin:16px 0;z-index:5;display:flex;gap:14px;flex-wrap:wrap}}
nav a{{color:var(--acc);text-decoration:none;font-weight:600}}
section{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:20px;margin:24px 0}}
h2{{margin:0 0 12px;font-size:1.5rem}}.n{{background:var(--acc);color:var(--bg);border-radius:6px;padding:2px 9px;margin-right:10px;font-size:1rem}}
h3{{margin:0 0 4px;font-size:.8rem;letter-spacing:.06em;text-transform:uppercase;color:var(--mut)}}h4{{margin:18px 0 6px}}
.block{{margin:16px 0}}.callout{{background:var(--call);border-left:4px solid var(--acc);padding:10px 14px;border-radius:6px;margin:16px 0}}
figure{{margin:14px 0}}img{{max-width:100%;height:auto;border:1px solid var(--line);border-radius:6px;background:#fff}}figcaption{{font-size:.85rem;color:var(--mut);margin-top:4px}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;display:block;overflow-x:auto}}th,td{{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}
tr.invest td:nth-child(2){{color:#1a7f37}}tr.review td:nth-child(2){{color:#c0392b}}summary{{cursor:pointer;font-weight:600;margin:8px 0}}
.note{{font-size:.9rem;color:var(--mut)}}footer{{padding-bottom:40px}}
</style></head><body>
<header><h1>Northwind Traders: Market, Shipping and Discount Analysis</h1>
<p class="sub">15.561 Digital Revolution &middot; HW3 &middot; Northwind database, orders July 1996 &ndash; May 1998</p>
<div class="callout"><h3>Summary</h3><p>Invest in USA, Germany and Brazil; do not build region-specific product campaigns (weak evidence); consolidate same-city shipments (up to 14% fewer); protect 14-day-deadline orders (the only driver of late delivery); and stop broad discounting (no real volume lift for the same customer, about &minus;13% revenue per line).</p></div>
<nav><a href="#q1">Q1 Markets</a><a href="#q2">Q2 Regions</a><a href="#q3">Q3 Late shipments</a><a href="#q4">Q4 Discounts</a><a href="#data">Data &amp; limits</a></nav></header>
<main>{q1}{q2}{q3}{q4}
<section id="data"><h2>Data and limitations</h2>
<ul><li>Revenue = UnitPrice &times; Quantity &times; (1 &minus; Discount) from Order Details; total <b>$1,265,793.04</b>, 830 orders, 2,155 lines, reconciled four ways.</li>
<li>Revenue, not profit: the data has no cost information.</li>
<li>Results are associations, not causal effects.</li>
<li>1998 Q2 is partial (88 orders), left out of growth comparisons; 21 unshipped orders left out of lateness analysis; 8 stray-discount lines (order 11077) left out of Q4.</li>
<li>Sales roughly doubled in 1998, so most countries show strong growth; growth is judged relative to the company's +83%.</li>
<li>Small samples (e.g. 37 late orders; countries with 1&ndash;2 customers) limit what can be concluded and are flagged.</li></ul>
<p class="note">All figures and tables are regenerated from the raw workbook by <code>python src/run_all.py</code>.</p></section></main>
<footer></footer></body></html>"""


def copy_figures() -> int:
    """Copy every PNG from ``outputs/figures/`` into the site folder.

    Args:
        None.

    Returns:
        int: Number of files copied; existing copies are overwritten.
    """
    SITE_FIG.mkdir(parents=True, exist_ok=True)
    pngs = sorted(OUT_FIG.glob("*.png"))
    for p in pngs:
        shutil.copyfile(p, SITE_FIG / p.name)
    return len(pngs)


def main() -> None:
    """Copy the figures and write ``outputs/site/index.html``.

    Args:
        None.

    Returns:
        None. Prints the number of figures copied and the page path.

    Raises:
        FileNotFoundError: If the Q1 table or a figure referenced by the
            page has not been generated yet (run ``run_all.py`` first).
    """
    n = copy_figures()
    page = render_page(q1_section(q1_table(OUT_TAB / "q1_market_classification.csv")),
                       q2_section(), q3_section(), q4_section())
    missing = sorted(set(re.findall(r"src='figures/([^']+)'", page))
                     - {p.name for p in SITE_FIG.glob("*.png")})
    if missing:
        raise FileNotFoundError(f"Figures missing from {SITE_FIG}: {missing}")
    (SITE / "index.html").write_text(page, encoding="utf-8")
    print(f"Site: copied {n} figures, wrote {SITE / 'index.html'} "
          f"({len(page):,} characters)")


if __name__ == "__main__":
    main()
