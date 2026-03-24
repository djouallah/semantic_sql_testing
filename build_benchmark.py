"""Build benchmark artifacts from DuckDB metadata.

Introspects a TPC-DS DuckDB database and generates:
  1. semantic_model.txt  - accurate schema for LLM system prompt
  2. questions.json      - tiered questions
  3. generate_baseline.py - validated baseline SQL
"""

import duckdb
import json
import os
import datetime
import time
import pathlib

# ── Config ──────────────────────────────────────────────────────────────────
SF = 1
OUTPUT_DIR = r"c:\llm"
BASELINE_MODEL = "claude-opus-4-6"

# Tables we care about (TPC-DS has many more but benchmark uses these)
FACT_TABLES = ["store_sales", "store_returns"]
DIM_TABLES = ["date_dim", "store", "customer", "item"]

# Known TPC-DS column descriptions (business-friendly)
COLUMN_DESCRIPTIONS = {
    # date_dim
    "d_date_sk": "Surrogate key",
    "d_date": "Calendar date",
    "d_date_id": "Date business key",
    "d_year": "Calendar year (4-digit)",
    "d_moy": "Month of year (1-12)",
    "d_dom": "Day of month (1-31)",
    "d_dow": "Day of week (0=Sunday, 1=Monday, ..., 6=Saturday)",
    "d_qoy": "Quarter of year (1-4)",
    "d_holiday": "Holiday indicator (Y/N)",
    "d_weekend": "Weekend indicator (Y=weekend, N=weekday)",
    "d_day_name": "Full day name (e.g. Sunday, Monday)",
    "d_month_seq": "Month sequence number",
    "d_week_seq": "Week sequence number",
    "d_quarter_seq": "Quarter sequence number",
    "d_fy_year": "Fiscal year",
    # store
    "s_store_sk": "Surrogate key",
    "s_store_id": "Store business key",
    "s_store_name": "Store display name",
    "s_number_employees": "Number of employees",
    "s_floor_space": "Floor space in square feet",
    "s_hours": "Operating hours",
    "s_manager": "Store manager name",
    "s_city": "City where store is located",
    "s_state": "State/province code",
    "s_country": "Country",
    "s_market_id": "Market identifier",
    "s_market_desc": "Market description",
    "s_market_manager": "Market manager name",
    "s_geography_class": "Geography classification",
    "s_division_id": "Division identifier",
    "s_division_name": "Division name",
    "s_company_id": "Company identifier",
    "s_company_name": "Company name",
    "s_zip": "ZIP/postal code",
    "s_county": "County name",
    "s_gmt_offset": "GMT offset",
    "s_tax_percentage": "Tax percentage",
    # customer
    "c_customer_sk": "Surrogate key",
    "c_customer_id": "Customer business key",
    "c_birth_year": "Customer birth year",
    "c_birth_month": "Customer birth month",
    "c_birth_day": "Customer birth day",
    "c_birth_country": "Country where customer was born",
    "c_preferred_cust_flag": "Preferred customer indicator (Y=preferred, N=standard)",
    "c_first_name": "Customer first name",
    "c_last_name": "Customer last name",
    "c_salutation": "Customer salutation (Mr., Mrs., etc.)",
    "c_email_address": "Customer email address",
    "c_login": "Customer login",
    # item
    "i_item_sk": "Surrogate key",
    "i_item_id": "Item business key",
    "i_item_desc": "Item description (generated text, not useful for filtering)",
    "i_brand_id": "Brand identifier",
    "i_brand": "Brand name",
    "i_class_id": "Class identifier",
    "i_class": "Item class",
    "i_category_id": "Category identifier",
    "i_category": "Item category",
    "i_color": "Item color",
    "i_units": "Unit of measure",
    "i_container": "Container type",
    "i_product_name": "Product name",
    "i_manufact_id": "Manufacturer identifier",
    "i_manufact": "Manufacturer name",
    "i_size": "Item size",
    "i_formulation": "Item formulation",
    "i_current_price": "Current selling price",
    "i_wholesale_cost": "Wholesale cost",
    "i_manager_id": "Manager identifier",
    # store_sales
    "ss_sold_date_sk": "FK to date_dim (sale date)",
    "ss_sold_time_sk": "FK to time_dim (sale time)",
    "ss_item_sk": "FK to item",
    "ss_customer_sk": "FK to customer",
    "ss_store_sk": "FK to store",
    "ss_ticket_number": "Transaction ticket identifier",
    "ss_quantity": "Quantity sold",
    "ss_sales_price": "Unit sales price",
    "ss_ext_sales_price": "Extended sales price",
    "ss_ext_discount_amt": "Extended discount amount",
    "ss_ext_wholesale_cost": "Extended wholesale cost",
    "ss_ext_list_price": "Extended list price",
    "ss_ext_tax": "Extended tax amount",
    "ss_coupon_amt": "Coupon amount",
    "ss_net_paid": "Net paid amount",
    "ss_net_paid_inc_tax": "Net paid including tax",
    "ss_net_profit": "Net profit",
    "ss_wholesale_cost": "Unit wholesale cost",
    "ss_list_price": "Unit list price",
    "ss_cdemo_sk": "FK to customer_demographics",
    "ss_hdemo_sk": "FK to household_demographics",
    "ss_addr_sk": "FK to customer_address",
    "ss_promo_sk": "FK to promotion",
    # store_returns
    "sr_returned_date_sk": "FK to date_dim (return date)",
    "sr_return_time_sk": "FK to time_dim (return time)",
    "sr_item_sk": "FK to item",
    "sr_customer_sk": "FK to customer",
    "sr_store_sk": "FK to store",
    "sr_ticket_number": "Return ticket identifier",
    "sr_return_quantity": "Quantity returned",
    "sr_return_amt": "Amount refunded",
    "sr_return_tax": "Return tax amount",
    "sr_return_amt_inc_tax": "Return amount including tax",
    "sr_fee": "Return fee",
    "sr_return_ship_cost": "Return shipping cost",
    "sr_refunded_cash": "Refunded cash amount",
    "sr_reversed_charge": "Reversed charge amount",
    "sr_store_credit": "Store credit amount",
    "sr_net_loss": "Net loss from return",
    "sr_reason_sk": "FK to reason",
    "sr_cdemo_sk": "FK to customer_demographics",
    "sr_hdemo_sk": "FK to household_demographics",
    "sr_addr_sk": "FK to customer_address",
}

# FK mapping: hardcoded from TPC-DS spec (DuckDB dsdgen doesn't create PK/FK constraints)
FK_MAP = {
    ("store_sales", "ss_sold_date_sk"): ("date_dim", "d_date_sk"),
    ("store_sales", "ss_store_sk"): ("store", "s_store_sk"),
    ("store_sales", "ss_customer_sk"): ("customer", "c_customer_sk"),
    ("store_sales", "ss_item_sk"): ("item", "i_item_sk"),
    ("store_returns", "sr_returned_date_sk"): ("date_dim", "d_date_sk"),
    ("store_returns", "sr_store_sk"): ("store", "s_store_sk"),
    ("store_returns", "sr_customer_sk"): ("customer", "c_customer_sk"),
    ("store_returns", "sr_item_sk"): ("item", "i_item_sk"),
}

# Columns to include in semantic model (subset most useful for queries)
# If a table is not listed here, ALL columns are included
INCLUDE_COLUMNS = {
    "date_dim": [
        "d_date_sk", "d_date", "d_year", "d_moy", "d_dom", "d_dow",
        "d_qoy", "d_day_name", "d_holiday", "d_weekend",
    ],
    "store": [
        "s_store_sk", "s_store_id", "s_store_name", "s_number_employees",
        "s_floor_space", "s_hours", "s_manager", "s_city", "s_state", "s_country",
    ],
    "customer": [
        "c_customer_sk", "c_customer_id", "c_preferred_cust_flag",
        "c_birth_year", "c_birth_month", "c_birth_day", "c_birth_country",
        "c_first_name", "c_last_name",
    ],
    "item": [
        "i_item_sk", "i_item_id", "i_product_name", "i_item_desc",
        "i_brand", "i_class", "i_category", "i_color", "i_units",
        "i_container", "i_current_price", "i_wholesale_cost",
    ],
    "store_sales": [
        "ss_sold_date_sk", "ss_store_sk", "ss_customer_sk", "ss_item_sk",
        "ss_ticket_number", "ss_quantity", "ss_sales_price", "ss_ext_sales_price",
    ],
    "store_returns": [
        "sr_returned_date_sk", "sr_store_sk", "sr_customer_sk", "sr_item_sk",
        "sr_ticket_number", "sr_return_quantity", "sr_return_amt",
    ],
}


# ── Schema introspection ────────────────────────────────────────────────────
def introspect_table(con, table_name):
    """Return metadata dict for one table."""
    cols_df = con.sql(f"DESCRIBE {table_name}").fetchdf()
    row_count = con.sql(f"SELECT COUNT(*) AS n FROM {table_name}").fetchone()[0]

    allowed = INCLUDE_COLUMNS.get(table_name)
    columns = []
    for _, row in cols_df.iterrows():
        col_name = row["column_name"]
        if allowed and col_name not in allowed:
            continue
        col_type = row["column_type"]

        # Sample values (up to 5 distinct non-NULL)
        # For date_dim year, sample from actual sales years not full calendar
        if table_name == "date_dim" and col_name == "d_year":
            sample_df = con.sql(
                "SELECT DISTINCT d.d_year FROM store_sales ss "
                "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
                "ORDER BY d.d_year LIMIT 5"
            ).fetchdf()
        elif table_name == "date_dim" and col_name == "d_date":
            sample_df = con.sql(
                "SELECT DISTINCT d.d_date FROM store_sales ss "
                "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk "
                "ORDER BY d.d_date LIMIT 5"
            ).fetchdf()
        else:
            sample_df = con.sql(
                f"SELECT DISTINCT {col_name} FROM {table_name} "
                f"WHERE {col_name} IS NOT NULL ORDER BY {col_name} LIMIT 5"
            ).fetchdf()
        samples = sample_df[col_name].tolist()

        # NULL count
        null_count = con.sql(
            f"SELECT COUNT(*) - COUNT({col_name}) FROM {table_name}"
        ).fetchone()[0]
        null_pct = round(null_count / max(row_count, 1) * 100, 1)

        # Distinct count
        distinct_count = con.sql(
            f"SELECT COUNT(DISTINCT {col_name}) FROM {table_name} "
            f"WHERE {col_name} IS NOT NULL"
        ).fetchone()[0]

        # Value range for numeric types
        value_range = None
        if "INT" in col_type.upper() or "DECIMAL" in col_type.upper():
            if table_name == "date_dim" and col_name == "d_year":
                rng = con.sql(
                    "SELECT MIN(d.d_year), MAX(d.d_year) FROM store_sales ss "
                    "JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk"
                ).fetchone()
            else:
                rng = con.sql(
                    f"SELECT MIN({col_name}), MAX({col_name}) FROM {table_name}"
                ).fetchone()
            if rng[0] is not None:
                value_range = {"min": rng[0], "max": rng[1]}

        columns.append({
            "name": col_name,
            "type": col_type,
            "samples": samples,
            "null_pct": null_pct,
            "distinct_count": distinct_count,
            "value_range": value_range,
            "description": COLUMN_DESCRIPTIONS.get(col_name, col_name),
        })

    return {"table_name": table_name, "row_count": row_count, "columns": columns}


def introspect_all(con):
    """Introspect all relevant tables."""
    schema = {}
    for t in DIM_TABLES + FACT_TABLES:
        print(f"  Introspecting {t}...")
        schema[t] = introspect_table(con, t)
    return schema


# ── Semantic model generation ───────────────────────────────────────────────
def format_sample(val):
    """Format a sample value for display."""
    if isinstance(val, str):
        return f'"{val}"'
    if isinstance(val, (datetime.date, datetime.datetime)):
        return f'"{val}"'
    return str(val)


def generate_semantic_model(schema, output_path, db_con):
    """Write semantic_model.txt from introspected schema."""
    # Get data context
    # Get actual sales year range (not the full date_dim calendar range)
    sales_years = db_con.sql(
        "SELECT MIN(d.d_year) AS min_yr, MAX(d.d_year) AS max_yr "
        "FROM store_sales ss JOIN date_dim d ON ss.ss_sold_date_sk = d.d_date_sk"
    ).fetchone()
    min_year = sales_years[0]
    max_year = sales_years[1]
    n_stores = next(c for c in schema["store"]["columns"] if c["name"] == "s_store_name")["distinct_count"]
    n_items = schema["item"]["row_count"]
    n_customers = schema["customer"]["row_count"]
    n_sales = schema["store_sales"]["row_count"]
    n_returns = schema["store_returns"]["row_count"]

    lines = []
    w = lines.append

    # ── Header ──
    w(f"# Auto-generated semantic model from DuckDB | SF={SF} | {datetime.date.today()}")
    w(f"# Sales: {n_sales:,} rows | Returns: {n_returns:,} rows | Items: {n_items:,} | Customers: {n_customers:,} | Stores: {n_stores}")
    w(f"# Data spans years {min_year}-{max_year}")
    w("")

    # ── Objective ──
    w("# === Model Objective ===")
    w("# Analyze store sales and return performance to identify trends and areas for improvement.")
    w("")

    # ── Instructions ──
    w("# === Instructions ===")
    w("# You are a SQL query generator. Given a user question, produce a single correct DuckDB SQL query.")
    w("# - Use ONLY the tables, columns, measures, and relationships defined below.")
    w("# - Add the question as a comment at the top of the SQL.")
    w("# - Return only the SQL query, no explanation.")
    w("")

    # ── Critical rules ──
    w("# === CRITICAL: NO FACT-TO-FACT JOINS ===")
    w("# NEVER join store_sales and store_returns directly.")
    w("# ALWAYS use separate CTEs for each fact table, aggregate separately, then FULL OUTER JOIN results.")
    w("#")
    w("# WRONG:")
    w("#   SELECT ... FROM store_sales ss JOIN store_returns sr ON ss.col = sr.col")
    w("#")
    w("# CORRECT pattern:")
    w("#   WITH sales AS (SELECT dim_key, SUM(...) FROM store_sales JOIN dim ON ... GROUP BY dim_key),")
    w("#        returns AS (SELECT dim_key, SUM(...) FROM store_returns JOIN dim ON ... GROUP BY dim_key)")
    w("#   SELECT COALESCE(s.dim_key, r.dim_key), ...")
    w("#   FROM sales s FULL OUTER JOIN returns r ON s.dim_key = r.dim_key")
    w("")

    # ── Measures ──
    w("# === Measures ===")
    w("# total_sales    = SUM(ss.ss_sales_price * ss.ss_quantity)   -- from store_sales")
    w("# total_quantity  = SUM(ss.ss_quantity)                       -- from store_sales")
    w("# total_returns   = SUM(sr.sr_return_amt)                     -- from store_returns")
    w("# net_sales       = COALESCE(total_sales,0) - COALESCE(total_returns,0)")
    w("#                   Requires CTE pattern: aggregate each fact separately, then FULL OUTER JOIN")
    w("# return_rate     = (COALESCE(total_returns,0) / NULLIF(COALESCE(total_sales,0), 0)) * 100")
    w("#                   Requires CTE pattern: aggregate each fact separately, then FULL OUTER JOIN")
    w("")

    # ── Expressions ──
    w("# === Reusable Expressions ===")
    w("# age_group (requires date_dim d and customer c):")
    w("#   CASE")
    w("#     WHEN (d.d_year - c.c_birth_year) < 20 THEN '< 20'")
    w("#     WHEN (d.d_year - c.c_birth_year) BETWEEN 20 AND 29 THEN '20-29'")
    w("#     WHEN (d.d_year - c.c_birth_year) BETWEEN 30 AND 39 THEN '30-39'")
    w("#     WHEN (d.d_year - c.c_birth_year) BETWEEN 40 AND 49 THEN '40-49'")
    w("#     WHEN (d.d_year - c.c_birth_year) BETWEEN 50 AND 59 THEN '50-59'")
    w("#     WHEN (d.d_year - c.c_birth_year) >= 60 THEN '60+'")
    w("#   END")
    w("")

    # ── NULL guidance ──
    w("# === NULL Handling ===")
    w("# Some dimension columns contain NULLs. Key nullable columns:")
    nullable_cols = []
    for tname in DIM_TABLES:
        for col in schema[tname]["columns"]:
            if col["null_pct"] > 0 and not col["name"].endswith("_sk"):
                nullable_cols.append(f"#   {tname}.{col['name']} ({col['null_pct']}% NULL)")
    for line in nullable_cols:
        w(line)
    w("# When questions ask for 'different categories', 'all brands', etc.:")
    w("#   - Filter NULLs with WHERE col IS NOT NULL unless question asks to include them")
    w("#   - FULL OUTER JOIN on nullable columns: NULL != NULL, so filter NULLs in CTEs first")
    w("")

    # ── Schema ──
    w("# ============================================================")
    w("# SCHEMA")
    w("# ============================================================")
    w("")

    # Dimensions
    w("# --- Dimension Tables ---")
    for tname in DIM_TABLES:
        tinfo = schema[tname]
        pk = next((c["name"] for c in tinfo["columns"] if c["name"].endswith("_sk") or c["name"].endswith("_sk")), tinfo["columns"][0]["name"])
        w(f"# {tname}  (PK: {pk}, {tinfo['row_count']:,} rows)")
        for col in tinfo["columns"]:
            if col["name"] == pk:
                continue  # skip PK, already shown
            samples_str = ", ".join(format_sample(v) for v in col["samples"][:5])
            null_note = f" | {col['null_pct']}% NULL" if col["null_pct"] > 0 else ""
            range_note = ""
            if col["value_range"] and col["distinct_count"] > 10:
                range_note = f" | range: {col['value_range']['min']}..{col['value_range']['max']}"
            w(f"#   {col['name']:30s} {col['type']:15s} -- {col['description']}")
            w(f"#     samples: [{samples_str}]{null_note}{range_note}")
        w("")

    # Fact tables
    w("# --- Fact Tables ---")
    for tname in FACT_TABLES:
        tinfo = schema[tname]
        w(f"# {tname}  ({tinfo['row_count']:,} rows)")

        # Foreign keys
        fks_for_table = {fk_col: (dim, pk) for (ft, fk_col), (dim, pk) in FK_MAP.items() if ft == tname}
        if fks_for_table:
            w(f"#   Foreign keys:")
            for fk_col, (dim, pk) in sorted(fks_for_table.items()):
                w(f"#     {fk_col} -> {dim}.{pk}")

        # Columns (non-FK)
        w(f"#   Columns:")
        for col in tinfo["columns"]:
            if col["name"] in fks_for_table:
                continue
            samples_str = ", ".join(format_sample(v) for v in col["samples"][:3])
            null_note = f" | {col['null_pct']}% NULL" if col["null_pct"] > 0 else ""
            w(f"#     {col['name']:30s} {col['type']:15s} -- {col['description']}")
            if samples_str:
                w(f"#       samples: [{samples_str}]{null_note}")
        w("")

    # ── Verified queries ──
    w("# ============================================================")
    w("# VERIFIED QUERY PATTERNS")
    w("# ============================================================")
    w("")
    w("# Pattern 1: Simple fact aggregation")
    w("# SELECT SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales FROM store_sales AS ss")
    w("")
    w("# Pattern 2: Fact + dimension join")
    w("# SELECT d.d_year, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales")
    w("# FROM store_sales AS ss")
    w("# JOIN date_dim AS d ON ss.ss_sold_date_sk = d.d_date_sk")
    w("# GROUP BY d.d_year ORDER BY d.d_year")
    w("")
    w("# Pattern 3: Net sales using CTE (combining two fact tables)")
    w("# WITH sales_agg AS (")
    w("#     SELECT s.s_store_name, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales")
    w("#     FROM store_sales AS ss JOIN store AS s ON ss.ss_store_sk = s.s_store_sk")
    w("#     GROUP BY s.s_store_name")
    w("# ), returns_agg AS (")
    w("#     SELECT s.s_store_name, SUM(sr.sr_return_amt) AS total_returns")
    w("#     FROM store_returns AS sr JOIN store AS s ON sr.sr_store_sk = s.s_store_sk")
    w("#     GROUP BY s.s_store_name")
    w("# )")
    w("# SELECT COALESCE(sa.s_store_name, ra.s_store_name) AS store_name,")
    w("#        COALESCE(sa.total_sales, 0) - COALESCE(ra.total_returns, 0) AS net_sales")
    w("# FROM sales_agg AS sa FULL OUTER JOIN returns_agg AS ra ON sa.s_store_name = ra.s_store_name")
    w("")
    w("# Pattern 4: Return rate by dimension")
    w("# WITH sales_agg AS (")
    w("#     SELECT i.i_category, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales")
    w("#     FROM store_sales AS ss JOIN item AS i ON ss.ss_item_sk = i.i_item_sk")
    w("#     WHERE i.i_category IS NOT NULL")
    w("#     GROUP BY i.i_category")
    w("# ), returns_agg AS (")
    w("#     SELECT i.i_category, SUM(sr.sr_return_amt) AS total_returns")
    w("#     FROM store_returns AS sr JOIN item AS i ON sr.sr_item_sk = i.i_item_sk")
    w("#     WHERE i.i_category IS NOT NULL")
    w("#     GROUP BY i.i_category")
    w("# )")
    w("# SELECT COALESCE(sa.i_category, ra.i_category) AS category,")
    w("#        (COALESCE(ra.total_returns, 0) / NULLIF(COALESCE(sa.total_sales, 0), 0)) * 100 AS return_rate")
    w("# FROM sales_agg AS sa FULL OUTER JOIN returns_agg AS ra ON sa.i_category = ra.i_category")
    w("# ORDER BY category")
    w("")

    # ── Anti-patterns ──
    w("# ============================================================")
    w("# ANTI-PATTERNS (FORBIDDEN)")
    w("# ============================================================")
    w("# 1. Direct fact-to-fact join:")
    w("#    WRONG: FROM store_sales ss JOIN store_returns sr ON ss.ss_store_sk = sr.sr_store_sk")
    w("#    This multiplies rows and inflates aggregations.")
    w("")
    w("# 2. Implicit cross join:")
    w("#    WRONG: FROM store_sales ss, store_returns sr WHERE ss.ss_store_sk = sr.sr_store_sk")
    w("#    Same problem as #1.")
    w("")
    w("# 3. Dimension-bridged fact join without CTEs:")
    w("#    WRONG: FROM store s LEFT JOIN store_sales ss ON ... LEFT JOIN store_returns sr ON ...")
    w("#    Creates cartesian product between sales and returns per store.")

    content = "\n".join(lines) + "\n"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Written: {output_path} ({len(lines)} lines)")


# ── Questions & Baseline SQL ────────────────────────────────────────────────
BENCHMARK = [
    # ── Tier 1: Single table, simple ──
    {
        "tier": 1,
        "question": "What is the overall total sales revenue?",
        "sql": """-- What is the overall total sales revenue?
SELECT SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales_revenue
FROM store_sales AS ss"""
    },
    {
        "tier": 1,
        "question": "What is the total number of items sold across all transactions?",
        "sql": """-- What is the total number of items sold across all transactions?
SELECT SUM(ss.ss_quantity) AS total_quantity
FROM store_sales AS ss"""
    },
    {
        "tier": 1,
        "question": "What is the total monetary value of all returned items?",
        "sql": """-- What is the total monetary value of all returned items?
SELECT SUM(sr.sr_return_amt) AS total_monetary_value_of_returns
FROM store_returns AS sr"""
    },
    {
        "tier": 1,
        "question": "List the names of all stores, order by store name.",
        "sql": """-- List the names of all stores, order by store name.
SELECT DISTINCT s.s_store_name
FROM store AS s
ORDER BY s.s_store_name"""
    },
    {
        "tier": 1,
        "question": "What are the different item categories available? Order alphabetically by category name.",
        "sql": """-- What are the different item categories available? Order alphabetically by category name.
SELECT DISTINCT i.i_category
FROM item AS i
WHERE i.i_category IS NOT NULL
ORDER BY i.i_category"""
    },
    # ── Tier 2: Fact + dimension joins ──
    {
        "tier": 2,
        "question": "Show total sales revenue for each year, ordered chronologically by year.",
        "sql": """-- Show total sales revenue for each year, ordered chronologically by year.
SELECT d.d_year, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales_revenue
FROM store_sales AS ss
JOIN date_dim AS d ON ss.ss_sold_date_sk = d.d_date_sk
GROUP BY d.d_year
ORDER BY d.d_year"""
    },
    {
        "tier": 2,
        "question": "Which store generated the most total sales revenue? Show store name and total sales. Order by revenue descending, then store name alphabetically for ties.",
        "sql": """-- Which store generated the most total sales revenue?
SELECT s.s_store_name, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
FROM store_sales AS ss
JOIN store AS s ON ss.ss_store_sk = s.s_store_sk
GROUP BY s.s_store_name
ORDER BY total_sales DESC, s.s_store_name
LIMIT 1"""
    },
    {
        "tier": 2,
        "question": "What is the total quantity of items sold, broken down by item brand? Order by quantity sold descending, then by brand name alphabetically for ties.",
        "sql": """-- What is the total quantity of items sold, broken down by item brand?
SELECT i.i_brand, SUM(ss.ss_quantity) AS total_quantity
FROM store_sales AS ss
JOIN item AS i ON ss.ss_item_sk = i.i_item_sk
WHERE i.i_brand IS NOT NULL
GROUP BY i.i_brand
ORDER BY total_quantity DESC, i.i_brand"""
    },
    {
        "tier": 2,
        "question": "Show total sales for preferred customers (c_preferred_cust_flag = 'Y') versus non-preferred customers (c_preferred_cust_flag = 'N') as two rows. Exclude customers where the preferred flag is NULL. Order by total sales descending.",
        "sql": """-- Show total sales for preferred vs non-preferred customers
SELECT c.c_preferred_cust_flag, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
FROM store_sales AS ss
JOIN customer AS c ON ss.ss_customer_sk = c.c_customer_sk
WHERE c.c_preferred_cust_flag IS NOT NULL
GROUP BY c.c_preferred_cust_flag
ORDER BY total_sales DESC"""
    },
    {
        "tier": 2,
        "question": "What is the total return amount for each city where stores are located, ordered alphabetically by city name.",
        "sql": """-- What is the total return amount for each city where stores are located?
SELECT s.s_city, SUM(sr.sr_return_amt) AS total_returns
FROM store_returns AS sr
JOIN store AS s ON sr.sr_store_sk = s.s_store_sk
GROUP BY s.s_city
ORDER BY s.s_city"""
    },
    # ── Tier 3: CTE pattern — combining facts ──
    {
        "tier": 3,
        "question": "What is the net sales for each store name, order by net sales.",
        "sql": """-- What is the net sales for each store name, order by net sales.
WITH sales_by_store AS (
    SELECT s.s_store_name, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN store AS s ON ss.ss_store_sk = s.s_store_sk
    GROUP BY s.s_store_name
),
returns_by_store AS (
    SELECT s.s_store_name, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN store AS s ON sr.sr_store_sk = s.s_store_sk
    GROUP BY s.s_store_name
)
SELECT
    COALESCE(sa.s_store_name, ra.s_store_name) AS store_name,
    COALESCE(sa.total_sales, 0) - COALESCE(ra.total_returns, 0) AS net_sales
FROM sales_by_store AS sa
FULL OUTER JOIN returns_by_store AS ra ON sa.s_store_name = ra.s_store_name
ORDER BY net_sales"""
    },
    {
        "tier": 3,
        "question": "Calculate the return rate for each item category (i_category). Show the category name and return rate, with one row per distinct category. Exclude NULL categories. Order alphabetically by category name.",
        "sql": """-- Calculate the return rate for each item category, ordered alphabetically.
WITH sales_by_category AS (
    SELECT i.i_category, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN item AS i ON ss.ss_item_sk = i.i_item_sk
    WHERE i.i_category IS NOT NULL
    GROUP BY i.i_category
),
returns_by_category AS (
    SELECT i.i_category, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN item AS i ON sr.sr_item_sk = i.i_item_sk
    WHERE i.i_category IS NOT NULL
    GROUP BY i.i_category
)
SELECT
    COALESCE(sa.i_category, ra.i_category) AS item_category,
    (COALESCE(ra.total_returns, 0) / NULLIF(COALESCE(sa.total_sales, 0), 0)) * 100 AS return_rate
FROM sales_by_category AS sa
FULL OUTER JOIN returns_by_category AS ra ON sa.i_category = ra.i_category
ORDER BY item_category"""
    },
    {
        "tier": 3,
        "question": "What is the monthly trend of net sales during the year 2001, order by net sales desc.",
        "sql": """-- What is the monthly trend of net sales during the year 2001, order by net sales desc
WITH sales_monthly AS (
    SELECT d.d_moy AS month, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN date_dim AS d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE d.d_year = 2001
    GROUP BY d.d_moy
),
returns_monthly AS (
    SELECT d.d_moy AS month, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN date_dim AS d ON sr.sr_returned_date_sk = d.d_date_sk
    WHERE d.d_year = 2001
    GROUP BY d.d_moy
)
SELECT
    COALESCE(sa.month, ra.month) AS month,
    COALESCE(sa.total_sales, 0) - COALESCE(ra.total_returns, 0) AS net_sales
FROM sales_monthly AS sa
FULL OUTER JOIN returns_monthly AS ra ON sa.month = ra.month
ORDER BY net_sales DESC"""
    },
    {
        "tier": 3,
        "question": "Which customer birth country has the highest return rate? Show the country name and return rate. Order by return rate descending, then by country name alphabetically. Return only the top 1 row.",
        "sql": """-- Which customer birth country has the highest return rate?
WITH sales_by_country AS (
    SELECT c.c_birth_country, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN customer AS c ON ss.ss_customer_sk = c.c_customer_sk
    WHERE c.c_birth_country IS NOT NULL
    GROUP BY c.c_birth_country
),
returns_by_country AS (
    SELECT c.c_birth_country, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN customer AS c ON sr.sr_customer_sk = c.c_customer_sk
    WHERE c.c_birth_country IS NOT NULL
    GROUP BY c.c_birth_country
)
SELECT
    COALESCE(sa.c_birth_country, ra.c_birth_country) AS birth_country,
    (COALESCE(ra.total_returns, 0) / NULLIF(COALESCE(sa.total_sales, 0), 0)) * 100 AS return_rate
FROM sales_by_country AS sa
FULL OUTER JOIN returns_by_country AS ra ON sa.c_birth_country = ra.c_birth_country
ORDER BY return_rate DESC, birth_country
LIMIT 1"""
    },
    {
        "tier": 3,
        "question": "List all item product names and total sales that have a return rate greater than 5%, ordered by item product name alphabetically.",
        "sql": """-- List all item product names and total sales that have a return rate greater than 5%
WITH sales_by_product AS (
    SELECT i.i_product_name, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN item AS i ON ss.ss_item_sk = i.i_item_sk
    WHERE i.i_product_name IS NOT NULL
    GROUP BY i.i_product_name
),
returns_by_product AS (
    SELECT i.i_product_name, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN item AS i ON sr.sr_item_sk = i.i_item_sk
    WHERE i.i_product_name IS NOT NULL
    GROUP BY i.i_product_name
)
SELECT
    COALESCE(sa.i_product_name, ra.i_product_name) AS i_product_name,
    COALESCE(sa.total_sales, 0) AS total_sales
FROM sales_by_product AS sa
FULL OUTER JOIN returns_by_product AS ra ON sa.i_product_name = ra.i_product_name
WHERE (COALESCE(ra.total_returns, 0) / NULLIF(COALESCE(sa.total_sales, 0), 0)) * 100 > 5
ORDER BY i_product_name"""
    },
    # ── Tier 4: Complex — multi-dimension, year-over-year, window functions ──
    {
        "tier": 4,
        "question": "For each store, what was the percentage change in net sales from year 2001 to year 2002? Show store name, net sales for 2001, net sales for 2002, and percentage change. Order alphabetically by store name.",
        "sql": """-- Percentage change in net sales from 2001 to 2002 per store
WITH sales_by_store_year AS (
    SELECT s.s_store_name, d.d_year, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN store AS s ON ss.ss_store_sk = s.s_store_sk
    JOIN date_dim AS d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE d.d_year IN (2001, 2002)
    GROUP BY s.s_store_name, d.d_year
),
returns_by_store_year AS (
    SELECT s.s_store_name, d.d_year, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN store AS s ON sr.sr_store_sk = s.s_store_sk
    JOIN date_dim AS d ON sr.sr_returned_date_sk = d.d_date_sk
    WHERE d.d_year IN (2001, 2002)
    GROUP BY s.s_store_name, d.d_year
),
net_sales_by_year AS (
    SELECT
        COALESCE(sa.s_store_name, ra.s_store_name) AS s_store_name,
        COALESCE(sa.d_year, ra.d_year) AS d_year,
        COALESCE(sa.total_sales, 0) - COALESCE(ra.total_returns, 0) AS net_sales
    FROM sales_by_store_year AS sa
    FULL OUTER JOIN returns_by_store_year AS ra
        ON sa.s_store_name = ra.s_store_name AND sa.d_year = ra.d_year
)
SELECT
    n1.s_store_name,
    n1.net_sales AS net_sales_2001,
    n2.net_sales AS net_sales_2002,
    ROUND(((n2.net_sales - n1.net_sales) / NULLIF(n1.net_sales, 0)) * 100, 2) AS pct_change
FROM net_sales_by_year AS n1
JOIN net_sales_by_year AS n2 ON n1.s_store_name = n2.s_store_name
WHERE n1.d_year = 2001 AND n2.d_year = 2002
ORDER BY n1.s_store_name"""
    },
    {
        "tier": 4,
        "question": "What is the return rate for items sold on weekends (d_weekend = 'Y') versus weekdays (d_weekend = 'N'), broken down by customer age group? Use these age groups based on (d_year - c_birth_year): '< 20', '20-29', '30-39', '40-49', '50-59', '60+'. Show age group, weekend flag, and return rate. Order by age group, then by weekend flag.",
        "sql": """-- Return rate for weekends vs weekdays by customer age group
WITH sales_by_group AS (
    SELECT
        CASE
            WHEN (d.d_year - c.c_birth_year) < 20 THEN '< 20'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 20 AND 29 THEN '20-29'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 30 AND 39 THEN '30-39'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 40 AND 49 THEN '40-49'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 50 AND 59 THEN '50-59'
            WHEN (d.d_year - c.c_birth_year) >= 60 THEN '60+'
        END AS age_group,
        d.d_weekend AS weekend_flag,
        SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN date_dim AS d ON ss.ss_sold_date_sk = d.d_date_sk
    JOIN customer AS c ON ss.ss_customer_sk = c.c_customer_sk
    WHERE c.c_birth_year IS NOT NULL AND d.d_year IS NOT NULL
    GROUP BY age_group, d.d_weekend
),
returns_by_group AS (
    SELECT
        CASE
            WHEN (d.d_year - c.c_birth_year) < 20 THEN '< 20'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 20 AND 29 THEN '20-29'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 30 AND 39 THEN '30-39'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 40 AND 49 THEN '40-49'
            WHEN (d.d_year - c.c_birth_year) BETWEEN 50 AND 59 THEN '50-59'
            WHEN (d.d_year - c.c_birth_year) >= 60 THEN '60+'
        END AS age_group,
        d.d_weekend AS weekend_flag,
        SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN date_dim AS d ON sr.sr_returned_date_sk = d.d_date_sk
    JOIN customer AS c ON sr.sr_customer_sk = c.c_customer_sk
    WHERE c.c_birth_year IS NOT NULL AND d.d_year IS NOT NULL
    GROUP BY age_group, d.d_weekend
)
SELECT
    COALESCE(sa.age_group, ra.age_group) AS age_group,
    COALESCE(sa.weekend_flag, ra.weekend_flag) AS weekend_flag,
    (COALESCE(ra.total_returns, 0) / NULLIF(COALESCE(sa.total_sales, 0), 0)) * 100 AS return_rate
FROM sales_by_group AS sa
FULL OUTER JOIN returns_by_group AS ra
    ON sa.age_group = ra.age_group AND sa.weekend_flag = ra.weekend_flag
ORDER BY age_group, weekend_flag"""
    },
    {
        "tier": 4,
        "question": "Which item brands had a decrease in return rate from 2001 to 2002 for stores in the 'TN' state? Show the brand name and decrease amount (2001 rate minus 2002 rate). Only include brands where the return rate decreased (positive difference). Order by decrease descending, then by brand name alphabetically.",
        "sql": """-- Item brands with decreased return rate from 2001 to 2002 for TN stores
WITH sales_by_brand_year AS (
    SELECT i.i_brand, d.d_year, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN item AS i ON ss.ss_item_sk = i.i_item_sk
    JOIN store AS s ON ss.ss_store_sk = s.s_store_sk
    JOIN date_dim AS d ON ss.ss_sold_date_sk = d.d_date_sk
    WHERE s.s_state = 'TN' AND d.d_year IN (2001, 2002) AND i.i_brand IS NOT NULL
    GROUP BY i.i_brand, d.d_year
),
returns_by_brand_year AS (
    SELECT i.i_brand, d.d_year, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN item AS i ON sr.sr_item_sk = i.i_item_sk
    JOIN store AS s ON sr.sr_store_sk = s.s_store_sk
    JOIN date_dim AS d ON sr.sr_returned_date_sk = d.d_date_sk
    WHERE s.s_state = 'TN' AND d.d_year IN (2001, 2002) AND i.i_brand IS NOT NULL
    GROUP BY i.i_brand, d.d_year
),
rates_by_brand_year AS (
    SELECT
        COALESCE(sa.i_brand, ra.i_brand) AS i_brand,
        COALESCE(sa.d_year, ra.d_year) AS d_year,
        (COALESCE(ra.total_returns, 0) / NULLIF(COALESCE(sa.total_sales, 0), 0)) * 100 AS return_rate
    FROM sales_by_brand_year AS sa
    FULL OUTER JOIN returns_by_brand_year AS ra
        ON sa.i_brand = ra.i_brand AND sa.d_year = ra.d_year
)
SELECT
    r1.i_brand,
    r1.return_rate - r2.return_rate AS decrease_in_return_rate
FROM rates_by_brand_year AS r1
JOIN rates_by_brand_year AS r2 ON r1.i_brand = r2.i_brand
WHERE r1.d_year = 2001 AND r2.d_year = 2002
  AND r1.return_rate IS NOT NULL AND r2.return_rate IS NOT NULL
  AND r1.return_rate - r2.return_rate > 0
ORDER BY decrease_in_return_rate DESC, r1.i_brand"""
    },
    {
        "tier": 4,
        "question": "For each item class and customer preferred flag (Y or N), show the average net sales per sales transaction (net sales divided by the number of distinct sales ticket numbers). Exclude rows where item class is NULL or preferred flag is NULL. Order by average net sales descending, then by item class alphabetically.",
        "sql": """-- Average net sales per transaction by item class and preferred flag
WITH sales_by_class_pref AS (
    SELECT
        i.i_class,
        c.c_preferred_cust_flag,
        SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales,
        COUNT(DISTINCT ss.ss_ticket_number) AS txn_count
    FROM store_sales AS ss
    JOIN item AS i ON ss.ss_item_sk = i.i_item_sk
    JOIN customer AS c ON ss.ss_customer_sk = c.c_customer_sk
    WHERE i.i_class IS NOT NULL AND c.c_preferred_cust_flag IS NOT NULL
    GROUP BY i.i_class, c.c_preferred_cust_flag
),
returns_by_class_pref AS (
    SELECT
        i.i_class,
        c.c_preferred_cust_flag,
        SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN item AS i ON sr.sr_item_sk = i.i_item_sk
    JOIN customer AS c ON sr.sr_customer_sk = c.c_customer_sk
    WHERE i.i_class IS NOT NULL AND c.c_preferred_cust_flag IS NOT NULL
    GROUP BY i.i_class, c.c_preferred_cust_flag
)
SELECT
    COALESCE(sa.i_class, ra.i_class) AS i_class,
    COALESCE(sa.c_preferred_cust_flag, ra.c_preferred_cust_flag) AS c_preferred_cust_flag,
    (COALESCE(sa.total_sales, 0) - COALESCE(ra.total_returns, 0)) / NULLIF(sa.txn_count, 0) AS avg_net_sales
FROM sales_by_class_pref AS sa
FULL OUTER JOIN returns_by_class_pref AS ra
    ON sa.i_class = ra.i_class AND sa.c_preferred_cust_flag = ra.c_preferred_cust_flag
ORDER BY avg_net_sales DESC, i_class"""
    },
    {
        "tier": 4,
        "question": "For each store, show the store name, its net sales, the overall average net sales across all stores, and the percentage calculated as (store net sales / average net sales) * 100. Order by percentage descending, then by store name alphabetically for ties.",
        "sql": """-- Store net sales vs average, with percentage
WITH sales_by_store AS (
    SELECT s.s_store_name, SUM(ss.ss_sales_price * ss.ss_quantity) AS total_sales
    FROM store_sales AS ss
    JOIN store AS s ON ss.ss_store_sk = s.s_store_sk
    GROUP BY s.s_store_name
),
returns_by_store AS (
    SELECT s.s_store_name, SUM(sr.sr_return_amt) AS total_returns
    FROM store_returns AS sr
    JOIN store AS s ON sr.sr_store_sk = s.s_store_sk
    GROUP BY s.s_store_name
),
net_sales_by_store AS (
    SELECT
        COALESCE(sa.s_store_name, ra.s_store_name) AS s_store_name,
        COALESCE(sa.total_sales, 0) - COALESCE(ra.total_returns, 0) AS net_sales
    FROM sales_by_store AS sa
    FULL OUTER JOIN returns_by_store AS ra ON sa.s_store_name = ra.s_store_name
)
SELECT
    s_store_name,
    net_sales,
    AVG(net_sales) OVER () AS avg_net_sales,
    ROUND((net_sales / NULLIF(AVG(net_sales) OVER (), 0)) * 100, 2) AS pct_net_sales_vs_avg
FROM net_sales_by_store
ORDER BY pct_net_sales_vs_avg DESC, s_store_name"""
    },
]


def generate_questions(output_path):
    """Write questions.json."""
    questions = [b["question"] for b in BENCHMARK]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4)
    print(f"  Written: {output_path} ({len(questions)} questions)")


def validate_and_save_baseline(con):
    """Validate all baseline SQL queries and save results to log."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    print("  Validating baseline SQL...")
    results_data = []
    all_ok = True
    for i, entry in enumerate(BENCHMARK):
        nbr = i + 1
        sql = entry["sql"]
        try:
            df = con.execute(sql).fetchdf()
            row_count = len(df)
            result_json = df.to_dict("records")
            error = None
            print(f"    Q{nbr:2d} (Tier {entry['tier']}): OK - {row_count} rows")
        except Exception as e:
            row_count = 0
            result_json = []
            error = str(e)
            all_ok = False
            print(f"    Q{nbr:2d} (Tier {entry['tier']}): FAILED - {e}")

        results_data.append({
            "model": BASELINE_MODEL,
            "SF": SF,
            "timestamp": timestamp,
            "nbr": nbr,
            "question": entry["question"],
            "duration_s": 0,
            "sql_query": sql,
            "attempts": 1,
            "feedback_iterations": 0,
            "result": result_json,
            "result_count": row_count,
            "error_details": error,
        })

    if not all_ok:
        print("  WARNING: Some baseline queries failed!")

    # Save baseline results to log
    log_dir = os.path.join(OUTPUT_DIR, "log")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{timestamp}_{BASELINE_MODEL}.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=4)
    print(f"  Written: {log_path}")

    return all_ok


# ── Main ────────────────────────────────────────────────────────────────────
def build_all():
    """Run the full pipeline."""
    print("=" * 60)
    print("Building benchmark artifacts")
    print("=" * 60)

    # Connect to DB
    if SF < 1:
        schema_name = f"DS{str(SF).replace('.', '_')}"
    else:
        schema_name = f"DS{SF:02d}"
    db_path = os.path.join(OUTPUT_DIR, f"{schema_name}.duckdb")

    if not pathlib.Path(db_path).exists():
        print(f"Generating TPC-DS data at SF={SF}...")
        c = duckdb.connect(db_path)
        c.sql("SET memory_limit = '14GB'")
        c.sql(f"CALL dsdgen(sf={SF})")
        c.close()

    con = duckdb.connect()
    con.sql(f"ATTACH '{db_path}' AS ds (READ_ONLY); USE ds;")
    print(f"Connected to {db_path}")

    # Step 1: Introspect
    print("\n[1/3] Introspecting schema...")
    schema = introspect_all(con)

    # Step 2: Generate semantic model
    print("\n[2/3] Generating semantic model...")
    sm_path = os.path.join(OUTPUT_DIR, "semantic_model.txt")
    generate_semantic_model(schema, sm_path, con)

    # Step 3: Generate questions + validate baseline
    print("\n[3/3] Generating questions & validating baseline...")
    q_path = os.path.join(OUTPUT_DIR, "questions.json")
    generate_questions(q_path)
    ok = validate_and_save_baseline(con)

    con.close()

    print("\n" + "=" * 60)
    if ok:
        print("BUILD COMPLETE - all queries validated successfully")
    else:
        print("BUILD COMPLETE - WARNING: some queries failed, check output above")
    print("=" * 60)
    print(f"  semantic_model.txt : {sm_path}")
    print(f"  questions.json     : {q_path}")


if __name__ == "__main__":
    build_all()
