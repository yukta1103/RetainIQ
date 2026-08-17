"""Central configuration: paths and the tunable knobs of the pipeline.

Everything that a reviewer might want to challenge ("why delivered-only?",
"why exclude 2016?") lives here as a named constant rather than being buried
as a magic value inside a transform.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

TRANSACTIONS_PARQUET = PROCESSED_DIR / "transactions.parquet"
ORDERS_PARQUET = PROCESSED_DIR / "orders.parquet"

for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --- Source ---------------------------------------------------------------

# Mirror of the raw Kaggle Olist tables. Byte sizes match the Kaggle originals
# (asserted in data/download.py), so this is the same data without needing a
# Kaggle API token — which matters because Streamlit Cloud can't hold one.
HF_REPO = "aviahYadler/Olist_Ecommerce_Dataset"
HF_BASE_URL = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"


# --- Business rules -------------------------------------------------------

# Order statuses that count as realised revenue.
#
# Olist has 8 statuses. 'delivered' (97.0% of orders) is the only one where we
# know money changed hands and the goods arrived. 'shipped'/'invoiced'/
# 'processing' are in-flight: including them would inflate revenue with orders
# that may still cancel. 'canceled'/'unavailable' are explicit non-revenue.
#
# The cost of delivered-only is a mild recency bias at the very end of the
# window (recent orders haven't had time to reach 'delivered' yet). We handle
# that separately by trimming the tail — see ANALYSIS_END below.
REVENUE_ORDER_STATUSES = ("delivered",)

# Olist's raw span is 2016-09-04 to 2018-10-17, but the tails are unusable:
#   * 2016-09/10/12 hold ~300 orders total — a pilot period, not a business.
#   * After 2018-08-31 the delivered-only filter truncates volume artificially,
#     because orders placed in Sept/Oct 2018 were still in transit at extract.
# Restricting to this window gives 24 clean months of comparable monthly volume.
ANALYSIS_START = "2017-01-01"
ANALYSIS_END = "2018-08-31"

# The customer identity column. THIS IS THE ONE THAT MATTERS — see schema.py.
CUSTOMER_KEY = "customer_unique_id"

# Revenue definition: item price + freight, summed to order level.
# See pipeline/build.py for why this rather than payments.
REVENUE_COMPONENTS = ("price", "freight_value")
