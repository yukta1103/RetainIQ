"""Declarative spec for the 9 raw Olist tables.

Keeping keys/dtypes/dates as data (rather than scattered read_csv kwargs) means
the loader, the cleaner and the validator all agree on what each table *should*
look like, and the referential-integrity checks can be generated from the same
source of truth.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class TableSpec:
    name: str
    filename: str
    expected_bytes: int
    # Columns that jointly identify a row. Empty tuple = no natural key.
    primary_key: tuple[str, ...]
    date_columns: tuple[str, ...] = ()
    # (local_column, parent_table, parent_column)
    foreign_keys: tuple[tuple[str, str, str], ...] = ()
    # Columns that must never be null in the cleaned output.
    required: tuple[str, ...] = ()
    dtypes: dict[str, str] = field(default_factory=dict)
    note: str = ""


CUSTOMERS = TableSpec(
    name="customers",
    filename="olist_customers_dataset.csv",
    expected_bytes=9_033_957,
    primary_key=("customer_id",),
    required=("customer_id", "customer_unique_id"),
    dtypes={"customer_zip_code_prefix": "int32"},
    note=(
        "CRITICAL: `customer_id` is a per-ORDER surrogate key — it is unique in "
        "this table and appears exactly once in orders. The stable identity of a "
        "human being is `customer_unique_id`. Doing RFM on customer_id yields "
        "frequency == 1 for all 99k customers and a meaningless model. All "
        "customer-level aggregation in this project keys on customer_unique_id."
    ),
)

ORDERS = TableSpec(
    name="orders",
    filename="olist_orders_dataset.csv",
    expected_bytes=17_654_914,
    primary_key=("order_id",),
    date_columns=(
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ),
    foreign_keys=(("customer_id", "customers", "customer_id"),),
    required=("order_id", "customer_id", "order_status", "order_purchase_timestamp"),
    note="One row per order. order_status drives the revenue filter.",
)

ORDER_ITEMS = TableSpec(
    name="order_items",
    filename="olist_order_items_dataset.csv",
    expected_bytes=15_438_671,
    primary_key=("order_id", "order_item_id"),
    date_columns=("shipping_limit_date",),
    foreign_keys=(
        ("order_id", "orders", "order_id"),
        ("product_id", "products", "product_id"),
        ("seller_id", "sellers", "seller_id"),
    ),
    required=("order_id", "order_item_id", "product_id", "seller_id", "price"),
    dtypes={"order_item_id": "int16", "price": "float64", "freight_value": "float64"},
    note=(
        "Line-item grain: order_item_id is a 1..n sequence within an order, not a "
        "quantity. Two units of the same product appear as two rows."
    ),
)

ORDER_PAYMENTS = TableSpec(
    name="order_payments",
    filename="olist_order_payments_dataset.csv",
    expected_bytes=5_777_138,
    primary_key=("order_id", "payment_sequential"),
    foreign_keys=(("order_id", "orders", "order_id"),),
    required=("order_id", "payment_sequential", "payment_type", "payment_value"),
    dtypes={
        "payment_sequential": "int16",
        "payment_installments": "int16",
        "payment_value": "float64",
    },
    note=(
        "An order can split across several payment rows (card + voucher), so this "
        "table does NOT join 1:1 to orders — it must be aggregated first."
    ),
)

ORDER_REVIEWS = TableSpec(
    name="order_reviews",
    filename="olist_order_reviews_dataset.csv",
    expected_bytes=14_451_670,
    primary_key=(),  # review_id is NOT unique — see note
    date_columns=("review_creation_date", "review_answer_timestamp"),
    foreign_keys=(("order_id", "orders", "order_id"),),
    required=("review_id", "order_id", "review_score"),
    dtypes={"review_score": "int8"},
    note=(
        "review_id repeats (a single survey can cover several orders), so there is "
        "no clean PK. Aggregated to one mean score per order before joining."
    ),
)

PRODUCTS = TableSpec(
    name="products",
    filename="olist_products_dataset.csv",
    expected_bytes=2_379_446,
    primary_key=("product_id",),
    required=("product_id",),
    note="product_category_name is Portuguese and ~1.85% null; both handled in clean.py.",
)

SELLERS = TableSpec(
    name="sellers",
    filename="olist_sellers_dataset.csv",
    expected_bytes=174_703,
    primary_key=("seller_id",),
    required=("seller_id",),
    dtypes={"seller_zip_code_prefix": "int32"},
)

CATEGORY_TRANSLATION = TableSpec(
    name="category_translation",
    filename="product_category_name_translation.csv",
    expected_bytes=2_613,
    primary_key=("product_category_name",),
    required=("product_category_name", "product_category_name_english"),
)

GEOLOCATION = TableSpec(
    name="geolocation",
    filename="olist_geolocation_dataset.csv",
    expected_bytes=61_273_883,
    primary_key=(),  # heavily duplicated by design
    required=("geolocation_zip_code_prefix",),
    dtypes={"geolocation_zip_code_prefix": "int32"},
    note=(
        "Zip-prefix -> lat/lng lookup with ~261k duplicate rows. Loaded and "
        "reported for completeness but NOT joined: it is many-to-many against "
        "customers and would fan out the transaction grain. Customer state/city "
        "already come from the customers table."
    ),
)

# Order matters: parents before children, so FK checks can rely on prior loads.
ALL_TABLES: tuple[TableSpec, ...] = (
    CUSTOMERS,
    ORDERS,
    PRODUCTS,
    SELLERS,
    CATEGORY_TRANSLATION,
    ORDER_ITEMS,
    ORDER_PAYMENTS,
    ORDER_REVIEWS,
    GEOLOCATION,
)

TABLES_BY_NAME = {t.name: t for t in ALL_TABLES}
