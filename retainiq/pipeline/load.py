"""Stage 1 — load raw CSVs into DataFrames and report what arrived."""

from __future__ import annotations

import pandas as pd

from retainiq.config import RAW_DIR
from retainiq.data.schema import ALL_TABLES, TableSpec


def load_table(spec: TableSpec) -> pd.DataFrame:
    path = RAW_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. Run: python scripts/download_data.py"
        )
    return pd.read_csv(
        path,
        dtype=spec.dtypes or None,
        parse_dates=list(spec.date_columns) or None,
    )


def load_all() -> dict[str, pd.DataFrame]:
    return {spec.name: load_table(spec) for spec in ALL_TABLES}


def describe(tables: dict[str, pd.DataFrame]) -> None:
    """Print schema + shape for every table so load can be eyeballed."""
    for spec in ALL_TABLES:
        df = tables[spec.name]
        print(f"\n{'=' * 78}")
        print(f"{spec.name.upper()}  ({spec.filename})")
        print(f"  shape: {df.shape[0]:,} rows x {df.shape[1]} cols"
              f"   memory: {df.memory_usage(deep=True).sum() / 1e6:,.1f} MB")
        if spec.primary_key:
            dup = df.duplicated(subset=list(spec.primary_key)).sum()
            flag = "OK" if dup == 0 else f"{dup:,} DUPLICATES"
            print(f"  primary key: {' + '.join(spec.primary_key)}  ->  {flag}")
        else:
            print("  primary key: (none declared)")
        print(f"{'-' * 78}")
        print(f"  {'column':<34} {'dtype':<16} {'nulls':>10} {'null %':>8}  {'distinct':>10}")
        for col in df.columns:
            n_null = int(df[col].isna().sum())
            pct = 100.0 * n_null / len(df) if len(df) else 0.0
            print(
                f"  {col:<34} {str(df[col].dtype):<16} {n_null:>10,} "
                f"{pct:>7.2f}%  {df[col].nunique(dropna=True):>10,}"
            )
        if spec.note:
            print(f"\n  NOTE: {_wrap(spec.note, 72)}")


def _wrap(text: str, width: int) -> str:
    import textwrap

    return ("\n" + " " * 8).join(textwrap.wrap(text, width))
