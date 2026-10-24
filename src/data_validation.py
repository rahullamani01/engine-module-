"""Input validation. Invalid data is REPORTED and REJECTED, never silently repaired.

Each issue names the table, row (DataFrame index label; None for table-level
problems), column and reason. Run `validate_or_raise` before forecasting.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src import config, schemas
from src.data_generator import SyntheticDataset


@dataclass(frozen=True)
class ValidationIssue:
    table: str
    row: int | None
    column: str | None
    reason: str

    def __str__(self) -> str:
        loc = f"row {self.row}" if self.row is not None else "table"
        col = f", column '{self.column}'" if self.column else ""
        return f"[{self.table}] {loc}{col}: {self.reason}"


@dataclass
class ValidationReport:
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.issues

    def add(self, table: str, row, column: str | None, reason: str) -> None:
        try:
            row = int(row) if row is not None else None
        except (TypeError, ValueError):
            row = None
        self.issues.append(ValidationIssue(table, row, column, reason))

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([(i.table, i.row, i.column, i.reason) for i in self.issues],
                            columns=["table", "row", "column", "reason"])

    def summary(self, limit: int = 10) -> str:
        if self.ok:
            return "valid"
        lines = [str(i) for i in self.issues[:limit]]
        if len(self.issues) > limit:
            lines.append(f"... and {len(self.issues) - limit} more issues")
        return f"{len(self.issues)} validation issue(s):\n" + "\n".join(lines)

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise DataValidationError(self)


class DataValidationError(ValueError):
    def __init__(self, report: ValidationReport):
        self.report = report
        super().__init__(report.summary())


# --- helpers ---------------------------------------------------------------
def _flag(report, table, df, mask, column, reason, show_value=True) -> None:
    for idx in df.index[np.asarray(mask, dtype=bool)]:
        extra = f" (value={df.at[idx, column]!r})" if show_value and column in df.columns else ""
        report.add(table, idx, column, reason + extra)


def _has_columns(df, table, required, report) -> bool:
    missing = [c for c in required if c not in df.columns]
    for c in missing:
        report.add(table, None, c, "required column is missing")
    return not missing


def _missing_mask(series: pd.Series) -> pd.Series:
    empty = series.astype("string").str.strip().eq("").fillna(False).astype(bool)
    return series.isna() | empty


def _check_text_ids(df, table, column, report) -> pd.Series:
    """Flag missing and duplicate identifiers; return mask of usable (non-missing) ids."""
    missing = _missing_mask(df[column])
    _flag(report, table, df, missing, column, "missing identifier", show_value=False)
    dup = df[column].duplicated(keep=False) & ~missing
    _flag(report, table, df, dup, column, "duplicate identifier")
    return ~missing


def _check_numeric(df, table, column, report, *, min_value=None, integer=False) -> pd.Series:
    """Return the column as float with NaN where unusable; report every problem."""
    raw = df[column]
    missing = _missing_mask(raw)
    numeric = pd.to_numeric(raw, errors="coerce").astype(float)
    _flag(report, table, df, missing, column, "missing value", show_value=False)
    _flag(report, table, df, ~missing & numeric.isna(), column, "non-numeric value")
    infinite = np.isinf(numeric.fillna(0.0))
    _flag(report, table, df, infinite, column, "infinite value")
    finite = numeric.notna() & ~infinite
    if min_value is not None:
        _flag(report, table, df, finite & (numeric < min_value), column, f"value below minimum {min_value}")
    if integer:
        _flag(report, table, df, finite & (numeric != np.floor(numeric)), column, "value must be a whole number")
    return numeric.where(finite)


# --- tables ----------------------------------------------------------------
def validate_locations(df, report) -> tuple[set[str], set[str]]:
    """Return (store_ids, warehouse_ids) found in locations.csv."""
    t = "locations"
    if not _has_columns(df, t, schemas.LOCATION_COLUMNS, report):
        return set(), set()
    usable = _check_text_ids(df, t, "location_id", report)
    _flag(report, t, df, _missing_mask(df["region"]), "region", "missing value", show_value=False)
    tmiss = _missing_mask(df["type"])
    _flag(report, t, df, tmiss, "type", "missing value", show_value=False)
    _flag(report, t, df, ~tmiss & ~df["type"].isin(config.LOCATION_TYPES), "type",
          f"type must be one of {config.LOCATION_TYPES}")
    ids = df["location_id"].astype(str)
    return (set(ids[usable & (df["type"] == "store")]), set(ids[usable & (df["type"] == "warehouse")]))


def validate_products(df, report) -> set[str]:
    t = "products"
    if not _has_columns(df, t, schemas.PRODUCT_COLUMNS, report):
        return set()
    usable = _check_text_ids(df, t, "sku_id", report)
    _check_numeric(df, t, "shelf_life_days", report, min_value=1, integer=True)
    _check_numeric(df, t, "lost_sale_cost", report, min_value=0)
    _check_numeric(df, t, "overstock_cost", report, min_value=0)
    return set(df.loc[usable, "sku_id"].astype(str))


def validate_demand_history(demand, store_ids, sku_ids, report=None,
                            max_week: int = config.N_HISTORY_WEEKS) -> ValidationReport:
    report = report if report is not None else ValidationReport()
    t = "demand_history"
    if not _has_columns(demand, t, schemas.DEMAND_HISTORY_COLUMNS, report):
        return report

    for col, valid in (("store_id", store_ids), ("sku_id", sku_ids)):
        missing = _missing_mask(demand[col])
        _flag(report, t, demand, missing, col, "missing identifier", show_value=False)
        orphan = ~missing & ~demand[col].astype(str).isin(valid)
        _flag(report, t, demand, orphan, col, "orphan reference: id not found in reference table")

    _check_numeric(demand, t, "units_sold", report, min_value=0)

    raw = demand["week_start"]
    date_missing = _missing_mask(raw)
    _flag(report, t, demand, date_missing, "week_start", "missing value", show_value=False)
    dates = pd.to_datetime(raw.where(~date_missing), format="%Y-%m-%d", errors="coerce")
    _flag(report, t, demand, ~date_missing & dates.isna(), "week_start", "invalid date (expected YYYY-MM-DD)")

    valid_date = dates.notna()
    week = pd.Series(np.nan, index=demand.index)
    if valid_date.any():
        anchor = dates[valid_date].min()  # week 1 = earliest week_start in the table
        offset = (dates - anchor).dt.days
        off_grid = valid_date & (offset % 7 != 0)
        _flag(report, t, demand, off_grid, "week_start",
              f"not on the weekly grid (must be a whole number of weeks after {anchor.date()})")
        week = (offset // 7 + 1).where(valid_date & ~off_grid)
        _flag(report, t, demand, week.notna() & (week > max_week), "week_start",
              f"week beyond history window (max {max_week} weeks from {anchor.date()})")

    keyed = ~_missing_mask(demand["store_id"]) & ~_missing_mask(demand["sku_id"]) & valid_date
    dup = keyed & demand.assign(_d=dates).duplicated(["store_id", "sku_id", "_d"], keep=False)
    _flag(report, t, demand, dup, "week_start", "duplicate store-SKU-week observation", show_value=False)

    # Completeness: every store x SKU needs every week 1..max_week (MA4/lags need contiguity).
    ok = keyed & week.notna()
    seen = set(zip(demand.loc[ok, "store_id"].astype(str), demand.loc[ok, "sku_id"].astype(str),
                   week[ok].astype(int)))
    for s in sorted(store_ids):
        for k in sorted(sku_ids):
            gaps = [w for w in range(1, max_week + 1) if (s, k, w) not in seen]
            if gaps:
                shown = gaps[:8] + (["..."] if len(gaps) > 8 else [])
                report.add(t, None, "week_start", f"series {s}/{k} missing {len(gaps)} week(s): {shown}")
    return report


def validate_dataset(dataset: SyntheticDataset) -> ValidationReport:
    report = ValidationReport()
    store_ids, _ = validate_locations(dataset.locations, report)
    sku_ids = validate_products(dataset.products, report)
    validate_demand_history(dataset.demand_history, store_ids, sku_ids, report)
    return report


def validate_or_raise(dataset: SyntheticDataset) -> ValidationReport:
    report = validate_dataset(dataset)
    report.raise_if_invalid()
    return report
