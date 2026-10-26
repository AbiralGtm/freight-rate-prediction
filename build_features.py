"""Feature engineering for the freight rate assessment (solution guide, Step 4).

Takes a frame that has already been through :mod:`clean_data` and adds the model inputs. The
same function serves train, validation and the December chart rows, so the 31 chart predictions
come out of exactly the same matrix as the 12,000 submitted ones.

What goes in, and why:

* **Size and scale** — ``distance``, ``log_distance``, ``weight``, ``weight_per_mile``. Mean
  rate per mile falls from $2.815 on sub-250-mile loads to $1.903 on 2,500-3,500-mile loads, so
  the short-haul premium is non-linear in distance and the log term gives the model a shape it
  can split on cheaply.
* **Circuity** — ``distance / haversine_miles``. The raw haversine distance is useless on its
  own (r = 0.9995 against the supplied ``distance``), but the *ratio* is not: it runs from 1.06
  to 9.6 with a median of 1.18 and correlates 0.12 with rate per mile. That is road indirectness
  the straight-line distance cannot express, so the ratio is kept and the raw haversine dropped.
* **Categoricals** — ``pickup``, ``delivery``, ``equipment`` as pandas categoricals, which
  LightGBM splits natively. With ~4,014 observed lanes over 48,000 rows the typical lane holds
  about 12 loads, far too thin for a per-lane target encoding, so the model pools across lanes
  sharing an origin or destination instead.
* **Coordinates are dropped.** Every city maps to exactly one lat/lon pair, so the four raw
  coordinate columns are a deterministic lookup of the city name. They are kept on the frame for
  the circuity calculation and then left out of the matrix.
* **Date** — ``day_of_week``, ``day_of_year``, ``month``, cyclical sin/cos pairs, and
  ``days_since_start`` as the linear trend term. The trend term is the only feature that can
  move the model past 31 October at all, and a tree can still only predict inside the range of
  trend values it saw; see :func:`trend_coverage` for the check that keeps that honest.
* **Market** — ``market_index`` and ``quote_signal``. They correlate only -0.07 with each other,
  so they carry different information and both are needed.
* **Cleaning flags** — the indicator columns :mod:`clean_data` produced, so an imputed value is
  never silently indistinguishable from a measured one.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from clean_data import CATEGORICAL_COLUMNS, COORD_COLUMNS

EARTH_RADIUS_MILES = 3958.7613
TREND_ORIGIN = pd.Timestamp("2025-01-01")

NUMERIC_FEATURES = [
    "distance",
    "log_distance",
    "weight",
    "weight_per_mile",
    "circuity",
    "market_index",
    "quote_signal",
    "days_since_start",
    "day_of_week",
    "day_of_year",
    "month",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

FLAG_FEATURES = [
    "is_weight_negative",
    "is_weight_missing",
    "is_market_missing",
    "is_quote_missing",
    "is_pickup_unseen",
    "is_delivery_unseen",
]

CATEGORICAL_FEATURES = list(CATEGORICAL_COLUMNS)
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + FLAG_FEATURES


def add_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``frame`` with every engineered column added.

    ``frame`` must already be cleaned, which guarantees a datetime ``date``, non-null
    coordinates and market signals, and a strictly positive ``distance``.
    """
    _assert_cleaned(frame)
    out = frame.copy()

    out["log_distance"] = np.log(out["distance"])
    out["weight_per_mile"] = out["weight"] / out["distance"]

    out["haversine_miles"] = _haversine_miles(out)
    out["circuity"] = out["distance"] / out["haversine_miles"]

    date = out["date"]
    out["days_since_start"] = (date - TREND_ORIGIN).dt.days
    out["day_of_week"] = date.dt.day_of_week
    out["day_of_year"] = date.dt.day_of_year
    out["month"] = date.dt.month

    # Cyclical encodings so that Sunday sits next to Monday, and 31 December next to 1 January,
    # rather than at the opposite end of an integer axis.
    out["day_of_week_sin"], out["day_of_week_cos"] = _cyclical(out["day_of_week"], 7)
    out["day_of_year_sin"], out["day_of_year_cos"] = _cyclical(out["day_of_year"], 365.25)

    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype("category")

    _assert_featured(out, frame)
    return out


def design_matrix(frame: pd.DataFrame, categories: dict | None = None) -> pd.DataFrame:
    """Return the model matrix: ``FEATURE_COLUMNS`` in a fixed order, nothing else.

    ``categories`` carries the category levels learnt on the training frame. Passing it when
    scoring validation or December keeps the integer codes aligned; without it, pandas would
    build the levels from whichever cities happen to appear in that frame and the model would
    read the wrong category entirely.
    """
    featured = frame if "circuity" in frame.columns else add_features(frame)

    missing = [column for column in FEATURE_COLUMNS if column not in featured.columns]
    if missing:
        raise KeyError(f"frame is missing engineered columns: {missing}")

    matrix = featured[FEATURE_COLUMNS].copy()
    if categories is not None:
        for column, levels in categories.items():
            # set_categories maps any level absent from training to NaN on purpose. LightGBM
            # reads that as a missing category, and is_pickup_unseen / is_delivery_unseen tell
            # the model which rows it happened to.
            matrix[column] = matrix[column].astype("category").cat.set_categories(levels)
    return matrix


def category_levels(frame: pd.DataFrame) -> dict[str, pd.Index]:
    """Capture the categorical levels of a training frame, for :func:`design_matrix`."""
    return {
        column: pd.Categorical(frame[column]).categories
        for column in CATEGORICAL_FEATURES
    }


def trend_coverage(train: pd.DataFrame, future: pd.DataFrame) -> dict[str, int]:
    """Report how far ``future`` reaches past the trend values a tree model has ever seen.

    A gradient-boosted tree cannot extrapolate: every ``days_since_start`` beyond the training
    maximum lands in the same terminal leaf as the last training day, so the trend term flattens
    out exactly where it is needed most. This returns the size of that gap rather than hiding
    it, and the number belongs in the report.
    """
    train_max = int(train["days_since_start"].max())
    future_max = int(future["days_since_start"].max())
    return {
        "train_max_day": train_max,
        "future_max_day": future_max,
        "days_beyond_training": max(0, future_max - train_max),
        "rows_beyond_training": int((future["days_since_start"] > train_max).sum()),
    }


def _haversine_miles(frame: pd.DataFrame) -> pd.Series:
    """Great-circle distance between pickup and delivery, in miles."""
    lat1, lon1, lat2, lon2 = (
        np.radians(frame[column].to_numpy()) for column in COORD_COLUMNS
    )
    chord = (
        np.sin((lat2 - lat1) / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    )
    miles = 2 * EARTH_RADIUS_MILES * np.arcsin(np.sqrt(chord))
    return pd.Series(miles, index=frame.index)


def _cyclical(values: pd.Series, period: float) -> tuple[pd.Series, pd.Series]:
    """Project an integer calendar position onto the unit circle."""
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def _assert_cleaned(frame: pd.DataFrame) -> None:
    """Fail loudly if a raw frame is passed instead of a cleaned one."""
    required = ["date", "distance", "weight", "market_index", "quote_signal", *COORD_COLUMNS]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"frame is not cleaned; missing {missing}. Run clean_data first.")
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise TypeError("date is not datetime; run clean_data first")


def _assert_featured(out: pd.DataFrame, source: pd.DataFrame) -> None:
    """Guard the invariants the model step depends on."""
    if len(out) != len(source):
        raise ValueError(f"row count changed: {len(source)} -> {len(out)}")

    numeric = out[NUMERIC_FEATURES]
    nulls = numeric.isna().sum()
    if nulls.any():
        raise ValueError(f"nulls in engineered features: {nulls[nulls > 0].to_dict()}")
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("non-finite values in engineered features")
    if (out["circuity"] < 1).any():
        # Road distance below straight-line distance means the coordinate join went wrong.
        worst = float(out["circuity"].min())
        raise ValueError(f"circuity below 1.0 ({worst:.3f}); check the coordinate lookup")


if __name__ == "__main__":
    from clean_data import load_clean

    train, validation, december, _ = load_clean()
    train, validation, december = (add_features(f) for f in (train, validation, december))

    levels = category_levels(train)
    matrices = {
        "train": design_matrix(train, levels),
        "validation": design_matrix(validation, levels),
        "december": design_matrix(december, levels),
    }
    for name, matrix in matrices.items():
        print(f"{name:11s} {matrix.shape[0]:>6,} rows x {matrix.shape[1]} features")

    print()
    print(f"numeric     ({len(NUMERIC_FEATURES):>2}): {', '.join(NUMERIC_FEATURES)}")
    print(f"categorical ({len(CATEGORICAL_FEATURES):>2}): {', '.join(CATEGORICAL_FEATURES)}")
    print(f"flags       ({len(FLAG_FEATURES):>2}): {', '.join(FLAG_FEATURES)}")
    print(f"dropped         : {', '.join(COORD_COLUMNS)}, haversine_miles")

    print()
    circuity = train["circuity"]
    print(f"circuity        : min {circuity.min():.3f}  median {circuity.median():.3f}  "
          f"max {circuity.max():.3f}")
    print("corr(circuity, rate_per_mile): "
          f"{train['circuity'].corr(train['rate_per_mile']):+.4f}")

    print()
    for name in ("validation", "december"):
        gap = trend_coverage(train, validation if name == "validation" else december)
        print(f"trend gap, {name:10s}: {gap['days_beyond_training']} days past training "
              f"(day {gap['train_max_day']} -> {gap['future_max_day']}), "
              f"{gap['rows_beyond_training']:,} rows affected")
    print("A tree cannot extrapolate a trend; every row above lands in the last training leaf.")
