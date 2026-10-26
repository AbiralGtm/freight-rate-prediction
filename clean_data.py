"""Loading and cleaning for the freight rate assessment.

One cleaning path serves all three frames (train, validation, December chart inputs), so the
notebook, the training script and the prediction script cannot drift apart. Every imputation
statistic is fitted on the training frame only; see :func:`fit_cleaner`.

The decisions implemented here come from ``eda.ipynb``:

* Negative weights are sign errors, not a separate population: taking the absolute value puts
  them on top of the positive distribution (mean 31,724 lb against 31,415 lb). So
  ``weight = weight.abs()``, with a flag kept.
* Missing ``weight`` and ``market_index`` are imputed with the training mean and flagged, never
  dropped: predictions are required for all 12,000 validation load_ids.
* City to coordinate is an exact lookup — every city maps to exactly one lat/lon pair — which is
  what lets the December rows be reconstructed.
* The December chart file carries only ``pickup, delivery, distance, equipment, weight, date``.
  The four coordinates come from the city lookup, and ``market_index`` / ``quote_signal`` come
  from the real December rows already present in ``validation.csv`` (per-date median). Neither
  gap needs a forecast.
* Eight cities (San Diego, Chicago, Allentown, Knoxville, Laredo, Norfolk, Charlotte, Jackson)
  appear in validation but never in training, so the coordinate lookup is built from both frames
  and unseen city levels are flagged for the model.

Outlier trimming on rate per mile is deliberately *not* part of :meth:`Cleaner.transform`. The
notebook decided it belongs inside a training fold only, so it lives in
:func:`trim_rate_per_mile` for the caller to apply to training rows alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

FILES = {
    "train": "train-test.csv",
    "validation": "validation.csv",
    "december": "december-chart-inputs.csv",
    "template": "validation-predictions-template.csv",
}

TARGET = "posted_rate"
COORD_COLUMNS = ["pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"]
SIGNAL_COLUMNS = ["market_index", "quote_signal"]
CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment"]
DECEMBER_START = "2025-12-01"


def load_raw(name: str) -> pd.DataFrame:
    """Read one of the four supplied CSVs by logical name."""
    if name not in FILES:
        raise KeyError(f"unknown frame {name!r}; expected one of {sorted(FILES)}")
    return pd.read_csv(DATA / FILES[name])


@dataclass
class Cleaner:
    """Imputation statistics and lookups fitted once, then applied to every frame.

    ``december_signals`` is the one table not derived from training data. It comes from the
    December rows of ``validation.csv``, which ship with real ``market_index`` and
    ``quote_signal`` values for exactly the 31 dates the chart asks about. That is supplied
    data, not an extrapolation, and it keeps the chart on the same model as the 12,000
    submitted predictions.
    """

    weight_mean: float
    market_mean: float
    quote_mean: float
    city_coords: pd.DataFrame
    known_cities: frozenset[str]
    december_signals: pd.DataFrame

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Return a cleaned copy of ``frame``. Row count and row order are preserved."""
        out = frame.copy()
        out["date"] = pd.to_datetime(out["date"])

        out = _fill_coordinates(out, self.city_coords)
        out = _fill_signals(out, self.december_signals)

        # Sign errors, not outliers: 292 train rows and 145 validation rows.
        out["is_weight_negative"] = (out["weight"] < 0).fillna(False).astype(int)
        out["weight"] = out["weight"].abs()

        out["is_weight_missing"] = out["weight"].isna().astype(int)
        out["weight"] = out["weight"].fillna(self.weight_mean)

        out["is_market_missing"] = out["market_index"].isna().astype(int)
        out["market_index"] = out["market_index"].fillna(self.market_mean)

        out["is_quote_missing"] = out["quote_signal"].isna().astype(int)
        out["quote_signal"] = out["quote_signal"].fillna(self.quote_mean)

        # A city absent from training has no usable category level. The flag lets the model
        # fall back on the rest of the row instead of trusting an unseen level.
        out["is_pickup_unseen"] = (~out["pickup"].isin(self.known_cities)).astype(int)
        out["is_delivery_unseen"] = (~out["delivery"].isin(self.known_cities)).astype(int)

        for column in CATEGORICAL_COLUMNS:
            out[column] = out[column].astype("category")

        if TARGET in out.columns:
            out["rate_per_mile"] = out[TARGET] / out["distance"]

        _assert_clean(out, frame)
        return out


def fit_cleaner(train: pd.DataFrame, validation: pd.DataFrame) -> Cleaner:
    """Fit imputation means on ``train`` and the December signal table on ``validation``.

    ``validation`` is used for two things only, neither of which touches the target: the
    coordinate lookup (8 cities appear nowhere in training) and the December market signals.
    """
    december = validation.loc[validation["date"].astype(str) >= DECEMBER_START]
    if december.empty:
        raise ValueError("no December rows in validation; the signal table would be empty")

    return Cleaner(
        # Fitted on the sign-corrected weights, so the 292 negative rows do not drag the mean.
        weight_mean=float(train["weight"].abs().mean()),
        market_mean=float(train["market_index"].mean()),
        quote_mean=float(train["quote_signal"].mean()),
        city_coords=_city_coordinates(train, validation),
        known_cities=frozenset(train["pickup"]) | frozenset(train["delivery"]),
        december_signals=(
            december.assign(date=pd.to_datetime(december["date"]))
            .groupby("date")[SIGNAL_COLUMNS]
            .median()
        ),
    )


def _city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """Build the city to latitude/longitude lookup.

    Each city maps to a single coordinate pair, so this is a lookup rather than an average, and
    the uniqueness check below is a real assertion rather than a formality.
    """
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            parts.append(
                frame[[side, f"{side}_lat", f"{side}_lon"]].rename(
                    columns={side: "city", f"{side}_lat": "lat", f"{side}_lon": "lon"}
                )
            )
    stacked = pd.concat(parts).dropna()

    counts = stacked.groupby("city")["lat"].nunique()
    if (counts > 1).any():
        raise ValueError(f"city coordinates are not unique: {counts[counts > 1].index.tolist()}")
    return stacked.drop_duplicates("city").set_index("city")


def _fill_coordinates(frame: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    """Add the four coordinate columns when absent, as they are in the December inputs."""
    if all(column in frame.columns for column in COORD_COLUMNS):
        return frame

    for side in ("pickup", "delivery"):
        frame[f"{side}_lat"] = frame[side].map(lookup["lat"])
        frame[f"{side}_lon"] = frame[side].map(lookup["lon"])

    unmapped = int(frame[COORD_COLUMNS].isna().any(axis=1).sum())
    if unmapped:
        raise ValueError(f"{unmapped} rows have a city missing from the coordinate lookup")
    return frame


def _fill_signals(frame: pd.DataFrame, december_signals: pd.DataFrame) -> pd.DataFrame:
    """Join the per-date market signals when absent, as they are in the December inputs."""
    missing = [column for column in SIGNAL_COLUMNS if column not in frame.columns]
    if not missing:
        return frame

    joined = frame.join(december_signals[missing], on="date")
    unmapped = int(joined[missing].isna().any(axis=1).sum())
    if unmapped:
        raise ValueError(
            f"{unmapped} rows fall outside the December signal table "
            f"({december_signals.index.min().date()} to {december_signals.index.max().date()})"
        )
    return joined


def trim_rate_per_mile(
    frame: pd.DataFrame, lower: float = 0.001, upper: float = 0.999
) -> pd.DataFrame:
    """Drop extreme rate-per-mile rows. Apply to training rows only.

    Raw ``posted_rate`` spans $57 to $25,533, but rate per mile sits in a tight band, so it is
    the frame in which an outlier is identifiable at all. The quantiles are computed on the
    frame passed in, so calling this per training fold keeps the cut-offs out of any held-out
    data; never call it on a frame you are about to score or evaluate against.
    """
    if "rate_per_mile" not in frame.columns:
        raise KeyError("rate_per_mile is absent; trimming needs a labelled frame")

    low, high = frame["rate_per_mile"].quantile([lower, upper])
    keep = frame["rate_per_mile"].between(low, high)
    return frame.loc[keep].copy()


def _assert_clean(out: pd.DataFrame, source: pd.DataFrame) -> None:
    """Guard the invariants the rest of the pipeline depends on."""
    if len(out) != len(source):
        raise ValueError(f"row count changed: {len(source)} -> {len(out)}")

    required = ["distance", "weight", *SIGNAL_COLUMNS, *COORD_COLUMNS]
    nulls = out[required].isna().sum()
    if nulls.any():
        raise ValueError(f"nulls survived cleaning: {nulls[nulls > 0].to_dict()}")
    if (out["weight"] < 0).any() or (out["distance"] <= 0).any():
        raise ValueError("non-physical weight or distance survived cleaning")
    if out["date"].isna().any():
        raise ValueError("unparseable dates survived cleaning")


def load_clean() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Cleaner]:
    """Load and clean all three modelling frames in one call."""
    train, validation = load_raw("train"), load_raw("validation")
    december = load_raw("december")

    cleaner = fit_cleaner(train, validation)
    return (
        cleaner.transform(train),
        cleaner.transform(validation),
        cleaner.transform(december),
        cleaner,
    )


if __name__ == "__main__":
    train, validation, december, cleaner = load_clean()

    for name, frame in [("train", train), ("validation", validation), ("december", december)]:
        print(f"{name:11s} {frame.shape[0]:>6,} rows x {frame.shape[1]:>2} cols  "
              f"{frame.date.min().date()} -> {frame.date.max().date()}")

    print()
    print(f"weight mean imputed to : {cleaner.weight_mean:,.0f} lb")
    print(f"market_index mean      : {cleaner.market_mean:.5f}")
    print(f"quote_signal mean      : {cleaner.quote_mean:.5f}")
    print(f"coordinate lookup      : {len(cleaner.city_coords)} cities")
    print(f"cities seen in training: {len(cleaner.known_cities)}")
    print(f"December signal dates  : {len(cleaner.december_signals)}")
    print()
    print(f"sign-corrected weights : {int(train.is_weight_negative.sum())} train, "
          f"{int(validation.is_weight_negative.sum())} validation")
    print(f"imputed weight         : {int(train.is_weight_missing.sum())} train, "
          f"{int(validation.is_weight_missing.sum())} validation")
    print(f"imputed market_index   : {int(train.is_market_missing.sum())} train, "
          f"{int(validation.is_market_missing.sum())} validation")
    unseen = (validation.is_pickup_unseen | validation.is_delivery_unseen).sum()
    print(f"unseen-city rows       : {int(unseen)} validation")
    print()
    trimmed = trim_rate_per_mile(train)
    print(f"trim_rate_per_mile     : {len(train):,} -> {len(trimmed):,} train rows "
          f"({len(train) - len(trimmed)} dropped)")
