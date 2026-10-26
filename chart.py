"""The 31-point December sensitivity curve (solution guide, Step 9).

All 31 rows of ``december-chart-inputs.csv`` are the same load — Lexington to Fort Wayne, 360
miles, Dry Van, 32,000 lb — with only ``date`` changing. The curve is therefore a pure picture
of what the model learned about time. Nothing else varies, so nothing else can move it.

The trap is that the file carries six input columns and the model was trained on more. Both
gaps are closed in :mod:`clean_data` rather than here, and neither needs a forecast:

* **Coordinates** come from the city lookup. Every city maps to exactly one lat/lon pair, so
  Lexington and Fort Wayne resolve exactly rather than approximately.
* **Market signals** come from the 6,164 December rows already sitting in ``validation.csv``,
  taken as a per-date median and joined on date. That is supplied data. Fitting a trend to
  ``market_index`` and projecting it forward would stack forecast error on top of model error
  when the real values were handed over in the data, and it would also force a second,
  reduced-feature model. This way the chart comes off the same model as the 12,000 submitted
  predictions.

Output goes to ``december-chart-inputs.csv`` beside this file, not to the copy under ``data/``,
so the supplied input stays pristine and a bad run is re-runnable. The six input columns are
copied from the raw file untouched and only ``predicted_rate`` is filled, which makes "the fixed
inputs are unchanged" true by construction rather than by assertion.

:func:`diagnose` is the acceptance test the guide asks for. A dead-flat line means the date
features were dropped somewhere in the pipeline; a violent sawtooth means they are overfit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from build_features import add_features
from clean_data import ROOT, load_clean, load_raw
from predict import fit_final_model
from target_transform import FLOOR

OUTPUT = ROOT / "december-chart-inputs.csv"

INPUT_COLUMNS = ["pickup", "delivery", "distance", "equipment", "weight", "date"]
FIXED_INPUTS = {
    "pickup": "Lexington",
    "delivery": "Fort Wayne",
    "distance": 360,
    "equipment": "Dry Van",
    "weight": 32000,
}
DECEMBER_DAYS = pd.date_range("2025-12-01", "2025-12-31", freq="D")

# A flat curve means the date features never reached the model; a jagged one means they are
# overfit. Both are judged on the peak-to-trough swing as a percentage of the mean.
FLAT_BELOW_PERCENT = 0.5
JAGGED_ABOVE_PERCENT = 25.0


def predict_december(model, december: pd.DataFrame) -> np.ndarray:
    """Dollar predictions for the 31 chart rows, floored like every other prediction."""
    return np.round(np.clip(model.predict(december), FLOOR, None), 2)


def lane_benchmark(train: pd.DataFrame) -> dict[str, float]:
    """What real loads on this exact lane cost, as the number to check the curve against.

    If the curve lands far from this, the December feature reconstruction is wrong and the
    place to debug is :mod:`clean_data`, not the model.
    """
    lane = train[(train["pickup"] == FIXED_INPUTS["pickup"])
                 & (train["delivery"] == FIXED_INPUTS["delivery"])]
    per_mile = (lane["posted_rate"] / lane["distance"]).mean()
    return {
        "loads": int(len(lane)),
        "mean_per_mile": float(per_mile),
        "implied_rate": float(per_mile * FIXED_INPUTS["distance"]),
        "mean_rate": float(lane["posted_rate"].mean()),
    }


def diagnose(predictions: pd.Series, dates: pd.Series) -> dict[str, float]:
    """Shape of the curve: trend, weekly ripple, and whether it is flat or jagged."""
    values = predictions.to_numpy(dtype=float)
    swing = (values.max() - values.min()) / values.mean() * 100

    by_weekday = pd.Series(values).groupby(dates.dt.day_of_week.to_numpy()).mean()
    weekend = by_weekday.reindex([5, 6]).mean()
    midweek = by_weekday.reindex([1, 2, 3]).mean()

    # Day-to-day absolute change, as a share of the level. A smooth seasonal curve with a
    # weekly ripple moves far less between consecutive days than it does end to end.
    steps = np.abs(np.diff(values)).mean() / values.mean() * 100

    return {
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "swing_percent": float(swing),
        "first_to_last": float(values[-1] - values[0]),
        "midweek_minus_weekend": float(midweek - weekend),
        "mean_daily_step_percent": float(steps),
    }


def assert_contract(filled: pd.DataFrame, raw: pd.DataFrame) -> None:
    """Every check ``score.py`` runs on the December file."""
    expected = INPUT_COLUMNS + ["predicted_rate"]
    if list(filled.columns) != expected:
        raise ValueError(f"columns must be {expected}; got {list(filled.columns)}")
    if len(filled) != 31:
        raise ValueError(f"expected 31 rows, got {len(filled)}")

    dates = pd.to_datetime(filled["date"])
    if not dates.equals(pd.Series(DECEMBER_DAYS)):
        raise ValueError("dates must be one row per day from 2025-12-01 to 2025-12-31, in order")

    for column, value in FIXED_INPUTS.items():
        if not (filled[column] == value).all():
            raise ValueError(f"fixed input {column} was modified; must stay {value!r}")
    for column in INPUT_COLUMNS:
        if not filled[column].equals(raw[column]):
            raise ValueError(f"input column {column} differs from the supplied file")

    rate = filled["predicted_rate"].to_numpy(dtype=float)
    if not np.isfinite(rate).all():
        raise ValueError("predicted_rate contains non-finite values")
    if (rate <= 0).any():
        raise ValueError(f"{int((rate <= 0).sum())} predictions are not strictly positive")


def main(kind: str = "ridge") -> pd.DataFrame:
    train, _, december, _ = load_clean()
    train, december = add_features(train), add_features(december)

    model = fit_final_model(train, kind=kind)

    # Start from the raw file so the six input columns are copied verbatim. clean_data preserves
    # row order, so the predictions line up position for position.
    filled = load_raw("december")
    filled["predicted_rate"] = predict_december(model, december)
    assert_contract(filled, load_raw("december"))
    filled.to_csv(OUTPUT, index=False)

    print(f"model      : {model.name}")
    print(f"written    : {OUTPUT}")
    print()

    benchmark = lane_benchmark(train)
    shape = diagnose(filled["predicted_rate"], pd.to_datetime(filled["date"]))

    print(f"curve      : ${shape['min']:,.2f} to ${shape['max']:,.2f}, "
          f"mean ${shape['mean']:,.2f}")
    print(f"benchmark  : {benchmark['loads']} real loads on this lane at "
          f"{benchmark['mean_per_mile']:.3f} $/mi, implying "
          f"${benchmark['implied_rate']:,.0f} over {FIXED_INPUTS['distance']} miles")
    gap = (shape["mean"] - benchmark["implied_rate"]) / benchmark["implied_rate"] * 100
    print(f"gap        : {gap:+.1f}% against the historical lane")
    print()
    print(f"swing peak to trough : {shape['swing_percent']:.2f}% of the mean")
    print(f"1 Dec to 31 Dec      : ${shape['first_to_last']:+,.2f}")
    print(f"midweek - weekend    : ${shape['midweek_minus_weekend']:+,.2f}")
    print(f"mean day-to-day step : {shape['mean_daily_step_percent']:.2f}%")

    if shape["swing_percent"] < FLAT_BELOW_PERCENT:
        print("\nFLAT. The date features are not reaching the model; debug build_features.")
    elif shape["swing_percent"] > JAGGED_ABOVE_PERCENT:
        print("\nJAGGED. The date features are overfit; reduce their resolution.")
    else:
        print("\nShape is a weekly ripple on a slow trend, as Step 2 predicted.")
    return filled


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "ridge")
