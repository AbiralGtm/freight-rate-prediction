"""Final fit and the 12,000 validation predictions (solution guide, Step 8).

Refits the chosen model on the whole of ``train-test.csv`` and writes
``validation_predictions.csv`` next to this file.

**The chosen model is Ridge, not LightGBM.** Step 7 measured both on the expanding-window
folds and the linear model won every metric: RMSE 627.5 against 630.1, MAE 119.3 against 130.7,
MAPE 5.17% against 5.87%, with a tighter spread across folds. The cause is in the feature
importances — ``distance`` carries a gain of 128,690 and ``log_distance`` 36,010 against 1,264
for the next feature, because ``log(posted_rate)`` is close to linear in ``log(distance)`` with
a slope of 0.871. Ridge fits that line exactly; trees approximate it with steps and pay for it,
and they cannot extrapolate the trend term 61 days past the end of training either. The guide
warns that a boosted model which merely ties the linear baseline means the engineered features
are not earning their place. Here it loses, so the baseline ships and LightGBM stays in the
report as the documented comparison.

Two settings carried over from the folds rather than guessed:

* ``alpha=10.0``. A sweep over 0.1, 1, 10 and 100 moved MAPE from 5.174 to 5.148, so the
  penalty is immaterial on this data; 10 had the best fold RMSE and the choice is reported as
  the non-decision it is.
* Trimming stays on for the final fit, because the final fit *is* a training fold. It drops 96
  extreme rate-per-mile rows and is worth 1.6 dollars of fold MAE (119.3 against 120.9).

The output contract is enforced here rather than discovered when ``score.py`` rejects the file:
exactly ``load_id,predicted_rate`` in that order, exactly 12,000 rows matching TE-000001 to
TE-012000 with no duplicates, every value finite and strictly positive. A single non-positive
prediction aborts the whole scoring run, so predictions are floored at $50 — below the $57.22
minimum in training, so the floor only ever catches a model failure.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from build_features import add_features
from clean_data import ROOT, load_clean, load_raw, trim_rate_per_mile
from models import LightGBMModel, RidgeBaseline
from target_transform import FLOOR

OUTPUT = ROOT / "validation_predictions.csv"

RIDGE_ALPHA = 10.0
# Chosen by early stopping on fold 3 in Step 7. Only used if the LightGBM path is selected.
LIGHTGBM_ROUNDS = 100


def build_model(kind: str = "ridge"):
    """Return the model to refit, configured as the folds selected it."""
    if kind == "ridge":
        return RidgeBaseline(alpha=RIDGE_ALPHA)
    if kind == "lightgbm":
        from models import LIGHTGBM_PARAMS

        params = dict(LIGHTGBM_PARAMS, n_estimators=LIGHTGBM_ROUNDS)
        return LightGBMModel(params=params)
    raise KeyError(f"unknown model {kind!r}; expected 'ridge' or 'lightgbm'")


def fit_final_model(train: pd.DataFrame, kind: str = "ridge", trim: bool = True):
    """Fit on the whole training frame. Step 9 reuses this, so the chart and the
    submission come out of one model rather than two."""
    fitted_on = trim_rate_per_mile(train) if trim else train
    return build_model(kind).fit(fitted_on)


def predict_validation(model, validation: pd.DataFrame) -> pd.DataFrame:
    """Return the two-column submission frame, in the order the scorer demands."""
    dollars = np.clip(model.predict(validation), FLOOR, None)
    return pd.DataFrame({
        "load_id": validation["load_id"].to_numpy(),
        "predicted_rate": np.round(dollars, 2),
    })


def assert_contract(predictions: pd.DataFrame, template: pd.DataFrame) -> None:
    """Every check ``score.py`` runs, run here first so a failure is cheap."""
    if list(predictions.columns) != ["load_id", "predicted_rate"]:
        raise ValueError(f"columns must be load_id,predicted_rate; got {list(predictions.columns)}")
    if len(predictions) != len(template):
        raise ValueError(f"expected {len(template)} rows, got {len(predictions)}")
    if predictions["load_id"].duplicated().any():
        raise ValueError("duplicate load_id values")

    expected = set(template["load_id"])
    actual = set(predictions["load_id"])
    if actual != expected:
        missing, extra = sorted(expected - actual), sorted(actual - expected)
        raise ValueError(f"load_id mismatch; missing {missing[:3]}, unexpected {extra[:3]}")

    rate = predictions["predicted_rate"].to_numpy(dtype=float)
    if not np.isfinite(rate).all():
        raise ValueError("predicted_rate contains non-finite values")
    if (rate <= 0).any():
        raise ValueError(f"{int((rate <= 0).sum())} predictions are not strictly positive")


def summarize(predictions: pd.DataFrame, validation: pd.DataFrame) -> None:
    """Print the sanity numbers worth looking at before shipping the file."""
    rate = predictions["predicted_rate"]
    per_mile = rate.to_numpy() / validation["distance"].to_numpy()

    print(f"rows          : {len(predictions):,}")
    print(f"posted_rate   : ${rate.min():,.2f} to ${rate.max():,.2f}, mean ${rate.mean():,.2f}")
    print(f"rate per mile : {np.percentile(per_mile, 1):.3f} / "
          f"{np.median(per_mile):.3f} / {np.percentile(per_mile, 99):.3f} $/mi (p1/p50/p99)")
    print(f"at the floor  : {int((rate <= FLOOR).sum())} rows")

    month = validation["date"].dt.to_period("M")
    by_month = pd.Series(per_mile).groupby(month.to_numpy()).mean()
    print("mean $/mi by month:",
          ", ".join(f"{str(k)} {v:.3f}" for k, v in by_month.items()))


def main(kind: str = "ridge") -> pd.DataFrame:
    train, validation, _, _ = load_clean()
    train, validation = add_features(train), add_features(validation)

    model = fit_final_model(train, kind=kind)
    predictions = predict_validation(model, validation)

    template = load_raw("template")
    assert_contract(predictions, template)

    # Reindex to the template's own order so the file matches it row for row, not just as a set.
    predictions = (predictions.set_index("load_id")
                   .reindex(template["load_id"])
                   .reset_index())

    predictions.to_csv(OUTPUT, index=False)
    print(f"model         : {model.name} (trained on {len(train):,} rows to "
          f"{train.date.max().date()})")
    print(f"smearing      : {model.transform_.smearing:.5f}")
    print(f"written       : {OUTPUT}")
    print()
    summarize(predictions, validation)
    return predictions


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else "ridge")
