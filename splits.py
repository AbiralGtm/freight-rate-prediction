"""Validation split and metrics for the freight rate assessment (solution guide, Step 6).

This is the decision being graded, so it gets its own module.

Training data ends 31 October and prediction begins 1 November. The validation scheme has to
reproduce that forward gap, which means expanding-window folds by month:

====  =====================  ===========
Fold  Train on               Validate on
====  =====================  ===========
1     January - July         August
2     January - August       September
3     January - September    October
====  =====================  ===========

Each fold trains on everything before its validation month and scores the month immediately
after, exactly as the final model trains to 31 October and scores November onward. Fold 3 is
the one the model step uses for early stopping, because it is the fold closest in shape to the
real task.

**Metrics are reported per fold, never blended.** The spread across the three folds is the
honest estimate of what month-to-month drift costs; a single averaged number hides it. See
:func:`summarize`.

**A random KFold would leak.** It puts October rows in training while scoring August, so the
model sees the future it is being asked to predict and the score comes back flattering and
wrong. :func:`random_kfold_folds` exists only so the model step can run the comparison and put
the size of that gap in the report, which is a large part of what this exercise is testing. It
is never the scheme used to pick anything.

Trimming belongs inside the fold: :func:`fold_frames` applies
:func:`clean_data.trim_rate_per_mile` to the training rows alone. Trimming a validation fold
would quietly delete the hard rows and flatter every metric in the table.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from clean_data import trim_rate_per_mile

TARGET = "posted_rate"
VALIDATION_MONTHS = ("2025-08", "2025-09", "2025-10")

# Fold 3 validates on October, the last month before the real prediction window, so it is the
# fold whose drift most resembles November-December. The model step early-stops on it.
EARLY_STOPPING_FOLD = 3


@dataclass(frozen=True)
class Fold:
    """One expanding-window fold, held as positional indices into the source frame."""

    number: int
    train_index: pd.Index
    valid_index: pd.Index
    train_end: pd.Timestamp
    valid_month: str

    @property
    def name(self) -> str:
        return f"fold {self.number} ({self.valid_month})"

    def __repr__(self) -> str:  # keeps the fold table readable when printed
        return (
            f"Fold({self.number}, train<= {self.train_end.date()} n={len(self.train_index):,}, "
            f"valid={self.valid_month} n={len(self.valid_index):,})"
        )


def expanding_month_folds(
    frame: pd.DataFrame, validation_months: tuple[str, ...] = VALIDATION_MONTHS
) -> list[Fold]:
    """Build the expanding-window folds. ``frame`` must be cleaned, so ``date`` is datetime."""
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise TypeError("date is not datetime; run clean_data first")

    month = frame["date"].dt.to_period("M")
    folds = []
    for number, label in enumerate(validation_months, start=1):
        period = pd.Period(label, freq="M")
        train_mask, valid_mask = month < period, month == period

        if not valid_mask.any():
            raise ValueError(f"no rows in validation month {label}")
        if not train_mask.any():
            raise ValueError(f"no rows before validation month {label}")

        folds.append(
            Fold(
                number=number,
                train_index=frame.index[train_mask],
                valid_index=frame.index[valid_mask],
                train_end=frame.loc[train_mask, "date"].max(),
                valid_month=label,
            )
        )
    _assert_no_leakage(frame, folds)
    return folds


def random_kfold_folds(
    frame: pd.DataFrame, n_splits: int = 3, seed: int = 0
) -> list[Fold]:
    """Shuffled KFold, for the leakage comparison in the report only.

    Do not select a model, a round count or a hyperparameter with these folds. They exist to be
    shown losing.
    """
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(len(frame))
    chunks = np.array_split(shuffled, n_splits)

    folds = []
    for number, chunk in enumerate(chunks, start=1):
        valid_index = frame.index[np.sort(chunk)]
        train_index = frame.index.difference(valid_index)
        folds.append(
            Fold(
                number=number,
                train_index=train_index,
                valid_index=valid_index,
                train_end=frame.loc[train_index, "date"].max(),
                valid_month="random",
            )
        )
    return folds


def fold_frames(
    frame: pd.DataFrame, fold: Fold, trim: bool = True
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(train, valid)`` for one fold, trimming the training rows only."""
    train = frame.loc[fold.train_index]
    valid = frame.loc[fold.valid_index]
    if trim:
        train = trim_rate_per_mile(train)
    return train, valid


def metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """RMSE, MAE and MAPE in dollars. All three, because they answer different questions.

    RMSE is what the model optimises and punishes the large misses on long hauls. MAE is the
    typical dollar error on a typical load. MAPE is the one a broker would quote, since being
    $80 out on a $700 load and on a $7,000 load are not the same mistake.
    """
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if actual.shape != predicted.shape:
        raise ValueError(f"shape mismatch: {actual.shape} against {predicted.shape}")
    if not np.isfinite(predicted).all():
        raise ValueError("predictions contain non-finite values")
    if (actual <= 0).any():
        raise ValueError("MAPE is undefined for a non-positive actual rate")

    error = predicted - actual
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "mape": float(np.mean(np.abs(error / actual)) * 100),
    }


def summarize(results: dict[str, dict[str, float]]) -> pd.DataFrame:
    """Per-fold metric table with a mean and a spread row appended.

    The ``spread`` row is max minus min across folds. It is the number that says how much
    month-to-month drift costs, and it belongs in the report next to the mean rather than
    behind it.
    """
    table = pd.DataFrame(results).T
    table.loc["mean"] = table.mean()
    table.loc["spread"] = table.iloc[:-1].max() - table.iloc[:-1].min()
    return table


def evaluate_folds(frame: pd.DataFrame, predict, folds: list[Fold] | None = None,
                   trim: bool = True) -> pd.DataFrame:
    """Run ``predict(train, valid) -> dollars`` over every fold and summarise.

    ``predict`` receives the two frames and returns one prediction per validation row, so any
    model — the Ridge baseline, LightGBM, or a constant — plugs in unchanged.
    """
    folds = folds if folds is not None else expanding_month_folds(frame)

    results = {}
    for fold in folds:
        train, valid = fold_frames(frame, fold, trim=trim)
        predicted = predict(train, valid)
        results[fold.name] = metrics(valid[TARGET].to_numpy(), predicted)
    return summarize(results)


def _assert_no_leakage(frame: pd.DataFrame, folds: list[Fold]) -> None:
    """Every training date must fall strictly before every validation date, in every fold."""
    for fold in folds:
        train_max = frame.loc[fold.train_index, "date"].max()
        valid_min = frame.loc[fold.valid_index, "date"].min()
        if train_max >= valid_min:
            raise ValueError(
                f"fold {fold.number} leaks: trains to {train_max.date()} "
                f"but validates from {valid_min.date()}"
            )
        overlap = fold.train_index.intersection(fold.valid_index)
        if len(overlap):
            raise ValueError(f"fold {fold.number} shares {len(overlap)} rows between sides")


if __name__ == "__main__":
    from clean_data import load_clean

    train, validation, _, _ = load_clean()
    folds = expanding_month_folds(train)

    print("Expanding-window folds:")
    for fold in folds:
        print(f"  {fold!r}")
    print(f"\nfinal fit: train<= {train.date.max().date()}, "
          f"predict {validation.date.min().date()} to {validation.date.max().date()}")
    print(f"early stopping on fold {EARLY_STOPPING_FOLD}")

    # Baseline: the mean rate per mile of the training fold, multiplied by distance. It uses no
    # feature except distance, so every later number has to beat it to have earned its place.
    def flat_rate_per_mile(train_fold: pd.DataFrame, valid_fold: pd.DataFrame) -> np.ndarray:
        rate = train_fold["rate_per_mile"].mean()
        return valid_fold["distance"].to_numpy() * rate

    print("\nBaseline (mean $/mi x distance), per fold:")
    print(evaluate_folds(train, flat_rate_per_mile).round(3).to_string())

    print("\nSame baseline under a random KFold, which leaks future months:")
    leaky = evaluate_folds(train, flat_rate_per_mile, folds=random_kfold_folds(train))
    print(leaky.round(3).to_string())
    print("A random split scores the same model against rows whose month it already trained on.")
