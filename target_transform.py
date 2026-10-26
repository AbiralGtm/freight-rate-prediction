"""Target transform for the freight rate assessment (solution guide, Step 5).

Raw ``posted_rate`` spans $57 to $25,533 with a coefficient of variation of 0.626, so fitting
squared error on it lets a handful of 3,000-mile hauls dominate the loss. The guide offers two
ways out: train on ``log(posted_rate)`` and invert, or train on ``posted_rate / distance`` and
multiply back. Both are implemented here behind one interface so the model code can switch with
a single argument, and :func:`compare_targets` is the measurement behind the choice.

**The choice: log(posted_rate).** Two reasons, both measured on ``train-test.csv``.

*Conditioning.* The log target is very close to normal — skew -0.49, excess kurtosis +0.14.
Rate per mile is not: skew +8.72, excess kurtosis +113.69, because dividing a short haul's fixed
costs by a small distance produces a long right tail. Squared error on a target with kurtosis
over 100 is squared error chasing a handful of rows.

*It does not assume the answer.* Regressing ``log(posted_rate)`` on ``log(distance)`` gives a
slope of 0.871, not 1.0 — the short-haul premium, visible as mean rate per mile falling from
$2.815 to $1.903 across distance bands. The rate-per-mile target hard-codes an exponent of
exactly 1.0 and leaves the model to undo the error through its distance features. The log target
keeps ``log_distance`` as a free feature and lets the model estimate 0.871 for itself.

The one cost of a log target is retransformation bias: ``exp`` of the mean of the logs
underestimates the mean of the levels, so predictions come back low on average. :class:`LogRate`
corrects it with a Duan smearing factor fitted on training residuals. See :func:`fit_smearing`.

Note on the guide's wording: it says ``log`` then ``expm1``. Those are not inverses — ``expm1``
pairs with ``log1p``. This module uses ``log``/``exp``, a consistent pair; with a minimum rate of
$57.22 the ``log1p`` variant would differ by under 2% anyway, but a mismatched pair would bias
every prediction downward by exactly 1 dollar.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

TARGET = "posted_rate"

# score.py aborts the whole run on a single non-positive prediction, and the lowest rate in
# training is $57.22, so anything under this floor is a model failure rather than a cheap load.
FLOOR = 50.0


@dataclass
class TargetTransform:
    """Maps ``posted_rate`` to the scale the model fits on, and back again.

    ``forward`` takes a frame and returns the training target. ``inverse`` takes raw model
    output plus the frame it was predicted from and returns dollars, because the rate-per-mile
    transforms need ``distance`` to get back to a price.
    """

    name: str = "identity"
    smearing: float = 1.0

    def forward(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[TARGET].to_numpy(dtype=float)

    def inverse(self, prediction: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return self._floor(np.asarray(prediction, dtype=float))

    def fit_smearing(self, frame: pd.DataFrame, prediction: np.ndarray) -> float:
        """Estimate the retransformation correction. A no-op on a linear scale."""
        self.smearing = 1.0
        return self.smearing

    @staticmethod
    def _floor(dollars: np.ndarray) -> np.ndarray:
        return np.clip(dollars, FLOOR, None)


@dataclass
class LogRate(TargetTransform):
    """Fit ``log(posted_rate)``, invert with ``exp`` and a smearing correction. The default."""

    name: str = "log_rate"

    def forward(self, frame: pd.DataFrame) -> np.ndarray:
        return np.log(frame[TARGET].to_numpy(dtype=float))

    def inverse(self, prediction: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        return self._floor(np.exp(np.asarray(prediction, dtype=float)) * self.smearing)

    def fit_smearing(self, frame: pd.DataFrame, prediction: np.ndarray) -> float:
        self.smearing = duan_smearing(self.forward(frame), prediction)
        return self.smearing


@dataclass
class RatePerMile(TargetTransform):
    """Fit ``posted_rate / distance``, invert by multiplying distance back in."""

    name: str = "rate_per_mile"

    def forward(self, frame: pd.DataFrame) -> np.ndarray:
        return (frame[TARGET] / frame["distance"]).to_numpy(dtype=float)

    def inverse(self, prediction: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        dollars = np.asarray(prediction, dtype=float) * frame["distance"].to_numpy(dtype=float)
        return self._floor(dollars)


@dataclass
class LogRatePerMile(TargetTransform):
    """Fit ``log(posted_rate / distance)``. Tames the rate-per-mile tail but still fixes the
    distance exponent at 1.0, so it is kept for comparison rather than as a candidate."""

    name: str = "log_rate_per_mile"

    def forward(self, frame: pd.DataFrame) -> np.ndarray:
        return np.log((frame[TARGET] / frame["distance"]).to_numpy(dtype=float))

    def inverse(self, prediction: np.ndarray, frame: pd.DataFrame) -> np.ndarray:
        per_mile = np.exp(np.asarray(prediction, dtype=float)) * self.smearing
        return self._floor(per_mile * frame["distance"].to_numpy(dtype=float))

    def fit_smearing(self, frame: pd.DataFrame, prediction: np.ndarray) -> float:
        self.smearing = duan_smearing(self.forward(frame), prediction)
        return self.smearing


TRANSFORMS = {
    t.name: t
    for t in (TargetTransform(), LogRate(), RatePerMile(), LogRatePerMile())
}


def get_transform(name: str = "log_rate") -> TargetTransform:
    """Return a fresh transform by name, so a fitted smearing factor is never shared."""
    if name not in TRANSFORMS:
        raise KeyError(f"unknown transform {name!r}; expected one of {sorted(TRANSFORMS)}")
    return type(TRANSFORMS[name])()


def duan_smearing(actual_log: np.ndarray, predicted_log: np.ndarray) -> float:
    """Duan's smearing estimator: the mean of ``exp(residual)`` on the log scale.

    ``exp(E[log y])`` is the geometric mean, which sits below the arithmetic mean whenever the
    residuals have any spread, so an uncorrected log model is biased low in dollars. Multiplying
    by ``mean(exp(residual))`` restores the level. Fit this on training residuals only; fitting
    it on the rows being scored would leak the target.
    """
    residual = np.asarray(actual_log, dtype=float) - np.asarray(predicted_log, dtype=float)
    factor = float(np.mean(np.exp(residual)))
    if not np.isfinite(factor) or factor <= 0:
        raise ValueError(f"smearing factor is not usable: {factor}")
    return factor


def compare_targets(train: pd.DataFrame) -> pd.DataFrame:
    """The conditioning table behind the choice of transform.

    ``spread_ratio`` is how much the within-band relative spread varies across distance bands:
    a value near 1 means the target's noise is the same size everywhere, which is the
    homoscedasticity squared error assumes.
    """
    bands = pd.cut(train["distance"], [0, 250, 500, 750, 1000, 1500, 2000, 2500, 3500])

    rows = []
    for name, transform in TRANSFORMS.items():
        values = pd.Series(transform.forward(train))
        grouped = values.groupby(bands, observed=True)
        relative = grouped.std() / grouped.mean().abs()
        rows.append({
            "target": name,
            "cv": values.std() / abs(values.mean()),
            "skew": values.skew(),
            "excess_kurtosis": values.kurt(),
            "spread_ratio": relative.max() / relative.min(),
        })
    return pd.DataFrame(rows).set_index("target")


def distance_exponent(train: pd.DataFrame) -> float:
    """Slope of ``log(posted_rate)`` on ``log(distance)``.

    A rate-per-mile target is only unbiased if this is 1.0. It is not.
    """
    slope, _ = np.polyfit(
        np.log(train["distance"].to_numpy(dtype=float)),
        np.log(train[TARGET].to_numpy(dtype=float)),
        1,
    )
    return float(slope)


if __name__ == "__main__":
    from clean_data import load_clean

    train, _, _, _ = load_clean()

    print("Conditioning of each candidate target (train-test.csv):")
    print(compare_targets(train).round(3).to_string())
    print()
    print(f"slope of log(posted_rate) on log(distance): {distance_exponent(train):.3f}")
    print("A rate-per-mile target assumes this slope is exactly 1.000.")
    print()

    # Round-trip check: forward then inverse must return the original dollars, so a transform
    # bug cannot hide behind a plausible-looking prediction later.
    for name in TRANSFORMS:
        transform = get_transform(name)
        recovered = transform.inverse(transform.forward(train), train)
        error = float(np.abs(recovered - train[TARGET].to_numpy()).max())
        print(f"{name:18s} round-trip max error: ${error:.6f}")

    print()
    chosen = get_transform("log_rate")
    print(f"chosen transform : {chosen.name}")
    print(f"prediction floor : ${FLOOR:.2f} (score.py rejects any value <= 0)")
    print("smearing factor is fitted on training residuals in the model step, not here.")
