"""Models for the freight rate assessment (solution guide, Step 7).

Two models, one interface. Both fit on the transformed target from :mod:`target_transform` and
return dollars, so :func:`splits.evaluate_folds` can score either without knowing which it has.

:class:`RidgeBaseline` is the honest baseline. It is a linear model on the log target, which
means it extrapolates the trend term in a straight line — the one thing a tree cannot do, and
the reason it is kept rather than discarded after the first comparison. If the boosted model
merely ties it, the engineered features are not earning their place and the report should say
so.

:class:`LightGBMModel` is the candidate. It takes ``pickup``, ``delivery`` and ``equipment`` as
native categoricals rather than 131 one-hot columns, which lets it pool across lanes sharing an
origin or destination instead of learning each of the ~4,014 thin lanes separately.

Round count is not guessed: :func:`tune_rounds` early-stops on fold 3 (October, the month
closest in shape to the real November-December task) and returns the best iteration for the
final refit in Step 8.

Both models correct the log retransformation bias with a Duan smearing factor fitted on their
own training residuals. Without it, ``exp`` of a mean of logs returns the geometric mean and
every prediction comes back low.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from build_features import (
    CATEGORICAL_FEATURES,
    FLAG_FEATURES,
    NUMERIC_FEATURES,
    add_features,
    category_levels,
    design_matrix,
)
from target_transform import get_transform

UNSEEN = "__unseen__"

LIGHTGBM_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 40,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "n_estimators": 800,
    "verbose": -1,
    "seed": 0,
    "n_jobs": -1,
}


@dataclass
class RidgeBaseline:
    """Ridge on the log target: one-hot categoricals, standardised numerics.

    A linear model is the only one here that can carry the trend term past the end of training,
    so its December curve is the sanity check on the boosted model's.
    """

    alpha: float = 1.0
    transform_name: str = "log_rate"
    name: str = "ridge"

    def fit(self, train: pd.DataFrame) -> "RidgeBaseline":
        train = _ensure_features(train)
        self.levels_ = category_levels(train)
        self.transform_ = get_transform(self.transform_name)

        matrix = _as_strings(design_matrix(train, self.levels_))
        self.pipeline_ = Pipeline([
            ("encode", ColumnTransformer([
                ("numeric", StandardScaler(), NUMERIC_FEATURES + FLAG_FEATURES),
                # handle_unknown='ignore' turns a city the model never saw into an all-zero
                # block rather than an error; is_pickup_unseen still tells it that happened.
                ("categorical",
                 OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                 CATEGORICAL_FEATURES),
            ])),
            ("ridge", Ridge(alpha=self.alpha)),
        ])

        target = self.transform_.forward(train)
        self.pipeline_.fit(matrix, target)
        self.transform_.fit_smearing(train, self.pipeline_.predict(matrix))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _ensure_features(frame)
        matrix = _as_strings(design_matrix(frame, self.levels_))
        return self.transform_.inverse(self.pipeline_.predict(matrix), frame)


@dataclass
class LightGBMModel:
    """Gradient boosting with native categorical handling, fitted on the log target."""

    params: dict = field(default_factory=lambda: dict(LIGHTGBM_PARAMS))
    transform_name: str = "log_rate"
    name: str = "lightgbm"

    def fit(self, train: pd.DataFrame, valid: pd.DataFrame | None = None,
            early_stopping_rounds: int = 50) -> "LightGBMModel":
        """Fit on ``train``. Passing ``valid`` turns on early stopping against that frame."""
        train = _ensure_features(train)
        self.levels_ = category_levels(train)
        self.transform_ = get_transform(self.transform_name)

        matrix = design_matrix(train, self.levels_)
        target = self.transform_.forward(train)

        callbacks, holdout = [], {}
        if valid is not None:
            valid = _ensure_features(valid)
            holdout = {"eval_X": design_matrix(valid, self.levels_),
                       "eval_y": self.transform_.forward(valid)}
            callbacks = [lgb.early_stopping(early_stopping_rounds, verbose=False),
                         lgb.log_evaluation(0)]

        self.model_ = lgb.LGBMRegressor(**self.params)
        self.model_.fit(matrix, target, callbacks=callbacks,
                        categorical_feature=CATEGORICAL_FEATURES, **holdout)

        self.best_iteration_ = self.model_.best_iteration_ or self.params["n_estimators"]
        self.transform_.fit_smearing(train, self.model_.predict(matrix))
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        frame = _ensure_features(frame)
        matrix = design_matrix(frame, self.levels_)
        return self.transform_.inverse(self.model_.predict(matrix), frame)

    def importances(self) -> pd.Series:
        """Gain-based importance, for the report."""
        booster = self.model_.booster_
        gains = booster.feature_importance(importance_type="gain")
        return pd.Series(gains, index=booster.feature_name()).sort_values(ascending=False)


def tune_rounds(train: pd.DataFrame, fold, params: dict | None = None,
                early_stopping_rounds: int = 50) -> int:
    """Return the boosting round count chosen by early stopping on one fold.

    The fold's own training rows are trimmed and its validation month is never trimmed, so the
    stopping point is chosen against untouched data from the month after training ends.
    """
    from splits import fold_frames

    train_fold, valid_fold = fold_frames(train, fold)
    model = LightGBMModel(params=dict(params or LIGHTGBM_PARAMS))
    model.fit(train_fold, valid=valid_fold, early_stopping_rounds=early_stopping_rounds)
    return int(model.best_iteration_)


def fold_predictor(build):
    """Adapt a zero-argument model factory to the ``predict(train, valid)`` shape.

    The model is rebuilt from scratch inside every fold, so nothing fitted on fold 3 can leak
    backwards into fold 1.
    """
    def predict(train_fold: pd.DataFrame, valid_fold: pd.DataFrame) -> np.ndarray:
        return build().fit(train_fold).predict(valid_fold)
    return predict


def early_stopping_fold_predictor(build, early_stopping_rounds: int = 50):
    """Same, but the round count is chosen inside the fold, never on the month being scored.

    The last month of the fold's own training window is held back as an inner validation set.
    Early-stopping against the outer validation month instead would pick the round count using
    the very rows the metric is computed on, which is the same leak as a random split wearing a
    different hat.
    """
    def predict(train_fold: pd.DataFrame, valid_fold: pd.DataFrame) -> np.ndarray:
        month = train_fold["date"].dt.to_period("M")
        last = month.max()
        inner_train, inner_valid = train_fold[month < last], train_fold[month == last]
        model = build().fit(inner_train, valid=inner_valid,
                            early_stopping_rounds=early_stopping_rounds)
        return model.predict(valid_fold)
    return predict


def _ensure_features(frame: pd.DataFrame) -> pd.DataFrame:
    return frame if "circuity" in frame.columns else add_features(frame)


def _as_strings(matrix: pd.DataFrame) -> pd.DataFrame:
    """Make the categoricals safe for OneHotEncoder, which cannot take a NaN category."""
    out = matrix.copy()
    for column in CATEGORICAL_FEATURES:
        out[column] = out[column].astype(object).fillna(UNSEEN).astype(str)
    return out


if __name__ == "__main__":
    from clean_data import load_clean
    from splits import (EARLY_STOPPING_FOLD, evaluate_folds, expanding_month_folds,
                        random_kfold_folds)

    train, _, _, _ = load_clean()
    train = add_features(train)
    folds = expanding_month_folds(train)

    print("Ridge baseline (log target), expanding-window folds:")
    ridge_table = evaluate_folds(train, fold_predictor(RidgeBaseline), folds=folds)
    print(ridge_table.round(3).to_string())

    print("\nLightGBM, expanding-window folds (round count early-stopped inside each fold):")
    gbm_table = evaluate_folds(train, early_stopping_fold_predictor(LightGBMModel), folds=folds)
    print(gbm_table.round(3).to_string())

    lift = (ridge_table.loc["mean"] - gbm_table.loc["mean"]) / ridge_table.loc["mean"] * 100
    print("\nLightGBM against the linear baseline (percent better):")
    print(lift.round(2).to_string())

    stopping_fold = folds[EARLY_STOPPING_FOLD - 1]
    rounds = tune_rounds(train, stopping_fold)
    print(f"\nearly stopping on {stopping_fold.name}: {rounds} rounds "
          f"(of {LIGHTGBM_PARAMS['n_estimators']} requested)")

    print("\nSame LightGBM under a random KFold, which leaks future months:")
    leaky = evaluate_folds(train, fold_predictor(LightGBMModel),
                           folds=random_kfold_folds(train))
    # Note the comparison is against the 800-round fit, since a random fold has no "last month"
    # to early-stop on. That is itself part of why the scheme is unusable here.
    print(leaky.round(3).to_string())
    optimism = (gbm_table.loc["mean"] - leaky.loc["mean"]) / gbm_table.loc["mean"] * 100
    print("\nHow much a random split flatters this model (percent):")
    print(optimism.round(2).to_string())

    fitted = LightGBMModel().fit(train)
    print("\nTop 12 features by gain:")
    print(fitted.importances().head(12).round(0).to_string())
    print(f"\nsmearing factor: {fitted.transform_.smearing:.5f}")
