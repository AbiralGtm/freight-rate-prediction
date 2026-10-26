# Freight Rate Prediction

Spotter ML Engineer assessment. Predicts the dollar rate of a freight load from its origin,
destination, distance, equipment type, weight, date and two market signals.

Train on 48,000 labelled loads covering 1 January to 31 October 2025, then predict 12,000
unlabelled loads covering 1 November to 31 December 2025. Training and prediction windows do not
overlap, so this is a forecasting problem, not interpolation, and every decision below follows
from that.

## Result

Validated on three expanding-window folds. The baseline uses no feature except distance.

| Model | RMSE ($) | MAE ($) | MAPE (%) | MAPE spread |
|---|---:|---:|---:|---:|
| Flat $/mi baseline | 705.0 | 291.0 | 12.13 | 0.39 |
| **Ridge on log target** (submitted) | **627.5** | **119.1** | **5.16** | **0.69** |
| LightGBM | 630.1 | 130.7 | 5.87 | 2.29 |

Mean across the three folds. MAPE spread is max minus min, which is the honest cost of
month-to-month drift. `score.py` computes no error metric — these come from held-out months, and
Spotter calculates the final figures after submission.

**The linear model won, which is not the expected outcome.** `distance` carries a LightGBM gain
of 128,690 and `log(distance)` 36,010 against 1,264 for the next feature; the target is close to
linear in `log(distance)` with a slope of 0.871, which ridge fits exactly and trees approximate
with step functions. Trees also cannot extrapolate the trend term 61 days past the end of
training. Section 6 of `report.pdf` has the three checks that confirm it is not an under-training
artefact.

## Run it

Python 3.13. From this directory:

```bash
python -m pip install -r requirements.txt

python predict.py    # fits ridge on all 48,000 rows, writes validation_predictions.csv
python chart.py      # writes december-chart-inputs.csv with predicted_rate filled

python score.py --predictions validation_predictions.csv \
                --december-predictions december-chart-inputs.csv
```

The scorer validates both files and writes `scorer_results/candidate_december.png`.

Note the paths: the filled output files sit in this directory, **not** under `data/`. The copies
in `data/` are the supplied inputs and their `predicted_rate` column is deliberately left empty.
Passing those to the scorer fails with `ERROR: predictions contains invalid predicted_rate
values`.

## Outputs

| File | What it is |
|---|---|
| `validation_predictions.csv` | 12,000 predictions, `load_id,predicted_rate` |
| `december-chart-inputs.csv` | The 31 chart rows with `predicted_rate` filled in place |
| `scorer_results/candidate_december.png` | The December sensitivity curve |
| `report.pdf` | Written report: split, validation approach, decisions, embedded chart |

## Code

Each module runs standalone and prints the evidence behind its own decisions, so
`python clean_data.py` shows the cleaning counts, `python splits.py` shows the fold table, and so
on. The pipeline runs in this order:

| Module | Role |
|---|---|
| `eda.ipynb` | Exploration. Every decision below traces back to a cell here. |
| `clean_data.py` | Sign errors, imputation, the city-to-coordinate lookup, December market signals. One cleaning path for all three frames. |
| `build_features.py` | 24 features. Size, circuity, categoricals, date terms, market signals, cleaning flags. |
| `target_transform.py` | `log(posted_rate)` with a Duan smearing correction, plus the rejected alternatives. |
| `splits.py` | Expanding-window folds, RMSE/MAE/MAPE, and the random-KFold contrast. |
| `models.py` | Ridge baseline and LightGBM behind one `fit`/`predict` interface. |
| `predict.py` | Final refit and the 12,000 predictions. |
| `chart.py` | The 31-point December curve and its acceptance checks. |

## Decisions

Full reasoning with evidence is in `report.pdf`. In short:

- **Split by date, never at random.** Three expanding-window folds (train January–July, validate
  August; then January–August, validate September; then January–September, validate October),
  each scoring the month immediately after its training window. A shuffled split puts October
  rows into training while scoring August; measured here it flatters LightGBM by 4.9% on RMSE.
- **Negative weights are sign errors, not outliers.** 292 training and 145 validation rows. Their
  absolute values average 31,724 lb against 31,415 lb for positive rows — the same distribution,
  flipped. Take the absolute value.
- **Impute and flag, never drop.** Missing `weight` (300 / 165) and `market_index` (374 / 249)
  get the training mean plus an indicator column. All 12,000 load IDs must be returned.
- **Target is `log(posted_rate)`.** Raw rate spans $57 to $25,533. Rate per mile is the obvious
  alternative but has excess kurtosis of +114 and hard-codes the distance exponent at exactly
  1.0 when the data says 0.871.
- **Coordinates are dropped as features.** Every city maps to exactly one lat/lon pair, so they
  are a deterministic lookup of the city name. The lookup itself is kept, to reconstruct the
  December rows.
- **The December gap is closed without forecasting.** The chart file has six input columns and no
  coordinates or market signals. Coordinates come from the city lookup; the signals come from the
  6,164 real December rows already present in `validation.csv`, as a per-date median. The chart
  therefore comes off the same model as the submission.
- **Trimming happens inside the training fold only.** Trimming a validation fold would delete the
  hard rows and flatter every metric.

## Known limitations

- **The December trend is extrapolated, not observed.** The curve rises $37 across the month
  purely because ridge continues the linear trend 61 days past the end of training. If the real
  market flattened or turned in Q4, this is confidently wrong in a way a tree's flat line would
  not have been. It is the single largest assumption in the submission.
- **RMSE barely separates the models** (705 → 628 → 630), because a few very long hauls dominate
  the squared term. MAE and MAPE are where they differ, which is why all three are reported per
  fold rather than blended into one number.
- **1,447 validation rows (12.1%) involve a city never seen in training.** 8 of the 72 cities
  appear only in validation. They are flagged and fall back on the remaining features, but carry
  more uncertainty than the fold metrics suggest, since every fold was scored on cities the model
  had already seen.

## Environment

Developed and measured on Python 3.13.11 with pandas 3.1.0, numpy 2.5.4, matplotlib 3.11.2,
scikit-learn 1.9.1 and lightgbm 4.7.0. `requirements.txt` is unpinned; pin these versions if you
need the numbers above to reproduce exactly.

Opening `eda.ipynb` additionally needs `jupyter`, which is not in `requirements.txt` because no
part of the pipeline imports it.
