from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models.autogluon_tabular import DirectTabularModel
from autogluon.timeseries.models.autogluon_tabular.utils import MLF_ITEMID, MLF_TARGET, MLF_TIMESTAMP
from autogluon.timeseries.utils.features import TimeSeriesFeatureGenerator


def generate_synthetic_data(
    num_items: int = 4,
    length: int = 80,
    freq: str = "D",
    seed: int = 0,
) -> TimeSeriesDataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for item_idx in range(num_items):
        item_id = f"item_{item_idx}"
        dates = pd.date_range("2021-01-01", periods=length, freq=freq)
        known_cov = np.linspace(0, 1, length) + rng.normal(scale=0.05, size=length)
        past_cov = rng.normal(scale=1.0, size=length)
        target = np.zeros(length, dtype=float)
        for t in range(length):
            prev_y = target[t - 1] if t > 0 else 0.0
            prev_past = past_cov[t - 1] if t > 0 else 0.0
            target[t] = 0.4 * prev_y + 1.25 * prev_past + 0.25 * known_cov[t] + rng.normal(scale=0.05)
        idx = pd.MultiIndex.from_arrays([[item_id] * length, dates], names=[TimeSeriesDataFrame.ITEMID, TimeSeriesDataFrame.TIMESTAMP])
        frames.append(pd.DataFrame({"target": target, "known_cov": known_cov, "past_cov": past_cov}, index=idx))
    return TimeSeriesDataFrame(pd.concat(frames))


def evaluate_models():
    freq = "D"
    prediction_length = 8

    raw_ts = generate_synthetic_data(freq=freq)
    feat_gen = TimeSeriesFeatureGenerator(target="target", known_covariates_names=["known_cov"])
    data = feat_gen.fit_transform(raw_ts)

    train_data, known_cov_future = data.get_model_inputs_for_scoring(prediction_length, known_covariates_names=["known_cov"])
    future_data = data.slice_by_timestep(-prediction_length, None)

    model = DirectTabularModel(
        path="ag_tmp_model",
        freq=freq,
        prediction_length=prediction_length,
        covariate_metadata=feat_gen.covariate_metadata,
        eval_metric="MAE",
        hyperparameters={"model_name": "LR"},
    )
    model.fit(train_data=train_data, time_limit=30)
    preds_ag = model.predict(train_data, known_covariates=known_cov_future)
    mae_ag = float((preds_ag["mean"] - future_data["target"]).abs().mean())

    train_df, _ = model._generate_train_val_dfs(train_data)
    past_cov_features = [c for c in train_df.columns if "past_cov_lag" in c]

    X_train = train_df.drop(columns=[MLF_TARGET, MLF_ITEMID], errors="ignore").fillna(0.0)
    y_train = train_df[MLF_TARGET]
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train, y_train)

    if known_cov_future is not None:
        data_future = known_cov_future.copy()
    else:
        horizon_index = model.get_forecast_horizon_index(train_data)
        data_future = pd.DataFrame(columns=[model.target], index=horizon_index, dtype="float32")
    data_future[model.target] = float("inf")
    data_extended = pd.concat([train_data, data_future])
    mlforecast_df = model._to_mlforecast_df(data_extended, train_data.static_features)
    if model._max_ts_length is not None:
        mlforecast_df = model._shorten_all_series(mlforecast_df, model._max_ts_length + model.prediction_length)
    df_future = model._mlf.preprocess(mlforecast_df, dropna=False, static_features=[])
    df_future = df_future.groupby(MLF_ITEMID, sort=False).tail(prediction_length)
    df_future = df_future.replace(float("inf"), float("nan"))
    X_future = df_future.drop(columns=[MLF_TARGET, MLF_ITEMID, MLF_TIMESTAMP], errors="ignore").fillna(0.0)
    ridge_preds = ridge.predict(X_future)

    preds_manual = pd.DataFrame({MLF_ITEMID: df_future[MLF_ITEMID].values, MLF_TIMESTAMP: df_future[MLF_TIMESTAMP].values, "mean": ridge_preds})

    if hasattr(model._mlf.ts, "target_transforms") and model._mlf.ts.target_transforms is not None:
        from autogluon.timeseries.models.autogluon_tabular.transforms import apply_inverse_transform

        pred_df = preds_manual.rename(columns={"mean": MLF_TARGET})
        for tfm in model._mlf.ts.target_transforms[::-1]:
            pred_df = apply_inverse_transform(pred_df, transform=tfm)
        preds_manual["mean"] = pred_df[MLF_TARGET]

    preds_manual_ts = TimeSeriesDataFrame(
        preds_manual.rename(columns={MLF_ITEMID: TimeSeriesDataFrame.ITEMID, MLF_TIMESTAMP: TimeSeriesDataFrame.TIMESTAMP})
    )
    manual_mae = float((preds_manual_ts["mean"] - future_data[model.target]).abs().mean())

    print(f"AutoGluon DirectTabularModel MAE: {mae_ag:.4f}")
    print(f"Past covariate lags used: {past_cov_features[:4]} ... ({len(past_cov_features)} total)")
    print(f"Standalone Ridge on MLForecast features MAE: {manual_mae:.4f}")


if __name__ == "__main__":
    evaluate_models()
