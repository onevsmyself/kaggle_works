import json
import shutil
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from catboost import CatBoostRegressor, Pool
from pathlib import Path
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error

# параметры
N_FOLDS = 5  # количество временных фолдов
TEMPORAL_CUTOFF = "2012-2"  # дата отсечки для оценки mae
CHECKPOINT_DIR = Path("checkpoints")  # прогресс между запусками
RESUME_FROM_CHECKPOINT = False
FRESH_START = True

# общее для моделей
ITERS = 5000
EARLY_STOPPING = 200
RANDOM_STATE = 42
MIN_DATA_IN_LEAF = 30
LEARNING_RATE = 0.03

CATBOOST_PARAMS = {
    "loss_function": "MAE",
    "eval_metric": "MAE",
    "iterations": ITERS,
    "learning_rate": LEARNING_RATE,
    "depth": 7,
    "l2_leaf_reg": 8,
    "min_data_in_leaf": MIN_DATA_IN_LEAF,
    "bagging_temperature": 1.0,
    "random_strength": 2.0,
    "verbose": 0,
    "early_stopping_rounds": EARLY_STOPPING,
    "random_state": RANDOM_STATE,
}

LIGHTGBM_PARAMS = {
    "objective": "regression_l1",
    "metric": "mae",
    "n_estimators": ITERS,
    "learning_rate": LEARNING_RATE,
    "num_leaves": 63,
    "max_depth": 8,
    "min_data_in_leaf": MIN_DATA_IN_LEAF,
    "feature_fraction": 0.6,
    "bagging_fraction": 0.6,
    "bagging_freq": 1,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "verbose": -1,
    "random_state": RANDOM_STATE,
}

XGBOOST_PARAMS = {
    "objective": "reg:absoluteerror",
    "eval_metric": "mae",
    "n_estimators": ITERS,
    "learning_rate": LEARNING_RATE,
    "max_depth": 7,
    "min_child_weight": MIN_DATA_IN_LEAF,
    "subsample": 0.6,
    "colsample_bytree": 0.6,
    "reg_alpha": 0.2,
    "reg_lambda": 0.2,
    "enable_categorical": True,
    "tree_method": "hist",
    "early_stopping_rounds": EARLY_STOPPING,
    "random_state": RANDOM_STATE,
}


def make_catboost() -> CatBoostRegressor:
    return CatBoostRegressor(**CATBOOST_PARAMS)


def make_lightgbm() -> lgb.LGBMRegressor:
    return lgb.LGBMRegressor(**LIGHTGBM_PARAMS)


def make_xgboost() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(**XGBOOST_PARAMS)


def checkpoint_path(name: str) -> Path:
    return CHECKPOINT_DIR / name


def save_checkpoint(name: str, arr: np.ndarray) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    np.save(checkpoint_path(name), arr)


def load_checkpoint(name: str) -> np.ndarray | None:
    path = checkpoint_path(f"{name}.npy")

    if path.exists():
        return np.load(path)
    return None


def save_cv_checkpoint(meta_weights: np.ndarray, temp_mae: float, cv_mae: float) -> None:
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    np.save(checkpoint_path("meta_weights"), meta_weights)

    with open(checkpoint_path("cv_metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"temp_mae": temp_mae, "cv_mae": cv_mae}, f)


def load_cv_checkpoint() -> tuple[np.ndarray, float, float] | None:
    weights_path = checkpoint_path("meta_weights.npy")
    metrics_path = checkpoint_path("cv_metrics.json")

    if not weights_path.exists() or not metrics_path.exists():
        return None
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    return np.load(weights_path), metrics["temp_mae"], metrics["cv_mae"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv("Train.csv")
    test = pd.read_csv("Test.csv")
    return train, test


def add_base_features(df: pd.DataFrame, origin_month: int | None = None) -> pd.DataFrame:
    out = df.copy()

    parts = out["date"].str.split("-", expand=True).astype(int)
    out["year"] = parts[0]
    out["month"] = parts[1]
    out["year_month_idx"] = out["year"] * 12 + out["month"]
    if origin_month is None:
        origin_month = int(out["year_month_idx"].min())
        
    out["months_from_start"] = out["year_month_idx"] - origin_month

    out["area_x_rooms"] = out["area"] * out["rooms"]
    out["floor_per_area"] = out["floor"] / out["area"].replace(0, np.nan)
    out["rooms_per_area"] = out["rooms"] / out["area"].replace(0, np.nan)
    out["area_per_room"] = out["area"] / out["rooms"].replace(0, np.nan)

    kw_cols = [f"kw{i}" for i in range(1, 14)]
    out["kw_sum"] = out[kw_cols].sum(axis=1)

    for col in ["build_tech", "metro_dist", "g_lift"]:
        out[f"{col}_missing"] = out[col].isna().astype(int)

    return out


def make_temporal_folds(df: pd.DataFrame, n_folds: int = N_FOLDS) -> list[tuple[np.ndarray, np.ndarray]]:
    # train = прошлые месяцы, val = следующий блок месяцев
    periods = sorted(df["year_month_idx"].unique())
    n_periods = len(periods)
    min_train = max(1, n_periods // (n_folds + 1))
    val_size = max(1, (n_periods - min_train) // n_folds)

    folds = []
    for i in range(n_folds):
        val_start = min_train + i * val_size
        val_end = n_periods if i == n_folds - 1 else min(val_start + val_size, n_periods)
        if val_start >= n_periods:
            break

        train_periods = set(periods[:val_start])
        val_periods = set(periods[val_start:val_end])
        tr_idx = np.where(df["year_month_idx"].isin(train_periods))[0]
        val_idx = np.where(df["year_month_idx"].isin(val_periods))[0]

        if len(tr_idx) > 0 and len(val_idx) > 0:
            folds.append((tr_idx, val_idx))

    return folds


def add_temporal_market_features(train: pd.DataFrame, test: pd.DataFrame,
folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train["market_price_past"] = np.nan

    for tr_idx, val_idx in folds:
        month_mean = train.iloc[tr_idx].groupby("year_month_idx")["price"].mean()
        train.loc[val_idx, "market_price_past"] = month_mean.mean()

    unfilled = train["market_price_past"].isna()
    if unfilled.any():
        month_mean = train.loc[unfilled].groupby("year_month_idx")["price"].mean()

        for month in month_mean.index:
            past = month_mean[month_mean.index < month]
            if len(past) > 0:
                train.loc[unfilled & (train["year_month_idx"] == month), "market_price_past"] = past.mean()

    month_mean = train.groupby("year_month_idx")["price"].mean()
    test["market_price_past"] = train["price"].mean()
    for month in test["year_month_idx"].unique():
        past = month_mean[month_mean.index < month]

        if len(past) > 0:
            test.loc[test["year_month_idx"] == month, "market_price_past"] = past.mean()

    return train, test


def oof_street_encoding(train: pd.DataFrame, test: pd.DataFrame,
folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()

    enc_cols = [
        "street_price_mean",
        "street_price_median",
        "street_price_per_sqm",
        "street_count",
    ]
    for col in enc_cols:
        train[col] = np.nan

    for tr_idx, val_idx in folds:
        tr = train.iloc[tr_idx]
        stats = tr.groupby("street_id").agg(
            street_price_mean=("price", "mean"),
            street_price_median=("price", "median"),
            street_price_per_sqm=("price", lambda s: (s / tr.loc[s.index, "area"]).mean()),
            street_count=("price", "count"),
        )

        val_part = train.iloc[val_idx][["street_id"]].merge(
            stats, left_on="street_id", right_index=True, how="left"
        )

        fill = {
            "street_price_mean": tr["price"].mean(),
            "street_price_median": tr["price"].median(),
            "street_price_per_sqm": (tr["price"] / tr["area"].replace(0, np.nan)).mean(),
            "street_count": 0,
        }
        for col in enc_cols:
            train.iloc[val_idx, train.columns.get_loc(col)] = val_part[col].fillna(fill[col]).values

    unfilled = train["street_price_mean"].isna()
    if unfilled.any():
        early = train.loc[unfilled]
        stats = early.groupby("street_id").agg(
            street_price_mean=("price", "mean"),
            street_price_median=("price", "median"),
            street_price_per_sqm=("price", lambda s: (s / early.loc[s.index, "area"]).mean()),
            street_count=("price", "count"),
        )
        early_part = early[["street_id"]].merge(stats, left_on="street_id", right_index=True, how="left")
        
        fill = {
            "street_price_mean": early["price"].mean(),
            "street_price_median": early["price"].median(),
            "street_price_per_sqm": (early["price"] / early["area"].replace(0, np.nan)).mean(),
            "street_count": 0,
        }
        for col in enc_cols:
            train.loc[unfilled, col] = early_part[col].fillna(fill[col]).values

    full_stats = train.groupby("street_id").agg(
        street_price_mean=("price", "mean"),
        street_price_median=("price", "median"),
        street_price_per_sqm=("price", lambda s: (s / train.loc[s.index, "area"]).mean()),
        street_count=("price", "count"),
    )

    test_part = test[["street_id"]].merge(
        full_stats, left_on="street_id", right_index=True, how="left"
    )

    fill = {
        "street_price_mean": train["price"].mean(),
        "street_price_median": train["price"].median(),
        "street_price_per_sqm": (train["price"] / train["area"].replace(0, np.nan)).mean(),
        "street_count": 0,
    }

    for col in enc_cols:
        test[col] = test_part[col].fillna(fill[col]).values

    train["street_price_by_area"] = train["street_price_per_sqm"] * train["area"]
    test["street_price_by_area"] = test["street_price_per_sqm"] * test["area"]

    return train, test


def get_feature_columns() -> tuple[list[str], list[str]]:
    cat_features = ["street_id"]
    num_features = [
        "year",
        "month",
        "year_month_idx",
        "months_from_start",
        "build_tech",
        "floor",
        "area",
        "rooms",
        "balcon",
        "metro_dist",
        "g_lift",
        "n_photos",
        "area_x_rooms",
        "floor_per_area",
        "rooms_per_area",
        "area_per_room",
        "kw_sum",
        "build_tech_missing",
        "metro_dist_missing",
        "g_lift_missing",
        "street_price_mean",
        "street_price_median",
        "street_price_per_sqm",
        "street_count",
        "street_price_by_area",
        "market_price_past",
    ] + [f"kw{i}" for i in range(1, 14)]
    return cat_features, num_features


def temporal_mae(train: pd.DataFrame, pred_cv: np.ndarray, oof_mask: np.ndarray | None = None) -> float:
    year, month = map(int, TEMPORAL_CUTOFF.split("-"))
    mask = train["year_month_idx"] >= year * 12 + month

    if oof_mask is not None:
        mask = mask & oof_mask
    if mask.sum() == 0:
        return float("nan")
    return mean_absolute_error(train.loc[mask, "price"], pred_cv[mask])


def prepare_xgb_frame(X: pd.DataFrame, cat_features: list[str]) -> pd.DataFrame:
    out = X.copy()

    for col in cat_features:
        out[col] = out[col].astype("category")
    return out


def fit_meta_weights(oof_preds: np.ndarray, y: np.ndarray) -> np.ndarray:
    # мета-алгоритм: подбираем веса 3 моделей, минимизируя MAE на OOF
    n_models = oof_preds.shape[1]

    def objective(w: np.ndarray) -> float:
        w = np.clip(w, 0, None)
        if w.sum() == 0:
            return 1e18
        w = w / w.sum()
        return mean_absolute_error(y, oof_preds @ w)

    result = minimize(objective, np.ones(n_models) / n_models, method="Nelder-Mead")
    w = np.clip(result.x, 0, None)
    if w.sum() == 0:
        return np.ones(n_models) / n_models
    return w / w.sum()


def train_catboost_cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cat_indices: list[int],
folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pred_cv = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
        model = make_catboost()
        model.fit(
            Pool(X.iloc[tr_idx], y[tr_idx], cat_features=cat_indices),
            eval_set=Pool(X.iloc[val_idx], y[val_idx], cat_features=cat_indices),
            use_best_model=True,
        )

        pred_cv[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / len(folds)  # среднее по фолдам - меньше переобучение
        print(f"  CatBoost fold {fold} MAE: {mean_absolute_error(y[val_idx], pred_cv[val_idx]):,.0f}")

    return pred_cv, test_preds


def train_lightgbm_cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cat_features: list[str],
folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pred_cv = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
        model = make_lightgbm()
        model.fit(
            X.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X.iloc[val_idx], y[val_idx])],
            eval_metric="mae",
            categorical_feature=cat_features,
            callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
        )

        pred_cv[val_idx] = model.predict(X.iloc[val_idx])
        test_preds += model.predict(X_test) / len(folds)
        print(f"  LightGBM fold {fold} MAE: {mean_absolute_error(y[val_idx], pred_cv[val_idx]):,.0f}")

    return pred_cv, test_preds


def train_xgboost_cv(X: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, cat_features: list[str],
folds: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    pred_cv = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))
    X_xgb = prepare_xgb_frame(X, cat_features)
    X_test_xgb = prepare_xgb_frame(X_test, cat_features)

    for fold, (tr_idx, val_idx) in enumerate(folds, start=1):
        model = make_xgboost()
        model.fit(
            X_xgb.iloc[tr_idx],
            y[tr_idx],
            eval_set=[(X_xgb.iloc[val_idx], y[val_idx])],
            verbose=False,
        )

        pred_cv[val_idx] = model.predict(X_xgb.iloc[val_idx])
        test_preds += model.predict(X_test_xgb) / len(folds)
        print(f"  XGBoost fold {fold} MAE: {mean_absolute_error(y[val_idx], pred_cv[val_idx]):,.0f}")

    return pred_cv, test_preds


def train_and_predict(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, float, np.ndarray]:
    train = add_base_features(train)
    test = add_base_features(test, origin_month=int(train["year_month_idx"].min()))

    folds = make_temporal_folds(train, N_FOLDS)
    print(f"Временных фолдов: {len(folds)}")

    train, test = add_temporal_market_features(train, test, folds)
    train, test = oof_street_encoding(train, test, folds)

    cat_features, num_features = get_feature_columns()
    feature_cols = cat_features + num_features

    X = train[feature_cols]
    y = train["price"].values
    X_test = test[feature_cols]
    cat_indices = [feature_cols.index(c) for c in cat_features]

    cv_loaded = None
    test_cb = test_lgb = test_xgb = None

    if RESUME_FROM_CHECKPOINT:
        cv_loaded = load_cv_checkpoint()
        test_cb = load_checkpoint("test_catboost")
        test_lgb = load_checkpoint("test_lightgbm")
        test_xgb = load_checkpoint("test_xgboost")

    if cv_loaded is not None and test_cb is not None and test_lgb is not None and test_xgb is not None:
        meta_weights, temp_mae, cv_mae = cv_loaded
        print("CV и test-предсказания загружены из checkpoints/")
        print(f"  веса: CatBoost={meta_weights[0]:.3f}, "
              f"LightGBM={meta_weights[1]:.3f}, XGBoost={meta_weights[2]:.3f}")
        print(f"  сохранённый temporal MAE: {temp_mae:,.0f}")
    else:
        print("\tCV: CatBoost:")
        pred_cv_cb, test_cb = train_catboost_cv(X, y, X_test, cat_indices, folds)
        save_checkpoint("test_catboost", test_cb)

        print("\tCV: LightGBM: ")
        pred_cv_lgb, test_lgb = train_lightgbm_cv(X, y, X_test, cat_features, folds)
        save_checkpoint("test_lightgbm", test_lgb)

        print("\tCV: XGBoost:")
        pred_cv_xgb, test_xgb = train_xgboost_cv(X, y, X_test, cat_features, folds)
        save_checkpoint("test_xgboost", test_xgb)

        oof_matrix = np.column_stack([pred_cv_cb, pred_cv_lgb, pred_cv_xgb])
        oof_mask = np.zeros(len(y), dtype=bool)
        for _, val_idx in folds:
            oof_mask[val_idx] = True

        meta_weights = fit_meta_weights(oof_matrix[oof_mask], y[oof_mask])
        pred_cv = oof_matrix @ meta_weights

        cv_mae = mean_absolute_error(y[oof_mask], pred_cv[oof_mask])
        temp_mae = temporal_mae(train, pred_cv, oof_mask)

        print()
        print(f"Веса мета-модели: CatBoost={meta_weights[0]:.3f}, LightGBM={meta_weights[1]:.3f}, XGBoost={meta_weights[2]:.3f}")

        print(f"Ensemble CV MAE (temporal OOF): {cv_mae:,.0f}")
        print(f"Ensemble MAE (>= {TEMPORAL_CUTOFF}): {temp_mae:,.0f}")

        print(f"\tCatBoost OOF MAE: {mean_absolute_error(y[oof_mask], pred_cv_cb[oof_mask]):,.0f}")
        print(f"\tLightGBM OOF MAE: {mean_absolute_error(y[oof_mask], pred_cv_lgb[oof_mask]):,.0f}")
        print(f"\tXGBoost OOF MAE: {mean_absolute_error(y[oof_mask], pred_cv_xgb[oof_mask]):,.0f}")

        save_cv_checkpoint(meta_weights, temp_mae, cv_mae)
        print("CV сохранён в checkpoints")

    test_preds = (
        meta_weights[0] * test_cb
        + meta_weights[1] * test_lgb
        + meta_weights[2] * test_xgb
    )

    return test_preds, temp_mae, meta_weights


def save_submission(test: pd.DataFrame, predictions: np.ndarray, path: str = "submission.csv") -> None:
    predictions = np.clip(predictions, 500_000, 50_000_000)

    submission = pd.DataFrame({"id": test["id"], "price": predictions})
    submission.to_csv(path, index=False)

    print()
    print(f"Submission в {path}")


train, test = load_data()

if FRESH_START and CHECKPOINT_DIR.exists():
    shutil.rmtree(CHECKPOINT_DIR)
    print("FRESH_START: чекпоинты удалены, обучение с нуля")
    print()

predictions, temp_mae, weights = train_and_predict(train, test)
save_submission(test, predictions)

print(f"Готово")
print(f"temporal MAE = {temp_mae:,.0f}")
