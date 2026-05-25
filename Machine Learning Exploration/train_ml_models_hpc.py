#train five machine learning models
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr

# xgboost
from xgboost import XGBRegressor


# =========================================================
# 1. Path configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

EMB_DIR = DATA_DIR / "embeddings_train_test_only"

TRAIN_EMB_FILE = EMB_DIR / "train_embeddings.pt"
TEST_EMB_FILE = EMB_DIR / "test_embeddings.pt"

# Change this to your own label file
LABEL_FILE = DATA_DIR / "all_with_duration.csv"

# Change this to your own target column name
TARGET_COL = "duration_days"

# If your label file uses a different ID column name instead of "nctid", change it here
ID_COL = "nctid"

OUTPUT_DIR = DATA_DIR / "model_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Utility functions
# =========================================================

def normalize_nctid(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    if not x:
        return None
    return x


def load_embedding_pt(pt_file):
    obj = torch.load(pt_file, map_location="cpu")

    nctids = obj["nctids"]
    embeddings = obj["embeddings"]

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.cpu().numpy()

    df = pd.DataFrame(embeddings)
    df.insert(0, "nctid", [normalize_nctid(x) for x in nctids])

    return df


def load_labels(label_file, id_col="nctid", target_col="duration_days"):
    df = pd.read_csv(label_file)

    if id_col not in df.columns:
        raise ValueError(f"ID column '{id_col}' not found in {label_file}")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {label_file}")

    out = df[[id_col, target_col]].copy()
    out[id_col] = out[id_col].map(normalize_nctid)
    out = out.dropna(subset=[id_col, target_col])

    return out


def merge_embeddings_with_labels(emb_df, label_df, group_name):
    merged = emb_df.merge(label_df, on="nctid", how="inner")

    print(f"\n[{group_name}] embedding rows: {len(emb_df)}")
    print(f"[{group_name}] matched rows after merge: {len(merged)}")
    print(f"[{group_name}] dropped rows: {len(emb_df) - len(merged)}")

    feature_cols = [c for c in merged.columns if c not in ["nctid", TARGET_COL]]
    X = merged[feature_cols].values.astype(np.float32)
    y = merged[TARGET_COL].values.astype(np.float32)

    return merged, X, y


def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
    else:
        pearson_r, pearson_p = np.nan, np.nan

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
        "Pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
    }


def save_predictions(nctids, y_true, y_pred, model_name, output_dir):
    df = pd.DataFrame({
        "nctid": nctids,
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": np.abs(y_pred - y_true)
    })
    df.to_csv(output_dir / f"{model_name}_predictions.csv", index=False)


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test, test_nctids, output_dir):
    print(f"\n==============================")
    print(f"Training: {model_name}")
    print(f"==============================")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = regression_metrics(y_test, y_pred)

    print(f"{model_name} results:")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    save_predictions(
        nctids=test_nctids,
        y_true=y_test,
        y_pred=y_pred,
        model_name=model_name,
        output_dir=output_dir
    )

    return metrics


# =========================================================
# 3. Main program
# =========================================================

def main():
    print("Loading embeddings...")
    train_emb_df = load_embedding_pt(TRAIN_EMB_FILE)
    test_emb_df = load_embedding_pt(TEST_EMB_FILE)

    print("Loading labels...")
    label_df = load_labels(LABEL_FILE, id_col=ID_COL, target_col=TARGET_COL)

    train_df, X_train, y_train = merge_embeddings_with_labels(train_emb_df, label_df, "train")
    test_df, X_test, y_test = merge_embeddings_with_labels(test_emb_df, label_df, "test")

    test_nctids = test_df["nctid"].tolist()

    print("\nFinal shapes:")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    # =====================================================
    # 4. Model definitions
    # =====================================================

    models = {
        "linear_regression": LinearRegression(),

        "random_forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
            n_jobs=-1
        ),

        "adaboost": AdaBoostRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=42
        ),

        "xgboost": XGBRegressor(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1
        ),

        "mlp": MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            batch_size=64,
            learning_rate_init=1e-3,
            max_iter=300,
            early_stopping=True,
            random_state=42
        ),
    }

    # =====================================================
    # 5. Training and evaluation
    # =====================================================

    all_results = {}

    for model_name, model in models.items():
        metrics = train_and_evaluate_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            test_nctids=test_nctids,
            output_dir=OUTPUT_DIR
        )
        all_results[model_name] = metrics

    # =====================================================
    # 6. Save summary results
    # =====================================================

    results_df = pd.DataFrame(all_results).T.reset_index().rename(columns={"index": "model"})
    results_df.to_csv(OUTPUT_DIR / "all_model_results.csv", index=False)

    with open(OUTPUT_DIR / "all_model_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    print("\nSaved summary:")
    print(OUTPUT_DIR / "all_model_results.csv")
    print(OUTPUT_DIR / "all_model_results.json")


if __name__ == "__main__":
    main()