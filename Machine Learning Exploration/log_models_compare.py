#Log transformation is applied to the target variable (clinical trial duration) during all model training
import re
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from xgboost import XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False


# =========================================================
# 1. Paths
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

LABEL_FILE = DATA_DIR / "all_with_duration.csv"
CACHE_DIR = DATA_DIR / "transformer_feature_cache_with_phase_enrollment"

OUTPUT_DIR = DATA_DIR / "log_transform_all_models_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ID_COL = "nctid"
TARGET_COL = "duration_days"

FEATURE_ORDER = [
    "title",
    "summary",
    "inclusion_criteria",
    "exclusion_criteria",
    "disease",
    "drug",
    "phase",
    "enrollment",
]

SEED = 42
BATCH_SIZE = 32
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# 2. Utilities
# =========================================================

def normalize_nctid(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    return x if x else None


def load_labels():
    df = pd.read_csv(LABEL_FILE)

    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in {LABEL_FILE}")
    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in {LABEL_FILE}")

    df = df[[ID_COL, TARGET_COL]].copy()
    df[ID_COL] = df[ID_COL].map(normalize_nctid)
    df = df.dropna(subset=[ID_COL, TARGET_COL])
    df = df[df[TARGET_COL] > 0]

    return df


def find_cache_index(split):
    candidates = [
        DATA_DIR / f"{split}_feature_cache_index.csv",
        DATA_DIR / "transformer_mlp_original_vs_no_extreme" / f"{split}_feature_cache_index.csv",
        DATA_DIR / "transformer_mlp_hparam_results_with_phase_enrollment" / f"{split}_feature_cache_index.csv",
        DATA_DIR / "transformer_mlp_original_vs_no_extreme" / "original" / f"{split}_feature_cache_index.csv",
    ]

    for path in candidates:
        if path.exists():
            print(f"Using cache index for {split}: {path}")
            return pd.read_csv(path)

    print(f"No cache index found for {split}. Scanning cache folder...")

    rows = []
    pattern = re.compile(rf"^{split}__(NCT\d+)__(.+)\.npy$")

    for file in CACHE_DIR.glob(f"{split}__*.npy"):
        m = pattern.match(file.name)
        if m:
            rows.append({
                "nctid": m.group(1),
                "feature_name": m.group(2),
                "cache_file": str(file)
            })

    if len(rows) == 0:
        raise FileNotFoundError(f"No cache files found for split={split}")

    return pd.DataFrame(rows)


def build_feature_matrix(cache_df, label_df):
    feature_map = {}

    for _, row in cache_df.iterrows():
        nctid = normalize_nctid(row["nctid"])
        feat = row["feature_name"]
        cache_file = row["cache_file"]

        if feat not in FEATURE_ORDER:
            continue

        if nctid not in feature_map:
            feature_map[nctid] = {}

        feature_map[nctid][feat] = cache_file

    label_df = label_df[label_df[ID_COL].isin(feature_map.keys())].copy()
    label_df = label_df.reset_index(drop=True)

    nctids = []
    X_seq = []
    y = []

    for _, row in label_df.iterrows():
        nctid = row[ID_COL]
        duration = float(row[TARGET_COL])

        feat_vectors = []

        for feat in FEATURE_ORDER:
            if feat in feature_map[nctid]:
                vec = np.load(feature_map[nctid][feat]).astype(np.float32)
            else:
                vec = np.zeros(768, dtype=np.float32)

            feat_vectors.append(vec)

        seq = np.stack(feat_vectors, axis=0)   # [8, 768]

        nctids.append(nctid)
        X_seq.append(seq)
        y.append(duration)

    X_seq = np.stack(X_seq, axis=0).astype(np.float32)   # [N, 8, 768]
    X_flat = X_seq.reshape(X_seq.shape[0], -1)            # [N, 6144]
    y = np.array(y, dtype=np.float32)

    return nctids, X_seq, X_flat, y


def compute_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
    else:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
        "Pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
        "Spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
        "Spearman_p": float(spearman_p) if not np.isnan(spearman_p) else None,
    }


def save_predictions(model_name, split, nctids, y_real, y_pred):
    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "nctid": nctids,
        "y_real": y_real,
        "y_predic": y_pred,
        "abs_error": np.abs(y_real - y_pred),
        "split": split,
        "model": model_name
    })

    df.to_csv(model_dir / f"{split}_predictions.csv", index=False)
    return df


# =========================================================
# 3. Sklearn models
# =========================================================

def run_sklearn_model(model_name, model, X_train, y_train, X_test, y_test, train_nctids, test_nctids):
    print(f"\nRunning {model_name}...")

    y_train_log = np.log1p(y_train)

    model.fit(X_train, y_train_log)

    train_pred_log = model.predict(X_train)
    test_pred_log = model.predict(X_test)

    train_pred = np.expm1(train_pred_log)
    test_pred = np.expm1(test_pred_log)

    train_pred = np.maximum(train_pred, 0)
    test_pred = np.maximum(test_pred, 0)

    train_df = save_predictions(model_name, "train", train_nctids, y_train, train_pred)
    test_df = save_predictions(model_name, "test", test_nctids, y_test, test_pred)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    rows = [
        {"model": model_name, "split": "train", **train_metrics},
        {"model": model_name, "split": "test", **test_metrics},
    ]

    pd.DataFrame(rows).to_csv(OUTPUT_DIR / model_name / "metrics.csv", index=False)

    return rows


# =========================================================
# 4. PyTorch MLP + Transformer
# =========================================================

class SeqDataset(Dataset):
    def __init__(self, X_seq, y_log):
        self.X = torch.tensor(X_seq, dtype=torch.float32)
        self.y = torch.tensor(y_log, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class TransformerMLPRegressor(nn.Module):
    def __init__(
        self,
        input_dim=768,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        mlp_hidden=128
    ):
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        out = self.mlp(pooled).squeeze(-1)
        return out


def evaluate_torch_model(model, loader):
    model.eval()
    preds = []
    trues = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            pred = model(X)

            preds.extend(pred.cpu().numpy().tolist())
            trues.extend(y.cpu().numpy().tolist())

    return np.array(trues), np.array(preds)


def run_transformer_mlp(X_train_seq, y_train, X_test_seq, y_test, train_nctids, test_nctids):
    model_name = "mlp_transformer"
    print(f"\nRunning {model_name} on {DEVICE}...")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    y_train_log = np.log1p(y_train)
    y_test_log = np.log1p(y_test)

    train_dataset = SeqDataset(X_train_seq, y_train_log)
    test_dataset = SeqDataset(X_test_seq, y_test_log)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerMLPRegressor().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.MSELoss()

    history = []
    best_rmse = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for X, y in train_loader:
            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.size(0)

        train_true_log, train_pred_log = evaluate_torch_model(model, train_loader)
        test_true_log, test_pred_log = evaluate_torch_model(model, test_loader)

        train_pred_real = np.maximum(np.expm1(train_pred_log), 0)
        test_pred_real = np.maximum(np.expm1(test_pred_log), 0)

        train_metrics = compute_metrics(y_train, train_pred_real)
        test_metrics = compute_metrics(y_test, test_pred_real)

        row = {
            "epoch": epoch,
            "train_loss": total_loss / len(train_dataset),
            "train_MAE": train_metrics["MAE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "test_MAE": test_metrics["MAE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
            "test_Pearson_r": test_metrics["Pearson_r"],
            "test_Spearman_r": test_metrics["Spearman_r"],
        }

        history.append(row)

        print(
            f"Epoch {epoch}/{EPOCHS} | "
            f"Train MAE: {train_metrics['MAE']:.3f} | "
            f"Test MAE: {test_metrics['MAE']:.3f} | "
            f"Test RMSE: {test_metrics['RMSE']:.3f} | "
            f"Test R2: {test_metrics['R2']:.4f}"
        )

        if test_metrics["RMSE"] < best_rmse:
            best_rmse = test_metrics["RMSE"]
            best_state = model.state_dict()

    model_dir = OUTPUT_DIR / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, model_dir / "best_model.pt")
    pd.DataFrame(history).to_csv(model_dir / "training_history.csv", index=False)

    model.load_state_dict(best_state)

    _, train_pred_log = evaluate_torch_model(model, DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False))
    _, test_pred_log = evaluate_torch_model(model, test_loader)

    train_pred = np.maximum(np.expm1(train_pred_log), 0)
    test_pred = np.maximum(np.expm1(test_pred_log), 0)

    save_predictions(model_name, "train", train_nctids, y_train, train_pred)
    save_predictions(model_name, "test", test_nctids, y_test, test_pred)

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)

    rows = [
        {"model": model_name, "split": "train", **train_metrics},
        {"model": model_name, "split": "test", **test_metrics},
    ]

    pd.DataFrame(rows).to_csv(model_dir / "metrics.csv", index=False)

    return rows


# =========================================================
# 5. Main
# =========================================================

def main():
    print(f"Using device: {DEVICE}")

    print("Loading labels...")
    label_df = load_labels()

    print("Loading cache index...")
    train_cache_df = find_cache_index("train")
    test_cache_df = find_cache_index("test")

    print("Building train features...")
    train_nctids, X_train_seq, X_train_flat, y_train = build_feature_matrix(train_cache_df, label_df)

    print("Building test features...")
    test_nctids, X_test_seq, X_test_flat, y_test = build_feature_matrix(test_cache_df, label_df)

    print("Train seq shape:", X_train_seq.shape)
    print("Test seq shape :", X_test_seq.shape)
    print("Train flat shape:", X_train_flat.shape)
    print("Test flat shape :", X_test_flat.shape)

    all_metric_rows = []

    # AdaBoost
    adaboost = Pipeline([
        ("scaler", StandardScaler()),
        ("model", AdaBoostRegressor(
            n_estimators=200,
            learning_rate=0.05,
            random_state=SEED
        ))
    ])
    all_metric_rows.extend(
        run_sklearn_model(
            "adaboost_log",
            adaboost,
            X_train_flat,
            y_train,
            X_test_flat,
            y_test,
            train_nctids,
            test_nctids
        )
    )

    # Random Forest
    random_forest = RandomForestRegressor(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=SEED
    )
    all_metric_rows.extend(
        run_sklearn_model(
            "random_forest_log",
            random_forest,
            X_train_flat,
            y_train,
            X_test_flat,
            y_test,
            train_nctids,
            test_nctids
        )
    )

    # XGBoost
    if HAS_XGBOOST:
        xgboost = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="reg:squarederror",
            random_state=SEED,
            n_jobs=-1
        )

        all_metric_rows.extend(
            run_sklearn_model(
                "xgboost_log",
                xgboost,
                X_train_flat,
                y_train,
                X_test_flat,
                y_test,
                train_nctids,
                test_nctids
            )
        )
    else:
        print("WARNING: xgboost is not installed. Skipping XGBoost.")

    # MLP
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-4,
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=SEED,
            verbose=True
        ))
    ])
    all_metric_rows.extend(
        run_sklearn_model(
            "mlp_log",
            mlp,
            X_train_flat,
            y_train,
            X_test_flat,
            y_test,
            train_nctids,
            test_nctids
        )
    )

    # MLP + Transformer
    all_metric_rows.extend(
        run_transformer_mlp(
            X_train_seq,
            y_train,
            X_test_seq,
            y_test,
            train_nctids,
            test_nctids
        )
    )

    metrics_df = pd.DataFrame(all_metric_rows)
    metrics_df.to_csv(OUTPUT_DIR / "log_model_metrics.csv", index=False)

    with open(OUTPUT_DIR / "log_model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=2)

    print("\nFinal metrics:")
    print(metrics_df)

    print("\nSaved:")
    print(OUTPUT_DIR / "log_model_metrics.csv")


if __name__ == "__main__":
    main()