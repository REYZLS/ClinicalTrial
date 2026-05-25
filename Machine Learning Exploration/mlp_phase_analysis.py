#Training the MLP model by phase
import re
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from scipy.stats import pearsonr, spearmanr


# =========================================================
# 1. Path settings
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

TRAIN_XML_DIR = DATA_DIR / "train_xml"
TEST_XML_DIR = DATA_DIR / "test_xml"

LABEL_FILE = DATA_DIR / "all_with_duration.csv"
ID_COL = "nctid"
TARGET_COL = "duration_days"

CACHE_DIR = DATA_DIR / "transformer_feature_cache_with_phase_enrollment"

OUTPUT_DIR = DATA_DIR / "mlp_phase_analysis_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


# =========================================================
# 2. Utility functions
# =========================================================

def normalize_nctid(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    return x if x else None


def clean_text(text):
    if text is None:
        return None
    text = str(text).strip()
    text = " ".join(text.split())
    return text if text else None


def get_text(node, path):
    found = node.find(path)
    if found is not None:
        return clean_text(found.text)
    return None


def extract_phase_from_xml(xml_path):
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        nctid = get_text(root, "id_info/nct_id")
        phase = get_text(root, "phase")

        return {
            "nctid": normalize_nctid(nctid or xml_path.stem),
            "phase": phase
        }

    except Exception:
        return {
            "nctid": normalize_nctid(xml_path.stem),
            "phase": None
        }


def load_phase_map(xml_dir):
    rows = []

    xml_files = sorted(xml_dir.rglob("*.xml"))
    print(f"Extracting phase from {len(xml_files)} XML files in {xml_dir}")

    for xml_file in xml_files:
        rows.append(extract_phase_from_xml(xml_file))

    phase_df = pd.DataFrame(rows)
    phase_df = phase_df.dropna(subset=["nctid"])
    phase_df = phase_df.drop_duplicates(subset=["nctid"])

    return phase_df


def load_labels(label_file):
    df = pd.read_csv(label_file)

    if ID_COL not in df.columns:
        raise ValueError(f"{ID_COL} not found in {label_file}")

    if TARGET_COL not in df.columns:
        raise ValueError(f"{TARGET_COL} not found in {label_file}")

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

    print(f"No cache index found for {split}. Scanning npy cache folder...")

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
        raise FileNotFoundError(f"No cache files found for split={split} in {CACHE_DIR}")

    return pd.DataFrame(rows)


# =========================================================
# 3. Build feature matrix
# =========================================================

def build_feature_matrix(cache_df, label_df, phase_df):
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

    merged = label_df[label_df[ID_COL].isin(feature_map.keys())].copy()
    merged = merged.merge(phase_df, on="nctid", how="left")
    merged = merged.reset_index(drop=True)

    X_list = []
    y_list = []
    nctids = []
    phases = []

    for _, row in merged.iterrows():
        nctid = row[ID_COL]
        duration = float(row[TARGET_COL])
        phase = row["phase"]

        feat_vectors = []

        for feat in FEATURE_ORDER:
            if feat in feature_map[nctid]:
                vec = np.load(feature_map[nctid][feat]).astype(np.float32)
            else:
                vec = np.zeros(768, dtype=np.float32)

            feat_vectors.append(vec)

        # [8, 768] -> [6144]
        x = np.concatenate(feat_vectors, axis=0)

        X_list.append(x)
        y_list.append(duration)
        nctids.append(nctid)
        phases.append(phase)

    X = np.vstack(X_list).astype(np.float32)
    y = np.array(y_list, dtype=np.float32)

    meta_df = pd.DataFrame({
        "nctid": nctids,
        "phase": phases,
        "real_duration": y
    })

    return meta_df, X, y


# =========================================================
# 4. Metrics
# =========================================================

def compute_metrics(y_true, y_pred):
    if len(y_true) == 0:
        return {
            "n": 0,
            "MAE": None,
            "RMSE": None,
            "R2": None,
            "Pearson_r": None,
            "Spearman_r": None,
        }

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r, _ = pearsonr(y_true, y_pred)
        spearman_r, _ = spearmanr(y_true, y_pred)
    else:
        pearson_r = np.nan
        spearman_r = np.nan

    return {
        "n": int(len(y_true)),
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2) if not np.isnan(r2) else None,
        "Pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
        "Spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
    }


def metrics_by_phase(pred_df, split_name):
    rows = []

    # Overall
    overall = compute_metrics(
        pred_df["real_duration"].values,
        pred_df["predicted_duration"].values
    )
    overall["split"] = split_name
    overall["phase"] = "All"
    rows.append(overall)

    # By phase
    for phase, g in pred_df.groupby("phase", dropna=False):
        phase_name = "Missing" if pd.isna(phase) else str(phase)

        m = compute_metrics(
            g["real_duration"].values,
            g["predicted_duration"].values
        )
        m["split"] = split_name
        m["phase"] = phase_name
        rows.append(m)

    return pd.DataFrame(rows)


# =========================================================
# 5. Main
# =========================================================

def main():
    print("Loading labels...")
    label_df = load_labels(LABEL_FILE)

    print("Loading phase information...")
    train_phase_df = load_phase_map(TRAIN_XML_DIR)
    test_phase_df = load_phase_map(TEST_XML_DIR)

    train_phase_df.to_csv(OUTPUT_DIR / "train_phase_info.csv", index=False)
    test_phase_df.to_csv(OUTPUT_DIR / "test_phase_info.csv", index=False)

    print("Loading cache index...")
    train_cache_df = find_cache_index("train")
    test_cache_df = find_cache_index("test")

    print("Building train matrix...")
    train_meta_df, X_train, y_train = build_feature_matrix(
        train_cache_df,
        label_df,
        train_phase_df
    )

    print("Building test matrix...")
    test_meta_df, X_test, y_test = build_feature_matrix(
        test_cache_df,
        label_df,
        test_phase_df
    )

    print(f"Train X shape: {X_train.shape}")
    print(f"Test X shape : {X_test.shape}")

    print("Training MLPRegressor...")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            learning_rate_init=1e-4,
            max_iter=300,
            random_state=SEED,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=True
        ))
    ])

    model.fit(X_train, y_train)

    print("Predicting train/test...")

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_pred = np.maximum(train_pred, 0)
    test_pred = np.maximum(test_pred, 0)

    train_pred_df = train_meta_df.copy()
    train_pred_df["predicted_duration"] = train_pred
    train_pred_df["abs_error"] = np.abs(
        train_pred_df["real_duration"] - train_pred_df["predicted_duration"]
    )
    train_pred_df["split"] = "train"

    test_pred_df = test_meta_df.copy()
    test_pred_df["predicted_duration"] = test_pred
    test_pred_df["abs_error"] = np.abs(
        test_pred_df["real_duration"] - test_pred_df["predicted_duration"]
    )
    test_pred_df["split"] = "test"

    # Reorder columns
    cols = [
        "split",
        "nctid",
        "phase",
        "real_duration",
        "predicted_duration",
        "abs_error"
    ]

    train_pred_df = train_pred_df[cols]
    test_pred_df = test_pred_df[cols]

    train_pred_df.to_csv(
        OUTPUT_DIR / "mlp_train_predictions_by_trial.csv",
        index=False
    )

    test_pred_df.to_csv(
        OUTPUT_DIR / "mlp_test_predictions_by_trial.csv",
        index=False
    )

    all_pred_df = pd.concat([train_pred_df, test_pred_df], axis=0)
    all_pred_df.to_csv(
        OUTPUT_DIR / "mlp_train_test_predictions_by_trial.csv",
        index=False
    )

    print("Computing phase-level metrics...")

    train_phase_metrics = metrics_by_phase(train_pred_df, "train")
    test_phase_metrics = metrics_by_phase(test_pred_df, "test")

    phase_metrics_df = pd.concat(
        [train_phase_metrics, test_phase_metrics],
        axis=0
    )

    phase_metrics_df = phase_metrics_df[
        [
            "split",
            "phase",
            "n",
            "MAE",
            "RMSE",
            "R2",
            "Pearson_r",
            "Spearman_r"
        ]
    ]

    phase_metrics_df.to_csv(
        OUTPUT_DIR / "mlp_metrics_by_phase.csv",
        index=False
    )

    with open(OUTPUT_DIR / "mlp_metrics_by_phase.json", "w", encoding="utf-8") as f:
        json.dump(
            phase_metrics_df.to_dict(orient="records"),
            f,
            indent=2
        )

    print("\nSaved files:")
    print(OUTPUT_DIR / "mlp_train_predictions_by_trial.csv")
    print(OUTPUT_DIR / "mlp_test_predictions_by_trial.csv")
    print(OUTPUT_DIR / "mlp_train_test_predictions_by_trial.csv")
    print(OUTPUT_DIR / "mlp_metrics_by_phase.csv")

    print("\nPhase-level metrics:")
    print(phase_metrics_df)


if __name__ == "__main__":
    main()