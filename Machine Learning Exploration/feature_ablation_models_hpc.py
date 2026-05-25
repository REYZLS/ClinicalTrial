#Feature ablation was conducted on title, summary, inclusion criteria, exclusion criteria, disease, and drug features,
#with performance evaluated across Linear Regression, Random Forest, AdaBoost, XGBoost, and MLP models.
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from scipy.stats import pearsonr, spearmanr
from xgboost import XGBRegressor


# =========================================================
# 1. Path configuration
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

TRAIN_XML_DIR = DATA_DIR / "train_xml"
TEST_XML_DIR = DATA_DIR / "test_xml"

LABEL_FILE = DATA_DIR / "all_with_duration.csv"
ID_COL = "nctid"
TARGET_COL = "duration_days"

OUTPUT_DIR = DATA_DIR / "feature_ablation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = DATA_DIR / "feature_embedding_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Model configuration
# Use the ClinicalBERT model that was successfully run
# =========================================================

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_FEATURES = [
    "title",
    "summary",
    "inclusion_criteria",
    "exclusion_criteria",
    "disease",
    "drug",
]

# Feature ablation settings to compare
EXPERIMENTS = {
    "all_features": [
        "title", "summary", "inclusion_criteria",
        "exclusion_criteria", "disease", "drug"
    ],
    "no_inclusion_criteria": [
        "title", "summary", "exclusion_criteria", "disease", "drug"
    ],
    "no_exclusion_criteria": [
        "title", "summary", "inclusion_criteria", "disease", "drug"
    ],
    "no_criteria": [
        "title", "summary", "disease", "drug"
    ],
    "no_drug": [
        "title", "summary", "inclusion_criteria", "exclusion_criteria", "disease"
    ],
    "no_disease": [
        "title", "summary", "inclusion_criteria", "exclusion_criteria", "drug"
    ],
    "no_title": [
        "summary", "inclusion_criteria", "exclusion_criteria", "disease", "drug"
    ],
    "no_summary": [
        "title", "inclusion_criteria", "exclusion_criteria", "disease", "drug"
    ],
}

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# =========================================================
# 3. Utility functions
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


def get_all_texts(node, path):
    results = []
    for item in node.findall(path):
        txt = clean_text(item.text)
        if txt is not None:
            results.append(txt)
    return results


def split_inclusion_exclusion(criteria_text):
    if not criteria_text:
        return None, None

    text = criteria_text.replace("\r", "\n").strip()

    patterns = [
        r"(?is)inclusion criteria\s*:?(.*?)(?:exclusion criteria\s*:?(.*))?$",
        r"(?is)inclusion\s*:?(.*?)(?:exclusion\s*:?(.*))?$",
    ]

    for pattern in patterns:
        m = re.search(pattern, text)
        if m:
            inclusion = clean_text(m.group(1))
            exclusion = clean_text(m.group(2)) if m.lastindex and m.lastindex >= 2 else None
            return inclusion, exclusion

    return clean_text(text), None


def extract_selected_features_from_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    nctid = get_text(root, "id_info/nct_id")

    diseases = get_all_texts(root, "condition")
    disease_text = "; ".join(diseases) if diseases else None

    title = get_text(root, "brief_title")
    summary = get_text(root, "brief_summary/textblock")

    criteria_text = get_text(root, "eligibility/criteria/textblock")
    inclusion_criteria, exclusion_criteria = split_inclusion_exclusion(criteria_text)

    drug_names = []
    for intervention in root.findall("intervention"):
        intervention_type = get_text(intervention, "intervention_type")
        intervention_name = get_text(intervention, "intervention_name")
        if intervention_type and intervention_name and intervention_type.lower() == "drug":
            drug_names.append(intervention_name)
    drug_text = "; ".join(drug_names) if drug_names else None

    return {
        "nctid": nctid,
        "title": title,
        "summary": summary,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "disease": disease_text,
        "drug": drug_text,
    }


def embed_text_mean_pooling(text, tokenizer, model, max_len=512, device="cpu"):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state

        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked_hidden = last_hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)
        valid_tokens = mask.sum(dim=1).clamp(min=1e-9)
        mean_vec = (sum_hidden / valid_tokens).squeeze(0).cpu().numpy().astype(np.float32)

    return mean_vec


def build_feature_cache(xml_dir, split_name):
    xml_files = sorted(Path(xml_dir).rglob("*.xml"))
    cache_records = []

    print(f"\nBuilding feature cache for {split_name}: {len(xml_files)} XML files")

    for xml_file in tqdm(xml_files, desc=f"Caching {split_name}"):
        try:
            feat_dict = extract_selected_features_from_xml(xml_file)
        except Exception as e:
            print(f"[ERROR] XML parse failed: {xml_file}\n{e}")
            continue

        nctid = feat_dict.get("nctid") or xml_file.stem.upper()

        for feat_name in TEXT_FEATURES:
            text = feat_dict.get(feat_name)
            if text is None:
                continue

            cache_file = CACHE_DIR / f"{split_name}__{nctid}__{feat_name}.npy"

            if not cache_file.exists():
                try:
                    vec = embed_text_mean_pooling(
                        text=text,
                        tokenizer=tokenizer,
                        model=model,
                        max_len=MAX_LEN,
                        device=DEVICE
                    )
                    np.save(cache_file, vec)
                except Exception as e:
                    print(f"[WARNING] Embedding failed: {nctid}, {feat_name}\n{e}")
                    continue

            cache_records.append({
                "nctid": nctid,
                "feature_name": feat_name,
                "cache_file": str(cache_file)
            })

    cache_df = pd.DataFrame(cache_records)
    cache_df.to_csv(OUTPUT_DIR / f"{split_name}_feature_cache_index.csv", index=False)
    return cache_df


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


def aggregate_trial_embeddings(cache_df, selected_features):
    grouped = {}

    for _, row in cache_df.iterrows():
        nctid = row["nctid"]
        feat = row["feature_name"]

        if feat not in selected_features:
            continue

        vec = np.load(row["cache_file"])

        if nctid not in grouped:
            grouped[nctid] = []
        grouped[nctid].append(vec)

    results = []
    for nctid, vecs in grouped.items():
        if len(vecs) == 0:
            continue
        trial_vec = np.mean(np.stack(vecs, axis=0), axis=0).astype(np.float32)
        results.append({"nctid": nctid, "embedding": trial_vec})

    return results


def results_to_df(results):
    if len(results) == 0:
        return pd.DataFrame(columns=["nctid"])

    emb = np.stack([x["embedding"] for x in results], axis=0)
    df = pd.DataFrame(emb)
    df.insert(0, "nctid", [normalize_nctid(x["nctid"]) for x in results])
    return df


def merge_embeddings_with_labels(emb_df, label_df, group_name, target_col):
    merged = emb_df.merge(label_df, on="nctid", how="inner")

    print(f"\n[{group_name}] embedding rows: {len(emb_df)}")
    print(f"[{group_name}] matched rows after merge: {len(merged)}")
    print(f"[{group_name}] dropped rows: {len(emb_df) - len(merged)}")

    feature_cols = [c for c in merged.columns if c not in ["nctid", target_col]]
    X = merged[feature_cols].values.astype(np.float32)
    y = merged[target_col].values.astype(np.float32)

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

    if len(y_true) > 1:
        spearman_rho, spearman_p = spearmanr(y_true, y_pred)
    else:
        spearman_rho, spearman_p = np.nan, np.nan

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
        "Pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
        "Spearman_rho": float(spearman_rho) if not np.isnan(spearman_rho) else None,
        "Spearman_p": float(spearman_p) if not np.isnan(spearman_p) else None,
    }


def save_predictions(nctids, y_true, y_pred, experiment_name, model_name, output_dir):
    df = pd.DataFrame({
        "nctid": nctids,
        "y_true": y_true,
        "y_pred": y_pred,
        "abs_error": np.abs(y_pred - y_true)
    })
    df.to_csv(output_dir / f"{experiment_name}__{model_name}_predictions.csv", index=False)


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test,
                             test_nctids, experiment_name, output_dir):
    print(f"\n==============================")
    print(f"Experiment: {experiment_name}")
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
        experiment_name=experiment_name,
        model_name=model_name,
        output_dir=output_dir
    )

    return metrics


def get_models():
    return {
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


# =========================================================
# 4. Main program
# =========================================================

def main():
    if not TRAIN_XML_DIR.exists():
        raise FileNotFoundError(f"TRAIN_XML_DIR not found: {TRAIN_XML_DIR}")
    if not TEST_XML_DIR.exists():
        raise FileNotFoundError(f"TEST_XML_DIR not found: {TEST_XML_DIR}")
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"LABEL_FILE not found: {LABEL_FILE}")

    print("Building / loading feature cache...")
    train_cache_df = build_feature_cache(TRAIN_XML_DIR, "train")
    test_cache_df = build_feature_cache(TEST_XML_DIR, "test")

    print("Loading labels...")
    label_df = load_labels(LABEL_FILE, id_col=ID_COL, target_col=TARGET_COL)

    all_rows = []

    for experiment_name, selected_features in EXPERIMENTS.items():
        print(f"\n\n########## {experiment_name} ##########")
        print(f"Selected features: {selected_features}")

        train_results = aggregate_trial_embeddings(train_cache_df, selected_features)
        test_results = aggregate_trial_embeddings(test_cache_df, selected_features)

        train_emb_df = results_to_df(train_results)
        test_emb_df = results_to_df(test_results)

        train_df, X_train, y_train = merge_embeddings_with_labels(
            train_emb_df, label_df, "train", TARGET_COL
        )
        test_df, X_test, y_test = merge_embeddings_with_labels(
            test_emb_df, label_df, "test", TARGET_COL
        )

        test_nctids = test_df["nctid"].tolist()

        print("Shapes:")
        print("X_train:", X_train.shape)
        print("y_train:", y_train.shape)
        print("X_test :", X_test.shape)
        print("y_test :", y_test.shape)

        models = get_models()

        for model_name, model in models.items():
            metrics = train_and_evaluate_model(
                model=model,
                model_name=model_name,
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                test_nctids=test_nctids,
                experiment_name=experiment_name,
                output_dir=OUTPUT_DIR
            )

            row = {
                "experiment": experiment_name,
                "excluded_features": ",".join(sorted(set(TEXT_FEATURES) - set(selected_features))),
                "used_features": ",".join(selected_features),
                "model": model_name,
            }
            row.update(metrics)
            all_rows.append(row)

    results_df = pd.DataFrame(all_rows)
    results_df.to_csv(OUTPUT_DIR / "feature_ablation_summary.csv", index=False)

    with open(OUTPUT_DIR / "feature_ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2)

    print("\nSaved:")
    print(OUTPUT_DIR / "feature_ablation_summary.csv")
    print(OUTPUT_DIR / "feature_ablation_summary.json")


if __name__ == "__main__":
    main()