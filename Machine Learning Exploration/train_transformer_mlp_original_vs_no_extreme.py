import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib.pyplot as plt


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

OUTPUT_DIR = DATA_DIR / "transformer_mlp_original_vs_no_extreme"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = DATA_DIR / "transformer_feature_cache_with_phase_enrollment"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_TRAIN_ID_FILE = DATA_DIR / "filtered_train_ids.csv"
FILTERED_TEST_ID_FILE = DATA_DIR / "filtered_test_ids.csv"


# =========================================================
# 2. Model and feature configuration
# =========================================================

MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"
MAX_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
BATCH_SIZE = 8
EPOCHS = 100


# =========================================================
# 3. Unified hyperparameter configuration
# =========================================================

CONFIG = {
    "name": "fixed_hparam",
    "d_model": 256,
    "nhead": 8,
    "num_layers": 2,
    "dim_feedforward": 512,
    "dropout": 0.1,
    "mlp_hidden": 128,
    "lr": 1e-4,
    "weight_decay": 1e-4,
}


# =========================================================
# 4. Fix random seed
# =========================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# 5. Load ClinicalBERT
# =========================================================

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder_model = AutoModel.from_pretrained(MODEL_NAME)
encoder_model.to(DEVICE)
encoder_model.eval()


# =========================================================
# 6. XML utility functions
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

    title = get_text(root, "brief_title")
    summary = get_text(root, "brief_summary/textblock")

    criteria_text = get_text(root, "eligibility/criteria/textblock")
    inclusion_criteria, exclusion_criteria = split_inclusion_exclusion(criteria_text)

    diseases = get_all_texts(root, "condition")
    disease_text = "; ".join(diseases) if diseases else None

    drug_names = []
    for intervention in root.findall("intervention"):
        intervention_type = get_text(intervention, "intervention_type")
        intervention_name = get_text(intervention, "intervention_name")
        if intervention_type and intervention_name and intervention_type.lower() == "drug":
            drug_names.append(intervention_name)

    drug_text = "; ".join(drug_names) if drug_names else None

    phase = get_text(root, "phase")
    enrollment = get_text(root, "enrollment")

    return {
        "nctid": nctid,
        "title": title,
        "summary": summary,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "disease": disease_text,
        "drug": drug_text,
        "phase": phase,
        "enrollment": enrollment,
    }


# =========================================================
# 7. Load filtered IDs
# =========================================================

def load_filtered_ids(id_file):
    df = pd.read_csv(id_file)

    if "nctid" not in df.columns:
        raise ValueError(f"'nctid' column not found in {id_file}")

    ids = df["nctid"].map(normalize_nctid).dropna().unique().tolist()
    return set(ids)


# =========================================================
# 8. Embedding generation
# =========================================================

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


def build_feature_cache(xml_dir, split_name, allowed_ids=None):
    xml_files = sorted(Path(xml_dir).rglob("*.xml"))
    records = []

    print(f"\nBuilding feature cache for {split_name}: {len(xml_files)} XML files")

    for xml_file in tqdm(xml_files, desc=f"Caching {split_name}"):
        try:
            feat_dict = extract_selected_features_from_xml(xml_file)
        except Exception as e:
            print(f"[ERROR] Failed to parse XML: {xml_file}\n{e}")
            continue

        nctid = normalize_nctid(feat_dict.get("nctid") or xml_file.stem.upper())

        if allowed_ids is not None and nctid not in allowed_ids:
            continue

        for feat_name in FEATURE_ORDER:
            text = feat_dict.get(feat_name)
            if text is None:
                continue

            cache_file = CACHE_DIR / f"{split_name}__{nctid}__{feat_name}.npy"

            if not cache_file.exists():
                try:
                    vec = embed_text_mean_pooling(
                        text=text,
                        tokenizer=tokenizer,
                        model=encoder_model,
                        max_len=MAX_LEN,
                        device=DEVICE
                    )
                    np.save(cache_file, vec)
                except Exception as e:
                    print(f"[WARNING] Embedding failed: {nctid}, {feat_name}\n{e}")
                    continue

            records.append({
                "nctid": nctid,
                "feature_name": feat_name,
                "cache_file": str(cache_file)
            })

    df = pd.DataFrame(records)
    df.to_csv(OUTPUT_DIR / f"{split_name}_feature_cache_index.csv", index=False)
    return df


# =========================================================
# 9. Load labels
# =========================================================

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


# =========================================================
# 10. Build (N, 8, 768) sequence dataset
# =========================================================

def build_sequence_dataset(cache_df, label_df, allowed_ids=None):
    feature_map = {}

    for _, row in cache_df.iterrows():
        nctid = normalize_nctid(row["nctid"])
        feat = row["feature_name"]
        vec = np.load(row["cache_file"])

        if allowed_ids is not None and nctid not in allowed_ids:
            continue

        if nctid not in feature_map:
            feature_map[nctid] = {}

        feature_map[nctid][feat] = vec

    merged = label_df[label_df["nctid"].isin(feature_map.keys())].copy()

    if allowed_ids is not None:
        merged = merged[merged["nctid"].isin(allowed_ids)]

    merged = merged.reset_index(drop=True)

    X = []
    masks = []
    y = []
    nctids = []

    for _, row in merged.iterrows():
        nctid = row["nctid"]
        target = float(row[TARGET_COL])

        feat_dict = feature_map.get(nctid, {})

        feat_vecs = []
        feat_mask = []

        for feat_name in FEATURE_ORDER:
            if feat_name in feat_dict:
                feat_vecs.append(feat_dict[feat_name])
                feat_mask.append(False)
            else:
                feat_vecs.append(np.zeros((768,), dtype=np.float32))
                feat_mask.append(True)

        X.append(np.stack(feat_vecs, axis=0))
        masks.append(np.array(feat_mask))
        y.append(target)
        nctids.append(nctid)

    X = np.stack(X, axis=0).astype(np.float32)
    masks = np.stack(masks, axis=0)
    y = np.array(y, dtype=np.float32)

    return nctids, X, masks, y


# =========================================================
# 11. Dataset definition
# =========================================================

class TrialDataset(Dataset):
    def __init__(self, X, masks, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.masks = torch.tensor(masks, dtype=torch.bool)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.masks[idx], self.y[idx]


# =========================================================
# 12. Transformer + MLP model
# =========================================================

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

    def forward(self, x, mask=None):
        x = self.input_proj(x)
        x = self.transformer(x, src_key_padding_mask=mask)

        if mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = (~mask).unsqueeze(-1).float()
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)

        out = self.mlp(pooled).squeeze(-1)
        return out


# =========================================================
# 13. Evaluation metrics
# =========================================================

def regression_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r, pearson_p = pearsonr(y_true, y_pred)
        spearman_r, spearman_p = spearmanr(y_true, y_pred)
    else:
        pearson_r, pearson_p = np.nan, np.nan
        spearman_r, spearman_p = np.nan, np.nan

    return {
        "MAE": float(mae),
        "MSE": float(mse),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_r": float(pearson_r) if not np.isnan(pearson_r) else None,
        "Pearson_p": float(pearson_p) if not np.isnan(pearson_p) else None,
        "Spearman_r": float(spearman_r) if not np.isnan(spearman_r) else None,
        "Spearman_p": float(spearman_p) if not np.isnan(spearman_p) else None,
    }


# =========================================================
# 14. Training and evaluation
# =========================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for X, mask, y in loader:
        X = X.to(device)
        mask = mask.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        pred = model(X, mask)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    y_true_all = []
    y_pred_all = []

    with torch.no_grad():
        for X, mask, y in loader:
            X = X.to(device)
            mask = mask.to(device)
            y = y.to(device)

            pred = model(X, mask)
            loss = criterion(pred, y)

            total_loss += loss.item() * X.size(0)
            y_true_all.extend(y.cpu().numpy().tolist())
            y_pred_all.extend(pred.cpu().numpy().tolist())

    avg_loss = total_loss / len(loader.dataset)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    metrics = regression_metrics(y_true_all, y_pred_all)

    return avg_loss, metrics, y_true_all, y_pred_all


# =========================================================
# 15. Plot training curves
# =========================================================

def plot_training_curves(history_df, exp_dir):
    metrics_to_plot = [
        ("train_loss", "test_loss", "Loss"),
        ("train_MAE", "test_MAE", "MAE"),
        ("train_RMSE", "test_RMSE", "RMSE"),
        ("train_R2", "test_R2", "R2"),
        ("train_Pearson_r", "test_Pearson_r", "Pearson_r"),
        ("train_Spearman_r", "test_Spearman_r", "Spearman_r"),
    ]

    for train_col, test_col, title in metrics_to_plot:
        plt.figure(figsize=(8, 5))
        plt.plot(history_df["epoch"], history_df[train_col], label=f"Train {title}")
        plt.plot(history_df["epoch"], history_df[test_col], label=f"Test {title}")
        plt.xlabel("Epoch")
        plt.ylabel(title)
        plt.title(f"{title} Curve")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(exp_dir / f"{title.lower()}_curve.png", dpi=300)
        plt.close()


# =========================================================
# 16. Single experiment
# =========================================================

def run_experiment(
    config,
    train_loader,
    test_loader,
    train_nctids,
    test_nctids,
    output_dir,
    device
):
    exp_name = config["name"]

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {exp_name}")
    print(json.dumps(config, indent=2))
    print(f"{'=' * 60}")

    set_seed(SEED)

    model = TransformerMLPRegressor(
        input_dim=768,
        d_model=config["d_model"],
        nhead=config["nhead"],
        num_layers=config["num_layers"],
        dim_feedforward=config["dim_feedforward"],
        dropout=config["dropout"],
        mlp_hidden=config["mlp_hidden"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"]
    )

    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    history = []

    for epoch in range(1, EPOCHS + 1):
        _ = train_one_epoch(model, train_loader, optimizer, criterion, device)

        train_eval_loss, train_metrics, _, _ = evaluate(model, train_loader, criterion, device)
        test_eval_loss, test_metrics, _, _ = evaluate(model, test_loader, criterion, device)

        history_row = {
            "experiment": exp_name,
            "epoch": epoch,

            "train_loss": train_eval_loss,
            "test_loss": test_eval_loss,

            "train_MAE": train_metrics["MAE"],
            "train_MSE": train_metrics["MSE"],
            "train_RMSE": train_metrics["RMSE"],
            "train_R2": train_metrics["R2"],
            "train_Pearson_r": train_metrics["Pearson_r"],
            "train_Pearson_p": train_metrics["Pearson_p"],
            "train_Spearman_r": train_metrics["Spearman_r"],
            "train_Spearman_p": train_metrics["Spearman_p"],

            "test_MAE": test_metrics["MAE"],
            "test_MSE": test_metrics["MSE"],
            "test_RMSE": test_metrics["RMSE"],
            "test_R2": test_metrics["R2"],
            "test_Pearson_r": test_metrics["Pearson_r"],
            "test_Pearson_p": test_metrics["Pearson_p"],
            "test_Spearman_r": test_metrics["Spearman_r"],
            "test_Spearman_p": test_metrics["Spearman_p"],
        }

        history.append(history_row)

        print(f"\n[{exp_name}] Epoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_eval_loss:.4f} | Test Loss: {test_eval_loss:.4f}")
        print(f"Train MAE : {train_metrics['MAE']:.4f} | Test MAE : {test_metrics['MAE']:.4f}")
        print(f"Train RMSE: {train_metrics['RMSE']:.4f} | Test RMSE: {test_metrics['RMSE']:.4f}")
        print(f"Train R2  : {train_metrics['R2']:.4f} | Test R2  : {test_metrics['R2']:.4f}")

        if test_metrics["RMSE"] < best_rmse:
            best_rmse = test_metrics["RMSE"]
            best_state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
            }

    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    torch.save(best_state, exp_dir / "best_model.pt")

    model.load_state_dict(best_state["model_state_dict"])

    _, final_train_metrics, y_train_true, y_train_pred = evaluate(
        model, train_loader, criterion, device
    )

    _, final_test_metrics, y_test_true, y_test_pred = evaluate(
        model, test_loader, criterion, device
    )

    train_pred_df = pd.DataFrame({
        "nctid": train_nctids,
        "y_true": y_train_true,
        "y_pred": y_train_pred,
        "abs_error": np.abs(y_train_pred - y_train_true),
        "split": "train"
    })

    test_pred_df = pd.DataFrame({
        "nctid": test_nctids,
        "y_true": y_test_true,
        "y_pred": y_test_pred,
        "abs_error": np.abs(y_test_pred - y_test_true),
        "split": "test"
    })

    train_pred_df.to_csv(exp_dir / "train_predictions.csv", index=False)
    test_pred_df.to_csv(exp_dir / "test_predictions.csv", index=False)

    all_pred_df = pd.concat([train_pred_df, test_pred_df], axis=0)
    all_pred_df.to_csv(exp_dir / "train_test_predictions.csv", index=False)

    final_metrics_df = pd.DataFrame([
        {"split": "train", **final_train_metrics},
        {"split": "test", **final_test_metrics},
    ])

    final_metrics_df.to_csv(exp_dir / "train_test_metrics.csv", index=False)

    history_df = pd.DataFrame(history)
    history_df.to_csv(exp_dir / "training_history.csv", index=False)

    plot_training_curves(history_df, exp_dir)

    with open(exp_dir / "train_test_metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "best_epoch": best_state["epoch"],
            "train_metrics": final_train_metrics,
            "test_metrics": final_test_metrics,
            "config": config
        }, f, indent=2)

    summary_rows = []

    train_row = config.copy()
    train_row.update(final_train_metrics)
    train_row["split"] = "train"
    train_row["best_epoch"] = best_state["epoch"]
    train_row["experiment"] = exp_name
    summary_rows.append(train_row)

    test_row = config.copy()
    test_row.update(final_test_metrics)
    test_row["split"] = "test"
    test_row["best_epoch"] = best_state["epoch"]
    test_row["experiment"] = exp_name
    summary_rows.append(test_row)

    return summary_rows


# =========================================================
# 17. Remove extreme values
# =========================================================

def filter_extreme_trials(nctids, X, masks, y, threshold=9000):
    keep_idx = y <= threshold

    filtered_nctids = [nctids[i] for i in range(len(nctids)) if keep_idx[i]]
    filtered_X = X[keep_idx]
    filtered_masks = masks[keep_idx]
    filtered_y = y[keep_idx]

    print(f"Original size: {len(y)}")
    print(f"After removing duration > {threshold}: {len(filtered_y)}")
    print(f"Removed: {len(y) - len(filtered_y)}")

    return filtered_nctids, filtered_X, filtered_masks, filtered_y


# =========================================================
# 18. Main program
# =========================================================

def main():
    if not TRAIN_XML_DIR.exists():
        raise FileNotFoundError(f"TRAIN_XML_DIR not found: {TRAIN_XML_DIR}")
    if not TEST_XML_DIR.exists():
        raise FileNotFoundError(f"TEST_XML_DIR not found: {TEST_XML_DIR}")
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"LABEL_FILE not found: {LABEL_FILE}")

    if not FILTERED_TRAIN_ID_FILE.exists():
        raise FileNotFoundError(f"FILTERED_TRAIN_ID_FILE not found: {FILTERED_TRAIN_ID_FILE}")
    if not FILTERED_TEST_ID_FILE.exists():
        raise FileNotFoundError(f"FILTERED_TEST_ID_FILE not found: {FILTERED_TEST_ID_FILE}")

    print("Step 0.5: loading filtered IDs...")
    train_allowed_ids = load_filtered_ids(FILTERED_TRAIN_ID_FILE)
    test_allowed_ids = load_filtered_ids(FILTERED_TEST_ID_FILE)

    print(f"Filtered train IDs: {len(train_allowed_ids)}")
    print(f"Filtered test IDs : {len(test_allowed_ids)}")

    print("Step 1: building/loading feature cache...")
    train_cache_df = build_feature_cache(
        TRAIN_XML_DIR,
        "train",
        allowed_ids=train_allowed_ids
    )

    test_cache_df = build_feature_cache(
        TEST_XML_DIR,
        "test",
        allowed_ids=test_allowed_ids
    )

    print("Step 2: loading labels...")
    label_df = load_labels(
        LABEL_FILE,
        id_col=ID_COL,
        target_col=TARGET_COL
    )

    print("Step 3: building sequence dataset...")
    train_nctids, X_train, mask_train, y_train = build_sequence_dataset(
        train_cache_df,
        label_df,
        allowed_ids=train_allowed_ids
    )

    test_nctids, X_test, mask_test, y_test = build_sequence_dataset(
        test_cache_df,
        label_df,
        allowed_ids=test_allowed_ids
    )

    print("Original Train shape:", X_train.shape)
    print("Original Test shape :", X_test.shape)

    all_summary_rows = []

    # =====================================================
    # Case 1: Original data
    # =====================================================

    print("\n==============================")
    print("CASE 1: Original data")
    print("==============================")

    original_output_dir = OUTPUT_DIR / "original"
    original_output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = TrialDataset(X_train, mask_train, y_train)
    test_dataset = TrialDataset(X_test, mask_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    original_config = CONFIG.copy()
    original_config["name"] = "original_fixed_hparam"

    rows = run_experiment(
        config=original_config,
        train_loader=train_loader,
        test_loader=test_loader,
        train_nctids=train_nctids,
        test_nctids=test_nctids,
        output_dir=original_output_dir,
        device=DEVICE
    )

    for row in rows:
        row["case"] = "original"

    all_summary_rows.extend(rows)

    # =====================================================
    # Case 2: Remove trials with duration > 9000
    # =====================================================

    print("\n==============================")
    print("CASE 2: Without extreme trials > 9000")
    print("==============================")

    (
        train_nctids_no_extreme,
        X_train_no_extreme,
        mask_train_no_extreme,
        y_train_no_extreme
    ) = filter_extreme_trials(
        train_nctids,
        X_train,
        mask_train,
        y_train,
        threshold=9000
    )

    (
        test_nctids_no_extreme,
        X_test_no_extreme,
        mask_test_no_extreme,
        y_test_no_extreme
    ) = filter_extreme_trials(
        test_nctids,
        X_test,
        mask_test,
        y_test,
        threshold=9000
    )

    print("No-extreme Train shape:", X_train_no_extreme.shape)
    print("No-extreme Test shape :", X_test_no_extreme.shape)

    no_extreme_output_dir = OUTPUT_DIR / "without_extreme_gt_9000"
    no_extreme_output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset_no_extreme = TrialDataset(
        X_train_no_extreme,
        mask_train_no_extreme,
        y_train_no_extreme
    )

    test_dataset_no_extreme = TrialDataset(
        X_test_no_extreme,
        mask_test_no_extreme,
        y_test_no_extreme
    )

    train_loader_no_extreme = DataLoader(
        train_dataset_no_extreme,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader_no_extreme = DataLoader(
        test_dataset_no_extreme,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    no_extreme_config = CONFIG.copy()
    no_extreme_config["name"] = "without_extreme_fixed_hparam"

    rows = run_experiment(
        config=no_extreme_config,
        train_loader=train_loader_no_extreme,
        test_loader=test_loader_no_extreme,
        train_nctids=train_nctids_no_extreme,
        test_nctids=test_nctids_no_extreme,
        output_dir=no_extreme_output_dir,
        device=DEVICE
    )

    for row in rows:
        row["case"] = "without_extreme_gt_9000"

    all_summary_rows.extend(rows)

    # =====================================================
    # Summary table
    # =====================================================

    summary_df = pd.DataFrame(all_summary_rows)

    wanted_cols = [
        "case",
        "experiment",
        "split",
        "best_epoch",
        "MAE",
        "MSE",
        "RMSE",
        "R2",
        "Pearson_r",
        "Pearson_p",
        "Spearman_r",
        "Spearman_p",
        "d_model",
        "nhead",
        "num_layers",
        "dim_feedforward",
        "dropout",
        "mlp_hidden",
        "lr",
        "weight_decay",
    ]

    summary_df = summary_df[wanted_cols]
    summary_df.to_csv(
        OUTPUT_DIR / "original_vs_no_extreme_summary.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "original_vs_no_extreme_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(all_summary_rows, f, indent=2)

    print("\nFinal Summary:")
    print(summary_df)

    print("\nSaved files:")
    print(OUTPUT_DIR / "original_vs_no_extreme_summary.csv")
    print(OUTPUT_DIR / "original_vs_no_extreme_summary.json")


if __name__ == "__main__":
    main()