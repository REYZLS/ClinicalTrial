# This script applies a Transformer to the MLP model and evaluates its performance without the phase and enrollment variables.
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.stats import pearsonr

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


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

OUTPUT_DIR = DATA_DIR / "transformer_mlp_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CACHE_DIR = DATA_DIR / "transformer_feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
]

SEED = 42
BATCH_SIZE = 8
EPOCHS = 100
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


# =========================================================
# 3. Fix random seed
# =========================================================

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)


# =========================================================
# 4. Load ClinicalBERT
# =========================================================

print(f"Using device: {DEVICE}")
print(f"Loading model: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
encoder_model = AutoModel.from_pretrained(MODEL_NAME)
encoder_model.to(DEVICE)
encoder_model.eval()


# =========================================================
# 5. XML utility functions
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

    return {
        "nctid": nctid,
        "title": title,
        "summary": summary,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "disease": disease_text,
        "drug": drug_text,
    }


# =========================================================
# 6. Embedding
# Use mean pooling to obtain a vector for each feature
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
        last_hidden = outputs.last_hidden_state  # [1, seq_len, hidden]

        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        masked_hidden = last_hidden * mask
        sum_hidden = masked_hidden.sum(dim=1)
        valid_tokens = mask.sum(dim=1).clamp(min=1e-9)

        mean_vec = (sum_hidden / valid_tokens).squeeze(0).cpu().numpy().astype(np.float32)

    return mean_vec


def build_feature_cache(xml_dir, split_name):
    xml_files = sorted(Path(xml_dir).rglob("*.xml"))
    records = []

    print(f"\nBuilding feature cache for {split_name}: {len(xml_files)} XML files")

    for xml_file in tqdm(xml_files, desc=f"Caching {split_name}"):
        try:
            feat_dict = extract_selected_features_from_xml(xml_file)
        except Exception as e:
            print(f"[ERROR] Failed to parse XML: {xml_file}\n{e}")
            continue

        nctid = feat_dict.get("nctid") or xml_file.stem.upper()

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
# 7. Load labels
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
# 8. Build (N, 6, 768)
# =========================================================

def build_sequence_dataset(cache_df, label_df):
    feature_map = {}

    for _, row in cache_df.iterrows():
        nctid = normalize_nctid(row["nctid"])
        feat = row["feature_name"]
        vec = np.load(row["cache_file"])

        if nctid not in feature_map:
            feature_map[nctid] = {}
        feature_map[nctid][feat] = vec

    merged = label_df[label_df["nctid"].isin(feature_map.keys())].copy()
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
                feat_mask.append(False)  # False = valid
            else:
                feat_vecs.append(np.zeros((768,), dtype=np.float32))
                feat_mask.append(True)   # True = mask out

        X.append(np.stack(feat_vecs, axis=0))      # [6, 768]
        masks.append(np.array(feat_mask))
        y.append(target)
        nctids.append(nctid)

    X = np.stack(X, axis=0).astype(np.float32)     # [N, 6, 768]
    masks = np.stack(masks, axis=0)                # [N, 6]
    y = np.array(y, dtype=np.float32)

    return nctids, X, masks, y


# =========================================================
# 9. Dataset definition
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
# 10. Transformer + MLP model
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
        # x: [B, 6, 768]
        x = self.input_proj(x)  # [B, 6, d_model]
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, 6, d_model]

        if mask is None:
            pooled = x.mean(dim=1)
        else:
            valid = (~mask).unsqueeze(-1).float()  # [B, 6, 1]
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1e-9)

        out = self.mlp(pooled).squeeze(-1)  # [B]
        return out


# =========================================================
# 11. Evaluation metrics
# =========================================================

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


# =========================================================
# 12. Training and evaluation
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
    metrics = regression_metrics(np.array(y_true_all), np.array(y_pred_all))

    return avg_loss, metrics, np.array(y_true_all), np.array(y_pred_all)


# =========================================================
# 13. Main program
# =========================================================

def main():
    if not TRAIN_XML_DIR.exists():
        raise FileNotFoundError(f"TRAIN_XML_DIR not found: {TRAIN_XML_DIR}")
    if not TEST_XML_DIR.exists():
        raise FileNotFoundError(f"TEST_XML_DIR not found: {TEST_XML_DIR}")
    if not LABEL_FILE.exists():
        raise FileNotFoundError(f"LABEL_FILE not found: {LABEL_FILE}")

    print("Step 1: building/loading feature cache...")
    train_cache_df = build_feature_cache(TRAIN_XML_DIR, "train")
    test_cache_df = build_feature_cache(TEST_XML_DIR, "test")

    print("Step 2: loading labels...")
    label_df = load_labels(LABEL_FILE, id_col=ID_COL, target_col=TARGET_COL)

    print("Step 3: building sequence dataset...")
    train_nctids, X_train, mask_train, y_train = build_sequence_dataset(train_cache_df, label_df)
    test_nctids, X_test, mask_test, y_test = build_sequence_dataset(test_cache_df, label_df)

    print("Train shape:", X_train.shape)   # [N, 6, 768]
    print("Test shape :", X_test.shape)

    train_dataset = TrialDataset(X_train, mask_train, y_train)
    test_dataset = TrialDataset(X_test, mask_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model = TransformerMLPRegressor(
        input_dim=768,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.1,
        mlp_hidden=128
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    criterion = nn.MSELoss()

    best_rmse = float("inf")
    best_state = None
    history = []

    print("Step 4: training...")
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, DEVICE)
        test_loss, test_metrics, y_true_epoch, y_pred_epoch = evaluate(model, test_loader, criterion, DEVICE)

        history_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
            **test_metrics
        }
        history.append(history_row)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Loss : {test_loss:.4f}")
        for k, v in test_metrics.items():
            print(f"{k}: {v}")

        if test_metrics["RMSE"] < best_rmse:
            best_rmse = test_metrics["RMSE"]
            best_state = {
                "model_state_dict": model.state_dict(),
                "metrics": test_metrics,
                "epoch": epoch
            }

    # Save the best model
    best_model_path = OUTPUT_DIR / "best_transformer_mlp.pt"
    torch.save(best_state, best_model_path)
    print(f"\nSaved best model to: {best_model_path}")

    # Load the best model for final evaluation
    model.load_state_dict(best_state["model_state_dict"])
    final_test_loss, final_metrics, y_true_final, y_pred_final = evaluate(
        model, test_loader, criterion, DEVICE
    )

    # Save prediction results
    pred_df = pd.DataFrame({
        "nctid": test_nctids,
        "y_true": y_true_final,
        "y_pred": y_pred_final,
        "abs_error": np.abs(y_pred_final - y_true_final)
    })
    pred_df.to_csv(OUTPUT_DIR / "transformer_mlp_predictions.csv", index=False)

    # Save training history
    history_df = pd.DataFrame(history)
    history_df.to_csv(OUTPUT_DIR / "training_history.csv", index=False)

    # Save final metrics
    metrics_df = pd.DataFrame([final_metrics])
    metrics_df.to_csv(OUTPUT_DIR / "transformer_mlp_metrics.csv", index=False)

    with open(OUTPUT_DIR / "transformer_mlp_metrics.json", "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)

    print("\nFinal metrics:")
    for k, v in final_metrics.items():
        print(f"{k}: {v}")

    print("\nSaved files:")
    print(OUTPUT_DIR / "transformer_mlp_predictions.csv")
    print(OUTPUT_DIR / "training_history.csv")
    print(OUTPUT_DIR / "transformer_mlp_metrics.csv")
    print(OUTPUT_DIR / "transformer_mlp_metrics.json")


if __name__ == "__main__":
    main()