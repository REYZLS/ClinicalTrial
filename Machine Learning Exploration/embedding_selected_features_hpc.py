import shutil
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# =========================================================
# 1. Path configuration (HPC version)
# Assume you run this script inside the data directory:
# cd /home/zs3563a-hpc/data
# python embedding_train_test_only.py
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR

ALL_XML_DIR = DATA_DIR / "trials"
TRAIN_ID_FILE = DATA_DIR / "id_train.csv"
TEST_ID_FILE = DATA_DIR / "id_test.csv"

TRAIN_XML_DIR = DATA_DIR / "train_xml"
TEST_XML_DIR = DATA_DIR / "test_xml"

OUTPUT_DIR = DATA_DIR / "embeddings_train_test_only"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# 2. Model configuration
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

print(f"Script directory: {BASE_DIR}")
print(f"Using device: {DEVICE}")
print("Loading BioBERT...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


# =========================================================
# 3. Read ID files
# =========================================================

def normalize_nctid(x):
    if pd.isna(x):
        return None
    x = str(x).strip().upper()
    if not x:
        return None
    return x


def find_id_column(df):
    candidate_cols = ["nctid", "nct_id", "id", "ID", "NCTID", "NCT_ID"]
    lower_map = {c.lower(): c for c in df.columns}

    for col in candidate_cols:
        if col.lower() in lower_map:
            return lower_map[col.lower()]

    return df.columns[0]


def read_id_file(file_path):
    try:
        df = pd.read_csv(file_path)
        if df.shape[1] == 1:
            id_col = df.columns[0]
        else:
            id_col = find_id_column(df)

        ids = df[id_col].map(normalize_nctid).dropna().unique().tolist()
        return set(ids), id_col

    except Exception:
        df = pd.read_csv(file_path, header=None, names=["id"])
        ids = df["id"].map(normalize_nctid).dropna().unique().tolist()
        return set(ids), "id"


# =========================================================
# 4. Build XML file index
# =========================================================

def build_xml_index(all_xml_dir):
    xml_files = sorted(Path(all_xml_dir).rglob("*.xml"))
    xml_index = {}

    for xml_file in xml_files:
        nctid = xml_file.stem.upper()
        if nctid not in xml_index:
            xml_index[nctid] = xml_file

    return xml_index


def copy_xmls_by_id(id_set, xml_index, output_dir, group_name):
    output_dir.mkdir(parents=True, exist_ok=True)

    matched = []
    missing = []

    for nctid in tqdm(sorted(id_set), desc=f"Copying {group_name} XMLs"):
        if nctid in xml_index:
            src = xml_index[nctid]
            dst = output_dir / src.name
            shutil.copy2(src, dst)
            matched.append(nctid)
        else:
            missing.append(nctid)

    pd.DataFrame({"nctid": matched}).to_csv(DATA_DIR / f"{group_name}_matched_ids.csv", index=False)
    pd.DataFrame({"nctid": missing}).to_csv(DATA_DIR / f"{group_name}_missing_ids.csv", index=False)

    print(f"\n[{group_name}] matched: {len(matched)}")
    print(f"[{group_name}] missing: {len(missing)}")

    return matched, missing


# =========================================================
# 5. XML utility functions
# =========================================================

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


# =========================================================
# 6. Extract selected fields
# =========================================================

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

    feature_dict = {
        "nctid": nctid,
        "disease": disease_text,
        "title": title,
        "summary": summary,
        "inclusion_criteria": inclusion_criteria,
        "exclusion_criteria": exclusion_criteria,
        "drug": drug_text,
    }

    return feature_dict


# =========================================================
# 7. BioBERT embedding
# Keep only trial-level embeddings
# Limit token length to 512
# =========================================================

def embed_text_with_cls(text, tokenizer, model, max_len=512, device="cpu"):
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        truncation=True,          # Truncate if text exceeds max length
        max_length=max_len,       # Limit sequence length to 512
        padding="max_length",
        return_attention_mask=True,
        return_tensors="pt"
    )

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :].squeeze(0).cpu().numpy()

    return cls_vec


def process_single_xml(xml_path, tokenizer, model, max_len=512, device="cpu"):
    xml_path = Path(xml_path)

    try:
        feature_dict = extract_selected_features_from_xml(xml_path)
    except Exception as e:
        print(f"[ERROR] Failed to parse XML: {xml_path}\n{e}")
        return None

    nctid = feature_dict.get("nctid")
    if nctid is None:
        nctid = xml_path.stem.upper()

    feature_vectors = []

    for feature_name in TEXT_FEATURES:
        content = feature_dict.get(feature_name)
        if content is None:
            continue

        try:
            cls_vec = embed_text_with_cls(
                content,
                tokenizer=tokenizer,
                model=model,
                max_len=max_len,
                device=device
            )
            feature_vectors.append(cls_vec)
        except Exception as e:
            print(f"[WARNING] Failed embedding feature in {nctid}, feature={feature_name}\n{e}")

    if len(feature_vectors) == 0:
        return None

    trial_embedding = np.mean(np.stack(feature_vectors, axis=0), axis=0).astype(np.float32)

    return {
        "nctid": nctid,
        "embedding": trial_embedding
    }


def process_xml_folder(xml_dir, group_name, tokenizer, model, max_len=512, device="cpu"):
    xml_dir = Path(xml_dir)
    xml_files = sorted(xml_dir.rglob("*.xml"))
    print(f"\nProcessing {group_name} XMLs: {len(xml_files)} files")

    all_results = []

    for xml_file in tqdm(xml_files, desc=f"Processing {group_name}"):
        result = process_single_xml(
            xml_path=xml_file,
            tokenizer=tokenizer,
            model=model,
            max_len=max_len,
            device=device
        )

        if result is not None:
            all_results.append(result)

    return all_results


# =========================================================
# 8. Save results
# Save only train/test embeddings
# =========================================================

def save_embeddings(results, group_name, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if len(results) > 0:
        nctids = [x["nctid"] for x in results]
        embeddings = np.stack([x["embedding"] for x in results], axis=0)
    else:
        nctids = []
        embeddings = np.empty((0, 768), dtype=np.float32)

    torch.save(
        {
            "nctids": nctids,
            "embeddings": torch.tensor(embeddings, dtype=torch.float32)
        },
        output_dir / f"{group_name}_embeddings.pt"
    )

    meta_df = pd.DataFrame({
        "nctid": nctids
    })
    meta_df.to_csv(output_dir / f"{group_name}_nctids.csv", index=False)

    print(f"[Saved] {group_name}: {len(nctids)} trials")
    print(f"[Saved file] {output_dir / f'{group_name}_embeddings.pt'}")


# =========================================================
# 9. Main program
# =========================================================

def main():
    if not ALL_XML_DIR.exists():
        raise FileNotFoundError(f"ALL_XML_DIR not found: {ALL_XML_DIR}")
    if not TRAIN_ID_FILE.exists():
        raise FileNotFoundError(f"TRAIN_ID_FILE not found: {TRAIN_ID_FILE}")
    if not TEST_ID_FILE.exists():
        raise FileNotFoundError(f"TEST_ID_FILE not found: {TEST_ID_FILE}")

    print("Checking paths...")
    print(f"ALL_XML_DIR: {ALL_XML_DIR}")
    print(f"TRAIN_ID_FILE: {TRAIN_ID_FILE}")
    print(f"TEST_ID_FILE: {TEST_ID_FILE}")
    print(f"MAX_LEN: {MAX_LEN}")

    train_ids, train_id_col = read_id_file(TRAIN_ID_FILE)
    test_ids, test_id_col = read_id_file(TEST_ID_FILE)

    print(f"Train ID column: {train_id_col}, count: {len(train_ids)}")
    print(f"Test ID column: {test_id_col}, count: {len(test_ids)}")

    overlap = train_ids.intersection(test_ids)
    if len(overlap) > 0:
        print(f"[WARNING] train/test IDs overlap: {len(overlap)}")
        pd.DataFrame({"nctid": sorted(overlap)}).to_csv(DATA_DIR / "overlap_train_test_ids.csv", index=False)

    xml_index = build_xml_index(ALL_XML_DIR)
    print(f"Total XML files indexed: {len(xml_index)}")

    TRAIN_XML_DIR.mkdir(parents=True, exist_ok=True)
    TEST_XML_DIR.mkdir(parents=True, exist_ok=True)

    copy_xmls_by_id(train_ids, xml_index, TRAIN_XML_DIR, "train")
    copy_xmls_by_id(test_ids, xml_index, TEST_XML_DIR, "test")

    train_results = process_xml_folder(
        xml_dir=TRAIN_XML_DIR,
        group_name="train",
        tokenizer=tokenizer,
        model=model,
        max_len=MAX_LEN,
        device=DEVICE
    )
    save_embeddings(train_results, "train", OUTPUT_DIR)

    test_results = process_xml_folder(
        xml_dir=TEST_XML_DIR,
        group_name="test",
        tokenizer=tokenizer,
        model=model,
        max_len=MAX_LEN,
        device=DEVICE
    )
    save_embeddings(test_results, "test", OUTPUT_DIR)

    print("\nDone.")


if __name__ == "__main__":
    main()