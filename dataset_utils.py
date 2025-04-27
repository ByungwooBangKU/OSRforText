"""
데이터셋 다운로드 및 전처리 유틸리티 모듈
(Dataset download and preprocessing utility module)
"""
import os
import re
import zipfile
import urllib.request
import numpy as np
import pandas as pd
import requests # Added for download_file
import shutil
from tqdm import tqdm # Added for download_file
from datasets import load_dataset, ClassLabel # Added for HF datasets
from sklearn.datasets import fetch_20newsgroups # For newsgroup20
from sklearn.preprocessing import LabelEncoder # Added for custom_syslog

# --- Constants ---
DATA_DIR = "data" # Default directory for downloaded data

# --- Utility Functions ---
def ensure_dir(directory):
    """디렉토리가 존재하지 않으면 생성합니다."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        # print(f"'{directory}' 디렉토리를 생성했습니다.") # Keep console less noisy

def download_file(url, file_path, description=None):
    """파일을 다운로드하고 진행 상황을 표시합니다."""
    try:
        if not description: description = os.path.basename(file_path)
        dirname = os.path.dirname(file_path)
        if dirname: ensure_dir(dirname)

        if os.path.exists(file_path):
            print(f"'{os.path.basename(file_path)}' 파일이 이미 존재합니다. 다운로드를 건너뜁니다.")
            return True

        print(f"'{description}' 다운로드 중... ({url})")
        response = requests.get(url, stream=True, headers={"User-Agent": "Mozilla/5.0"}) # Added User-Agent
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024 * 1024 # 1MB block size for faster download display

        with open(file_path, 'wb') as f, tqdm(
                desc=description, total=total_size, unit='B', unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                f.write(data)

        # Final check on size
        if total_size != 0 and bar.n != total_size:
             print(f"경고: 다운로드된 파일 크기 불일치! ({bar.n}/{total_size} bytes)")
             # Consider deleting the file if size mismatch is critical
             # os.remove(file_path)
             # return False
        print(f"'{os.path.basename(file_path)}' 다운로드 완료.")
        return True
    except requests.exceptions.RequestException as e:
        print(f"다운로드 중 네트워크 오류 발생 ({url}): {e}")
        if os.path.exists(file_path): os.remove(file_path)
        return False
    except Exception as e:
        print(f"다운로드 중 예상치 못한 오류 발생 ({url}): {e}")
        if os.path.exists(file_path): os.remove(file_path)
        return False

def load_csv_universal(file_path, text_col_candidates, label_col_candidates):
    """Attempts to load CSV with flexible column names."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"CSV 파일 없음: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        raise ValueError(f"CSV 파일 로딩 오류 ({file_path}): {e}") from e

    text_col = next((col for col in text_col_candidates if col in df.columns), None)
    label_col = next((col for col in label_col_candidates if col in df.columns), None)

    if not text_col: raise ValueError(f"텍스트 열을 찾을 수 없음 (후보: {text_col_candidates}) in {file_path}")
    if not label_col: raise ValueError(f"레이블 열을 찾을 수 없음 (후보: {label_col_candidates}) in {file_path}")

    print(f"'{os.path.basename(file_path)}' 로드 완료. Text: '{text_col}', Label: '{label_col}'.")
    # Ensure text column is string type, handle potential NaN/None
    texts = df[text_col].fillna('').astype(str).tolist()
    # Ensure label column is string type for consistent processing before encoding
    classes = df[label_col].fillna('unknown').astype(str).tolist()
    return texts, classes


# --- Dataset Specific Download & Preparation Functions ---

# 1. ACM
def download_acm_dataset(output_dir=DATA_DIR):
    """Downloads ACM dataset, extracts, and saves as CSV."""
    dataset_name = "acm"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    zip_url = "https://zenodo.org/records/7555249/files/acm.zip"
    zip_path = os.path.join(dataset_dir, "acm.zip")
    csv_path = os.path.join(dataset_dir, "acm.csv")
    txt_file_name = "texts.txt"
    score_file_name = "score.txt"

    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    if not os.path.exists(zip_path):
        if not download_file(zip_url, zip_path, "ACM ZIP"):
            raise ConnectionError("ACM ZIP 다운로드 실패.")

    print(f"'{os.path.basename(zip_path)}' 압축 해제 및 처리 중...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
            # Find files potentially inside a subdirectory
            txt_member = next((m for m in members if m.endswith(txt_file_name)), None)
            score_member = next((m for m in members if m.endswith(score_file_name)), None)

            if not txt_member or not score_member:
                raise FileNotFoundError(f"ZIP 파일 내에서 '{txt_file_name}' 또는 '{score_file_name}'을 찾을 수 없습니다.")

            texts = z.read(txt_member).decode("utf-8").splitlines()
            labels = z.read(score_member).decode("utf-8").splitlines()

        pd.DataFrame({"text": texts, "label": labels}).to_csv(csv_path, index=False, encoding="utf-8")
        print(f"ACM 데이터가 '{csv_path}'에 저장되었습니다.")
        # Optionally remove zip after extraction
        # os.remove(zip_path)
        return csv_path
    except zipfile.BadZipFile:
        print(f"오류: '{zip_path}' 파일이 손상되었습니다. 삭제 후 다시 시도하세요.")
        if os.path.exists(zip_path): os.remove(zip_path)
        raise
    except Exception as e:
        print(f"ACM ZIP 압축 해제 또는 처리 중 오류: {e}")
        raise

def prepare_acm_dataset(data_dir=DATA_DIR):
    """Prepares ACM dataset: downloads, processes, maps labels."""
    print("ACM 데이터셋 준비 중...")
    try:
        csv_path = download_acm_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['label']) # Use 'label' as loaded
    except Exception as e:
        print(f"ACM 데이터셋 준비 실패: {e}")
        return [], [], []

    class_mapping = {
        '0': 'artificial intelligence', '1': 'computer networks', '2': 'computer security',
        '3': 'database', '4': 'distributed systems', '5': 'graphics & vision',
        '6': 'human‑computer interaction', '7': 'information retrieval', '8': 'operating systems',
        '9': 'programming languages', '10': 'software engineering'
    }
    mapped_classes = [class_mapping.get(cls, f"Unknown_{cls}") for cls in classes]
    unique_classes = sorted(list(set(mapped_classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[c] for c in mapped_classes]
    print(f"ACM 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 2. Reuters-8
R8_LABEL2TOPIC = { 0: "acq", 1: "crude", 2: "earn", 3: "grain", 4: "interest", 5: "money-fx", 6: "ship", 7: "trade" }

def download_reuters8_dataset(output_dir=DATA_DIR):
    """Downloads Reuters-8 from Hugging Face and saves as CSV."""
    dataset_name = "reuters8"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "reuters8.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    try:
        print("Reuters-8 데이터셋 다운로드 중 (Hugging Face dxgp/R8)...")
        # Specify cache directory within our data structure if desired
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("dxgp/R8", split="train+test") # Combine splits, cache_dir=cache_dir
        df = ds.to_pandas()
        # Map numeric label to topic string
        df["topic"] = df["label"].map(R8_LABEL2TOPIC).astype(str)
        df[["text", "topic"]].to_csv(csv_path, index=False, encoding="utf-8")
        print(f"Reuters-8 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 Reuters-8 다운로드 실패: {e}")
        raise ConnectionError("Reuters-8 다운로드 실패.")

def prepare_reuters8_dataset(data_dir=DATA_DIR):
    """Prepares Reuters-8 dataset."""
    print("Reuters-8 데이터셋 준비 중...")
    try:
        csv_path = download_reuters8_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['topic'])
    except Exception as e:
        print(f"Reuters-8 데이터셋 준비 실패: {e}")
        return [], [], []

    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"Reuters-8 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 3. ChemProt
def download_chemprot_dataset(output_dir=DATA_DIR):
    """Downloads ChemProt from Hugging Face and saves as CSV."""
    dataset_name = "chemprot"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "chemprot.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    try:
        print("ChemProt 데이터셋 다운로드 중 (Hugging Face AdaptLLM/ChemProt)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        # Load all available splits and concatenate
        ds_dict = load_dataset("AdaptLLM/ChemProt") #, cache_dir=cache_dir)
        dfs = [ds_dict[split].to_pandas() for split in ds_dict.keys()]
        df = pd.concat(dfs, ignore_index=True)

        # Normalize label column
        if "label" not in df.columns: raise ValueError("'label' column not found.")
        feat = ds_dict[list(ds_dict.keys())[0]].features["label"] # Get features from first split
        if isinstance(feat, ClassLabel):
            df["relation"] = df["label"].apply(feat.int2str)
        else:
            df["relation"] = df["label"].astype(str)

        # Ensure text column exists
        if "text" not in df.columns: raise ValueError("'text' column not found.")

        df[["text", "relation"]].to_csv(csv_path, index=False, encoding="utf-8")
        print(f"ChemProt 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 ChemProt 다운로드 실패: {e}")
        raise ConnectionError("ChemProt 다운로드 실패.")

def prepare_chemprot_dataset(data_dir=DATA_DIR):
    """Prepares ChemProt dataset."""
    print("ChemProt 데이터셋 준비 중...")
    try:
        csv_path = download_chemprot_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['relation'])
    except Exception as e:
        print(f"ChemProt 데이터셋 준비 실패: {e}")
        return [], [], []

    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"ChemProt 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 4. BBC News
def prepare_bbc_news_dataset(data_dir=DATA_DIR):
    """Downloads, extracts, and prepares the BBC News dataset."""
    print("BBC News 데이터셋 준비 중...")
    dataset_name = "bbc_news"
    dataset_dir = os.path.join(data_dir, dataset_name)
    ensure_dir(dataset_dir)
    url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
    zip_path = os.path.join(dataset_dir, 'bbc-fulltext.zip')
    # Target directory for extracted content *within* the dataset's folder
    extracted_base_dir = os.path.join(dataset_dir, 'bbc')

    if not download_file(url, zip_path, "BBC News ZIP"):
         print("BBC News 다운로드 실패.")
         return [], [], []

    # Check if already extracted correctly
    if not os.path.exists(extracted_base_dir) or not os.listdir(extracted_base_dir):
        print(f"압축 파일 해제 중... {zip_path} -> {dataset_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all contents into the dataset-specific directory
                zip_ref.extractall(dataset_dir)
            print(f"압축 해제 완료. 내용 확인 중...")

            # Handle potential nested structure: zip might contain 'bbc-fulltext/bbc/'
            potential_nested_dir = os.path.join(dataset_dir, 'bbc-fulltext', 'bbc')
            if os.path.exists(potential_nested_dir) and not os.path.exists(extracted_base_dir):
                 print(f"  중첩된 디렉토리 구조 감지됨. '{potential_nested_dir}' -> '{extracted_base_dir}' 이동 중...")
                 shutil.move(potential_nested_dir, extracted_base_dir)
                 # Clean up the intermediate 'bbc-fulltext' directory if it's now empty
                 intermediate_parent = os.path.join(dataset_dir, 'bbc-fulltext')
                 if os.path.exists(intermediate_parent) and not os.listdir(intermediate_parent):
                      try: os.rmdir(intermediate_parent)
                      except OSError: pass # Ignore if removal fails (e.g., hidden files)
            elif not os.path.exists(extracted_base_dir):
                 # Check if it extracted directly as 'bbc'
                 if os.path.exists(os.path.join(dataset_dir, 'bbc')):
                      print("  압축 해제 결과가 이미 올바른 위치에 있습니다.")
                 else:
                      print(f"오류: 압축 해제 후 예상 디렉토리 없음: {extracted_base_dir}")
                      # List contents for debugging
                      print(f"  '{dataset_dir}' 내용: {os.listdir(dataset_dir)}")
                      return [], [], []
        except zipfile.BadZipFile:
             print(f"오류: '{zip_path}' 파일 손상됨. 삭제 후 재시도.")
             if os.path.exists(zip_path): os.remove(zip_path)
             return [], [], []
        except Exception as e:
            print(f"압축 해제 중 오류: {e}"); return [], [], []
    else:
        print(f"이미 압축 해제된 디렉토리 사용: {extracted_base_dir}")

    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    texts, labels = [], []
    print("데이터 로딩 중...")
    for i, category in enumerate(categories):
        category_dir = os.path.join(extracted_base_dir, category)
        if not os.path.isdir(category_dir):
            print(f"경고: 카테고리 디렉토리 없음: {category_dir}")
            continue
        try:
            for filename in os.listdir(category_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_dir, filename)
                    try:
                        # Use latin1 as it's common for this dataset, ignore errors
                        with open(file_path, 'r', encoding='latin1', errors='ignore') as file:
                            # Replace multiple whitespace chars with a single space
                            text = re.sub(r'\s+', ' ', file.read()).strip()
                            if text: texts.append(text); labels.append(i)
                    except Exception as e: print(f"\n파일 읽기 오류 ({filename}): {e}")
        except Exception as e: print(f"\n디렉토리 리스팅 오류 ({category_dir}): {e}")

    if not texts: print("경고: BBC News 텍스트 로드 실패!"); return [], [], []
    print(f"BBC News 데이터셋 로딩 완료: {len(texts)} 샘플, {len(categories)} 클래스")
    return texts, labels, categories # Return category names as class_names

# 5. TREC
def prepare_trec_dataset(data_dir=DATA_DIR):
    """Downloads and prepares the TREC dataset."""
    print("TREC 데이터셋 준비 중...")
    dataset_name = "trec"
    dataset_dir = os.path.join(data_dir, dataset_name)
    ensure_dir(dataset_dir)
    train_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label'
    test_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label'
    train_path = os.path.join(dataset_dir, 'train_trec.txt')
    test_path = os.path.join(dataset_dir, 'test_trec.txt')

    if not download_file(train_url, train_path, "TREC Train"): return ([], []), ([], []), []
    if not download_file(test_url, test_path, "TREC Test"): return ([], []), ([], []), []

    def load_trec_file(file_path):
        texts, labels_str = [], []
        # Use only the 6 coarse labels
        valid_coarse_labels = {'ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM'}
        try:
            # Try latin1 first, fallback to utf-8 ignoring errors
            encodings_to_try = ['latin1', 'utf-8']
            content = None
            for enc in encodings_to_try:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        content = file.readlines()
                    print(f"  TREC 파일 '{os.path.basename(file_path)}' 로드 성공 (인코딩: {enc})")
                    break
                except UnicodeDecodeError:
                    continue
            if content is None:
                print(f"  경고: TREC 파일 '{os.path.basename(file_path)}' 디코딩 실패. 건너뜁니다.")
                return [], []

            for line in content:
                line = line.strip()
                if not line: continue
                # Split only on the first space to handle questions with spaces
                parts = line.split(' ', 1)
                if len(parts) != 2: continue # Skip malformed lines

                label_part, question = parts
                # Extract the coarse label (before the colon)
                coarse_label = label_part.split(':', 1)[0]

                # Keep only samples belonging to the valid coarse labels
                if coarse_label in valid_coarse_labels:
                    texts.append(question)
                    labels_str.append(coarse_label)
        except Exception as e: print(f"TREC 파일 로딩 오류 ({file_path}): {e}")
        return texts, labels_str

    train_texts, train_labels_str = load_trec_file(train_path)
    test_texts, test_labels_str = load_trec_file(test_path)
    if not train_texts or not test_texts: print("오류: TREC 로딩 실패."); return ([], []), ([], []), []

    # Combine train/test for consistent label mapping
    all_labels_str = train_labels_str + test_labels_str
    unique_labels = sorted(list(set(all_labels_str)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}

    train_labels = [label_to_id[label] for label in train_labels_str]
    test_labels = [label_to_id[label] for label in test_labels_str]

    print(f"TREC 로딩 완료: Train={len(train_texts)}, Test={len(test_texts)}, Classes={unique_labels}")
    # Return combined data for DataModule to split, and original class names
    all_texts = train_texts + test_texts
    all_labels = train_labels + test_labels
    return all_texts, all_labels, unique_labels # Modified to return combined data

# 6. Banking77
def download_banking77_dataset(output_dir=DATA_DIR):
    """Downloads Banking77 dataset, trying HF then GitHub, saves as CSV."""
    dataset_name = "banking77"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "banking77.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    # Try Hugging Face URL first
    hf_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/banking77.csv"
    # Fallback GitHub URL (might point to train split only)
    github_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"

    if download_file(hf_url, csv_path, "Banking77 (HF)"):
        # Verify columns after download
        try:
            df = pd.read_csv(csv_path)
            # Normalize columns robustly
            text_col = next((c for c in ['text', 'query'] if c in df.columns), None)
            label_col = next((c for c in ['category', 'label'] if c in df.columns), None)

            if text_col and label_col:
                 df_processed = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'category'})
                 df_processed.to_csv(csv_path, index=False, encoding='utf-8')
                 print(f"Banking77 (HF) 데이터가 '{csv_path}'에 저장되었습니다.")
                 return csv_path
            else: # Try fallback if columns are wrong
                 print("HF Banking77 CSV 컬럼 불일치, GitHub 대체 시도 중...")
                 os.remove(csv_path) # Remove incorrect file
        except Exception as e:
             print(f"다운로드된 Banking77 (HF) 처리 오류: {e}. 대체 시도 중...")
             if os.path.exists(csv_path): os.remove(csv_path)

    # Try GitHub fallback
    print("GitHub 대체 URL 시도 중...")
    if download_file(github_url, csv_path, "Banking77 (GitHub Fallback)"):
         try:
            df = pd.read_csv(csv_path)
            text_col = next((c for c in ['text', 'query'] if c in df.columns), None)
            label_col = next((c for c in ['category', 'label'] if c in df.columns), None)

            if text_col and label_col:
                 df_processed = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'category'})
                 df_processed.to_csv(csv_path, index=False, encoding='utf-8')
                 print(f"Banking77 (GitHub) 데이터가 '{csv_path}'에 저장되었습니다.")
                 return csv_path
            else:
                 raise ValueError("GitHub 대체 파일에서 텍스트 또는 레이블 열을 찾을 수 없습니다.")
         except Exception as e:
              print(f"다운로드된 Banking77 (GitHub) 처리 오류: {e}")
              if os.path.exists(csv_path): os.remove(csv_path)

    raise ConnectionError("Banking77 다운로드 실패.")

def prepare_banking77_dataset(data_dir=DATA_DIR):
    """Prepares Banking77 dataset."""
    print("Banking77 데이터셋 준비 중...")
    try:
        csv_path = download_banking77_dataset(data_dir)
        # Use flexible column names for loading
        texts, classes = load_csv_universal(csv_path, ['text', 'query'], ['category', 'label'])
    except Exception as e:
        print(f"Banking77 데이터셋 준비 실패: {e}")
        return [], [], []

    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"Banking77 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 7. OOS (CLINC150)
def download_oos_dataset(output_dir=DATA_DIR, config="plus"): # Use 'plus' for more data including OOS
    """Downloads OOS (CLINC150) from Hugging Face and saves as CSV."""
    dataset_name = "oos_clinc150"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, f"oos_{config}.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    try:
        print(f"OOS (CLINC150) 데이터셋 다운로드 중 (Hugging Face clinc_oos/{config})...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        # Load all splits if available, otherwise just train
        try:
             ds = load_dataset("clinc_oos", config) #, cache_dir=cache_dir)
             splits = list(ds.keys())
             print(f"  사용 가능한 분할: {splits}")
             df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)
        except ValueError: # If config doesn't exist, try 'imbalanced' or default
             print(f"구성 '{config}'을(를) 찾을 수 없습니다. 'imbalanced' 시도 중...")
             try:
                  ds = load_dataset("clinc_oos", "imbalanced") #, cache_dir=cache_dir)
                  splits = list(ds.keys())
                  df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)
             except Exception as e_imb:
                  print(f"'imbalanced' 구성 로드 실패: {e_imb}. 기본 로드 시도 중.")
                  ds = load_dataset("clinc_oos") # Load whatever default is available, cache_dir=cache_dir
                  splits = list(ds.keys())
                  df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)

        # Normalize label column -> 'intent'
        label_col_found = None
        if "intent" in df.columns:
            label_col_found = "intent"
        elif "label" in df.columns:
             feat = ds[splits[0]].features["label"]
             if isinstance(feat, ClassLabel): df["intent"] = df["label"].apply(feat.int2str)
             else: df["intent"] = df["label"].astype(str)
             label_col_found = "intent"
        if not label_col_found: raise ValueError("레이블/의도 열을 찾을 수 없습니다.")

        # Normalize text column -> 'text'
        text_col_found = None
        if "text" in df.columns:
            text_col_found = "text"
        else:
             text_col_cand = next((c for c in ["sentence", "utterance", "query"] if c in df.columns), None)
             if text_col_cand:
                 df.rename(columns={text_col_cand: "text"}, inplace=True)
                 text_col_found = "text"
        if not text_col_found: raise ValueError("텍스트 열을 찾을 수 없습니다.")

        df[["text", "intent"]].to_csv(csv_path, index=False, encoding="utf-8")
        print(f"OOS (CLINC150) 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 OOS (CLINC150) 다운로드 실패: {e}")
        raise ConnectionError("OOS (CLINC150) 다운로드 실패.")

def prepare_oos_dataset(data_dir=DATA_DIR):
    """Prepares OOS (CLINC150) dataset."""
    print("OOS (CLINC150) 데이터셋 준비 중...")
    try:
        # Using 'plus' config which includes out-of-scope samples often marked with a specific label
        csv_path = download_oos_dataset(data_dir, config="plus")
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence', 'utterance', 'query'], ['intent', 'label'])
    except Exception as e:
        print(f"OOS (CLINC150) 데이터셋 준비 실패: {e}")
        return [], [], []

    # OOS samples are often labeled as 'oos' or 'unknown' or similar.
    # We load all unique class names including the OOS label.
    # The DataModule's test set preparation will map this label to -1.
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]

    print(f"OOS (CLINC150) 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스 (포함 가능성: 'oos')")
    # Return the original string names; DataModule handles mapping 'oos'/'unknown' during setup
    return texts, labels, unique_classes


# 8. StackOverflow
def download_stackoverflow_dataset(output_dir=DATA_DIR):
    """Downloads StackOverflow dataset from HF and saves as CSV."""
    dataset_name = "stackoverflow"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "stackoverflow.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    try:
        print("StackOverflow 데이터셋 다운로드 중 (Hugging Face)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        df = None
        # Try a larger, more standard one first if available
        try:
             print("  pvcnn/stackoverflow-questions-title-tags 시도 중...")
             ds = load_dataset("pvcnn/stackoverflow-questions-title-tags", split="train") #, cache_dir=cache_dir) # Example
             print("  pvcnn/stackoverflow-questions-title-tags 로드됨.")
             df = ds.to_pandas()
             # Normalize columns
             if 'text' not in df.columns and 'title' in df.columns: df.rename(columns={'title': 'text'}, inplace=True)
             if 'tags' not in df.columns and 'label' in df.columns: df.rename(columns={'label': 'tags'}, inplace=True)

        except Exception as e1:
             print(f"  pvcnn/stackoverflow 로딩 실패 ({e1}), c17hawke/stackoverflow-dataset 시도 중...")
             try:
                 ds = load_dataset("c17hawke/stackoverflow-dataset", split="train") #, cache_dir=cache_dir)
                 print("  c17hawke/stackoverflow-dataset 로드됨.")
                 df = ds.to_pandas()
                 # Combine title and body for 'text'
                 df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
                 df['text'] = df['text'].str.strip()
                 if 'tags' not in df.columns: raise ValueError("c17hawke에서 태그 열 누락.")
             except Exception as e2:
                 print(f"  c17hawke/stackoverflow 로딩 실패 ({e2}).")
                 raise ConnectionError("StackOverflow 다운로드 실패 (두 소스 모두).")

        if df is None or 'text' not in df.columns or 'tags' not in df.columns:
             raise ValueError("StackOverflow 데이터 처리 실패: 텍스트 또는 태그 열 없음.")

        # Ensure tags are strings (handle potential lists)
        if not df.empty and isinstance(df['tags'].iloc[0], list):
            df['tags'] = df['tags'].apply(lambda x: ' '.join(map(str, x)) if isinstance(x, list) else str(x))
        else:
            df['tags'] = df['tags'].astype(str)

        df = df[['text', 'tags']].dropna(subset=['text']) # Keep relevant columns, drop rows with no text
        df = df[df['text'].str.len() > 10] # Remove very short texts

        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"StackOverflow 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path

    except Exception as e:
        print(f"Hugging Face에서 StackOverflow 다운로드 실패: {e}")
        raise ConnectionError("StackOverflow 다운로드 실패.")

def prepare_stackoverflow_dataset(data_dir=DATA_DIR):
    """Prepares StackOverflow dataset, potentially limiting classes."""
    print("StackOverflow 데이터셋 준비 중...")
    try:
        csv_path = download_stackoverflow_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'title'], ['tags'])
        # StackOverflow often has many tags per question.
        # Treat the full tag string as the class for now.
    except Exception as e:
        print(f"StackOverflow 데이터셋 준비 실패: {e}")
        return [], [], []

    # Limit number of classes if too many unique tag combinations exist
    MAX_SO_CLASSES = 100 # Example limit
    unique_classes_full = sorted(list(set(classes)))
    if len(unique_classes_full) > MAX_SO_CLASSES:
         print(f"경고: StackOverflow 태그 조합이 너무 많음 ({len(unique_classes_full)}). 상위 {MAX_SO_CLASSES}개만 사용합니다.")
         # Get most frequent tags
         tag_counts = pd.Series(classes).value_counts()
         top_tags = tag_counts.nlargest(MAX_SO_CLASSES).index.tolist()
         # Filter data to keep only top tags
         mask = pd.Series(classes).isin(top_tags)
         # Use list comprehensions for filtering
         texts = [t for i, t in enumerate(texts) if mask.iloc[i]]
         classes = [c for i, c in enumerate(classes) if mask.iloc[i]]
         unique_classes = sorted(top_tags)
         print(f"  상위 태그로 필터링 후 샘플 수: {len(texts)}")
    else:
         unique_classes = unique_classes_full

    if not texts:
        print("경고: 필터링 후 StackOverflow 데이터 없음.")
        return [], [], []

    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"StackOverflow 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes


# 9. ATIS
def download_atis_dataset(output_dir=DATA_DIR):
    """Downloads ATIS dataset from HF and saves as CSV."""
    dataset_name = "atis"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "atis.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("ATIS 데이터셋 다운로드 중 (Hugging Face tuetschek/atis)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("tuetschek/atis") #, cache_dir=cache_dir)
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)

        # Normalize columns
        text_col = next((c for c in ['text', 'sentence', 'query'] if c in df.columns), None)
        label_col_int = 'label' if 'label' in df.columns else None
        label_col_str = 'intent' if 'intent' in df.columns else None

        if not text_col: raise ValueError("텍스트 열 없음.")
        if not label_col_int and not label_col_str: raise ValueError("레이블/의도 열 없음.")

        # Prefer intent if available, otherwise map label
        if label_col_str:
            df['intent_norm'] = df[label_col_str]
        elif label_col_int:
            feat = ds[list(ds.keys())[0]].features[label_col_int]
            if isinstance(feat, ClassLabel): df["intent_norm"] = df[label_col_int].apply(feat.int2str)
            else: df["intent_norm"] = df[label_col_int].astype(str)

        df_processed = df[[text_col, 'intent_norm']].rename(columns={text_col: 'text', 'intent_norm': 'intent'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"ATIS 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 ATIS 다운로드 실패: {e}")
        raise ConnectionError("ATIS 다운로드 실패.")

def prepare_atis_dataset(data_dir=DATA_DIR):
    """Prepares ATIS dataset."""
    print("ATIS 데이터셋 준비 중...")
    try:
        csv_path = download_atis_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence'], ['intent', 'label'])
    except Exception as e:
        print(f"ATIS 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"ATIS 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 10. SNIPS
def download_snips_dataset(output_dir=DATA_DIR):
    """Downloads SNIPS dataset from HF and saves as CSV."""
    dataset_name = "snips"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "snips.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("SNIPS 데이터셋 다운로드 중 (Hugging Face snips_built_in_intents)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        ds = load_dataset("snips_built_in_intents") #, cache_dir=cache_dir)
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)

        # Normalize columns
        text_col = next((c for c in ['text', 'sentence', 'query'] if c in df.columns), None)
        label_col_int = 'label' if 'label' in df.columns else None
        label_col_str = 'intent' if 'intent' in df.columns else None

        if not text_col: raise ValueError("텍스트 열 없음.")
        if not label_col_int and not label_col_str: raise ValueError("레이블/의도 열 없음.")

        if label_col_str:
            df['intent_norm'] = df[label_col_str]
        elif label_col_int:
            feat = ds[list(ds.keys())[0]].features[label_col_int]
            if isinstance(feat, ClassLabel): df["intent_norm"] = df[label_col_int].apply(feat.int2str)
            else: df["intent_norm"] = df[label_col_int].astype(str)

        df_processed = df[[text_col, 'intent_norm']].rename(columns={text_col: 'text', 'intent_norm': 'intent'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"SNIPS 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 SNIPS 다운로드 실패: {e}")
        # Add fallback here if needed (e.g., GitHub JSON download from your code)
        raise ConnectionError("SNIPS 다운로드 실패.")

def prepare_snips_dataset(data_dir=DATA_DIR):
    """Prepares SNIPS dataset."""
    print("SNIPS 데이터셋 준비 중...")
    try:
        csv_path = download_snips_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text'], ['intent', 'label'])
    except Exception as e:
        print(f"SNIPS 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"SNIPS 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 11. Financial PhraseBank
def download_financial_phrasebank_dataset(output_dir=DATA_DIR):
    """Downloads Financial PhraseBank from HF and saves as CSV."""
    dataset_name = "financial_phrasebank"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "financial_phrasebank.csv")
    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path
    try:
        print("Financial PhraseBank 데이터셋 다운로드 중 (Hugging Face takala/financial_phrasebank)...")
        # cache_dir = os.path.join(dataset_dir, "hf_cache")
        # Use 'sentences_allagree' configuration
        ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train") #, cache_dir=cache_dir) # Only train split available
        df = ds.to_pandas()

        # Normalize columns
        text_col = next((c for c in ['text', 'sentence'] if c in df.columns), None)
        label_col_int = 'label' if 'label' in df.columns else None
        label_col_str = 'sentiment' if 'sentiment' in df.columns else None

        if not text_col: raise ValueError("텍스트/문장 열 없음.")
        if not label_col_int and not label_col_str: raise ValueError("감성/레이블 열 없음.")

        if label_col_str:
            df['sentiment_norm'] = df[label_col_str]
        elif label_col_int:
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            df['sentiment_norm'] = df[label_col_int].map(label_map).fillna('unknown')

        df_processed = df[[text_col, 'sentiment_norm']].rename(columns={text_col: 'text', 'sentiment_norm': 'sentiment'})
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Financial PhraseBank 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 Financial PhraseBank 다운로드 실패: {e}")
        raise ConnectionError("Financial PhraseBank 다운로드 실패.")

def prepare_financial_phrasebank_dataset(data_dir=DATA_DIR):
    """Prepares Financial PhraseBank dataset."""
    print("Financial PhraseBank 데이터셋 준비 중...")
    try:
        csv_path = download_financial_phrasebank_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence'], ['sentiment', 'label'])
    except Exception as e:
        print(f"Financial PhraseBank 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"Financial PhraseBank 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 12. ArXiv-10
def download_arxiv10_dataset(output_dir=DATA_DIR):
    """Downloads ArXiv-10 dataset, extracts, normalizes, and saves as CSV."""
    dataset_name = "arxiv10"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    zip_url = "https://github.com/ashfarhangi/Protoformer/raw/main/data/ArXiv-10.zip"
    zip_path = os.path.join(dataset_dir, "arxiv10.zip")
    csv_path = os.path.join(dataset_dir, "arxiv10.csv") # Target CSV

    if os.path.exists(csv_path):
        print(f"'{os.path.basename(csv_path)}' 파일이 이미 존재합니다. 처리를 건너뜁니다.")
        return csv_path

    if not os.path.exists(zip_path):
        if not download_file(zip_url, zip_path, "ArXiv-10 ZIP"):
            raise ConnectionError("ArXiv-10 ZIP 다운로드 실패.")

    print(f"'{os.path.basename(zip_path)}' 압축 해제 및 처리 중...")
    extracted_folder_name = "ArXiv-10" # Expected folder name inside zip
    extracted_folder_path = os.path.join(dataset_dir, extracted_folder_name)
    temp_extract_path = os.path.join(dataset_dir, "temp_extract") # Extract to temp location first
    ensure_dir(temp_extract_path)

    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(temp_extract_path)
        print(f"  '{temp_extract_path}'에 압축 해제 완료.")

        # Find the CSV file within the extracted content (could be nested)
        found_csv_path = None
        target_csv_name = 'arxiv-10.csv' # Expected filename (case-insensitive check)
        for root, _, files in os.walk(temp_extract_path):
            for file in files:
                if file.lower() == target_csv_name:
                    found_csv_path = os.path.join(root, file)
                    print(f"  CSV 파일 찾음: {found_csv_path}")
                    break
            if found_csv_path: break # Stop searching once found

        if not found_csv_path:
            # Fallback: look for any CSV file if specific name not found
            print(f"  '{target_csv_name}'을(를) 찾지 못했습니다. 폴더 내 다른 CSV 검색 중...")
            for root, _, files in os.walk(temp_extract_path):
                 for file in files:
                      if file.lower().endswith('.csv'):
                           found_csv_path = os.path.join(root, file)
                           print(f"  대체 CSV 파일 찾음: {found_csv_path}")
                           break
                 if found_csv_path: break

        if not found_csv_path:
            raise FileNotFoundError("압축 해제된 폴더에서 ArXiv-10 CSV 파일을 찾을 수 없습니다.")

        # Read, process, and save to target CSV path
        print(f"  CSV 파일 처리 중: {found_csv_path}")
        df = pd.read_csv(found_csv_path)

        # Normalize columns (adjust based on actual CSV columns)
        text_col = None
        if 'abstract' in df.columns: text_col = 'abstract'
        elif 'text' in df.columns: text_col = 'text'
        elif 'title' in df.columns: text_col = 'title' # Use title if abstract/text missing
        else: raise ValueError("ArXiv-10 CSV에서 텍스트 열(abstract/text/title)을 찾을 수 없습니다.")

        label_col = None
        if 'category' in df.columns: label_col = 'category'
        elif 'label' in df.columns: label_col = 'label'
        elif 'class' in df.columns: label_col = 'class'
        else: raise ValueError("ArXiv-10 CSV에서 카테고리 열(category/label/class)을 찾을 수 없습니다.")

        print(f"  정규화된 열 사용: Text='{text_col}', Label='{label_col}'")
        df_processed = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'category'})

        # Overwrite the target CSV with normalized columns
        df_processed.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"ArXiv-10 데이터가 '{csv_path}'에 최종 저장되었습니다.")

        # Clean up extracted folder and zip
        try:
            shutil.rmtree(temp_extract_path)
            print(f"  임시 추출 폴더 삭제됨: {temp_extract_path}")
        except Exception as e_rm: print(f"경고: 임시 추출 폴더 삭제 실패 {temp_extract_path}: {e_rm}")
        # os.remove(zip_path) # Keep the zip?

        return csv_path

    except zipfile.BadZipFile:
        print(f"오류: '{zip_path}' 파일이 손상되었습니다. 삭제 후 다시 시도하세요.")
        if os.path.exists(zip_path): os.remove(zip_path)
        if os.path.exists(temp_extract_path): shutil.rmtree(temp_extract_path)
        raise
    except Exception as e:
        print(f"ArXiv-10 처리 오류: {e}")
        if os.path.exists(temp_extract_path): shutil.rmtree(temp_extract_path)
        raise

def prepare_arxiv10_dataset(data_dir=DATA_DIR):
    """Prepares ArXiv-10 dataset."""
    print("ArXiv-10 데이터셋 준비 중...")
    try:
        csv_path = download_arxiv10_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'abstract', 'title'], ['category', 'label', 'class'])
    except Exception as e:
        print(f"ArXiv-10 데이터셋 준비 실패: {e}")
        return [], [], []
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"ArXiv-10 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes

# 13. NewsGroup20 (Using sklearn)
def prepare_newsgroup20_dataset(data_dir=DATA_DIR):
    """Prepares 20 Newsgroups dataset using sklearn's fetcher."""
    # Note: sklearn handles download/caching, but we might want to control the location
    # For simplicity, we'll let sklearn manage it for now.
    # If strict control over DATA_DIR is needed, this would require more work.
    print("20 Newsgroups 데이터셋 준비 중 (sklearn 사용)...")
    try:
        # Load combined data for consistent splitting later
        data_all = fetch_20newsgroups(
            subset='all', remove=('headers', 'footers', 'quotes'),
            random_state=42 # Use a fixed state for reproducibility if desired
            # data_home=os.path.join(data_dir, "newsgroup20", "sklearn_cache") # Optional: Control cache location
        )
        texts = data_all.data
        labels = data_all.target # Numeric labels 0-19
        class_names = data_all.target_names
        print(f"20 Newsgroups 로딩 완료: {len(texts)} 샘플, {len(class_names)} 클래스")
        return texts, labels, class_names
    except Exception as e:
        print(f"20 Newsgroups 로딩 실패: {e}")
        return [], [], []

# 14. Custom Syslog
def download_custom_syslog_dataset(output_dir=DATA_DIR):
    """
    Checks for the existence of the custom syslog dataset file.
    Does not actually download, assumes user provides the file.
    """
    dataset_name = "custom_syslog"
    dataset_dir = os.path.join(output_dir, dataset_name)
    ensure_dir(dataset_dir)
    csv_path = os.path.join(dataset_dir, "custom_syslog.csv") # Expected filename

    if os.path.exists(csv_path):
        print(f"Custom Syslog 데이터셋 파일 '{csv_path}'이(가) 존재합니다.")
        return csv_path
    else:
        error_message = (
            f"오류: Custom Syslog 데이터셋 파일을 찾을 수 없습니다.\n"
            f"'{csv_path}' 경로에 'custom_syslog.csv' 파일을 위치시켜 주세요.\n"
            f"이 파일은 'text' 열과 클래스 레이블을 포함하는 'class' 또는 'label' 열을 가져야 합니다."
        )
        raise FileNotFoundError(error_message)

def prepare_custom_syslog_dataset(data_dir=DATA_DIR):
    """Prepares custom syslog dataset: checks existence, loads, encodes labels."""
    print("Custom Syslog 데이터셋 준비 중...")
    try:
        csv_path = download_custom_syslog_dataset(data_dir) # Checks existence
        # Load using flexible column names
        texts, classes_str = load_csv_universal(
            csv_path,
            text_col_candidates=['text', 'message', 'log', 'content'],
            label_col_candidates=['class', 'label', 'category', 'event', 'template']
        )
    except Exception as e:
        print(f"Custom Syslog 데이터셋 준비 실패: {e}")
        return [], [], []

    if not texts:
        print("경고: Custom Syslog 텍스트 데이터가 비어 있습니다.")
        return [], [], []

    # Encode string labels to integers (0 to N-1)
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(classes_str)
    class_names = label_encoder.classes_.tolist() # Get original string names

    print(f"Custom Syslog 데이터셋 준비 완료: {len(texts)} 샘플, {len(class_names)} 클래스")
    print(f"  클래스 예시: {class_names[:10]}{'...' if len(class_names) > 10 else ''}")
    return texts, labels_encoded, class_names


# --- Add any other dataset functions if needed ---

# --- END OF dataset_utils.py ---