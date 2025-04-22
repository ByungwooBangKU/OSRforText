# --- START OF dataset_utils.py (Integrated Version) ---
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

    df = pd.read_csv(file_path)
    text_col = next((col for col in text_col_candidates if col in df.columns), None)
    label_col = next((col for col in label_col_candidates if col in df.columns), None)

    if not text_col: raise ValueError(f"텍스트 열을 찾을 수 없음 (후보: {text_col_candidates}) in {file_path}")
    if not label_col: raise ValueError(f"레이블 열을 찾을 수 없음 (후보: {label_col_candidates}) in {file_path}")

    print(f"'{os.path.basename(file_path)}' 로드 완료. Text: '{text_col}', Label: '{label_col}'.")
    return df[text_col].astype(str).tolist(), df[label_col].astype(str).tolist()


# --- Dataset Specific Download & Preparation Functions ---

# 1. ACM
# --- START OF dataset_utils.py (download_acm_dataset fix only) ---
# ... (다른 임포트 및 함수들) ...

def download_acm_dataset(output_dir=DATA_DIR):
    ensure_dir(output_dir)
    zip_url = "https://zenodo.org/records/7555249/files/acm.zip" # Corrected URL structure
    zip_path = os.path.join(output_dir, "acm.zip")
    csv_path = os.path.join(output_dir, "acm.csv")
    txt_file_name = "texts.txt"
    score_file_name = "score.txt"

    if os.path.exists(csv_path): return csv_path # Already processed

    if not os.path.exists(zip_path):
        if not download_file(zip_url, zip_path, "ACM ZIP"):
            raise ConnectionError("ACM ZIP 다운로드 실패.")

    print(f"'{os.path.basename(zip_path)}' 압축 해제 중...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            members = z.namelist()
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
        # 수정된 부분: 각 동작을 별도의 줄로 분리
        print(f"오류: '{zip_path}' 파일이 손상되었습니다. 삭제 후 다시 시도하세요.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise # 예외를 다시 발생시켜 호출자에게 알림
    except Exception as e:
        print(f"ACM ZIP 압축 해제 또는 처리 중 오류: {e}")
        raise

# ... (파일의 나머지 부분) ...
# --- END OF dataset_utils.py (download_acm_dataset fix only) ---

def prepare_acm_dataset(data_dir=DATA_DIR):
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
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "reuters8.csv")
    if os.path.exists(csv_path): return csv_path

    try:
        print("Reuters-8 데이터셋 다운로드 중 (Hugging Face dxgp/R8)...")
        ds = load_dataset("dxgp/R8", split="train+test") # Combine splits
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
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "chemprot.csv")
    if os.path.exists(csv_path): return csv_path

    try:
        print("ChemProt 데이터셋 다운로드 중 (Hugging Face AdaptLLM/ChemProt)...")
        # Load all available splits and concatenate
        ds_dict = load_dataset("AdaptLLM/ChemProt")
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

# 4. BBC News (Self-contained downloader already exists)
def prepare_bbc_news_dataset(data_dir=DATA_DIR):
    # Uses the existing function from the previous response
    print("BBC News 데이터셋 준비 중 (기존 로직 사용)...")
    # --- (Include the full prepare_bbc_news_dataset function from previous response here) ---
    print(f"prepare_bbc_news_dataset 함수 시작 - 대상 디렉토리: {data_dir}")
    os.makedirs(data_dir, exist_ok=True)
    url = 'http://mlg.ucd.ie/files/datasets/bbc-fulltext.zip'
    zip_path = os.path.join(data_dir, 'bbc-fulltext.zip')
    extracted_base_dir = os.path.join(data_dir, 'bbc')

    if not download_file(url, zip_path, "BBC News ZIP"):
         print("BBC News 다운로드 실패.")
         return [], [], []

    actual_extracted_dir = os.path.join(data_dir, 'bbc')
    if not os.path.exists(actual_extracted_dir) or not os.listdir(actual_extracted_dir):
        print(f"압축 파일 해제 중... {zip_path} -> {data_dir}")
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            print(f"압축 해제 완료. 내용 확인 중...")
            potential_nested_dir = os.path.join(data_dir, 'bbc-fulltext', 'bbc')
            if os.path.exists(potential_nested_dir) and not os.path.exists(actual_extracted_dir):
                 print(f"  중첩된 디렉토리 구조 감지됨. 이동 중...")
                 shutil.move(potential_nested_dir, actual_extracted_dir)
                 if os.path.exists(os.path.join(data_dir, 'bbc-fulltext')):
                      try: os.rmdir(os.path.join(data_dir, 'bbc-fulltext'))
                      except OSError: pass
            if not os.path.exists(actual_extracted_dir):
                 print(f"오류: 압축 해제 후 예상 디렉토리 없음: {actual_extracted_dir}")
                 return [], [], []
        except zipfile.BadZipFile:
             print(f"오류: '{zip_path}' 파일 손상됨. 삭제 후 재시도.")
             if os.path.exists(zip_path): os.remove(zip_path)
             return [], [], []
        except Exception as e:
            print(f"압축 해제 중 오류: {e}"); return [], [], []
    else:
        print(f"이미 압축 해제된 디렉토리 사용: {actual_extracted_dir}")

    categories = ['business', 'entertainment', 'politics', 'sport', 'tech']
    texts, labels = [], []
    print("데이터 로딩 중...")
    for i, category in enumerate(categories):
        category_dir = os.path.join(actual_extracted_dir, category)
        if not os.path.isdir(category_dir): continue
        try:
            for filename in os.listdir(category_dir):
                if filename.endswith('.txt'):
                    file_path = os.path.join(category_dir, filename)
                    try:
                        with open(file_path, 'r', encoding='latin1', errors='ignore') as file:
                            text = re.sub(r'\s+', ' ', file.read()).strip()
                            if text: texts.append(text); labels.append(i)
                    except Exception as e: print(f"\n파일 읽기 오류 ({filename}): {e}")
        except Exception as e: print(f"\n디렉토리 리스팅 오류 ({category_dir}): {e}")

    if not texts: print("경고: BBC News 텍스트 로드 실패!"); return [], [], []
    print(f"BBC News 데이터셋 로딩 완료: {len(texts)} 샘플, {len(categories)} 클래스")
    return texts, labels, categories # Return category names as class_names

# 5. TREC (Self-contained downloader already exists)
def prepare_trec_dataset(data_dir=DATA_DIR):
     # Uses the existing function from the previous response
    print("TREC 데이터셋 준비 중 (기존 로직 사용)...")
    # --- (Include the full prepare_trec_dataset function from previous response here) ---
    os.makedirs(data_dir, exist_ok=True)
    train_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/train_5500.label'
    test_url = 'https://cogcomp.seas.upenn.edu/Data/QA/QC/TREC_10.label'
    train_path = os.path.join(data_dir, 'train_trec.txt') # Use distinct names
    test_path = os.path.join(data_dir, 'test_trec.txt')

    if not download_file(train_url, train_path, "TREC Train"): return ([], []), ([], []), []
    if not download_file(test_url, test_path, "TREC Test"): return ([], []), ([], []), []

    def load_trec_file(file_path):
        texts, labels_str = [], []
        valid_coarse_labels = {'ABBR', 'DESC', 'ENTY', 'HUM', 'LOC', 'NUM'}
        try:
            with open(file_path, 'r', encoding='latin1') as file:
                for line in file:
                    line = line.strip();
                    if not line: continue
                    parts = line.split(' ', 1);
                    if len(parts) != 2: continue
                    label_part, question = parts
                    label = label_part.split(':', 1)[0]
                    if label in valid_coarse_labels: texts.append(question); labels_str.append(label)
        except Exception as e: print(f"TREC 파일 로딩 오류 ({file_path}): {e}")
        return texts, labels_str

    train_texts, train_labels_str = load_trec_file(train_path)
    test_texts, test_labels_str = load_trec_file(test_path)
    if not train_texts or not test_texts: print("오류: TREC 로딩 실패."); return ([], []), ([], []), []

    unique_labels = sorted(list(set(train_labels_str + test_labels_str)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    train_labels = [label_to_id[label] for label in train_labels_str]
    test_labels = [label_to_id[label] for label in test_labels_str]

    print(f"TREC 로딩 완료: Train={len(train_texts)}, Test={len(test_texts)}, Classes={unique_labels}")
    return (train_texts, train_labels), (test_texts, test_labels), unique_labels

# --- Add prepare functions for other datasets (Banking77, OOS, StackOverflow, etc.) ---
# --- using their respective download functions from the provided code ---

# 6. Banking77
def download_banking77_dataset(output_dir=DATA_DIR):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "banking.csv")
    if os.path.exists(csv_path): return csv_path

    # Try Hugging Face URL first
    hf_url = "https://huggingface.co/datasets/PolyAI/banking77/resolve/main/banking77.csv"
    # Fallback GitHub URL (might point to train split only)
    github_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"

    if download_file(hf_url, csv_path, "Banking77 (HF)"):
        # Verify columns after download
        try:
            df = pd.read_csv(csv_path)
            if 'text' in df.columns and 'category' in df.columns: return csv_path
            elif 'text' in df.columns and 'label' in df.columns:
                 df.rename(columns={'label': 'category'}, inplace=True)
                 df.to_csv(csv_path, index=False); return csv_path
            elif 'query' in df.columns and 'category' in df.columns:
                 df.rename(columns={'query': 'text'}, inplace=True)
                 df.to_csv(csv_path, index=False); return csv_path
            elif 'query' in df.columns and 'label' in df.columns:
                 df.rename(columns={'query': 'text', 'label': 'category'}, inplace=True)
                 df.to_csv(csv_path, index=False); return csv_path
            else: # Try fallback if columns are wrong
                 print("HF Banking77 CSV columns incorrect, trying GitHub fallback...")
                 os.remove(csv_path) # Remove incorrect file
        except Exception as e:
             print(f"Error processing downloaded Banking77 (HF): {e}. Trying fallback...")
             if os.path.exists(csv_path): os.remove(csv_path)

    # Try GitHub fallback
    if download_file(github_url, csv_path, "Banking77 (GitHub Fallback)"):
         try:
            df = pd.read_csv(csv_path)
            # Assume GitHub file has 'text', 'category' or similar standard names
            if 'text' not in df.columns: # Simple check, might need more robust column finding
                 if 'query' in df.columns: df.rename(columns={'query': 'text'}, inplace=True)
                 else: raise ValueError("Cannot find text column in GitHub fallback.")
            if 'category' not in df.columns:
                 if 'label' in df.columns: df.rename(columns={'label': 'category'}, inplace=True)
                 else: raise ValueError("Cannot find category/label column in GitHub fallback.")
            df[['text', 'category']].to_csv(csv_path, index=False); return csv_path
         except Exception as e:
              print(f"Error processing downloaded Banking77 (GitHub): {e}")
              if os.path.exists(csv_path): os.remove(csv_path)

    raise ConnectionError("Banking77 다운로드 실패.")


def prepare_banking77_dataset(data_dir=DATA_DIR):
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
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, f"oos_{config}.csv")
    if os.path.exists(csv_path): return csv_path

    try:
        print(f"OOS (CLINC150) 데이터셋 다운로드 중 (Hugging Face clinc_oos/{config})...")
        # Load all splits if available, otherwise just train
        try:
             ds = load_dataset("clinc_oos", config)
             splits = list(ds.keys())
             print(f"  Available splits: {splits}")
             df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)
        except ValueError: # If config doesn't exist, try 'imbalanced' or default
             print(f"Config '{config}' not found, trying 'imbalanced'...")
             try:
                  ds = load_dataset("clinc_oos", "imbalanced")
                  splits = list(ds.keys())
                  df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)
             except Exception as e_imb:
                  print(f"Failed to load 'imbalanced' config: {e_imb}. Trying default load.")
                  ds = load_dataset("clinc_oos") # Load whatever default is available
                  splits = list(ds.keys())
                  df = pd.concat([ds[split].to_pandas() for split in splits], ignore_index=True)


        # Normalize label column
        if "intent" not in df.columns and "label" in df.columns:
             feat = ds[splits[0]].features["label"]
             if isinstance(feat, ClassLabel): df["intent"] = df["label"].apply(feat.int2str)
             else: df["intent"] = df["label"].astype(str)
        elif "intent" not in df.columns: raise ValueError("Label/Intent column not found.")

        # Normalize text column
        if "text" not in df.columns:
             text_col = next((c for c in ["sentence", "utterance", "query"] if c in df.columns), None)
             if text_col: df.rename(columns={text_col: "text"}, inplace=True)
             else: raise ValueError("Text column not found.")

        df[["text", "intent"]].to_csv(csv_path, index=False, encoding="utf-8")
        print(f"OOS (CLINC150) 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 OOS (CLINC150) 다운로드 실패: {e}")
        raise ConnectionError("OOS (CLINC150) 다운로드 실패.")

def prepare_oos_dataset(data_dir=DATA_DIR):
    print("OOS (CLINC150) 데이터셋 준비 중...")
    try:
        # Using 'plus' config which includes out-of-scope samples often marked with a specific label
        csv_path = download_oos_dataset(data_dir, config="plus")
        texts, classes = load_csv_universal(csv_path, ['text', 'sentence', 'utterance', 'query'], ['intent', 'label'])
    except Exception as e:
        print(f"OOS (CLINC150) 데이터셋 준비 실패: {e}")
        return [], [], []

    # OOS samples are often labeled as 'oos' or a specific index. We need to identify them.
    # Let's assume the label 'oos' marks the out-of-scope samples.
    # We will map 'oos' to a special numeric label like -1 later in DataModule.setup if needed.
    # For now, just load all unique class names including 'oos'.
    unique_classes = sorted(list(set(classes)))
    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]

    print(f"OOS (CLINC150) 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스 (포함 가능성: 'oos')")
    # Return the original string names, DataModule will handle mapping 'oos' if necessary
    return texts, labels, unique_classes


# 8. StackOverflow
def download_stackoverflow_dataset(output_dir=DATA_DIR):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "stackoverflow.csv")
    if os.path.exists(csv_path): return csv_path

    try:
        print("StackOverflow 데이터셋 다운로드 중 (Hugging Face)...")
        # Try a larger, more standard one first if available
        try:
             ds = load_dataset("pvcnn/stackoverflow-questions-title-tags", split="train") # Example
             print("Loaded pvcnn/stackoverflow-questions-title-tags.")
             df = ds.to_pandas()
             if 'tags' not in df.columns and 'label' in df.columns: df.rename(columns={'label': 'tags'}, inplace=True)
             if 'text' not in df.columns and 'title' in df.columns: df.rename(columns={'title': 'text'}, inplace=True)

        except Exception:
             print("Failed loading pvcnn/stackoverflow, trying c17hawke/stackoverflow-dataset...")
             ds = load_dataset("c17hawke/stackoverflow-dataset", split="train")
             df = ds.to_pandas()
             # Combine title and body for 'text'
             df['text'] = df['title'].fillna('') + ' ' + df['body'].fillna('')
             df['text'] = df['text'].str.strip()
             if 'tags' not in df.columns: raise ValueError("Tags column missing in c17hawke.")

        # Ensure tags are strings
        if isinstance(df['tags'].iloc[0], list): df['tags'] = df['tags'].apply(lambda x: ' '.join(map(str, x)))
        else: df['tags'] = df['tags'].astype(str)

        df = df[['text', 'tags']].dropna(subset=['text']) # Keep relevant columns, drop rows with no text
        df.to_csv(csv_path, index=False, encoding="utf-8")
        print(f"StackOverflow 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path

    except Exception as e:
        print(f"Hugging Face에서 StackOverflow 다운로드 실패: {e}")
        raise ConnectionError("StackOverflow 다운로드 실패.")

def prepare_stackoverflow_dataset(data_dir=DATA_DIR):
    print("StackOverflow 데이터셋 준비 중...")
    try:
        csv_path = download_stackoverflow_dataset(data_dir)
        texts, classes = load_csv_universal(csv_path, ['text', 'title'], ['tags'])
        # StackOverflow often has many tags per question. We might need to handle this.
        # For simplicity, let's maybe take the first tag or treat it as multi-label?
        # For now, let's treat the full tag string as the class. DataModule might need adjustment.
    except Exception as e:
        print(f"StackOverflow 데이터셋 준비 실패: {e}")
        return [], [], []

    unique_classes = sorted(list(set(classes)))
    # Limit number of classes if too many unique tag combinations exist?
    MAX_SO_CLASSES = 100 # Example limit
    if len(unique_classes) > MAX_SO_CLASSES:
         print(f"경고: StackOverflow 태그 조합이 너무 많음 ({len(unique_classes)}). 상위 {MAX_SO_CLASSES}개만 사용합니다.")
         # Get most frequent tags
         tag_counts = pd.Series(classes).value_counts()
         top_tags = tag_counts.nlargest(MAX_SO_CLASSES).index.tolist()
         # Filter data to keep only top tags
         mask = pd.Series(classes).isin(top_tags)
         texts = [t for i, t in enumerate(texts) if mask[i]]
         classes = [c for i, c in enumerate(classes) if mask[i]]
         unique_classes = sorted(top_tags)

    class_to_idx = {cls: i for i, cls in enumerate(unique_classes)}
    labels = [class_to_idx[cls] for cls in classes]
    print(f"StackOverflow 데이터셋 준비 완료: {len(texts)} 샘플, {len(unique_classes)} 클래스")
    return texts, labels, unique_classes


# --- Add prepare functions for ATIS, SNIPS, FinancialPhraseBank, ArXiv10 ---
# --- following the pattern: call download, load csv, process labels ---

# 9. ATIS
def download_atis_dataset(output_dir=DATA_DIR):
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "atis.csv")
    if os.path.exists(csv_path): return csv_path
    try:
        print("ATIS 데이터셋 다운로드 중 (Hugging Face tuetschek/atis)...")
        ds = load_dataset("tuetschek/atis")
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)
        # Rename columns if necessary
        if 'text' not in df.columns and 'sentence' in df.columns: df.rename(columns={'sentence': 'text'}, inplace=True)
        if 'intent' not in df.columns and 'label' in df.columns:
             feat = ds[list(ds.keys())[0]].features["label"]
             if isinstance(feat, ClassLabel): df["intent"] = df["label"].apply(feat.int2str)
             else: df["intent"] = df["label"].astype(str)
        elif 'intent' not in df.columns: raise ValueError("Intent/Label column missing.")
        if 'text' not in df.columns: raise ValueError("Text/Sentence column missing.")

        df[['text', 'intent']].to_csv(csv_path, index=False, encoding='utf-8')
        print(f"ATIS 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 ATIS 다운로드 실패: {e}")
        raise ConnectionError("ATIS 다운로드 실패.")

def prepare_atis_dataset(data_dir=DATA_DIR):
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
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "snips.csv")
    if os.path.exists(csv_path): return csv_path
    try:
        print("SNIPS 데이터셋 다운로드 중 (Hugging Face snips_built_in_intents)...")
        ds = load_dataset("snips_built_in_intents")
        df = pd.concat([ds[split].to_pandas() for split in ds.keys()], ignore_index=True)
        if 'text' not in df.columns: raise ValueError("Text column missing.")
        if 'intent' not in df.columns and 'label' in df.columns:
             feat = ds[list(ds.keys())[0]].features["label"]
             if isinstance(feat, ClassLabel): df["intent"] = df["label"].apply(feat.int2str)
             else: df["intent"] = df["label"].astype(str)
        elif 'intent' not in df.columns: raise ValueError("Intent/Label column missing.")

        df[['text', 'intent']].to_csv(csv_path, index=False, encoding='utf-8')
        print(f"SNIPS 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 SNIPS 다운로드 실패: {e}")
        # Add fallback here if needed (e.g., GitHub JSON download from your code)
        raise ConnectionError("SNIPS 다운로드 실패.")

def prepare_snips_dataset(data_dir=DATA_DIR):
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
    ensure_dir(output_dir)
    csv_path = os.path.join(output_dir, "financial_phrasebank.csv")
    if os.path.exists(csv_path): return csv_path
    try:
        print("Financial PhraseBank 데이터셋 다운로드 중 (Hugging Face takala/financial_phrasebank)...")
        ds = load_dataset("takala/financial_phrasebank", "sentences_allagree", split="train") # Only train split available
        df = ds.to_pandas()
        if 'text' not in df.columns and 'sentence' in df.columns: df.rename(columns={'sentence': 'text'}, inplace=True)
        if 'sentiment' not in df.columns and 'label' in df.columns:
            label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
            df['sentiment'] = df['label'].map(label_map).fillna('unknown')
        elif 'sentiment' not in df.columns: raise ValueError("Sentiment/Label column missing.")
        if 'text' not in df.columns: raise ValueError("Text/Sentence column missing.")

        df[['text', 'sentiment']].to_csv(csv_path, index=False, encoding='utf-8')
        print(f"Financial PhraseBank 데이터가 '{csv_path}'에 저장되었습니다.")
        return csv_path
    except Exception as e:
        print(f"Hugging Face에서 Financial PhraseBank 다운로드 실패: {e}")
        raise ConnectionError("Financial PhraseBank 다운로드 실패.")

def prepare_financial_phrasebank_dataset(data_dir=DATA_DIR):
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

# 12. ArXiv-10맑# --- START OF dataset_utils.py (download_arxiv10_dataset fix only) ---
# ... (다른 임포트 및 함수들) ...

# 12. ArXiv-10
def download_arxiv10_dataset(output_dir=DATA_DIR):
    ensure_dir(output_dir)
    zip_url = "https://github.com/ashfarhangi/Protoformer/raw/main/data/ArXiv-10.zip"
    zip_path = os.path.join(output_dir, "arxiv10.zip")
    csv_path = os.path.join(output_dir, "arxiv10.csv") # Target CSV

    if os.path.exists(csv_path): return csv_path

    if not os.path.exists(zip_path):
        if not download_file(zip_url, zip_path, "ArXiv-10 ZIP"):
            raise ConnectionError("ArXiv-10 ZIP 다운로드 실패.")

    print(f"'{os.path.basename(zip_path)}' 압축 해제 중...")
    extracted_folder_name = "ArXiv-10" # Expected folder name inside zip
    extracted_folder_path = os.path.join(output_dir, extracted_folder_name)
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(output_dir)
        if not os.path.exists(extracted_folder_path):
             # Maybe extracted directly without the folder? Check for CSV.
             potential_csv = os.path.join(output_dir, "ArXiv-10.csv") # Or similar name
             if os.path.exists(potential_csv):
                  # Ensure target csv_path doesn't exist before moving
                  if os.path.exists(csv_path) and potential_csv != csv_path:
                       os.remove(csv_path)
                  shutil.move(potential_csv, csv_path) # Rename to target
                  print(f"ArXiv-10 데이터가 '{csv_path}'에 저장되었습니다.")
                  return csv_path
             else:
                  raise FileNotFoundError(f"압축 해제 후 '{extracted_folder_name}' 폴더 또는 관련 CSV를 찾을 수 없습니다.")

        # Find the CSV file within the extracted folder
        found_csv = None
        for root, _, files in os.walk(extracted_folder_path):
            for file in files:
                # Be more specific if possible, e.g., if the file is always named 'ArXiv-10.csv'
                if file.lower() == 'arxiv-10.csv': # Check for specific name first
                    found_csv = os.path.join(root, file)
                    break
                elif file.lower().endswith('.csv'): # Fallback to any CSV
                    found_csv = os.path.join(root, file)
                    # Don't break immediately on fallback, prefer specific name
            if found_csv and os.path.basename(found_csv).lower() == 'arxiv-10.csv':
                 break # Found the specific file, stop searching
            # If only a generic CSV found, continue searching other dirs just in case

        if not found_csv: raise FileNotFoundError("압축 해제된 폴더에서 ArXiv-10 CSV 파일을 찾을 수 없습니다.")

        # If found_csv is not the target csv_path, move it
        if os.path.abspath(found_csv) != os.path.abspath(csv_path):
             print(f"Moving found CSV '{found_csv}' to target '{csv_path}'")
             if os.path.exists(csv_path): os.remove(csv_path) # Remove target if exists
             shutil.move(found_csv, csv_path)
        else:
             print(f"Using found CSV directly: '{csv_path}'")


        # Read, process, and save to target CSV path
        df = pd.read_csv(csv_path) # Now read from the final csv_path
        # Normalize columns (adjust based on actual CSV columns)
        if 'text' not in df.columns and 'abstract' in df.columns: df.rename(columns={'abstract': 'text'}, inplace=True)
        elif 'text' not in df.columns and 'title' in df.columns: df['text'] = df['title'] # Use title if abstract missing
        if 'category' not in df.columns and 'label' in df.columns: df.rename(columns={'label': 'category'}, inplace=True)
        elif 'category' not in df.columns and 'class' in df.columns: df.rename(columns={'class': 'category'}, inplace=True)

        if 'text' not in df.columns or 'category' not in df.columns:
             print(f"ArXiv-10 CSV Columns: {df.columns.tolist()}") # Print columns for debugging
             raise ValueError("ArXiv-10 CSV에서 'text' 또는 'category' 열을 찾거나 생성할 수 없습니다.")

        # Overwrite the CSV with potentially renamed columns
        df[['text', 'category']].to_csv(csv_path, index=False, encoding='utf-8')
        print(f"ArXiv-10 데이터가 '{csv_path}'에 최종 저장되었습니다.")
        # Clean up extracted folder and zip
        if os.path.exists(extracted_folder_path):
             try: shutil.rmtree(extracted_folder_path)
             except Exception as e_rm: print(f"Warning: Could not remove extracted folder {extracted_folder_path}: {e_rm}")
        # os.remove(zip_path) # Keep the zip?
        return csv_path

    except zipfile.BadZipFile:
        # 수정된 부분: 각 동작을 별도의 줄로 분리
        print(f"오류: '{zip_path}' 파일이 손상되었습니다. 삭제 후 다시 시도하세요.")
        if os.path.exists(zip_path):
            os.remove(zip_path)
        raise
    except Exception as e:
        # 수정된 부분: 각 동작을 별도의 줄로 분리
        print(f"ArXiv-10 처리 오류: {e}")
        raise

# ... (파일의 나머지 부분) ...
# --- END OF dataset_utils.py (download_arxiv10_dataset fix only) ---
def prepare_arxiv10_dataset(data_dir=DATA_DIR):
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


# --- END OF dataset_utils.py ---