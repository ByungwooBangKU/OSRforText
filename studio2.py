# -*- coding: utf-8 -*-
import os
import sys
import json
import time
import math
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
from scipy.stats import weibull_min, norm # Added norm for DOC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm # 한글 글꼴 설정용 임포트
import seaborn as sns
import shutil # Added for cleaning up trial checkpoints

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, CSVLogger

from transformers import (
    RobertaModel, RobertaTokenizer, RobertaConfig,
    get_linear_schedule_with_warmup,
    logging as hf_logging
)

# Custom Modules
from hyperparameter_tuning import (
    OptunaHyperparameterTuner, load_best_params, get_default_best_params
)
# Import the specific prepare functions needed
from dataset_utils import (
    prepare_newsgroup20_dataset, prepare_bbc_news_dataset, prepare_trec_dataset,
    prepare_reuters8_dataset, prepare_acm_dataset, prepare_chemprot_dataset,
    prepare_banking77_dataset, prepare_oos_dataset, prepare_stackoverflow_dataset,
    prepare_atis_dataset, prepare_snips_dataset, prepare_financial_phrasebank_dataset,
    prepare_arxiv10_dataset, prepare_custom_syslog_dataset # Added custom_syslog
)

# --- Basic Setup ---
# Set default encoding (less common now, but can be useful)
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suppress excessive warnings from transformers/tokenizers
hf_logging.set_verbosity_error()
# Suppress PyTorch Lightning UserWarnings about processes/workers
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*DataLoader running processes.*")
warnings.filterwarnings("ignore", ".*Checkpoint directory.*exists but is not empty.*") # Ignore checkpoint exists warning

# --- Matplotlib Korean Font Setup ---
def setup_korean_font():
    font_name = None
    try:
        possible_fonts = ['Malgun Gothic', 'NanumGothic', 'Apple SD Gothic Neo', 'Noto Sans KR']
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        for font in possible_fonts:
            if font in available_fonts:
                font_name = font
                break
        if font_name:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False
            # print(f"Korean font '{font_name}' set for Matplotlib.") # Less verbose
        else:
            print("Warning: No common Korean font found. Install 'Malgun Gothic' or 'NanumGothic'. Plots might not display Korean characters correctly.")
            plt.rcParams['axes.unicode_minus'] = False
    except Exception as e:
        print(f"Error setting up Korean font: {e}")
        plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()


# =============================================================================
# Data Processing (TextDataset, DataModule)
# =============================================================================
class TextDataset(Dataset):
    """PyTorch Dataset for text classification."""
    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels # Can be original labels (including -1) or remapped (0..N-1)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        # Ensure text is a string, handle potential None or non-string types
        text = str(self.texts[idx]) if self.texts[idx] is not None else ""
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            # RoBERTa does not use token_type_ids, explicitly set to False
            return_token_type_ids=False,
            return_tensors='pt'
        )

        # Construct item, excluding token_type_ids
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

        return item

class DataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for handling datasets."""
    def __init__(
        self,
        dataset_name,
        tokenizer,
        batch_size=64,
        seen_class_ratio=0.5,
        random_seed=42,
        max_length=384,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        data_dir="data" # Add data_dir argument
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seen_class_ratio = seen_class_ratio
        self.random_seed = random_seed
        self.max_length = max_length
        self.data_dir = data_dir # Store data directory
        self.num_classes = None # Total number of original classes before split
        self.num_seen_classes = None # Number of classes used for training (Known classes)
        self.seen_classes = None # Indices of known classes (in original labeling, 0..N-1)
        self.unseen_classes = None # Indices of unknown classes (in original labeling) - Not directly used by models typically
        # self.label_encoder = LabelEncoder() # No longer needed here, handled in prepare_custom_syslog

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        # Validate and normalize split ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            print(f"Warning: Data split ratios sum to {total}. Normalizing...")
            self.train_ratio /= total
            self.val_ratio /= total
            self.test_ratio /= total
            print(f"Normalized ratios: train={self.train_ratio:.3f}, val={self.val_ratio:.3f}, test={self.test_ratio:.3f}")

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.class_names = None # List of string names for all original classes

        # Store original split info for OSR evaluation
        self.original_seen_indices = None # Original indices considered seen
        self.original_unseen_indices = None # Original indices considered unseen
        self.seen_class_mapping = None # Mapping from original seen index -> new index (0..num_seen-1)

        # Map dataset name to its prepare function
        self.prepare_func_map = {
            "newsgroup20": prepare_newsgroup20_dataset,
            "bbc_news": prepare_bbc_news_dataset,
            "trec": prepare_trec_dataset,
            "reuters8": prepare_reuters8_dataset,
            "acm": prepare_acm_dataset,
            "chemprot": prepare_chemprot_dataset,
            "banking77": prepare_banking77_dataset,
            "oos": prepare_oos_dataset,
            "stackoverflow": prepare_stackoverflow_dataset,
            "atis": prepare_atis_dataset,
            "snips": prepare_snips_dataset,
            "financial_phrasebank": prepare_financial_phrasebank_dataset,
            "arxiv10": prepare_arxiv10_dataset,
            "custom_syslog": prepare_custom_syslog_dataset, # Added custom_syslog
        }

    def prepare_data(self):
        """Calls the appropriate prepare function which handles download/processing."""
        print(f"Preparing data for dataset: {self.dataset_name}...")
        if self.dataset_name in self.prepare_func_map:
            try:
                # The prepare function now handles download and returns processed data path or data itself
                # We just need to ensure it runs once. The actual loading happens in setup.
                _ = self.prepare_func_map[self.dataset_name](data_dir=self.data_dir)
                print(f"{self.dataset_name} data preparation check complete.")
            except FileNotFoundError as e: # Catch specific error for syslog
                 if self.dataset_name == 'custom_syslog':
                      print(f"\n{'='*20} ACTION REQUIRED {'='*20}")
                      print(e) # Print the informative error message from prepare_custom_syslog
                      print(f"{'='*58}\n")
                 else:
                      print(f"Error during prepare_data for {self.dataset_name}: {e}")
                 sys.exit(1) # Exit if required file is missing
            except Exception as e:
                print(f"Error during prepare_data for {self.dataset_name}: {e}")
                import traceback
                traceback.print_exc()
                raise # Re-raise the exception to stop execution if preparation fails
        else:
            print(f"Warning: No specific prepare_data action defined for dataset '{self.dataset_name}'. Assuming data exists or is handled by setup.")


    def setup(self, stage=None):
        """Loads and splits data. Called on every GPU."""
        if self.train_dataset is not None and stage == 'fit': return # Avoid redundant setup
        if self.test_dataset is not None and stage == 'test': return

        # Set seeds for reproducibility within setup
        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        print(f"\n--- Setting up DataModule for dataset: {self.dataset_name} (Seen Ratio: {self.seen_class_ratio}) ---")

        # --- Load Data using the mapped prepare function ---
        if self.dataset_name in self.prepare_func_map:
            print(f"Loading data using prepare_{self.dataset_name}_dataset...")
            try:
                # ******** 수정된 부분 ********
                # prepare_* 함수는 이제 (texts, labels, class_names) 3개의 값을 반환합니다.
                texts, labels, self.class_names = self.prepare_func_map[self.dataset_name](data_dir=self.data_dir)
                # ***************************
            except Exception as e:
                 print(f"Failed to load data for {self.dataset_name}: {e}")
                 raise ValueError(f"Data loading failed for {self.dataset_name}") from e
        else:
            raise ValueError(f"Unknown or unprepared dataset: {self.dataset_name}")

        if not texts:
             raise ValueError(f"Failed to load any text data for dataset '{self.dataset_name}'. Please check the loading function and data source.")

        # Ensure labels are numpy array of integers
        labels = np.array(labels, dtype=int)
        if labels.ndim == 0 or len(labels) == 0:
             raise ValueError(f"Loaded labels for {self.dataset_name} are invalid or empty.")
        if len(texts) != len(labels):
             raise ValueError(f"Mismatch between number of texts ({len(texts)}) and labels ({len(labels)}) for {self.dataset_name}.")

        # --- Train/Val/Test Split ---
        # (이 부분은 이전 코드와 동일하게 유지됩니다. 이제 texts와 labels 변수에
        #  올바르게 전체 데이터가 로드되었으므로 train_test_split이 정상 작동합니다.)
        print("Splitting data into train/validation/test sets...")
        min_samples_per_class = 2
        unique_labels_split, counts = np.unique(labels, return_counts=True)
        if np.any(counts < min_samples_per_class):
            print(f"Warning: Some classes have fewer than {min_samples_per_class} samples. Stratification might fail or be unreliable.")
            stratify_param = None
        else:
            stratify_param = labels

        try:
            X_train_val, X_test, y_train_val, y_test = train_test_split(
                texts, labels,
                test_size=self.test_ratio,
                random_state=self.random_seed,
                stratify=stratify_param
            )
            if self.train_ratio + self.val_ratio <= 1e-6:
                 val_size_relative = 0
                 X_train, X_val, y_train, y_val = X_train_val, [], y_train_val, []
            else:
                 val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
                 unique_labels_train_val, counts_train_val = np.unique(y_train_val, return_counts=True)
                 if np.any(counts_train_val < min_samples_per_class):
                      print("Warning: Stratification for train/val split might be unreliable.")
                      stratify_train_val = None
                 else:
                      stratify_train_val = y_train_val

                 X_train, X_val, y_train, y_val = train_test_split(
                     X_train_val, y_train_val,
                     test_size=val_size_relative,
                     random_state=self.random_seed,
                     stratify=stratify_train_val
                 )
        except ValueError as e:
             print(f"Error during train/test split (possibly due to stratification issues): {e}")
             print("Attempting split without stratification...")
             X_train_val, X_test, y_train_val, y_test = train_test_split(
                 texts, labels, test_size=self.test_ratio, random_state=self.random_seed)
             if self.train_ratio + self.val_ratio <= 1e-6: val_size_relative = 0
             else: val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
             X_train, X_val, y_train, y_val = train_test_split(
                 X_train_val, y_train_val, test_size=val_size_relative, random_state=self.random_seed)


        # Get all unique original class labels (0..N-1) from the *initial* loaded labels
        all_original_indices = np.unique(labels) # These are already 0..N-1 integers
        self.num_classes = len(all_original_indices)
        print(f"Total original classes found: {self.num_classes} -> {all_original_indices}")
        if self.class_names is None: # Should be set by specific loaders
             self.class_names = [str(i) for i in all_original_indices]
             print(f"Warning: class_names not explicitly set. Using: {self.class_names}")
        elif len(self.class_names) != self.num_classes:
             print(f"Warning: Mismatch between number of class names ({len(self.class_names)}) and unique labels ({self.num_classes}). Adjusting names.")
             # Attempt to align names with actual indices found
             self.class_names = [self.class_names[i] if i < len(self.class_names) else f"Class_{i}" for i in all_original_indices]


        # --- Seen/Unseen Class Split Logic ---
        # (이 부분은 이전 코드와 동일하게 유지됩니다.)
        if self.seen_class_ratio < 1.0:
            print(f"Splitting classes: {self.seen_class_ratio*100:.1f}% Seen / {(1-self.seen_class_ratio)*100:.1f}% Unseen")
            num_seen = max(1, int(np.round(self.num_classes * self.seen_class_ratio)))
            if num_seen >= self.num_classes:
                 print("Warning: Calculated num_seen >= total classes. Setting seen_class_ratio to 1.0.")
                 self.seen_class_ratio = 1.0

        if self.seen_class_ratio < 1.0:
            np.random.seed(self.random_seed)
            all_classes_shuffled = np.random.permutation(all_original_indices)

            self.original_seen_indices = np.sort(all_classes_shuffled[:num_seen])
            self.original_unseen_indices = np.sort(all_classes_shuffled[num_seen:])

            print(f"  Original Seen Indices (0..Total-1): {self.original_seen_indices}")
            print(f"  Original Unseen Indices (0..Total-1): {self.original_unseen_indices}")

            self.seen_class_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.original_seen_indices)}
            self.num_seen_classes = len(self.original_seen_indices)
            self.seen_classes = np.arange(self.num_seen_classes)

            train_seen_mask = np.isin(y_train, self.original_seen_indices)
            X_train = [X_train[i] for i, keep in enumerate(train_seen_mask) if keep]
            y_train_original_kept = y_train[train_seen_mask]
            y_train_mapped = np.array([self.seen_class_mapping[lbl] for lbl in y_train_original_kept])

            val_seen_mask = np.isin(y_val, self.original_seen_indices)
            X_val = [X_val[i] for i, keep in enumerate(val_seen_mask) if keep]
            y_val_original_kept = y_val[val_seen_mask]
            y_val_mapped = np.array([self.seen_class_mapping[lbl] for lbl in y_val_original_kept])

            y_test_final = y_test.copy().astype(int)
            unseen_test_mask = np.isin(y_test, self.original_unseen_indices)
            y_test_final[unseen_test_mask] = -1

            y_train_final = y_train_mapped
            y_val_final = y_val_mapped

        else: # seen_class_ratio == 1.0
            print("All classes are considered Known (seen_class_ratio = 1.0)")
            self.original_seen_indices = all_original_indices.copy()
            self.original_unseen_indices = np.array([])
            self.num_seen_classes = self.num_classes
            self.seen_classes = all_original_indices.copy()
            self.seen_class_mapping = {orig_idx: orig_idx for orig_idx in all_original_indices}

            y_train_final = y_train
            y_val_final = y_val
            y_test_final = y_test

        # --- Create TextDataset instances ---
        self.train_dataset = TextDataset(list(X_train), y_train_final, self.tokenizer, self.max_length)
        self.val_dataset = TextDataset(list(X_val), y_val_final, self.tokenizer, self.max_length)
        self.test_dataset = TextDataset(list(X_test), y_test_final, self.tokenizer, self.max_length)

        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Validation: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        print(f"Number of classes for model training (Known): {self.num_seen_classes}")
        if self.seen_class_ratio < 1.0:
             num_unseen_in_test = np.sum(y_test_final == -1)
             print(f"Number of original unseen classes: {len(self.original_unseen_indices)}")
             print(f"Number of samples marked as Unknown (-1) in test set: {num_unseen_in_test}")
        if self.class_names and self.original_seen_indices is not None:
             try:
                  seen_names_list = [self.class_names[i] for i in self.original_seen_indices]
                  print(f"Known class names (original indices {self.original_seen_indices}): {seen_names_list}")
             except IndexError:
                  print(f"Warning: Could not map all original_seen_indices ({self.original_seen_indices}) to class_names (len={len(self.class_names)}).")
                  print(f"Using indices as names for seen classes.")
        print("--- Finished DataModule setup ---")
        
    def train_dataloader(self):
        if self.train_dataset is None: raise ValueError("Train dataset not initialized.")
        # Adjust num_workers based on CPU count, limit to avoid resource exhaustion
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0)
        persistent = num_workers > 0
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=num_workers, persistent_workers=persistent, pin_memory=True)

    def val_dataloader(self):
        if self.val_dataset is None: raise ValueError("Validation dataset not initialized.")
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0)
        persistent = num_workers > 0
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=num_workers, persistent_workers=persistent, pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is None: raise ValueError("Test dataset not initialized.")
        num_workers = min(4, os.cpu_count() // 2 if os.cpu_count() and os.cpu_count() > 1 else 0)
        persistent = num_workers > 0
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=num_workers, persistent_workers=persistent, pin_memory=True)

    def _determine_unknown_labels(self, labels_np):
        """Helper to consistently determine true unknown labels for evaluation based on setup."""
        # Test dataset labels are already marked with -1 if they belong to an original unseen class.
        unknown_mask = (np.array(labels_np) == -1)
        return unknown_mask

# =============================================================================
# Model Definitions (Remain Unchanged - Inherited from previous response)
# =============================================================================
class RobertaClassifier(pl.LightningModule):
    """Standard RoBERTa model with a classification head."""
    def __init__(
        self,
        model_name="roberta-base",
        num_classes=20, # Number of *known* classes for the classifier
        learning_rate=2e-5, # Renamed from lr for consistency
        weight_decay=0.01,
        warmup_steps=0,
        total_steps=0
    ):
        super().__init__()
        # Use save_hyperparameters() correctly
        self.save_hyperparameters(ignore=['model_name']) # Saves num_classes, learning_rate, etc.
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.hparams.num_classes) # Use hparams
        # Store scheduler params if needed later, though configure_optimizers uses hparams
        # self.learning_rate = learning_rate # Already saved in hparams
        # self.weight_decay = weight_decay # Already saved in hparams
        self.warmup_steps = warmup_steps # Store for configure_optimizers
        self.total_steps = total_steps   # Store for configure_optimizers
        self.num_classes = self.hparams.num_classes # Explicitly set for OSRAlgorithm use

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass returning logits and CLS token embedding."""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids # Will be None if tokenizer doesn't provide it
        )
        cls_output = outputs.last_hidden_state[:, 0, :] # [CLS] embedding
        logits = self.classifier(cls_output)
        return logits, cls_output

    def training_step(self, batch, batch_idx):
        """Single training step."""
        # Assumes labels are 0..N-1 mapped known classes
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are not -1 during training
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None # Skip batch if no valid labels

        loss = F.cross_entropy(valid_logits, valid_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Validation set should also only contain known classes (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {'val_loss': torch.tensor(0.0), 'val_acc': torch.tensor(0.0)}

        loss = F.cross_entropy(valid_logits, valid_labels)
        preds = torch.argmax(valid_logits, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        """Single test step."""
        logits, embeddings = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels_original = batch['label'] # Original labels (can include -1 and original seen indices 0..Total-1)
        preds_mapped = torch.argmax(logits, dim=1) # Predictions indices (0..num_seen-1)

        # Calculate loss & acc only on known samples (where label >= 0)
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             # Map true known labels (original indices 0..Total-1) to the model's output space (0..num_seen-1) for loss/acc
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 # Use the mapping stored in the datamodule
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 loss = F.cross_entropy(logits[known_mask], mapped_true_labels_known)
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size) # Log accuracy on knowns
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")
                 # Handle error, maybe log or skip accuracy calculation for this batch


        return {
            'preds_mapped': preds_mapped, # Predictions indices (0..num_seen-1) for all samples
            'labels_original': labels_original,    # Original labels (can include -1 and original seen indices 0..Total-1)
            'logits': logits,             # Raw logits (size num_seen_classes)
            'embeddings': embeddings      # CLS embeddings
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        # Ensure total_steps is calculated correctly before scheduler creation
        if self.total_steps <= 0:
             print("Warning: total_steps not set for scheduler. Estimating from trainer.")
             try:
                 self.total_steps = self.trainer.estimated_stepping_batches
                 if self.total_steps is None or self.total_steps <= 0: # Check if estimation failed
                      self.total_steps = 10000 # Fallback
             except Exception:
                 self.total_steps = 10000 # Fallback if trainer not available yet
             print(f"Using estimated/fallback total_steps: {self.total_steps}")

        # Ensure warmup_steps doesn't exceed total_steps
        actual_warmup_steps = min(self.warmup_steps, self.total_steps)
        if actual_warmup_steps != self.warmup_steps:
             print(f"Warning: Adjusted warmup_steps from {self.warmup_steps} to {actual_warmup_steps}")

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class RobertaAutoencoder(pl.LightningModule):
    """RoBERTa-based Autoencoder for CROSR."""
    def __init__( self, model_name="roberta-base", num_classes=20, learning_rate=2e-5, weight_decay=0.01,
                  warmup_steps=0, total_steps=0, latent_dim=256, reconstruction_weight=0.5 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_name'])
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, self.hparams.num_classes)
        # Encoder/Decoder Layers
        self.encoder = nn.Sequential( nn.Linear(self.config.hidden_size, self.hparams.latent_dim), nn.ReLU() )
        self.decoder = nn.Sequential( nn.Linear(self.hparams.latent_dim, self.config.hidden_size) )
        # Store scheduler params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = self.hparams.num_classes # For OSRAlgorithm use

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        encoded = self.encoder(cls_output)
        reconstructed = self.decoder(encoded)
        return logits, cls_output, encoded, reconstructed

    def training_step(self, batch, batch_idx):
        logits, cls_output, _, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        valid_cls_output = cls_output[batch['label'] >= 0]
        valid_reconstructed = reconstructed[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None

        class_loss = F.cross_entropy(valid_logits, valid_labels)
        recon_loss = F.mse_loss(valid_reconstructed, valid_cls_output)
        loss = class_loss + self.hparams.reconstruction_weight * recon_loss
        self.log_dict({'train_loss': loss, 'train_class_loss': class_loss, 'train_recon_loss': recon_loss},
                      prog_bar=False, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, cls_output, _, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        valid_cls_output = cls_output[batch['label'] >= 0]
        valid_reconstructed = reconstructed[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {'val_loss': torch.tensor(0.0), 'val_acc': torch.tensor(0.0)}

        class_loss = F.cross_entropy(valid_logits, valid_labels)
        recon_loss = F.mse_loss(valid_reconstructed, valid_cls_output)
        loss = class_loss + self.hparams.reconstruction_weight * recon_loss
        preds = torch.argmax(valid_logits, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log_dict({'val_loss': loss, 'val_class_loss': class_loss, 'val_recon_loss': recon_loss, 'val_acc': acc},
                       prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        logits, cls_output, encoded, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels_original = batch['label'] # Original labels (-1 possible, original seen indices 0..Total-1)
        preds_mapped = torch.argmax(logits, dim=1) # Predictions (0..num_seen-1)
        recon_errors = torch.norm(reconstructed - cls_output, p=2, dim=1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 class_loss = F.cross_entropy(logits[known_mask], mapped_true_labels_known)
                 recon_loss = F.mse_loss(reconstructed[known_mask], cls_output[known_mask])
                 loss = class_loss + self.hparams.reconstruction_weight * recon_loss
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")

        return {
            'preds_mapped': preds_mapped, 'labels_original': labels_original, 'logits': logits,
            'embeddings': cls_output, 'encoded': encoded, 'reconstructed': reconstructed,
            'recon_errors': recon_errors # Crucial for CROSR evaluation
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if self.total_steps <= 0:
             print("Warning: total_steps not set for scheduler. Estimating from trainer.")
             try: self.total_steps = self.trainer.estimated_stepping_batches or 10000
             except Exception: self.total_steps = 10000
             print(f"Using estimated/fallback total_steps: {self.total_steps}")
        actual_warmup_steps = min(self.warmup_steps, self.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}


class DOCRobertaClassifier(pl.LightningModule):
    """RoBERTa model adapted for DOC (one-vs-rest binary classifiers)."""
    def __init__( self, model_name="roberta-base", num_classes=20, learning_rate=2e-5, weight_decay=0.01,
                  warmup_steps=0, total_steps=0 ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_name'])
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        # One binary classifier head per known class
        self.classifiers = nn.ModuleList([nn.Linear(self.config.hidden_size, 1) for _ in range(self.hparams.num_classes)])
        # Store scheduler params
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = self.hparams.num_classes # For OSRAlgorithm use

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply each binary classifier and concatenate logits
        logits = torch.cat([clf(cls_output) for clf in self.classifiers], dim=1) # Shape: (batch_size, num_classes)
        return logits, cls_output

    def training_step(self, batch, batch_idx):
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None

        # Create multi-label binary targets (one-hot)
        one_hot_labels = F.one_hot(valid_labels, num_classes=self.hparams.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(valid_logits, one_hot_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_logits = logits[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {'val_loss': torch.tensor(0.0), 'val_acc': torch.tensor(0.0)}

        one_hot_labels = F.one_hot(valid_labels, num_classes=self.hparams.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(valid_logits, one_hot_labels)
        # Predict based on highest sigmoid score
        sigmoid_scores = torch.sigmoid(valid_logits)
        preds = torch.argmax(sigmoid_scores, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        logits, embeddings = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels_original = batch['label'] # Original labels (-1 possible, original seen indices 0..Total-1)
        sigmoid_scores = torch.sigmoid(logits)
        preds_mapped = torch.argmax(sigmoid_scores, dim=1) # Predicted class index (0..num_seen-1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 one_hot_labels = F.one_hot(mapped_true_labels_known, num_classes=self.hparams.num_classes).float()
                 loss = F.binary_cross_entropy_with_logits(logits[known_mask], one_hot_labels)
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")

        return {
            'preds_mapped': preds_mapped,           # Predicted class index (0..num_seen-1)
            'labels_original': labels_original,     # Original labels
            'logits': logits,                       # Raw one-vs-rest logits
            'sigmoid_scores': sigmoid_scores,       # Sigmoid scores (crucial for DOC eval)
            'embeddings': embeddings                # CLS embeddings
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        if self.total_steps <= 0:
             print("Warning: total_steps not set for scheduler. Estimating from trainer.")
             try: self.total_steps = self.trainer.estimated_stepping_batches or 10000
             except Exception: self.total_steps = 10000
             print(f"Using estimated/fallback total_steps: {self.total_steps}")
        actual_warmup_steps = min(self.warmup_steps, self.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class RobertaADB(pl.LightningModule):
    """RoBERTa model with Adaptive Decision Boundary (ADB) components."""
    def __init__( self,
                  model_name: str = "roberta-base",
                  num_classes: int = 20,
                  lr: float = 2e-5, # Backbone fine-tuning LR (from args.lr)
                  lr_adb: float = 5e-4, # ADB parameter LR (from args.lr_adb)
                  weight_decay: float = 0.0,
                  warmup_steps: int = 0, # For backbone scheduler
                  total_steps: int = 0,  # For backbone scheduler
                  param_adb_delta: float = 0.1, # Renamed to match arg name
                  param_adb_alpha: float = 0.5, # Renamed to match arg name
                  adb_freeze_backbone: bool = True ): # Renamed to match arg name
        super().__init__()
        # Save hyperparameters passed during initialization
        self.save_hyperparameters(ignore=['model_name']) # Saves num_classes, lr, lr_adb, etc.
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.feat_dim = self.config.hidden_size

        # Learnable centers (initialized using hparams)
        self.centers = nn.Parameter(torch.empty(self.hparams.num_classes, self.feat_dim), requires_grad=True)
        nn.init.normal_(self.centers, std=0.05)

        # Learnable Logits for Radii (Delta' in paper)
        initial_delta_prime = -1.5 # Initial value aiming for radius ~0.2
        self.delta_prime = nn.Parameter(torch.full((self.hparams.num_classes,), initial_delta_prime), requires_grad=True)

        # Store scheduler params (needed by configure_optimizers)
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = self.hparams.num_classes # For OSRAlgorithm use

        # Freeze backbone based on hparams
        if self.hparams.adb_freeze_backbone:
            print("[RobertaADB Init] Freezing RoBERTa backbone parameters.")
            for param in self.roberta.parameters(): param.requires_grad = False
        else: print("[RobertaADB Init] RoBERTa backbone parameters will be fine-tuned.")

    def get_radii(self):
        """Calculate actual radii using Softplus."""
        return F.softplus(self.delta_prime)

    @staticmethod
    def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates pairwise cosine distance (1 - similarity). Input x assumed normalized."""
        # Ensure y is normalized for cosine distance calculation
        y_norm = F.normalize(y, p=2, dim=-1)
        similarity = torch.matmul(x, y_norm.t())
        similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7)
        distance = 1.0 - similarity
        return distance

    def adb_margin_loss(self, feat_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates ADB margin loss: max(0, d(feat, c_y) - r_y + delta)."""
        distances = self._cosine_distance(feat_norm, self.centers) # feat_norm is normalized
        # Get distances corresponding to the true labels
        d_y = distances.gather(1, labels.unsqueeze(1)).squeeze(1) # More robust way

        radii = self.get_radii() # Get positive radii via Softplus
        r_y = radii[labels] # Get radii for the true labels

        # Loss = max(0, distance_to_correct_center - radius_of_correct_center + margin)
        # Use hparams for delta
        loss_per_sample = torch.relu(d_y - r_y + self.hparams.param_adb_delta)
        loss = loss_per_sample.mean()
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass: get CLS embedding, normalize it, calculate similarity logits."""
        # Control gradient flow based on freeze_backbone hparam
        with torch.set_grad_enabled(not self.hparams.adb_freeze_backbone):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        feat = outputs.last_hidden_state[:, 0, :]
        feat_norm = F.normalize(feat, p=2, dim=-1)
        # Logits are similarity scores (1 - cosine distance)
        logits = 1.0 - self._cosine_distance(feat_norm, self.centers)
        return logits, feat_norm

    def training_step(self, batch, batch_idx):
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_input_ids = batch["input_ids"][batch['label'] >= 0]
        valid_attention_mask = batch["attention_mask"][batch['label'] >= 0]
        valid_token_type_ids = batch.get("token_type_ids")
        if valid_token_type_ids is not None:
            valid_token_type_ids = valid_token_type_ids[batch['label'] >= 0]
        if valid_labels.numel() == 0: return None

        logits, feat_norm = self(valid_input_ids, valid_attention_mask, valid_token_type_ids)
        ce_loss = F.cross_entropy(logits, valid_labels)
        adb_loss = self.adb_margin_loss(feat_norm, valid_labels)
        # Use hparams for alpha
        loss = ce_loss + self.hparams.param_adb_alpha * adb_loss
        self.log_dict({'train_loss': loss, 'train_ce_loss': ce_loss, 'train_adb_loss': adb_loss},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        # Log actual average radius
        avg_radius = self.get_radii().mean().item()
        self.log("avg_radius", avg_radius, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # Ensure labels are valid (0..N-1)
        valid_labels = batch['label'][batch['label'] >= 0]
        valid_input_ids = batch["input_ids"][batch['label'] >= 0]
        valid_attention_mask = batch["attention_mask"][batch['label'] >= 0]
        valid_token_type_ids = batch.get("token_type_ids")
        if valid_token_type_ids is not None:
            valid_token_type_ids = valid_token_type_ids[batch['label'] >= 0]
        if valid_labels.numel() == 0: return {"val_loss": torch.tensor(0.0), "val_acc": torch.tensor(0.0)}

        logits, feat_norm = self(valid_input_ids, valid_attention_mask, valid_token_type_ids)
        ce_loss = F.cross_entropy(logits, valid_labels)
        adb_loss = self.adb_margin_loss(feat_norm, valid_labels)
        loss = ce_loss + self.hparams.param_adb_alpha * adb_loss
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(valid_labels.cpu(), preds.cpu())
        self.log_dict({'val_loss': loss, 'val_ce_loss': ce_loss, 'val_adb_loss': adb_loss, 'val_acc': acc},
                      prog_bar=True, on_step=False, on_epoch=True, logger=True, batch_size=self.trainer.datamodule.batch_size)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        labels_original = batch["label"] # Original labels (-1 possible, original seen indices 0..Total-1)
        logits, feat_norm = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        preds_mapped = torch.argmax(logits, dim=1) # Predictions based on highest similarity (0..num_seen-1)

        # Calculate loss/acc on knowns only
        known_mask = labels_original >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             original_indices_known = labels_original[known_mask].cpu().numpy()
             try:
                 mapped_true_labels_known = torch.tensor([self.trainer.datamodule.seen_class_mapping[idx] for idx in original_indices_known], device=self.device)
                 ce_loss = F.cross_entropy(logits[known_mask], mapped_true_labels_known)
                 adb_loss = self.adb_margin_loss(feat_norm[known_mask], mapped_true_labels_known)
                 loss = ce_loss + self.hparams.param_adb_alpha * adb_loss
                 acc = accuracy_score(mapped_true_labels_known.cpu(), preds_mapped[known_mask].cpu())
                 self.log('test_loss', loss, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
                 self.log('test_acc_known', acc, prog_bar=False, logger=True, batch_size=self.trainer.datamodule.batch_size)
             except KeyError as e:
                 print(f"Error: Test label {e} not found in seen_class_mapping during test_step.")

        return {
            'preds_mapped': preds_mapped,        # Predicted class index (0..num_seen-1) based on similarity
            'labels_original': labels_original,  # Original labels
            'logits': logits,                    # Similarity-based logits
            'features': feat_norm                # Normalized features (crucial for ADB eval)
        }

    def configure_optimizers(self):
        """Configure optimizer(s) and scheduler(s)."""
        params_to_optimize = []
        if not self.hparams.adb_freeze_backbone:
             # Use the general 'lr' hyperparameter for the backbone
             backbone_lr = self.hparams.lr
             params_to_optimize.append({'params': self.roberta.parameters(), 'lr': backbone_lr})
             print(f"[ADB Optim] Fine-tuning RoBERTa with LR: {backbone_lr}")

        # Optimize centers and delta_prime with the specific ADB LR ('lr_adb' hyperparameter)
        adb_param_lr = self.hparams.lr_adb
        params_to_optimize.append({'params': self.centers, 'lr': adb_param_lr})
        params_to_optimize.append({'params': self.delta_prime, 'lr': adb_param_lr})
        print(f"[ADB Optim] Optimizing centers/delta_prime with LR: {adb_param_lr}")

        optimizer = AdamW(params_to_optimize, weight_decay=self.hparams.weight_decay)

        # Add scheduler only if backbone is being trained and total_steps is valid
        if not self.hparams.adb_freeze_backbone and self.total_steps > 0:
             if self.total_steps <= 0:
                 print("Warning: total_steps not set for scheduler. Estimating from trainer.")
                 try: self.total_steps = self.trainer.estimated_stepping_batches or 10000
                 except Exception: self.total_steps = 10000
                 print(f"Using estimated/fallback total_steps: {self.total_steps}")

             actual_warmup_steps = min(self.warmup_steps, self.total_steps)
             scheduler = get_linear_schedule_with_warmup(
                 optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
             )
             print("[ADB Optim] Using learning rate scheduler for backbone.")
             return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        else:
             print("[ADB Optim] No learning rate scheduler used (backbone frozen or total_steps=0).")
             return optimizer

# =============================================================================
# OSR Algorithms (Remain Unchanged - Inherited from previous response)
# =============================================================================

class OSRAlgorithm:
    """Base class for Open Set Recognition algorithms."""
    def __init__(self, model, datamodule, args):
        self.model = model
        self.datamodule = datamodule
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() and args.gpu_id >= 0 else 'cpu')
        # Determine the number of known classes the model was trained on
        self.num_known_classes = model.num_classes if hasattr(model, 'num_classes') else datamodule.num_seen_classes
        if self.num_known_classes is None:
             raise ValueError("Could not determine number of known classes for OSR algorithm.")
        print(f"[{self.__class__.__name__}] Initialized for {self.num_known_classes} known classes.")

    def predict(self, dataloader):
        """Predicts labels, including potential 'unknown' (-1)."""
        raise NotImplementedError("Predict method must be implemented by subclass.")

    def evaluate(self, dataloader):
        """Evaluates OSR performance on the dataloader."""
        raise NotImplementedError("Evaluate method must be implemented by subclass.")

    def visualize(self, results):
        """Visualizes the OSR evaluation results."""
        raise NotImplementedError("Visualize method must be implemented by subclass.")

    def _get_seen_class_names(self):
         """Helper to get string names for seen classes based on original indices."""
         if self.datamodule.class_names is None or self.datamodule.original_seen_indices is None:
              # Fallback if names/indices aren't set
              print("Warning: Class names or seen indices not available in datamodule. Using generic names.")
              return {i: f"Known_{i}" for i in range(self.num_known_classes)}

         seen_names = {}
         original_seen_indices = sorted(list(self.datamodule.original_seen_indices)) # These are 0..Total-1 indices
         for original_idx in original_seen_indices:
              if 0 <= original_idx < len(self.datamodule.class_names):
                   # The key should be the original index for CM labels
                   seen_names[original_idx] = self.datamodule.class_names[original_idx]
              else:
                   print(f"Warning: Original index {original_idx} out of bounds for class names list (len={len(self.datamodule.class_names)}).")
                   seen_names[original_idx] = f"Class_{original_idx}" # Fallback name
         return seen_names

    def _get_cm_labels(self):
         """Gets integer labels and string names for confusion matrix axes."""
         seen_class_names_map = self._get_seen_class_names()
         # CM labels should include -1 for Unknown and the *original* indices of seen classes (0..Total-1)
         cm_axis_labels_int = [-1] + sorted(list(self.datamodule.original_seen_indices))
         cm_axis_labels_names = ["Unknown"] + [seen_class_names_map.get(lbl, str(lbl)) for lbl in cm_axis_labels_int if lbl != -1]
         return cm_axis_labels_int, cm_axis_labels_names

    def _map_preds_to_original(self, preds_mapped_batch):
        """Maps model's output indices (0..num_seen-1) back to original dataset indices (0..Total-1)."""
        original_seen_indices = self.datamodule.original_seen_indices # These are the original indices chosen as seen
        if original_seen_indices is None:
            print("Warning: original_seen_indices not found in datamodule. Returning mapped indices.")
            return preds_mapped_batch # Fallback

        # Ensure original_indices is a numpy array for efficient indexing
        if not isinstance(original_seen_indices, np.ndarray):
            original_seen_indices = np.array(original_seen_indices)

        # Create a result array initialized with a placeholder (e.g., -999)
        preds_original_batch = np.full_like(preds_mapped_batch, -999, dtype=int)

        for i, mapped_idx in enumerate(preds_mapped_batch):
            # mapped_idx is the prediction in the 0..num_seen-1 range
            if 0 <= mapped_idx < len(original_seen_indices):
                # Use mapped_idx to look up the corresponding original index
                preds_original_batch[i] = original_seen_indices[mapped_idx]
            else:
                # This case should ideally not happen if model predicts 0..num_seen-1
                print(f"Warning: Predicted mapped index {mapped_idx} out of bounds for original_seen_indices (len={len(original_seen_indices)}).")
                preds_original_batch[i] = -1 # Assign to unknown if mapping fails

        return preds_original_batch


class ThresholdOSR(OSRAlgorithm):
    """OSR using a simple threshold on the maximum softmax probability."""
    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        if not isinstance(model, RobertaClassifier):
             print(f"Warning: ThresholdOSR typically uses RobertaClassifier, but received {type(model)}. Proceeding anyway.")
        # Get threshold from args, fallback handled by get_default_best_params or argparse default
        self.threshold = getattr(args, 'param_threshold', 0.5) # Use default 0.5 if not set
        print(f"[ThresholdOSR Init] Using softmax threshold: {self.threshold:.4f}")

    def predict(self, dataloader):
        self.model.eval().to(self.device) # 모델 GPU 이동
        all_max_probs = []
        all_probs = []
        all_preds_final = []
        all_labels_original = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (Threshold OSR)"):
                # ******** 수정: 데이터 GPU 이동 ********
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None: token_type_ids = token_type_ids.to(self.device)
                labels_orig = batch['label'].cpu().numpy()
                # *************************************

                # forward 직접 호출 (test_step 대신)
                logits, _ = self.model(input_ids, attention_mask, token_type_ids)
                preds_mapped = torch.argmax(logits, dim=1) # Model predictions (0..num_seen-1)

                probs = F.softmax(logits, dim=1)
                max_probs, _ = torch.max(probs, dim=1)

                preds_mapped_cpu = preds_mapped.cpu().numpy()
                max_probs_cpu = max_probs.cpu().numpy()
                final_batch_preds = np.full_like(preds_mapped_cpu, -1, dtype=int)

                accept_mask = max_probs_cpu >= self.threshold
                if np.any(accept_mask):
                     accepted_mapped_indices = preds_mapped_cpu[accept_mask]
                     original_accepted_indices = self._map_preds_to_original(accepted_mapped_indices)
                     final_batch_preds[accept_mask] = original_accepted_indices

                all_max_probs.append(max_probs_cpu)
                all_probs.append(probs.cpu().numpy())
                all_preds_final.extend(final_batch_preds)
                all_labels_original.extend(labels_orig)

        all_max_probs = np.concatenate(all_max_probs) if all_max_probs else np.array([])
        all_probs = np.concatenate(all_probs) if all_probs else np.array([])
        all_preds_final = np.array(all_preds_final)
        all_labels_original = np.array(all_labels_original)
        return all_probs, all_preds_final, all_labels_original, all_max_probs

    def evaluate(self, dataloader):
        all_probs, all_preds, all_labels, all_max_probs = self.predict(dataloader)
        if len(all_labels) == 0: print("Warning: No data to evaluate for ThresholdOSR."); return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0}

        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels) # True where label is -1
        unknown_preds_mask = (all_preds == -1) # True where prediction is -1
        known_mask = ~unknown_labels_mask # True where label is an original index (0..Total-1)

        # Accuracy on known samples: Compare predicted original index with true original index
        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0

        # Unknown Detection Rate: Correctly predicted -1 / Total true -1
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0

        # AUROC: Use negative max probability (lower prob -> more likely unknown)
        auroc = roc_auc_score(unknown_labels_mask, -all_max_probs) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        # CM and F1: Use labels mapped for CM (True labels are -1 or original indices 0..Total-1)
        labels_mapped_for_cm = all_labels.copy() # Already in the correct format (-1 or orig_idx)
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        # Ensure predictions and labels are within the expected set for confusion_matrix
        valid_cm_labels = set(cm_axis_labels_int)
        filtered_labels_true = [l if l in valid_cm_labels else -1 for l in labels_mapped_for_cm]
        filtered_labels_pred = [p if p in valid_cm_labels else -1 for p in all_preds]

        conf_matrix = confusion_matrix(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int)
        precision, recall, f1_by_class, _ = precision_recall_fscore_support(
            filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int, average=None, zero_division=0
        )
        macro_f1 = np.mean(f1_by_class) if len(f1_by_class) > 0 else 0.0

        # Print Summary
        print("\nThreshold OSR Evaluation Summary:")
        print(f"  Threshold: {self.threshold:.4f}")
        print(f"  Accuracy (Known): {accuracy:.4f}")
        print(f"  AUROC: {auroc:.4f}")
        print(f"  Unknown Detection Rate: {unknown_detection_rate:.4f}")
        print(f"  Macro F1 Score: {macro_f1:.4f}")

        results = {
            'accuracy': accuracy, 'auroc': auroc, 'f1_score': macro_f1,
            'unknown_detection_rate': unknown_detection_rate,
            'confusion_matrix': conf_matrix, 'confusion_matrix_labels': cm_axis_labels_int,
            'confusion_matrix_names': cm_axis_labels_names,
            'predictions': all_preds, 'labels': all_labels,
            'probs': all_probs, 'max_probs': all_max_probs
        }
        return results

    def visualize(self, results):
        print("[ThresholdOSR Visualize] Generating result plots...")
        os.makedirs("results", exist_ok=True)
        base_filename = f"results/threshold_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        labels_np = results['labels']
        max_probs = results['max_probs']
        unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)

        # ROC Curve
        if len(np.unique(unknown_labels_mask)) > 1 and len(max_probs) == len(unknown_labels_mask):
            fpr, tpr, _ = roc_curve(unknown_labels_mask, -max_probs) # Low score = unknown
            roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (Threshold)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close()
            print(f"  ROC curve saved.")
        else: print("  Skipping ROC (only one class or data mismatch).")

        # Confidence Distribution
        if len(max_probs) > 0:
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': max_probs, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50); plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold={self.threshold:.2f}'); plt.title('Confidence Distribution (Threshold)'); plt.xlabel('Max Softmax Probability'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_confidence.png"); plt.close()
            print(f"  Confidence distribution saved.")
        else: print("  Skipping confidence distribution (no scores).")

        # Confusion Matrix
        if 'confusion_matrix' in results:
            plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (Threshold)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close()
            print(f"  Confusion matrix saved.")
        else: print("  Skipping confusion matrix (not found in results).")
        print("[ThresholdOSR Visualize] Finished.")


class OpenMaxOSR(OSRAlgorithm):
    """OpenMax OSR algorithm implementation."""
    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        if not isinstance(model, RobertaClassifier):
             print(f"Warning: OpenMaxOSR typically uses RobertaClassifier, but received {type(model)}. Proceeding anyway.")
        self.tail_size = getattr(args, 'param_openmax_tailsize', 50)
        self.alpha = getattr(args, 'param_openmax_alpha', 10)
        print(f"[OpenMaxOSR Init] Tail size: {self.tail_size}, Alpha: {self.alpha}, Known Classes: {self.num_known_classes}")
        # Initialize MAVs and Weibull models
        self.mav: dict[int, np.ndarray] = {}
        self.weibull_models: dict[int, tuple[float, float, float]] = {}
        self.feat_dim = model.config.hidden_size if hasattr(model, 'config') and hasattr(model.config, 'hidden_size') else 768

    def fit_weibull(self, dataloader):
        """Fits Weibull models using correctly classified training samples."""
        print("[OpenMaxOSR Fit] Fitting Weibull models...")
        self.model.eval().to(self.device)
        av_per_class = {c: [] for c in range(self.num_known_classes)} # Uses 0..num_seen-1 indices

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="OpenMax Fit: Collecting embeddings"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_mapped = batch["label"].to(self.device) # Expecting 0..num_seen-1 labels

                # Get logits and embeddings (penultimate features, AVs)
                logits, embeddings = self.model(ids, attn, tok)
                preds_mapped = torch.argmax(logits, dim=1) # Predicted class index (0..num_seen-1)

                for i, true_mapped_idx in enumerate(labels_mapped):
                    true_idx = true_mapped_idx.item()
                    pred_idx = preds_mapped[i].item()
                    # Only use samples that are correctly classified for the current class
                    if pred_idx == true_idx and 0 <= true_idx < self.num_known_classes:
                        av_per_class[true_idx].append(embeddings[i].cpu().numpy())

        self.mav.clear(); self.weibull_models.clear()
        print("[OpenMaxOSR Fit] Calculating MAVs and fitting Weibull models...")
        for c_idx, av_list in tqdm(av_per_class.items(), desc="OpenMax Fit: Weibull Fitting"):
            if not av_list:
                print(f"  Warning: No correctly classified samples found for internal class index {c_idx}. Skipping Weibull fitting."); continue
            avs = np.stack(av_list)
            self.mav[c_idx] = np.mean(avs, axis=0)
            distances = np.linalg.norm(avs - self.mav[c_idx], axis=1)
            distances_sorted = np.sort(distances)
            current_tail_size = min(self.tail_size, len(distances_sorted))
            if current_tail_size < 2:
                print(f"  Warning: Insufficient tail points ({current_tail_size}) for class index {c_idx}. Using default Weibull.")
                mean_dist = np.mean(distances_sorted) if len(distances_sorted) > 0 else 1.0
                shape, loc, scale = 1.0, 0.0, mean_dist
            else:
                tail_distances = distances_sorted[-current_tail_size:]
                try:
                    shape, loc, scale = weibull_min.fit(tail_distances, floc=0)
                    if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9:
                        print(f"  Warning: Weibull fit failed for class {c_idx} (invalid params). Using default.")
                        shape, loc, scale = 1.0, 0.0, np.mean(tail_distances)
                except Exception as e:
                    print(f"  Warning: Weibull fit exception for class {c_idx}: {e}. Using default.")
                    shape, loc, scale = 1.0, 0.0, np.mean(tail_distances)
            self.weibull_models[c_idx] = (shape, loc, scale)
        print("[OpenMaxOSR Fit] Weibull fitting complete.")

    def openmax_probability(self, embedding_av: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """Recalibrates logits based on Weibull CDF scores."""
        if not self.mav or not self.weibull_models:
            raise RuntimeError("MAV or Weibull models not calculated. Call fit_weibull first.")
        num_known = len(logits)
        if num_known != self.num_known_classes:
             print(f"Warning: Logits dim ({num_known}) != expected known classes ({self.num_known_classes}). Adjusting alpha.")
             current_alpha = min(self.alpha, num_known)
        else: current_alpha = self.alpha

        distances = np.full(num_known, np.inf)
        for c_idx in range(num_known):
             if c_idx in self.mav: distances[c_idx] = np.linalg.norm(embedding_av - self.mav[c_idx])

        cdf_scores = np.ones(num_known)
        for c_idx in range(num_known):
            if c_idx in self.weibull_models and np.isfinite(distances[c_idx]):
                shape, loc, scale = self.weibull_models[c_idx]
                cdf_scores[c_idx] = weibull_min.cdf(distances[c_idx], shape, loc=loc, scale=scale)

        revised_logits = logits.copy()
        sorted_indices = np.argsort(logits)[::-1]
        for rank, c_idx in enumerate(sorted_indices):
            if rank < current_alpha:
                weight = 1.0 - cdf_scores[c_idx]
                revised_logits[c_idx] *= weight

        unknown_logit_score = np.sum(logits[sorted_indices[:current_alpha]] * cdf_scores[sorted_indices[:current_alpha]])
        final_logits = np.append(revised_logits, unknown_logit_score)
        exp_logits = np.exp(final_logits - np.max(final_logits))
        openmax_probs = exp_logits / np.sum(exp_logits)
        return openmax_probs

    def predict(self, dataloader):
        # ... (fit_weibull 호출 로직) ...
        self.model.eval().to(self.device) # 모델 GPU 이동
        openmax_probs_list = []
        preds_final_list = []
        labels_original_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (OpenMax OSR)"):
                # ******** 수정: 데이터 GPU 이동 ********
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch["label"].cpu().numpy()
                # *************************************

                # forward 직접 호출
                logits_batch_gpu, embeddings_batch_gpu = self.model(ids, attn, tok)
                logits_batch = logits_batch_gpu.cpu().numpy()
                embeddings_batch = embeddings_batch_gpu.cpu().numpy()

                for i in range(len(labels_orig)):
                    embedding_np = embeddings_batch[i]
                    logits_np = logits_batch[i] # Logits for 0..num_seen-1 classes
                    if embedding_np.ndim == 0 or logits_np.ndim == 0: continue

                    om_probs = self.openmax_probability(embedding_np, logits_np)
                    openmax_probs_list.append(om_probs)
                    pred_idx_with_unknown = np.argmax(om_probs)

                    if pred_idx_with_unknown == self.num_known_classes:
                        pred_final = -1
                    else:
                        pred_final = self._map_preds_to_original([pred_idx_with_unknown])[0]

                    preds_final_list.append(pred_final)
                    labels_original_list.append(labels_orig[i])

        if not openmax_probs_list: return np.array([]), np.array([]), np.array([]), np.array([])
        all_openmax_probs = np.vstack(openmax_probs_list)
        all_preds_final = np.array(preds_final_list)
        all_labels_original = np.array(labels_original_list)
        all_unknown_probs = all_openmax_probs[:, -1] if all_openmax_probs.shape[1] > self.num_known_classes else np.zeros(len(all_labels_original))
        return all_openmax_probs, all_preds_final, all_labels_original, all_unknown_probs
    
    def evaluate(self, dataloader):
        # ******** 수정된 부분 시작 ********
        # predict 전에 fit_weibull을 명시적으로 호출
        if not hasattr(self, 'mav') or not self.mav or not self.weibull_models:
            print("[OpenMaxOSR Evaluate] Weibull models not fitted. Fitting now...")
            try:
                # train_dataloader를 사용하여 피팅
                train_loader = self.datamodule.train_dataloader()
                self.fit_weibull(train_loader)
                if not self.mav or not self.weibull_models: # 피팅 후에도 없으면 에러 발생
                     raise RuntimeError("OpenMax fitting failed.")
            except Exception as e:
                 print(f"Error during automatic OpenMax fitting in evaluate: {e}")
                 # 피팅 실패 시 평가 중단 또는 기본값 반환
                 return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0, 'error': 'Weibull fitting failed'}
        # ******** 수정된 부분 끝 ********

        all_probs, all_preds, all_labels, all_unknown_probs = self.predict(dataloader)
        if len(all_labels) == 0: print("Warning: No data to evaluate for OpenMax."); return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0}

        # ... (나머지 evaluate 로직은 동일) ...
        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1)
        known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        auroc = roc_auc_score(unknown_labels_mask, all_unknown_probs) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        labels_mapped_for_cm = all_labels.copy()
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        valid_cm_labels = set(cm_axis_labels_int)
        filtered_labels_true = [l if l in valid_cm_labels else -1 for l in labels_mapped_for_cm]
        filtered_labels_pred = [p if p in valid_cm_labels else -1 for p in all_preds]

        conf_matrix = confusion_matrix(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int)
        precision, recall, f1_by_class, _ = precision_recall_fscore_support(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int, average=None, zero_division=0)
        macro_f1 = np.mean(f1_by_class) if len(f1_by_class) > 0 else 0.0

        print("\nOpenMax OSR Evaluation Summary:")
        print(f"  Tail size: {self.tail_size}, Alpha: {self.alpha}")
        print(f"  Accuracy (Known): {accuracy:.4f}"); print(f"  AUROC: {auroc:.4f}"); print(f"  Unknown Detection Rate: {unknown_detection_rate:.4f}"); print(f"  Macro F1 Score: {macro_f1:.4f}")

        results = { 'accuracy': accuracy, 'auroc': auroc, 'f1_score': macro_f1, 'unknown_detection_rate': unknown_detection_rate,
                    'confusion_matrix': conf_matrix, 'confusion_matrix_labels': cm_axis_labels_int, 'confusion_matrix_names': cm_axis_labels_names,
                    'predictions': all_preds, 'labels': all_labels, 'probs': all_probs, 'unknown_probs': all_unknown_probs }
        return results
    
    def visualize(self, results):
        print("[OpenMaxOSR Visualize] Generating result plots...")
        os.makedirs("results", exist_ok=True)
        base_filename = f"results/openmax_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np); unknown_probs = results["unknown_probs"]

        if len(np.unique(unknown_labels_mask)) > 1 and len(unknown_probs) == len(unknown_labels_mask): # ROC
            fpr, tpr, _ = roc_curve(unknown_labels_mask, unknown_probs); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (OpenMax)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close()
            print("  ROC curve saved.")
        else: print("  Skipping ROC (only one class or data mismatch).")

        # Unknown Prob Dist
        if len(unknown_probs) > 0:
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': unknown_probs, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50); plt.title('Unknown Probability Distribution (OpenMax)'); plt.xlabel('OpenMax Unknown Probability'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_confidence.png"); plt.close()
            print("  Unknown probability distribution saved.")
        else: print("  Skipping unknown probability distribution (no scores).")

        # Confusion Matrix
        if 'confusion_matrix' in results:
            plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (OpenMax)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close()
            print("  Confusion matrix saved.")
        else: print("  Skipping confusion matrix (not found in results).")
        print("[OpenMaxOSR Visualize] Finished.")


class CROSROSR(OSRAlgorithm):
    """CROSR algorithm using reconstruction error."""
    def __init__(self, model, datamodule, args):
        if not isinstance(model, RobertaAutoencoder): raise TypeError("CROSR needs RobertaAutoencoder.")
        super().__init__(model, datamodule, args)
        self.threshold = getattr(args, 'param_crosr_reconstruction_threshold', 0.9) # CDF threshold
        self.tail_size = getattr(args, 'param_crosr_tailsize', 100)
        print(f"[CROSROSR Init] Threshold (CDF): {self.threshold:.4f}, Tail Size: {self.tail_size}")
        self.weibull_model = None # (shape, loc, scale)

    def fit_evt_model(self, dataloader):
        """Fits EVT model (Weibull) on reconstruction errors from training data."""
        print("[CROSROSR Fit] Fitting EVT model on reconstruction errors...")
        self.model.eval().to(self.device)
        errors = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="CROSR Fit: Collecting errors"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                # Use model's forward pass directly
                _, cls_output, _, reconstructed = self.model(ids, attn, tok)
                errors.extend(torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy())
        if not errors: print("Warning: No errors collected for EVT fit."); self.weibull_model = (1.0, 0.0, 1.0); return

        errors_np = np.sort(np.array(errors))
        tail = errors_np[-min(self.tail_size, len(errors_np)):]
        if len(tail) < 2: print(f"Warning: Insufficient tail ({len(tail)}) for EVT fit. Using default."); self.weibull_model = (1.0, 0.0, np.mean(errors_np) if errors_np.size > 0 else 1.0); return
        try:
            shape, loc, scale = weibull_min.fit(tail, floc=0) # Fix location to 0
            if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9: raise ValueError("Invalid fit params")
            self.weibull_model = (shape, loc, scale)
            print(f"  CROSR Fitted Weibull: shape={shape:.4f}, scale={scale:.4f}")
        except Exception as e: print(f"Warning: CROSR Weibull fit exception: {e}. Using default."); self.weibull_model = (1.0, 0.0, np.mean(tail))
        print("[CROSROSR Fit] Complete.")

    def predict(self, dataloader):
        # ... (fit_evt_model 호출 로직) ...
        self.model.eval().to(self.device) # 모델 GPU 이동
        all_recon_errors = [] ; all_unknown_probs = [] ; all_preds_final = [] ; all_labels_original = []
        shape, loc, scale = self.weibull_model

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (CROSR)"):
                # ******** 수정: 데이터 GPU 이동 ********
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch['label'].cpu().numpy()
                # *************************************

                # forward 직접 호출
                logits, cls_output, _, reconstructed = self.model(ids, attn, tok)
                preds_mapped = torch.argmax(logits, dim=1) # Model predictions (0..num_seen-1)
                recon_errors_batch = torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy()

                unknown_probs_batch = weibull_min.cdf(recon_errors_batch, shape, loc=loc, scale=scale)

                preds_mapped_cpu = preds_mapped.cpu().numpy()
                batch_preds_final = np.full_like(preds_mapped_cpu, -1, dtype=int)
                accept_mask = unknown_probs_batch <= self.threshold

                if np.any(accept_mask):
                     accepted_mapped_indices = preds_mapped_cpu[accept_mask]
                     original_accepted_indices = self._map_preds_to_original(accepted_mapped_indices)
                     batch_preds_final[accept_mask] = original_accepted_indices

                all_recon_errors.extend(recon_errors_batch)
                all_unknown_probs.extend(unknown_probs_batch)
                all_preds_final.extend(batch_preds_final)
                all_labels_original.extend(labels_orig)
        return np.array(all_recon_errors), np.array(all_unknown_probs), np.array(all_preds_final), np.array(all_labels_original)

    def evaluate(self, dataloader):
        # ******** 수정된 부분 시작 ********
        # predict 전에 fit_evt_model을 명시적으로 호출
        if self.weibull_model is None:
            print("[CROSROSR Evaluate] EVT model not fitted. Fitting now...")
            try:
                # train_dataloader를 사용하여 피팅
                train_loader = self.datamodule.train_dataloader()
                self.fit_evt_model(train_loader)
                if self.weibull_model is None: # 피팅 후에도 없으면 에러 발생
                    raise RuntimeError("CROSR EVT fitting failed.")
            except Exception as e:
                print(f"Error during automatic CROSR EVT fitting in evaluate: {e}")
                # 피팅 실패 시 평가 중단 또는 기본값 반환
                return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0, 'error': 'EVT fitting failed'}
        # ******** 수정된 부분 끝 ********

        all_errors, all_unknown_probs, all_preds, all_labels = self.predict(dataloader)
        if len(all_labels) == 0: print("Warning: No data to evaluate for CROSR."); return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0}

        # ... (나머지 evaluate 로직은 동일) ...
        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1); known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        auroc = roc_auc_score(unknown_labels_mask, all_unknown_probs) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        labels_mapped_for_cm = all_labels.copy()
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        valid_cm_labels = set(cm_axis_labels_int)
        filtered_labels_true = [l if l in valid_cm_labels else -1 for l in labels_mapped_for_cm]
        filtered_labels_pred = [p if p in valid_cm_labels else -1 for p in all_preds]

        conf_matrix = confusion_matrix(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int)
        _, _, f1_by_class, _ = precision_recall_fscore_support(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int, average=None, zero_division=0)
        macro_f1 = np.mean(f1_by_class) if len(f1_by_class) > 0 else 0.0

        print("\nCROSR OSR Evaluation Summary:")
        print(f"  Threshold (CDF): {self.threshold:.4f}, Tail size: {self.tail_size}")
        print(f"  Accuracy (Known): {accuracy:.4f}"); print(f"  AUROC: {auroc:.4f}"); print(f"  Unknown Detection Rate: {unknown_detection_rate:.4f}"); print(f"  Macro F1 Score: {macro_f1:.4f}")

        results = {'accuracy': accuracy, 'auroc': auroc, 'f1_score': macro_f1, 'unknown_detection_rate': unknown_detection_rate,
                   'confusion_matrix': conf_matrix, 'confusion_matrix_labels': cm_axis_labels_int, 'confusion_matrix_names': cm_axis_labels_names,
                   'predictions': all_preds, 'labels': all_labels, 'reconstruction_errors': all_errors, 'unknown_probs': all_unknown_probs}
        return results
    
    def visualize(self, results):
        """Visualizes CROSR OSR results."""
        print("[CROSROSR Visualize] Generating plots..."); os.makedirs("results", exist_ok=True)
        base_filename = f"results/crosr_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
        unknown_probs = results['unknown_probs']; recon_errors = results['reconstruction_errors']

        # ROC Curve
        if len(np.unique(unknown_labels_mask)) > 1 and len(unknown_probs) == len(unknown_labels_mask):
            fpr, tpr, _ = roc_curve(unknown_labels_mask, unknown_probs); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (CROSR)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close(); print("  ROC saved.")
        else: print("  Skipping ROC.")

        # Reconstruction Error Distribution
        if len(recon_errors) > 0:
            plt.figure(figsize=(7, 5))
            sns.histplot(data=pd.DataFrame({'error': recon_errors, 'Known': ~unknown_labels_mask}),
                         x='error', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
            plt.title('Reconstruction Error Distribution (CROSR)'); plt.xlabel('L2 Reconstruction Error'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_error.png"); plt.close(); print("  Error dist saved.")
        else: print("  Skipping error distribution.")

        # Unknown Probability (CDF) Distribution
        if len(unknown_probs) > 0:
            plt.figure(figsize=(7, 5))
            sns.histplot(data=pd.DataFrame({'score': unknown_probs, 'Known': ~unknown_labels_mask}),
                         x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
            plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold={self.threshold:.2f}'); plt.title('Unknown Probability Distribution (CROSR)'); plt.xlabel('Weibull CDF of Recon Error'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_prob.png"); plt.close(); print("  Prob dist saved.")
        else: print("  Skipping probability distribution.")

        # Confusion Matrix
        if 'confusion_matrix' in results:
            plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (CROSR)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close(); print("  CM saved.")
        else: print("  Skipping confusion matrix.")
        print("[CROSROSR Visualize] Finished.")


class DOCOSR(OSRAlgorithm):
    """DOC algorithm using class-specific thresholds on sigmoid scores."""
    def __init__(self, model, datamodule, args):
        if not isinstance(model, DOCRobertaClassifier): raise TypeError("DOC needs DOCRobertaClassifier.")
        super().__init__(model, datamodule, args)
        self.k_sigma = getattr(args, 'param_doc_k', 3.0)
        print(f"[DOCOSR Init] k-sigma: {self.k_sigma}")
        self.gaussian_params: dict[int, tuple[float, float]] = {} # key = internal class index (0..N-1) -> (mean_orig, std_combined)
        self.class_thresholds: dict[int, float] = {} # key = internal class index (0..N-1) -> threshold

    def fit_gaussian(self, dataloader):
        """Fits Gaussian models to sigmoid scores for each known class."""
        print("[DOCOSR Fit] Fitting Gaussian models to sigmoid scores...")
        self.model.eval().to(self.device)
        scores_per_class = {c: [] for c in range(self.num_known_classes)} # Use 0..num_seen-1 indices

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="DOC Fit: Collecting scores"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_mapped = batch['label'].to(self.device) # These are 0..num_seen-1

                logits, _ = self.model(ids, attn, tok)
                sigmoid_scores = torch.sigmoid(logits)

                for i, true_mapped_idx in enumerate(labels_mapped):
                     idx = true_mapped_idx.item()
                     if 0 <= idx < self.num_known_classes:
                         scores_per_class[idx].append(sigmoid_scores[i, idx].item())

        self.gaussian_params.clear(); self.class_thresholds.clear()
        print("[DOCOSR Fit] Calculating Gaussian parameters and thresholds...")
        for c_idx, scores in tqdm(scores_per_class.items(), desc="DOC Fit: Fitting"):
            if len(scores) >= 2:
                # Use mirrored scores approach from DOC paper
                scores_np = np.array([s for s in scores if s > 0.1]) # Filter low scores
                if len(scores_np) < 2:
                     print(f"Warning: Insufficient valid scores ({len(scores_np)}) for class {c_idx} after filtering. Using default threshold.");
                     self.gaussian_params[c_idx] = (0.5, 0.5); self.class_thresholds[c_idx] = 0.5
                     continue

                mirrored_scores = 1.0 + (1.0 - scores_np)
                combined_scores = np.concatenate([scores_np, mirrored_scores])
                _, std_combined = norm.fit(combined_scores)
                std_combined = max(std_combined, 1e-6) # Avoid zero std dev
                threshold = max(0.5, 1.0 - self.k_sigma * std_combined)

                self.gaussian_params[c_idx] = (np.mean(scores_np), std_combined) # Store original mean, combined std
                self.class_thresholds[c_idx] = threshold
            else:
                print(f"Warning: Insufficient samples ({len(scores)}) for class {c_idx}. Using default threshold.")
                self.gaussian_params[c_idx] = (0.5, 0.5); self.class_thresholds[c_idx] = 0.5
        print("[DOCOSR Fit] Complete.")

    def predict(self, dataloader):
        # ... (fit_gaussian 호출 로직) ...
        self.model.eval().to(self.device) # 모델 GPU 이동
        all_sigmoid_scores = []; all_max_scores = []; all_preds_final = []; all_labels_original = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (DOC OSR)"):
                # ******** 수정: 데이터 GPU 이동 ********
                ids = batch['input_ids'].to(self.device)
                attn = batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch['label'].cpu().numpy()
                # *************************************

                # forward 직접 호출
                logits, _ = self.model(ids, attn, tok)
                sigmoid_scores_batch = torch.sigmoid(logits) # B x num_seen_classes
                preds_mapped = torch.argmax(sigmoid_scores_batch, dim=1) # Predicted index (0..num_seen-1)

                max_scores_batch, _ = torch.max(sigmoid_scores_batch, dim=1)

                pred_indices_np = preds_mapped.cpu().numpy()
                max_scores_np = max_scores_batch.cpu().numpy()
                batch_preds_final = np.full_like(pred_indices_np, -1, dtype=int)
                accept_mask = np.zeros_like(pred_indices_np, dtype=bool)

                for i in range(len(labels_orig)):
                     pred_mapped_idx = pred_indices_np[i]
                     threshold = self.class_thresholds.get(pred_mapped_idx, 0.5)
                     if max_scores_np[i] >= threshold:
                         accept_mask[i] = True

                if np.any(accept_mask):
                    accepted_mapped_indices = pred_indices_np[accept_mask]
                    original_accepted_indices = self._map_preds_to_original(accepted_mapped_indices)
                    batch_preds_final[accept_mask] = original_accepted_indices

                all_sigmoid_scores.append(sigmoid_scores_batch.cpu().numpy())
                all_max_scores.extend(max_scores_np)
                all_preds_final.extend(batch_preds_final)
                all_labels_original.extend(labels_orig)

        # Calculate Z-scores (optional, using original mean and combined std)
        all_z_scores = np.full(len(all_max_scores), -np.inf)
        pred_indices_all = np.argmax(np.vstack(all_sigmoid_scores), axis=1) if all_sigmoid_scores else np.array([])
        for i in range(len(all_max_scores)):
            pred_idx = pred_indices_all[i]
            if pred_idx in self.gaussian_params:
                 mean_orig, std_combined = self.gaussian_params[pred_idx]
                 all_z_scores[i] = (all_max_scores[i] - mean_orig) / std_combined if std_combined > 1e-6 else 0
        return (np.vstack(all_sigmoid_scores) if all_sigmoid_scores else np.array([])), \
               np.array(all_max_scores), np.array(all_preds_final), np.array(all_labels_original), np.array(all_z_scores)

    def evaluate(self, dataloader):
        all_scores, all_max_scores, all_preds, all_labels, all_z_scores = self.predict(dataloader)
        if len(all_labels) == 0: print("Warning: No data to evaluate for DOC."); return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0}

        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1); known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        # Use negative max score for AUROC (lower score -> more likely unknown)
        auroc = roc_auc_score(unknown_labels_mask, -all_max_scores) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        labels_mapped_for_cm = all_labels.copy()
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        valid_cm_labels = set(cm_axis_labels_int)
        filtered_labels_true = [l if l in valid_cm_labels else -1 for l in labels_mapped_for_cm]
        filtered_labels_pred = [p if p in valid_cm_labels else -1 for p in all_preds]

        conf_matrix = confusion_matrix(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int)
        _, _, f1_by_class, _ = precision_recall_fscore_support(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int, average=None, zero_division=0)
        macro_f1 = np.mean(f1_by_class) if len(f1_by_class) > 0 else 0.0

        print("\nDOC OSR Evaluation Summary:")
        print(f"  k-sigma: {self.k_sigma}")
        print(f"  Accuracy (Known): {accuracy:.4f}"); print(f"  AUROC: {auroc:.4f}"); print(f"  Unknown Detection Rate: {unknown_detection_rate:.4f}"); print(f"  Macro F1 Score: {macro_f1:.4f}")

        results = {'accuracy': accuracy, 'auroc': auroc, 'f1_score': macro_f1, 'unknown_detection_rate': unknown_detection_rate,
                   'confusion_matrix': conf_matrix, 'confusion_matrix_labels': cm_axis_labels_int, 'confusion_matrix_names': cm_axis_labels_names,
                   'predictions': all_preds, 'labels': all_labels, 'scores': all_scores, 'max_scores': all_max_scores, 'z_scores': all_z_scores}
        return results

    def visualize(self, results):
        print("[DOCOSR Visualize] Generating plots..."); os.makedirs("results", exist_ok=True)
        base_filename = f"results/doc_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
        max_scores = results['max_scores']; z_scores = results.get('z_scores', None) # Handle if z_scores missing

        if len(np.unique(unknown_labels_mask)) > 1 and len(max_scores) == len(unknown_labels_mask): # ROC
            fpr, tpr, _ = roc_curve(unknown_labels_mask, -max_scores); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (DOC)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close(); print("  ROC saved.")
        else: print("  Skipping ROC.")

        # Max Score Dist
        if len(max_scores) > 0:
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': max_scores, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
            avg_threshold = np.mean(list(self.class_thresholds.values())) if self.class_thresholds else 0.5
            plt.axvline(avg_threshold, color='g', linestyle=':', label=f'Avg Thresh~{avg_threshold:.2f}')
            plt.title('Max Sigmoid Score Distribution (DOC)'); plt.xlabel('Max Sigmoid Score'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_score.png"); plt.close(); print("  Score dist saved.")
        else: print("  Skipping score distribution.")

        # Z-Score Dist
        if z_scores is not None and len(z_scores) > 0:
            z_scores_clipped = np.clip(z_scores, -10, 10) # Clip for visualization
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': z_scores_clipped, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50); plt.axvline(-self.k_sigma, color='r', linestyle='--', label=f'Z=-k ({-self.k_sigma:.1f})'); plt.title('Z-Score Distribution (DOC)'); plt.xlabel('Z-Score (Approx)'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_zscore.png"); plt.close(); print("  Z-score dist saved.")
        else: print("  Skipping Z-score distribution.")

        # CM
        if 'confusion_matrix' in results:
            plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (DOC)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close(); print("  CM saved.")
        else: print("  Skipping confusion matrix.")
        print("[DOCOSR Visualize] Finished.")


class ADBOSR(OSRAlgorithm):
    """OSR algorithm using Adaptive Decision Boundaries (ADB)."""
    def __init__(self, model, datamodule, args):
        if not isinstance(model, RobertaADB): raise TypeError("ADBOSR needs RobertaADB.")
        super().__init__(model, datamodule, args)
        self.distance_metric = getattr(args, 'param_adb_distance', 'cosine')
        print(f"[ADBOSR Init] Distance metric: {self.distance_metric}")

    def compute_distances(self, features_norm: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
        """Computes distances. Features are already normalized."""
        if self.distance_metric == 'cosine':
            centers_norm = F.normalize(centers, p=2, dim=-1)
            similarity = torch.matmul(features_norm, centers_norm.t())
            similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7)
            return 1.0 - similarity
        elif self.distance_metric == 'euclidean':
             centers_norm = F.normalize(centers, p=2, dim=-1) # Normalize centers for fair comparison
             return torch.cdist(features_norm, centers_norm, p=2)
        else: raise ValueError(f"Unknown distance: {self.distance_metric}")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval().to(self.device) # 모델을 GPU로 이동
        all_features = []; all_distances = []; all_preds_final = []; all_labels_original = []; all_min_distances = []

        centers = self.model.centers.detach()
        radii = self.model.get_radii().detach()

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (ADB OSR)"):
                # ******** 수정된 부분 시작 ********
                # batch 내의 텐서들을 모델과 동일한 device로 이동
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                # token_type_ids는 없을 수 있으므로 확인 후 이동
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(self.device)
                labels_orig = batch['label'].cpu().numpy() # 레이블은 CPU에 두어도 됨

                # 이동된 텐서들을 사용하여 모델 호출 준비
                batch_on_device = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'label': batch['label'] # test_step은 원본 레이블을 받을 수 있음
                }
                if token_type_ids is not None:
                    batch_on_device['token_type_ids'] = token_type_ids
                # ******** 수정된 부분 끝 ********

                # Use model's test_step with tensors on the correct device
                # test_step 내부에서도 device 처리가 필요할 수 있으나,
                # forward 호출 시 device에 있는 텐서를 사용하게 됨
                # test_step 대신 forward를 직접 호출하는 것이 더 명확할 수 있음

                # --- 대안: forward 직접 호출 ---
                logits, features_norm = self.model(input_ids, attention_mask, token_type_ids)
                preds_mapped = torch.argmax(logits, dim=1) # Predicted index (0..num_seen-1)
                # -----------------------------

                # --- 기존 방식 유지 시 (test_step 호출) ---
                # test_output = self.model.test_step(batch_on_device, 0) # Pass batch_on_device
                # features_norm = test_output['features'] # Normalized features
                # preds_mapped = test_output['preds_mapped'] # Predicted index (0..num_seen-1)
                # ---------------------------------------

                distances_batch = self.compute_distances(features_norm, centers) # B x num_seen_classes
                min_distances_batch, _ = torch.min(distances_batch, dim=1) # Min distance per sample

                # Get radius for the *predicted* class (index 0..num_seen-1)
                closest_radii_batch = radii[preds_mapped]

                pred_indices_np = preds_mapped.cpu().numpy() # 0..num_seen-1
                min_distances_np = min_distances_batch.cpu().numpy()
                closest_radii_np = closest_radii_batch.cpu().numpy()

                batch_preds_final = np.full_like(pred_indices_np, -1, dtype=int)
                # Accept if min distance <= radius of *predicted* class
                accept_mask = min_distances_np <= closest_radii_np

                if np.any(accept_mask):
                    accepted_mapped_indices = pred_indices_np[accept_mask] # Indices are 0..num_seen-1
                    # Map 0..num_seen-1 back to original 0..Total-1
                    original_accepted_indices = self._map_preds_to_original(accepted_mapped_indices)
                    batch_preds_final[accept_mask] = original_accepted_indices

                all_features.append(features_norm.cpu().numpy())
                all_distances.append(distances_batch.cpu().numpy())
                all_preds_final.extend(batch_preds_final)
                all_labels_original.extend(labels_orig)
                all_min_distances.extend(min_distances_np) # Store min distance for AUROC

        return (np.concatenate(all_features) if all_features else np.array([])), \
               (np.concatenate(all_distances) if all_distances else np.array([])), \
               np.array(all_preds_final), np.array(all_labels_original), np.array(all_min_distances)

    def evaluate(self, dataloader) -> dict:
        all_features, all_distances, all_preds, all_labels, all_min_distances = self.predict(dataloader)
        if len(all_labels) == 0: print("Warning: No data to evaluate for ADB."); return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0}

        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1); known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        # Higher distance -> more likely unknown
        auroc = roc_auc_score(unknown_labels_mask, all_min_distances) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        labels_mapped_for_cm = all_labels.copy()
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        valid_cm_labels = set(cm_axis_labels_int)
        filtered_labels_true = [l if l in valid_cm_labels else -1 for l in labels_mapped_for_cm]
        filtered_labels_pred = [p if p in valid_cm_labels else -1 for p in all_preds]

        conf_matrix = confusion_matrix(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int)
        precision, recall, f1_by_class, _ = precision_recall_fscore_support(filtered_labels_true, filtered_labels_pred, labels=cm_axis_labels_int, average=None, zero_division=0)
        macro_f1 = np.mean(f1_by_class) if len(f1_by_class) > 0 else 0.0

        print("\nADB OSR Evaluation Summary:")
        print(f"  Distance Metric: {self.distance_metric}")
        print(f"  Accuracy (Known): {accuracy:.4f}"); print(f"  AUROC: {auroc:.4f}"); print(f"  Unknown Detection Rate: {unknown_detection_rate:.4f}"); print(f"  Macro F1 Score: {macro_f1:.4f}")

        results = {'accuracy': accuracy, 'auroc': auroc, 'f1_score': macro_f1, 'unknown_detection_rate': unknown_detection_rate,
                   'confusion_matrix': conf_matrix, 'confusion_matrix_labels': cm_axis_labels_int, 'confusion_matrix_names': cm_axis_labels_names,
                   'predictions': all_preds, 'labels': all_labels, 'features': all_features, 'distances': all_distances, 'min_distances': all_min_distances}
        return results

    def visualize(self, results: dict):
        print("[ADBOSR Visualize] Generating plots..."); os.makedirs("results", exist_ok=True)
        base_filename = f"results/adb_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
        min_distances = results['min_distances']

        if len(np.unique(unknown_labels_mask)) > 1 and len(min_distances) == len(unknown_labels_mask): # ROC
            fpr, tpr, _ = roc_curve(unknown_labels_mask, min_distances); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (ADB)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close(); print("  ROC saved.")
        else: print("  Skipping ROC.")

        # Min Distance Dist
        if len(min_distances) > 0:
            plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'dist': min_distances, 'Known': ~unknown_labels_mask}), x='dist', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
            # Use get_radii() to calculate mean radius
            mean_radius = self.model.get_radii().detach().mean().item()
            plt.axvline(mean_radius, color='g', linestyle=':', label=f'Avg Radius~{mean_radius:.3f}')
            plt.title(f'Min Distance Distribution (ADB - {self.distance_metric})'); plt.xlabel(f'Min {self.distance_metric.capitalize()} Distance'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_distance.png"); plt.close(); print("  Distance dist saved.")
        else: print("  Skipping distance distribution.")

        # t-SNE
        if 'features' in results and len(results['features']) > 100 and results['features'].shape[1] > 2:
             try:
                 from sklearn.manifold import TSNE
                 print("  Generating t-SNE plot (this may take a while)...")
                 features = results['features']; centers = self.model.centers.detach().cpu().numpy()
                 n_samples = features.shape[0]; max_tsne = 5000
                 indices = np.random.choice(n_samples, min(n_samples, max_tsne), replace=False)
                 features_sub = features[indices]; unknown_sub = unknown_labels_mask[indices]
                 centers_norm = F.normalize(torch.from_numpy(centers), p=2, dim=-1).numpy()

                 combined = np.vstack([features_sub, centers_norm])
                 tsne = TSNE(n_components=2, random_state=self.args.random_seed, perplexity=min(30, combined.shape[0]-1), n_iter=300, init='pca', learning_rate='auto', metric=self.distance_metric)
                 reduced = tsne.fit_transform(combined)
                 reduced_feats, reduced_centers = reduced[:-len(centers)], reduced[-len(centers):]

                 plt.figure(figsize=(10, 8))
                 plt.scatter(reduced_feats[~unknown_sub, 0], reduced_feats[~unknown_sub, 1], c='blue', alpha=0.4, s=8, label='Known')
                 if np.any(unknown_sub): plt.scatter(reduced_feats[unknown_sub, 0], reduced_feats[unknown_sub, 1], c='red', alpha=0.4, s=8, label='Unknown')
                 plt.scatter(reduced_centers[:, 0], reduced_centers[:, 1], c='black', marker='X', s=100, edgecolors='w', linewidth=1, label='Centers')
                 plt.title('t-SNE Visualization (ADB Features & Centers)'); plt.xlabel("Dim 1"); plt.ylabel("Dim 2"); plt.legend(markerscale=1.5); plt.grid(alpha=0.4); plt.tight_layout(); plt.savefig(f"{base_filename}_tsne.png"); plt.close(); print("  t-SNE saved.")
             except ImportError: print("  Skipping t-SNE: scikit-learn needed.")
             except Exception as e: print(f"  t-SNE error: {e}")
        else: print("  Skipping t-SNE (few samples or low dim).")

        # CM
        if 'confusion_matrix' in results:
            plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (ADB)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close(); print("  CM saved.")
        else: print("  Skipping confusion matrix.")
        print("[ADBOSR Visualize] Finished.")


# =============================================================================
# Training and Evaluation Functions (Remain Unchanged - Inherited)
# =============================================================================

def train_model(model, datamodule, args):
    """Trains a PyTorch Lightning model."""
    print(f"\n--- Training Model: {model.__class__.__name__} ---")
    # Use a unique directory for each training run, potentially including timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Use a more specific run name based on args provided to this function
    run_name = f"{args.dataset}_{args.osr_method}_{args.seen_class_ratio}_{timestamp}"
    output_dir = os.path.join("checkpoints", run_name)
    log_dir = os.path.join("logs", run_name) # Separate log dir for each run
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Ensure data is setup for finding total steps
    # datamodule.setup(stage='fit') # Setup should happen before calling train_model
    total_steps = 0
    train_batches = 0
    try:
        # Calculate steps based on the length of the dataloader
        train_loader = datamodule.train_dataloader()
        train_batches = len(train_loader)
        # Adjust for accumulate_grad_batches if used (not currently in args)
        # effective_batches = train_batches // args.accumulate_grad_batches
        total_steps = train_batches * args.epochs
        warmup_steps = min(int(args.warmup_ratio * total_steps), args.max_warmup_steps)
        if total_steps == 0: raise ValueError("Total steps cannot be zero.")
    except Exception as e:
         print(f"Warning: Could not determine dataloader length or total steps: {e}. Using defaults.")
         total_steps = 10000; warmup_steps = 500; train_batches = total_steps // args.epochs

    # Set scheduler steps in the model if attributes exist
    if hasattr(model, 'total_steps'): model.total_steps = total_steps
    if hasattr(model, 'warmup_steps'): model.warmup_steps = warmup_steps
    print(f"Scheduler: Total steps={total_steps}, Warmup steps={warmup_steps}")

    # Callbacks
    monitor_metric = "val_loss"
    monitor_mode = "min"
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{model.__class__.__name__}-{{epoch:02d}}-{{{monitor_metric}:.4f}}",
        save_top_k=1, verbose=True, monitor=monitor_metric, mode=monitor_mode
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, patience=args.early_stopping_patience,
        min_delta=args.early_stopping_delta, verbose=True, mode=monitor_mode
    )

    # Logger (use the specific log_dir for this run)
    try:
        # Use the specific log_dir created for this run
        logger = TensorBoardLogger(save_dir=os.path.dirname(log_dir), name=os.path.basename(log_dir), version="")
        print(f"Using TensorBoardLogger in: {logger.log_dir}")
    except ImportError:
        print("TensorBoard not available. Using CSVLogger.")
        logger = CSVLogger(save_dir=os.path.dirname(log_dir), name=os.path.basename(log_dir), version="")
        print(f"Using CSVLogger in: {logger.log_dir}")


    # Trainer Config
    use_gpu = args.force_gpu or torch.cuda.is_available()
    trainer_kwargs = {
        "max_epochs": args.epochs,
        "callbacks": [checkpoint_callback, early_stopping_callback],
        "logger": logger,
        "log_every_n_steps": max(1, train_batches // 10) if train_batches > 0 else 50,
        "precision": "16-mixed" if use_gpu else 32,
        "gradient_clip_val": args.gradient_clip_val,
        "deterministic": "warn", # For reproducibility, but can slow down
        "benchmark": False if args.random_seed else True # Benchmark faster if seed not fixed
    }
    if use_gpu:
        if not torch.cuda.is_available(): raise RuntimeError("GPU forced but not available.")
        trainer_kwargs["accelerator"] = "gpu"
        trainer_kwargs["devices"] = [args.gpu_id]
    else:
        print("Using CPU for training.")
        trainer_kwargs["accelerator"] = "cpu"

    trainer = pl.Trainer(**trainer_kwargs)
    print(f"Starting training for {args.epochs} epochs...")
    best_checkpoint_path = None
    try:
        trainer.fit(model, datamodule=datamodule)
        best_checkpoint_path = checkpoint_callback.best_model_path
        print(f"Training finished. Best model saved at: {best_checkpoint_path}")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        print("Attempting to use last saved checkpoint (if any)...")
        best_checkpoint_path = checkpoint_callback.best_model_path # May exist even if error occurred
    finally:
        # Ensure logger closes files properly
        if hasattr(logger, 'finalize'): logger.finalize("finished")


    if best_checkpoint_path and os.path.exists(best_checkpoint_path):
         return best_checkpoint_path
    else:
         print(f"Warning: No valid checkpoint found after training in {output_dir}.")
         # Check if any checkpoint exists in the directory
         existing_ckpts = [f for f in os.listdir(output_dir) if f.endswith('.ckpt')]
         if existing_ckpts:
             # Try to find the one with 'epoch' and 'val_loss' in the name
             best_guess = None
             for ckpt in existing_ckpts:
                 if 'epoch' in ckpt and 'val_loss' in ckpt:
                     best_guess = os.path.join(output_dir, ckpt)
                     break
             if best_guess:
                 print(f"  Returning best guess checkpoint: {best_guess}")
                 return best_guess
             else:
                 print(f"  Found other checkpoints: {existing_ckpts}. Returning the first one.")
                 return os.path.join(output_dir, existing_ckpts[0])
         return None


# --- OSR Evaluation Wrappers (Remain Unchanged - Inherited) ---

MODEL_CLASS_MAP = {
    'standard': RobertaClassifier,
    'crosr': RobertaAutoencoder,
    'doc': DOCRobertaClassifier,
    'adb': RobertaADB
}
MODEL_NEEDS_SPECIAL_TRAINING = ['crosr', 'doc', 'adb']

def _prepare_evaluation(method_name, current_model, datamodule, args, osr_algorithm_class):
    """
    Handles model checking, potential retraining, and parameter setup
    (tuning with retraining OR loading defaults/saved). Returns the model to use for final evaluation.
    """
    print(f"\n--- Preparing for {method_name.upper()} OSR Evaluation ---")
    target_model_class = MODEL_CLASS_MAP.get(method_name if method_name in MODEL_NEEDS_SPECIAL_TRAINING else 'standard')
    model_to_evaluate_finally = None
    needs_final_training = False
    best_params = {} # Store best params found/loaded

    # --- Hyperparameter Tuning Logic (with Retraining) ---
    if args.parameter_search and (args.osr_method == 'all' or method_name == args.osr_method):
        print(f"Starting Optuna hyperparameter search for {method_name.upper()} (with retraining)...")
        tuner = OptunaHyperparameterTuner(method_name, datamodule, args)

        def train_and_evaluate_trial(trial_args):
            print(f"\n  Starting Training & Eval for Trial...")
            trial_start_time = time.time()
            # 1. Initialize the target model with trial hyperparameters
            num_classes = datamodule.num_seen_classes

            # ******** 수정된 부분 시작 ********
            # Start with common arguments for the trial model
            init_kwargs = {
                'model_name': trial_args.model,
                'num_classes': num_classes,
                'weight_decay': trial_args.weight_decay,
                 # Add scheduler params needed by the model's configure_optimizers later
                'warmup_steps': trial_args.max_warmup_steps, # Pass max_warmup_steps initially
                'total_steps': 0 # Will be calculated in train_model
            }

            # Add model-specific arguments with correct names based on target_model_class
            if target_model_class == RobertaClassifier:
                init_kwargs['learning_rate'] = trial_args.lr # Use 'learning_rate'
            elif target_model_class == RobertaAutoencoder:
                init_kwargs['learning_rate'] = trial_args.lr # Use 'learning_rate'
                init_kwargs['reconstruction_weight'] = trial_args.param_crosr_recon_weight
            elif target_model_class == DOCRobertaClassifier:
                init_kwargs['learning_rate'] = trial_args.lr # Use 'learning_rate'
            elif target_model_class == RobertaADB:
                # ADB expects 'lr' for backbone and 'lr_adb' for params
                init_kwargs['lr'] = trial_args.lr
                init_kwargs['lr_adb'] = trial_args.lr_adb
                init_kwargs['param_adb_delta'] = trial_args.param_adb_delta
                init_kwargs['param_adb_alpha'] = trial_args.param_adb_alpha
                init_kwargs['adb_freeze_backbone'] = trial_args.adb_freeze_backbone
            # ******** 수정된 부분 끝 ********

            print(f"  Initializing {target_model_class.__name__} with: {init_kwargs}")
            try:
                trial_model = target_model_class(**init_kwargs)
            except TypeError as e:
                print(f"  Error initializing model during trial: {e}")
                print(f"  Provided init_kwargs: {init_kwargs}")
                # Return failure score if initialization fails
                return {}, -1e9

            # 2. Train the model for this trial
            tuning_epochs = max(3, args.epochs // 2) # Use half epochs for tuning speed
            print(f"  Training trial model for {tuning_epochs} epochs...")
            trial_args_copy = argparse.Namespace(**vars(trial_args))
            trial_args_copy.epochs = tuning_epochs
            # Use a temporary, unique directory for trial checkpoints
            trial_run_id = f"trial_{method_name}_{trial_args.dataset}_{datetime.now().strftime('%H%M%S%f')}"
            trial_args_copy.osr_method = trial_run_id # Unique identifier for logging/checkpoints

            checkpoint_path = train_model(trial_model, datamodule, trial_args_copy)
            trial_ckpt_dir = os.path.dirname(checkpoint_path) if checkpoint_path else None # Get the actual dir used

            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                 print("  Trial Training Failed. Returning failure score.")
                 if trial_ckpt_dir and os.path.exists(trial_ckpt_dir): shutil.rmtree(trial_ckpt_dir)
                 return {}, -1e9 # Return large negative score for failure

            # 3. Load the best model from trial training
            print(f"  Loading best model from trial checkpoint: {checkpoint_path}")
            try:
                # Try loading directly first, map to current device
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                trained_trial_model = target_model_class.load_from_checkpoint(checkpoint_path, map_location=device)
            except Exception as load_e:
                 print(f"  Failed to load trial checkpoint: {load_e}")
                 if trial_ckpt_dir and os.path.exists(trial_ckpt_dir): shutil.rmtree(trial_ckpt_dir)
                 return {}, -1e9

            # 4. Evaluate the trained model
            print("  Evaluating trial model...")
            # Pass trial_args to evaluator for method-specific params (e.g., threshold, k_sigma)
            evaluator = osr_algorithm_class(trained_trial_model, datamodule, trial_args)
            try:
                results = evaluator.evaluate(datamodule.test_dataloader())
            except RuntimeError as eval_e:
                print(f"  Error during trial evaluation: {eval_e}")
                # Clean up and return failure score
                if trial_ckpt_dir and os.path.exists(trial_ckpt_dir): shutil.rmtree(trial_ckpt_dir)
                # Clean up logs as well
                trial_log_base_dir = "logs"
                trial_logger_name = f"{trial_args.model.replace('/','_')}_{trial_args.dataset}_{trial_run_id}_{trial_args.seen_class_ratio}"
                trial_log_dir = os.path.join(trial_log_base_dir, trial_logger_name)
                if os.path.exists(trial_log_dir):
                     try: shutil.rmtree(trial_log_dir)
                     except Exception as e: print(f"  Warning: Failed to remove trial log dir {trial_log_dir}: {e}")
                return {}, -1e9


            # 5. Clean up trial checkpoint directory and logs
            if trial_ckpt_dir and os.path.exists(trial_ckpt_dir):
                try: shutil.rmtree(trial_ckpt_dir)
                except Exception as e: print(f"  Warning: Failed to remove trial checkpoint dir {trial_ckpt_dir}: {e}")
            # Construct potential log dir path based on logger structure
            trial_log_base_dir = "logs"
            trial_logger_name = f"{trial_args.model.replace('/','_')}_{trial_args.dataset}_{trial_run_id}_{trial_args.seen_class_ratio}"
            trial_log_dir = os.path.join(trial_log_base_dir, trial_logger_name)
            if os.path.exists(trial_log_dir):
                 try: shutil.rmtree(trial_log_dir)
                 except Exception as e: print(f"  Warning: Failed to remove trial log dir {trial_log_dir}: {e}")


            # 6. Extract score and return
            score = results.get(args.tuning_metric)
            if score is None or not np.isfinite(score):
                 print(f"  Warning: Metric '{args.tuning_metric}' invalid ({score}) for trial.")
                 score = -1e9 # Return failure score
            print(f"  Trial completed in {time.time() - trial_start_time:.2f}s. Score ({args.tuning_metric}): {score:.4f}")
            return results, float(score)
        # --- End of train_and_evaluate_trial function ---

        # Pass the combined function to the tuner
        best_params, best_trial_metrics = tuner.tune(train_and_evaluate_trial)

        # Apply best parameters for the *final* run
        print(f"\nApplying best tuned parameters for final {method_name.upper()} evaluation:")
        for name, value in best_params.items():
            setattr(args, name, value); print(f"  {name}: {value}")
        # Set flag to retrain one last time with best params and full epochs
        needs_final_training = True

    else:
        # --- Logic for No Tuning ---
        # (이 부분은 이전과 동일하게 유지)
        needs_final_training = False
        best_params = load_best_params(method_name, args.dataset, args.seen_class_ratio)
        param_source = "loaded from previous tuning"
        if not best_params:
            best_params = get_default_best_params(method_name) # Get defaults if no saved params
            param_source = "defaults"
        print(f"Applying parameters ({param_source}) for final {method_name.upper()} evaluation:")
        for name, value in best_params.items():
            # Apply loaded/default params to the main args if they exist in args
            if hasattr(args, name):
                setattr(args, name, value); print(f"  {name}: {value}")
            else:
                print(f"  (Skipping param '{name}': Not in args namespace)")
        # Store the applied params
        best_params = {k: getattr(args, k) for k in best_params if hasattr(args, k)}


        # Check model type mismatch even when not tuning
        if not isinstance(current_model, target_model_class) and method_name in MODEL_NEEDS_SPECIAL_TRAINING:
             print(f"Warning: Model type mismatch ({type(current_model).__name__} vs {target_model_class.__name__}). Retraining required.")
             needs_final_training = True


    # --- Final Model Preparation ---
    if needs_final_training:
        print(f"\nTraining final {target_model_class.__name__} model with best/default parameters...")
        num_classes = datamodule.num_seen_classes
        # ******** 수정된 부분 시작 ********
        # Use the parameters stored in 'args' which have been updated either by tuning or loading defaults
        # Start with common arguments
        init_kwargs = {
            'model_name': args.model, 'num_classes': num_classes,
            'weight_decay': args.weight_decay,
            'warmup_steps': args.max_warmup_steps,
            'total_steps': 0 # Will be calculated in train_model
        }
        # Add model-specific arguments with correct names
        if target_model_class == RobertaClassifier:
            init_kwargs['learning_rate'] = args.lr
        elif target_model_class == RobertaAutoencoder:
            init_kwargs['learning_rate'] = args.lr
            init_kwargs['reconstruction_weight'] = args.param_crosr_recon_weight
        elif target_model_class == DOCRobertaClassifier:
            init_kwargs['learning_rate'] = args.lr
        elif target_model_class == RobertaADB:
            init_kwargs['lr'] = args.lr
            init_kwargs['lr_adb'] = args.lr_adb
            init_kwargs['param_adb_delta'] = args.param_adb_delta
            init_kwargs['param_adb_alpha'] = args.param_adb_alpha
            init_kwargs['adb_freeze_backbone'] = args.adb_freeze_backbone
        # ******** 수정된 부분 끝 ********

        final_model_instance = target_model_class(**init_kwargs)

        # Use full epochs for final training
        final_args = argparse.Namespace(**vars(args)) # Use current args state
        final_args.epochs = args.epochs # Ensure full epochs are used
        # Use a specific osr_method name for the final training run's checkpoints/logs
        final_args.osr_method = f"final_{method_name}"

        final_checkpoint_path = train_model(final_model_instance, datamodule, final_args)
        if final_checkpoint_path is None or not os.path.exists(final_checkpoint_path):
            raise RuntimeError(f"Failed to train final model for {method_name.upper()}.")
        print(f"Loading final trained model from: {final_checkpoint_path}")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_to_evaluate_finally = target_model_class.load_from_checkpoint(final_checkpoint_path, map_location=device)
        except Exception as load_e:
            raise RuntimeError(f"Failed to load final checkpoint: {load_e}")
    else:
         print("Using the initially provided/loaded model for final evaluation.")
         model_to_evaluate_finally = current_model

    # Return the model that should be used for the final evaluation run
    return model_to_evaluate_finally

def evaluate_threshold_osr(base_model, datamodule, args, all_results):
    """Evaluates Threshold OSR, handles tuning."""
    method_name = 'threshold'
    model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, ThresholdOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
    evaluator = ThresholdOSR(model_for_eval, datamodule, args) # Use final args
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results[method_name] = results
    return results

def evaluate_openmax_osr(base_model, datamodule, args, all_results):
    """Evaluates OpenMax OSR, handles tuning."""
    method_name = 'openmax'
    model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, OpenMaxOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
    evaluator = OpenMaxOSR(model_for_eval, datamodule, args) # Use final args
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results[method_name] = results
    return results

def evaluate_crosr_osr(base_model, datamodule, args, all_results):
    """Evaluates CROSR OSR, handles tuning and model training."""
    method_name = 'crosr'
    model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, CROSROSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
    evaluator = CROSROSR(model_for_eval, datamodule, args) # Use final args
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results[method_name] = results
    return results

def evaluate_doc_osr(base_model, datamodule, args, all_results):
    """Evaluates DOC OSR, handles tuning and model training."""
    method_name = 'doc'
    model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, DOCOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
    evaluator = DOCOSR(model_for_eval, datamodule, args) # Use final args
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results[method_name] = results
    return results

def evaluate_adb_osr(base_model, datamodule, args, all_results):
    """Evaluates ADB OSR, handles tuning and model training."""
    method_name = 'adb'
    model_for_eval = _prepare_evaluation(method_name, base_model, datamodule, args, ADBOSR)
    print(f"\n--- Running Final {method_name.upper()} Evaluation ---")
    evaluator = ADBOSR(model_for_eval, datamodule, args) # Use final args
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results[method_name] = results
    return results


# --- OSCR Curve Calculation and Visualization (Remain Unchanged - Inherited) ---
def calculate_oscr_curve(results, datamodule):
    """Calculates CCR vs FPR for the OSCR curve."""
    if 'predictions' not in results or 'labels' not in results: return np.array([0,1]), np.array([0,0])
    preds = np.array(results['predictions'])
    labels = np.array(results['labels'])
    if len(preds) != len(labels): return np.array([0,1]), np.array([0,0])

    # Determine score for ranking (lower score -> more likely unknown)
    score_key = None
    if 'max_probs' in results: scores_for_ranking = -np.array(results['max_probs']); score_key = 'max_probs' # Threshold
    elif 'unknown_probs' in results: scores_for_ranking = np.array(results['unknown_probs']); score_key = 'unknown_probs' # OpenMax, CROSR
    elif 'max_scores' in results: scores_for_ranking = -np.array(results['max_scores']); score_key = 'max_scores' # DOC
    elif 'min_distances' in results: scores_for_ranking = np.array(results['min_distances']); score_key = 'min_distances' # ADB
    else: print("Warning: No suitable score found for OSCR."); return np.array([0,1]), np.array([0,0])

    if len(scores_for_ranking) != len(labels):
        print(f"Warning: Score array length ({len(scores_for_ranking)}) mismatch with labels ({len(labels)}) for OSCR using key '{score_key}'.")
        return np.array([0,1]), np.array([0,0])

    unknown_labels_mask = datamodule._determine_unknown_labels(labels)
    known_mask = ~unknown_labels_mask
    is_correct_known = (preds == labels) & known_mask
    is_false_positive = (preds != -1) & unknown_labels_mask # Predicted known, but is unknown

    # Handle potential NaNs in scores
    valid_score_mask = ~np.isnan(scores_for_ranking)
    if not np.all(valid_score_mask):
        print(f"Warning: Found NaNs in scores for OSCR calculation. Filtering {np.sum(~valid_score_mask)} samples.")
        scores_for_ranking = scores_for_ranking[valid_score_mask]
        is_correct_known = is_correct_known[valid_score_mask]
        is_false_positive = is_false_positive[valid_score_mask]
        # Recalculate n_known and n_unknown based on filtered data? Or use original counts?
        # Using original counts might slightly skew CCR/FPR if many NaNs exist.
        # Let's use original counts for now.

    n_known = np.sum(known_mask) # Use original count
    n_unknown = np.sum(unknown_labels_mask) # Use original count

    if n_known == 0 or n_unknown == 0:
        print("Warning: No known or no unknown samples found for OSCR calculation.")
        return np.array([0,1]), np.array([0,0])

    sorted_indices = np.argsort(scores_for_ranking) # Sorts ascendingly (more unknown first)
    sorted_correct_known = is_correct_known[sorted_indices]
    sorted_false_positive = is_false_positive[sorted_indices]

    ccr = np.cumsum(sorted_correct_known) / n_known
    fpr = np.cumsum(sorted_false_positive) / n_unknown

    # Add (0,0) point for AUC calculation and plotting
    fpr = np.insert(fpr, 0, 0.0)
    ccr = np.insert(ccr, 0, 0.0)
    # Add (1, CCR_at_1) point? Optional, helps visualize full range.
    # fpr = np.append(fpr, 1.0)
    # ccr = np.append(ccr, ccr[-1])

    return fpr, ccr


def visualize_oscr_curves(all_results, datamodule, args):
    """Plots OSCR curves for comparing multiple OSR methods."""
    print("\nGenerating OSCR Comparison Curve...")
    plt.figure(figsize=(8, 7))
    method_found = False
    plotted_methods = []

    # Sort methods for consistent legend order (optional)
    sorted_methods = sorted(all_results.keys())

    for method in sorted_methods:
        results = all_results.get(method)
        if results and isinstance(results, dict) and 'error' not in results:
            try:
                fpr, ccr = calculate_oscr_curve(results, datamodule)
                if len(fpr) > 1 and len(ccr) > 1: # Ensure valid curve data
                    oscr_auc = np.trapz(ccr, fpr) # Use trapezoidal rule for AUC
                    plt.plot(fpr, ccr, lw=2.5, label=f'{method.upper()} (AUC = {oscr_auc:.3f})', alpha=0.8)
                    method_found = True
                    plotted_methods.append(method)
                else:
                    print(f"  Skipping OSCR plot for {method}: Insufficient data points.")
            except Exception as e: print(f"  Error calculating OSCR for {method}: {e}")
        # else: print(f"Skipping OSCR for {method} (no results or error).") # Less verbose

    if not method_found: print("No valid results found to plot OSCR."); plt.close(); return

    plt.plot([0, 1], [1, 0], color='grey', lw=1.5, linestyle='--', label='Ideal Closed-Set') # Removed AUC here
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('Correct Classification Rate (CCR)', fontsize=12)
    plt.title(f'OSCR Curves Comparison ({args.dataset}, Seen Ratio: {args.seen_class_ratio*100:.0f}%)', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    save_path = f"results/oscr_comparison_{args.dataset}_{args.seen_class_ratio}.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"OSCR comparison curve saved to: {save_path}")


# --- Main Evaluation Orchestrator (Remain Unchanged - Inherited) ---
def evaluate_osr_main(base_model, datamodule, args):
    """Runs evaluation for the selected OSR method(s)."""
    all_results = {}
    os.makedirs("results", exist_ok=True)

    if args.parameter_search:
        print("\n" + "="*70 + f"\n{' ' * 15}Hyperparameter Tuning Mode (Optuna)\n" + "="*70)
        print(f"Tuning Metric: {args.tuning_metric}, Trials: {args.n_trials}, Methods: {args.osr_method}")
        print("="*70 + "\n")

    method_map = {
        "threshold": evaluate_threshold_osr, "openmax": evaluate_openmax_osr,
        "crosr": evaluate_crosr_osr, "doc": evaluate_doc_osr, "adb": evaluate_adb_osr
    }
    methods_to_run = list(method_map.keys()) if args.osr_method == "all" else [args.osr_method]

    # Keep track of the model used for the last method if running 'all'
    last_model_used = base_model

    for method in methods_to_run:
        if method in method_map:
            try:
                print(f"\n>>> Starting evaluation for: {method.upper()} <<<")
                # Pass the *last used* model to the evaluation function.
                # _prepare_evaluation inside will decide if retraining is needed.
                method_map[method](last_model_used, datamodule, args, all_results)
                # Update last_model_used if a new model was trained and returned implicitly
                # (This assumes the evaluate_* functions modify the model in-place or we reload it if needed,
                # but _prepare handles it)
                # We don't strictly need to track last_model_used here as _prepare handles it.
            except Exception as e:
                print(f"\n!!!!! Error evaluating method {method.upper()}: {e} !!!!!")
                import traceback
                traceback.print_exc()
                all_results[method] = {"error": str(e)} # Store error message
        else:
            print(f"Warning: Unknown OSR method '{method}' skipped.")


    # --- Save Consolidated Results ---
    results_suffix = "_tuned" if args.parameter_search else ""
    results_filename = f"results/final_{args.model.replace('/','_')}_{args.dataset}_{args.osr_method}_{args.seen_class_ratio}{results_suffix}.json"

    # Enhanced JSON serialization
    def json_converter(obj):
        if isinstance(obj, (np.integer, np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)): return int(obj)
        elif isinstance(obj, (np.floating, np.float_, np.float16, np.float32, np.float64)): return float(obj) if np.isfinite(obj) else str(obj) # Handle NaN/Inf
        elif isinstance(obj, (np.ndarray,)): return obj.tolist() # Convert arrays
        elif isinstance(obj, (datetime, pd.Timestamp)): return obj.isoformat()
        elif isinstance(obj, (torch.Tensor)): return obj.cpu().numpy().tolist() # Convert tensors
        elif isinstance(obj, set): return list(obj) # Convert sets
        # Add handling for Path objects if they appear
        elif isinstance(obj, pathlib.Path): return str(obj)
        try: return obj.__dict__ # For simple objects
        except AttributeError: raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    # Import pathlib if needed for the converter above
    import pathlib

    try:
        # Remove potentially huge data before saving summary JSON
        summary_results = {}
        keys_to_exclude = [
            'predictions', 'labels', 'probs', 'scores', 'features', 'distances',
            'max_probs', 'unknown_probs', 'max_scores', 'min_distances',
            'reconstruction_errors', 'z_scores', 'embeddings', 'logits',
            'sigmoid_scores', 'encoded', 'reconstructed', 'preds_mapped',
            'labels_original', 'confusion_matrix' # Exclude raw CM too
        ]
        for method, res in all_results.items():
             if isinstance(res, dict):
                 summary_results[method] = {k: v for k, v in res.items() if k not in keys_to_exclude}
                 # Keep CM labels/names if they exist
                 if 'confusion_matrix_labels' in res: summary_results[method]['confusion_matrix_labels'] = res['confusion_matrix_labels']
                 if 'confusion_matrix_names' in res: summary_results[method]['confusion_matrix_names'] = res['confusion_matrix_names']
             else:
                 summary_results[method] = res # Store error message etc.


        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_results, f, indent=2, ensure_ascii=False, default=json_converter)
        print(f"\nConsolidated results summary saved to: {results_filename}")
    except Exception as e:
        print(f"\nError saving summary results to JSON: {e}")
        # Fallback: Save raw results using pickle
        pickle_filename = results_filename.replace(".json", "_full.pkl")
        try:
            import pickle
            with open(pickle_filename, 'wb') as pf: pickle.dump(all_results, pf)
            print(f"Warning: JSON saving failed. Saved full results as pickle: {pickle_filename}")
        except Exception as pe: print(f"Error saving full results as pickle: {pe}")

    # --- Print Summary Table ---
    metrics_to_display = ["accuracy", "auroc", "unknown_detection_rate", "f1_score"]
    metric_names_display = ["Acc(Known)", "AUROC", "UnkDetect", "F1(Macro)"]
    methods_evaluated = [m for m in all_results if isinstance(all_results.get(m), dict) and 'error' not in all_results[m]]

    if not methods_evaluated: print("\nNo successful evaluation results to display."); return all_results

    print("\n" + "="*100); print(f"{' ' * 38}Experiment Results Summary"); print("="*100)
    header = "{:<20}".format("Metric")
    for method in methods_evaluated: header += "{:<18}".format(method.upper())
    print(header); print("-"*len(header))

    for i, metric_key in enumerate(metrics_to_display):
        row = "{:<20}".format(metric_names_display[i])
        is_tuning_metric = args.parameter_search and metric_key == args.tuning_metric
        if is_tuning_metric: row = "* " + row.strip(); row = "{:<20}".format(row) # Mark tuning metric

        for method in methods_evaluated:
            val = all_results[method].get(metric_key, "N/A")
            try: formatted_val = "{:<18.4f}".format(float(val)) if pd.notna(val) else "{:<18}".format("NaN")
            except (TypeError, ValueError): formatted_val = "{:<18}".format(str(val))
            row += formatted_val
        print(row)

    if args.parameter_search: print("\n* Metric used for hyperparameter tuning.")
    print("="*len(header))

    # --- Generate Comparison OSCR Curve ---
    if len(methods_evaluated) > 1:
        visualize_oscr_curves(all_results, datamodule, args)

    print("\nEvaluation finished!")
    return all_results

# =============================================================================
# Argument Parser and Main Execution Block
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Open-Set Recognition Experiments with RoBERTa')
    # --- Core Arguments ---
    parser.add_argument('-dataset', type=str, default='acm',
                        choices=['newsgroup20', 'bbc_news', 'trec', 'reuters8', 'acm',
                                 'chemprot', 'banking77', 'oos', 'stackoverflow',
                                 'atis', 'snips', 'financial_phrasebank', 'arxiv10',
                                 'custom_syslog'], # Added custom_syslog
                        help='Dataset to use.')
    parser.add_argument('-model', type=str, default='roberta-base', help='Pre-trained RoBERTa model name (e.g., roberta-base, roberta-large).')
    parser.add_argument('-osr_method', type=str, default='all', choices=['threshold', 'openmax', 'crosr', 'doc', 'adb', 'all'], help='OSR method(s) to evaluate.')
    parser.add_argument('-seen_class_ratio', type=float, default=0.5, help='Ratio of classes used as known/seen (0.0 to 1.0).')
    parser.add_argument('-random_seed', type=int, default=42, help='Random seed for reproducibility.')
    # --- Training Arguments ---
    parser.add_argument('-epochs', type=int, default=5, help='Number of training epochs.')
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size (adjust based on GPU memory).') # Slightly reduced default
    parser.add_argument('-lr', type=float, default=2e-5, help='Learning rate for backbone/standard classifier.')
    parser.add_argument('-lr_adb', type=float, default=5e-4, help='Specific learning rate for ADB centers/radii.')
    parser.add_argument('-weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')
    parser.add_argument('-warmup_ratio', type=float, default=0.1, help='Ratio of total steps for LR warmup.')
    parser.add_argument('-max_warmup_steps', type=int, default=500, help='Max warmup steps.')
    parser.add_argument('-gradient_clip_val', type=float, default=1.0, help='Gradient clipping value (0 to disable).')
    parser.add_argument('-early_stopping_patience', type=int, default=3, help='Patience for early stopping.')
    parser.add_argument('-early_stopping_delta', type=float, default=0.001, help='Min delta for early stopping improvement.')
    # --- Data Split Arguments ---
    parser.add_argument('-train_ratio', type=float, default=0.7, help='Proportion for training.')
    parser.add_argument('-val_ratio', type=float, default=0.15, help='Proportion for validation.')
    parser.add_argument('-test_ratio', type=float, default=0.15, help='Proportion for testing.')
    # --- Hardware Arguments ---
    parser.add_argument('-force_gpu', action='store_true', help='Force GPU usage even if CUDA not detected (will error if unavailable).')
    parser.add_argument('-gpu_id', type=int, default=0, help='GPU ID to use.')
    # --- OSR Method Specific Parameters (Defaults handled by get_default_best_params) ---
    parser.add_argument('-param_threshold', type=float, default=None, help='Softmax threshold for ThresholdOSR.')
    parser.add_argument('-param_openmax_tailsize', type=int, default=None, help='Tail size for OpenMax.')
    parser.add_argument('-param_openmax_alpha', type=int, default=None, help='Alpha parameter for OpenMax.')
    parser.add_argument('-param_crosr_reconstruction_threshold', type=float, default=None, help='CDF threshold for CROSR.')
    parser.add_argument('-param_crosr_tailsize', type=int, default=None, help='Tail size for CROSR EVT.')
    parser.add_argument('-param_crosr_recon_weight', type=float, default=0.5, help='Reconstruction loss weight for CROSR model training.')
    parser.add_argument('-param_doc_k', type=float, default=None, help='k-sigma factor for DOC.')
    parser.add_argument('-param_adb_distance', type=str, default='cosine', choices=['cosine', 'euclidean'], help='Distance metric for ADB.')
    parser.add_argument('-param_adb_delta', type=float, default=0.1, help='Margin delta for ADB training loss.')
    parser.add_argument('-param_adb_alpha', type=float, default=0.5, help='Weight alpha for ADB training loss.')
    parser.add_argument('--adb_freeze_backbone', action=argparse.BooleanOptionalAction, default=True,
                        help='Freeze backbone during ADB training (default: True). Use --no-adb_freeze_backbone to disable freezing.')
    # --- Hyperparameter Tuning Arguments ---
    parser.add_argument('-parameter_search', action='store_true', help='Enable Optuna hyperparameter tuning.')
    parser.add_argument('-tuning_metric', type=str, default='f1_score', choices=['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate'], help='Metric to optimize during tuning.')
    parser.add_argument('-n_trials', type=int, default=20, help='Number of Optuna trials.')

    return parser.parse_args()

def check_gpu():
    print("\n----- GPU Diagnostics -----")
    if torch.cuda.is_available():
        print(f"CUDA Available: Yes, Devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()): print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        try:
            current_dev = torch.cuda.current_device()
            print(f"Current Device: {current_dev} ({torch.cuda.get_device_name(current_dev)})")
        except AssertionError as e:
            print(f"Could not get current device: {e}")
    else: print("CUDA Available: No")
    print("-------------------------\n")

def main():
    args = parse_args()
    print("\n----- Command Line Arguments -----"); print(json.dumps(vars(args), indent=2)); print("----------------------------------\n")
    check_gpu()
    print(f"Setting random seed: {args.random_seed}")
    pl.seed_everything(args.random_seed, workers=True) # Use pl seed_everything

    print(f"Loading tokenizer: {args.model}...")
    try: tokenizer = RobertaTokenizer.from_pretrained(args.model)
    except Exception as e: print(f"Error loading tokenizer '{args.model}': {e}"); sys.exit(1)

    print(f"Preparing DataModule for dataset: {args.dataset}...")
    datamodule = DataModule(
        dataset_name=args.dataset, tokenizer=tokenizer, batch_size=args.batch_size,
        seen_class_ratio=args.seen_class_ratio, random_seed=args.random_seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio,
        max_length=384, # Adjust as needed
        data_dir="data" # Pass data dir
    )
    # Prepare data runs the download/prep functions once
    datamodule.prepare_data() # This will exit if custom_syslog file is missing
    # Setup loads the data into splits
    datamodule.setup(stage=None) # Setup all splits
    num_model_classes = datamodule.num_seen_classes
    if num_model_classes is None or num_model_classes <= 0:
        raise ValueError(f"Number of seen classes is invalid ({num_model_classes}) after DataModule setup.")
    print(f"Model training will target {num_model_classes} known classes.")

    # --- Initialize Base Model ---
    # Decide initial model based on the *first* method to be run or default.
    initial_method = args.osr_method if args.osr_method != 'all' else 'threshold' # Default initial if 'all'
    print(f"\nInitializing base model architecture (targeting initial method: {initial_method.upper()})...")
    initial_model_class = MODEL_CLASS_MAP.get(initial_method if initial_method in MODEL_NEEDS_SPECIAL_TRAINING else 'standard')
    # Use args directly for initialization, _prepare_evaluation will handle tuning overrides
    init_kwargs = {
        'model_name': args.model,
        'num_classes': num_model_classes,
        'weight_decay': args.weight_decay,
        # Add scheduler params needed by the model's configure_optimizers later
        'warmup_steps': args.max_warmup_steps, # Pass max_warmup_steps initially
        'total_steps': 0 # Will be calculated in train_model
    }

    # Add model-specific arguments with correct names
    if initial_model_class == RobertaClassifier:
        init_kwargs['learning_rate'] = args.lr # RobertaClassifier expects learning_rate
    elif initial_model_class == RobertaAutoencoder:
        init_kwargs['learning_rate'] = args.lr # RobertaAutoencoder expects learning_rate
        init_kwargs['reconstruction_weight'] = args.param_crosr_recon_weight
        # latent_dim is fixed in the class definition, could be made an arg if needed
    elif initial_model_class == DOCRobertaClassifier:
        init_kwargs['learning_rate'] = args.lr # DOCRobertaClassifier expects learning_rate
    elif initial_model_class == RobertaADB:
        # ADB expects 'lr' for backbone and 'lr_adb' for params
        init_kwargs['lr'] = args.lr
        init_kwargs['lr_adb'] = args.lr_adb
        init_kwargs['param_adb_delta'] = args.param_adb_delta
        init_kwargs['param_adb_alpha'] = args.param_adb_alpha
        init_kwargs['adb_freeze_backbone'] = args.adb_freeze_backbone

    base_model = initial_model_class(**init_kwargs)

    # --- Train Initial Model ---
    # Only train initially if not doing parameter search OR if the first method needs it
    base_model = initial_model_class(**init_kwargs)

    # --- Train Initial Model ---
    # Only train initially if not doing parameter search OR if the first method needs it
    best_checkpoint_path = None
    loaded_model = None # Initialize loaded_model

    # Corrected logic: Always train initially if parameter search is OFF.
    # If parameter search is ON, only train initially if the method requires special training (crosr, doc, adb).
    # If parameter search is ON and method is standard (threshold, openmax), skip initial training.
    needs_initial_training = (not args.parameter_search) or (args.parameter_search and initial_method in MODEL_NEEDS_SPECIAL_TRAINING)

    if needs_initial_training:
        print("\nStep 1: Training the initial model...")
        # Pass args copy for training to avoid modification
        train_args = argparse.Namespace(**vars(args))
        # Use a specific osr_method name for the initial training run's checkpoints/logs
        train_args.osr_method = f"initial_{initial_method}"
        best_checkpoint_path = train_model(base_model, datamodule, train_args)
        if not best_checkpoint_path:
            print("Error: Initial training failed. Exiting."); sys.exit(1)
    else:
        print(f"\nStep 1: Skipping initial training (parameter search enabled for method '{initial_method}').")
        # Use the initialized instance as the starting point for evaluation/tuning
        loaded_model = base_model


    # --- Load Best Model (if trained initially) ---
    if best_checkpoint_path:
        print(f"\nStep 2: Loading best model from initial training: {best_checkpoint_path}")
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            # Load using the correct class
            loaded_model = initial_model_class.load_from_checkpoint(best_checkpoint_path, map_location=device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model from {best_checkpoint_path}: {e}"); sys.exit(1)
    elif loaded_model is None:
        # This case should not happen if logic above is correct, but as a safeguard:
        print("Error: No model available for evaluation (initial training skipped and no placeholder). Exiting.")
        sys.exit(1)
    # else: model already assigned if training was skipped

    # --- Evaluate OSR Algorithms ---
    print("\nStep 3: Evaluating OSR algorithm(s)...")
    # Pass the loaded or initialized model to the main evaluation function
    evaluate_osr_main(loaded_model, datamodule, args)

    print("\nExperiment finished.")

if __name__ == "__main__":
    main()
