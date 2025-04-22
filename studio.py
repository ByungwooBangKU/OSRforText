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
from dataset_utils import (
    prepare_bbc_news_dataset, prepare_trec_dataset, prepare_reuters8_dataset,
    prepare_acm_dataset, prepare_chemprot_dataset # Add others as needed
)
import os
import re
import zipfile
import urllib.request
import numpy as np
import pandas as pd # Make sure pandas is imported
from datasets import load_dataset                       # 🤗 Datasets
from sklearn.datasets import fetch_20newsgroups # Added for newsgroup20
import shutil # Added import
# --- Basic Setup ---
# Set default encoding
# import io
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
# sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Suppress excessive warnings from transformers/tokenizers
hf_logging.set_verbosity_error()
# Suppress PyTorch Lightning UserWarnings about processes/workers
import warnings
warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*DataLoader running processes.*")

# --- Matplotlib Korean Font Setup ---
# Check available fonts and set one that supports Korean
# Requires user to have a Korean font installed.
def setup_korean_font():
    font_name = None
    try:
        # Try common Korean font names
        possible_fonts = ['Malgun Gothic', 'NanumGothic', 'Apple SD Gothic Neo', 'Noto Sans KR']
        available_fonts = [f.name for f in fm.fontManager.ttflist]

        for font in possible_fonts:
            if font in available_fonts:
                font_name = font
                break

        if font_name:
            plt.rcParams['font.family'] = font_name
            plt.rcParams['axes.unicode_minus'] = False # Handle minus sign display
            # print(f"Korean font '{font_name}' set for Matplotlib.")
        else:
            print("Warning: No common Korean font found. Install 'Malgun Gothic' or 'NanumGothic'. Plots might not display Korean characters correctly.")
            # Fallback to default font
            plt.rcParams['axes.unicode_minus'] = False

    except Exception as e:
        print(f"Error setting up Korean font: {e}")
        plt.rcParams['axes.unicode_minus'] = False

setup_korean_font()


# =============================================================================
# Data Processing (TextDataset, DataModule) - Keep as provided previously
# Ensure DataModule correctly handles seen/unseen splits and label remapping
# =============================================================================
# --- START OF studio.py (TextDataset fix only) ---
# ... (다른 임포트 및 클래스 정의) ...

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
        text = str(self.texts[idx]) if self.texts[idx] is not None else "" # Handle potential None
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            # 수정된 부분: RoBERTa는 token_type_ids를 사용하지 않으므로 False로 고정
            return_token_type_ids=False,
            return_tensors='pt'
        )

        # token_type_ids가 반환되지 않으므로 item 구성 시에도 제거
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
        # if 'token_type_ids' in encoding: # 이 부분은 이제 필요 없음
        #      item['token_type_ids'] = encoding['token_type_ids'].flatten()

        return item

# ... (파일의 나머지 부분) ...
# --- END OF studio.py (TextDataset fix only) ---

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
        test_ratio=0.15
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.seen_class_ratio = seen_class_ratio
        self.random_seed = random_seed
        self.max_length = max_length
        self.num_classes = None # Total number of original classes before split
        self.num_seen_classes = None # Number of classes used for training (Known classes)
        self.seen_classes = None # Indices of known classes (in original labeling, 0..N-1)
        self.unseen_classes = None # Indices of unknown classes (in original labeling) - Not directly used by models typically
        self.label_encoder = LabelEncoder() # Primarily for custom_syslog

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

    def prepare_data(self):
        """Downloads or prepares dataset files (called once per node)."""
        print(f"Preparing data for dataset: {self.dataset_name}...")
        # Add logic here if dataset requires explicit download steps
        # For datasets loaded via scripts/libraries, this might be empty.
        if self.dataset_name == "bbc_news":
             prepare_bbc_news_dataset() # Ensures download/extraction
        elif self.dataset_name == "trec":
             prepare_trec_dataset()
        # Add others if needed
        print("Data preparation step complete.")


    def setup(self, stage=None):
        """Loads and splits data. Called on every GPU."""
        if self.train_dataset is not None and stage == 'fit': return # Avoid redundant setup
        if self.test_dataset is not None and stage == 'test': return

        np.random.seed(self.random_seed)
        random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_seed)

        print(f"\n--- Setting up DataModule for dataset: {self.dataset_name} (Seen Ratio: {self.seen_class_ratio}) ---")

        # --- Load Data ---
        if self.dataset_name == "newsgroup20":
            print("Loading 20 Newsgroups dataset...")
            # Load combined data for consistent splitting
            data_all = fetch_20newsgroups(
                subset='all', remove=('headers', 'footers', 'quotes'),
                random_state=self.random_seed # Ensure consistent load if possible
            )
            texts = data_all.data
            labels = data_all.target # Numeric labels 0-19
            self.class_names = data_all.target_names
        elif self.dataset_name == "bbc_news":
            print("Loading BBC News dataset...")
            texts, labels, self.class_names = prepare_bbc_news_dataset()
        elif self.dataset_name == "trec":
            print("Loading TREC dataset...")
            # TREC loader gives pre-split, combine and resplit
            (train_texts, train_labels), (test_texts, test_labels), self.class_names = prepare_trec_dataset()
            texts = train_texts + test_texts
            labels = np.concatenate([train_labels, test_labels])
        elif self.dataset_name == "reuters8":
            print("Loading Reuters-8 dataset...")
            texts, labels, self.class_names = prepare_reuters8_dataset()
        elif self.dataset_name == "acm":
            print("Loading ACM dataset...")
            texts, labels, self.class_names = prepare_acm_dataset()
        elif self.dataset_name == "chemprot":
            print("Loading ChemProt dataset...")
            texts, labels, self.class_names = prepare_chemprot_dataset()
        # Add other dataset loading logic here
        # elif self.dataset_name == "custom_syslog":
        #     self._setup_custom_syslog() # Specific logic for syslog
        #     print("--- Finished DataModule setup ---")
        #     return
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        if not texts:
             raise ValueError(f"Failed to load any text data for dataset '{self.dataset_name}'. Please check the loading function and data source.")

        # Ensure labels are numpy array
        labels = np.array(labels)

        # --- Train/Val/Test Split ---
        print("Splitting data into train/validation/test sets...")
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            texts, labels,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            stratify=labels
        )
        val_size_relative = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val,
            test_size=val_size_relative,
            random_state=self.random_seed,
            stratify=y_train_val
        )

        # Get all unique original class labels
        all_original_indices = np.unique(labels)
        self.num_classes = len(all_original_indices)
        print(f"Total original classes found: {self.num_classes} -> {all_original_indices}")
        if self.class_names is None: # Should be set by specific loaders
             self.class_names = [str(i) for i in all_original_indices]
             print(f"Warning: class_names not explicitly set. Using: {self.class_names}")

        # --- Seen/Unseen Class Split Logic ---
        if self.seen_class_ratio < 1.0:
            print(f"Splitting classes: {self.seen_class_ratio*100:.1f}% Seen / {(1-self.seen_class_ratio)*100:.1f}% Unseen")
            num_seen = max(1, int(np.round(self.num_classes * self.seen_class_ratio))) # Round to nearest int
            # Shuffle original indices consistently
            np.random.seed(self.random_seed) # Ensure same shuffle each time setup is called
            all_classes_shuffled = np.random.permutation(all_original_indices)

            self.original_seen_indices = np.sort(all_classes_shuffled[:num_seen])
            self.original_unseen_indices = np.sort(all_classes_shuffled[num_seen:])

            print(f"  Original Seen Indices  : {self.original_seen_indices}")
            print(f"  Original Unseen Indices: {self.original_unseen_indices}")

            # Create mapping for training: original seen index -> 0..num_seen-1
            self.seen_class_mapping = {orig_idx: new_idx for new_idx, orig_idx in enumerate(self.original_seen_indices)}
            self.num_seen_classes = len(self.original_seen_indices)
            self.seen_classes = np.arange(self.num_seen_classes) # Model trains on these indices

            # Filter Train set: Keep only original_seen_indices and remap labels
            train_seen_mask = np.isin(y_train, self.original_seen_indices)
            X_train = [X_train[i] for i, keep in enumerate(train_seen_mask) if keep]
            y_train_original = y_train[train_seen_mask]
            y_train_mapped = np.array([self.seen_class_mapping[lbl] for lbl in y_train_original])

            # Filter Validation set: Keep only original_seen_indices and remap labels
            val_seen_mask = np.isin(y_val, self.original_seen_indices)
            X_val = [X_val[i] for i, keep in enumerate(val_seen_mask) if keep]
            y_val_original = y_val[val_seen_mask]
            y_val_mapped = np.array([self.seen_class_mapping[lbl] for lbl in y_val_original])

            # Test set: Keep all original samples, but map unseen original labels to -1 for evaluation
            # The model will predict 0..N-1, but OSR evaluation needs to know the true state (known/unknown)
            y_test_mapped = y_test.copy() # Start with original test labels
            unseen_test_mask = np.isin(y_test, self.original_unseen_indices)
            y_test_mapped[unseen_test_mask] = -1 # Mark original unseen classes as -1
            # Seen classes in test set retain their *original* indices for easier CM interpretation later
            # Let OSR algorithms handle mapping during prediction/evaluation if needed

            # Assign final datasets
            y_train_final = y_train_mapped
            y_val_final = y_val_mapped
            y_test_final = y_test_mapped

        else: # seen_class_ratio == 1.0 (All classes are known, standard classification)
            print("All classes are considered Known (seen_class_ratio = 1.0)")
            self.original_seen_indices = all_original_indices.copy()
            self.original_unseen_indices = np.array([])
            self.num_seen_classes = self.num_classes
            self.seen_classes = all_original_indices.copy() # Use original indices directly
            self.seen_class_mapping = {orig_idx: orig_idx for orig_idx in all_original_indices} # Identity map

            # Keep original labels for all splits
            y_train_final = y_train
            y_val_final = y_val
            y_test_final = y_test # Test labels are original indices

        # --- Create TextDataset instances ---
        self.train_dataset = TextDataset(X_train, y_train_final, self.tokenizer, self.max_length)
        self.val_dataset = TextDataset(X_val, y_val_final, self.tokenizer, self.max_length)
        self.test_dataset = TextDataset(X_test, y_test_final, self.tokenizer, self.max_length) # Uses labels with -1 for unknowns

        print(f"\nDataset sizes:")
        print(f"  Train: {len(self.train_dataset)} samples")
        print(f"  Validation: {len(self.val_dataset)} samples")
        print(f"  Test: {len(self.test_dataset)} samples")
        print(f"Number of classes for model training (Known): {self.num_seen_classes}")
        if self.seen_class_ratio < 1.0:
             num_unseen_in_test = np.sum(y_test_final == -1)
             print(f"Number of original unseen classes: {len(self.original_unseen_indices)}")
             print(f"Number of samples marked as Unknown (-1) in test set: {num_unseen_in_test}")
        # print(f"Known class names (original indices {self.original_seen_indices}): {[self.class_names[i] for i in self.original_seen_indices]}")
        # print(f"Unknown class index used in test set: -1")
        print("--- Finished DataModule setup ---")

    def train_dataloader(self):
        if self.train_dataset is None: raise ValueError("Train dataset not initialized.")
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1), # Limit workers
                          persistent_workers=True if min(4, os.cpu_count() // 2 if os.cpu_count() else 1) > 0 else False, pin_memory=True)

    def val_dataloader(self):
        if self.val_dataset is None: raise ValueError("Validation dataset not initialized.")
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1),
                          persistent_workers=True if min(4, os.cpu_count() // 2 if os.cpu_count() else 1) > 0 else False, pin_memory=True)

    def test_dataloader(self):
        if self.test_dataset is None: raise ValueError("Test dataset not initialized.")
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          num_workers=min(4, os.cpu_count() // 2 if os.cpu_count() else 1),
                          persistent_workers=True if min(4, os.cpu_count() // 2 if os.cpu_count() else 1) > 0 else False, pin_memory=True)

    def _determine_unknown_labels(self, labels_np):
        """Helper to consistently determine true unknown labels for evaluation based on setup."""
        # In the current setup, the test dataset labels (`y_test_final`) are already marked
        # with -1 if they belong to an original unseen class (when seen_ratio < 1.0)
        # or if they were explicitly unknown (like in custom_syslog).
        # Therefore, the true unknown mask is simply where labels_np == -1.
        unknown_mask = (labels_np == -1)
        return unknown_mask

# =============================================================================
# Model Definitions (RobertaClassifier, RobertaAutoencoder, DOCRobertaClassifier, RobertaADB)
# Keep as provided previously, ensure initialization uses num_seen_classes
# =============================================================================
class RobertaClassifier(pl.LightningModule):
    """Standard RoBERTa model with a classification head."""
    def __init__(
        self,
        model_name="roberta-base",
        num_classes=20, # Number of *known* classes for the classifier
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=0,
        total_steps=0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_name'])
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = num_classes

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
        loss = F.cross_entropy(logits, batch['label'])
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Single validation step."""
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        loss = F.cross_entropy(logits, batch['label'])
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(batch['label'].cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        """Single test step."""
        logits, embeddings = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels = batch['label'] # Original labels (can include -1)
        preds = torch.argmax(logits, dim=1)

        # Calculate loss & acc only on known samples
        known_mask = labels >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             loss = F.cross_entropy(logits[known_mask], labels[known_mask])
             acc = accuracy_score(labels[known_mask].cpu(), preds[known_mask].cpu())
             self.log('test_loss', loss, prog_bar=False, logger=True)
             self.log('test_acc_known', acc, prog_bar=False, logger=True) # Log accuracy on knowns

        return {
            'preds': preds,      # Predictions indices (0..N-1) for all samples
            'labels': labels,    # Original labels (can include -1)
            'logits': logits,    # Raw logits
            'embeddings': embeddings # CLS embeddings
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        # Ensure total_steps is calculated correctly before scheduler creation
        if self.total_steps <= 0:
             print("Warning: total_steps not set for scheduler. Estimating from trainer.")
             # Try to estimate from trainer if available (might require trainer reference or manual setting)
             if self.trainer and self.trainer.estimated_stepping_batches:
                 self.total_steps = self.trainer.estimated_stepping_batches
             else:
                 self.total_steps = 10000 # Fallback
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
                  warmup_steps=0, total_steps=0, latent_dim=256, reconstruction_weight=0.5 ): # Smaller default latent_dim
        super().__init__()
        self.save_hyperparameters(ignore=['model_name'])
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        # Encoder/Decoder Layers
        self.encoder = nn.Sequential( nn.Linear(self.config.hidden_size, latent_dim), nn.ReLU() )
        self.decoder = nn.Sequential( nn.Linear(latent_dim, self.config.hidden_size) ) # Often no activation on final decoder layer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.reconstruction_weight = reconstruction_weight
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        encoded = self.encoder(cls_output)
        reconstructed = self.decoder(encoded)
        return logits, cls_output, encoded, reconstructed

    def training_step(self, batch, batch_idx):
        logits, cls_output, _, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        class_loss = F.cross_entropy(logits, batch['label'])
        recon_loss = F.mse_loss(reconstructed, cls_output)
        loss = class_loss + self.reconstruction_weight * recon_loss
        self.log_dict({'train_loss': loss, 'train_class_loss': class_loss, 'train_recon_loss': recon_loss},
                      prog_bar=False, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, cls_output, _, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        class_loss = F.cross_entropy(logits, batch['label'])
        recon_loss = F.mse_loss(reconstructed, cls_output)
        loss = class_loss + self.reconstruction_weight * recon_loss
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(batch['label'].cpu(), preds.cpu())
        self.log_dict({'val_loss': loss, 'val_class_loss': class_loss, 'val_recon_loss': recon_loss, 'val_acc': acc},
                       prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        logits, cls_output, encoded, reconstructed = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels = batch['label']
        preds = torch.argmax(logits, dim=1)
        recon_errors = torch.norm(reconstructed - cls_output, p=2, dim=1)

        # Calculate loss/acc on knowns only
        known_mask = labels >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             class_loss = F.cross_entropy(logits[known_mask], labels[known_mask])
             recon_loss = F.mse_loss(reconstructed[known_mask], cls_output[known_mask])
             loss = class_loss + self.reconstruction_weight * recon_loss
             acc = accuracy_score(labels[known_mask].cpu(), preds[known_mask].cpu())
             self.log('test_loss', loss, prog_bar=False, logger=True)
             self.log('test_acc_known', acc, prog_bar=False, logger=True)

        return {
            'preds': preds, 'labels': labels, 'logits': logits,
            'embeddings': cls_output, 'encoded': encoded, 'reconstructed': reconstructed,
            'recon_errors': recon_errors # Crucial for CROSR evaluation
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.total_steps <= 0: self.total_steps = 10000 # Fallback
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
        self.classifiers = nn.ModuleList([nn.Linear(self.config.hidden_size, 1) for _ in range(num_classes)])
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.num_classes = num_classes

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        cls_output = outputs.last_hidden_state[:, 0, :]
        # Apply each binary classifier and concatenate logits
        logits = torch.cat([clf(cls_output) for clf in self.classifiers], dim=1) # Shape: (batch_size, num_classes)
        return logits, cls_output

    def training_step(self, batch, batch_idx):
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels = batch['label']
        # Create multi-label binary targets (one-hot)
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(logits, one_hot_labels)
        self.log('train_loss', loss, prog_bar=True, on_step=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits, _ = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels = batch['label']
        one_hot_labels = F.one_hot(labels, num_classes=self.num_classes).float()
        loss = F.binary_cross_entropy_with_logits(logits, one_hot_labels)
        # Predict based on highest sigmoid score
        sigmoid_scores = torch.sigmoid(logits)
        preds = torch.argmax(sigmoid_scores, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log('val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, batch_idx):
        logits, embeddings = self(batch['input_ids'], batch['attention_mask'], batch.get('token_type_ids'))
        labels = batch['label'] # Original labels (-1 possible)
        sigmoid_scores = torch.sigmoid(logits)
        preds = torch.argmax(sigmoid_scores, dim=1) # Predicted class index (0..N-1)

        # Calculate loss/acc on knowns only
        known_mask = labels >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             one_hot_labels = F.one_hot(labels[known_mask], num_classes=self.num_classes).float()
             loss = F.binary_cross_entropy_with_logits(logits[known_mask], one_hot_labels)
             acc = accuracy_score(labels[known_mask].cpu(), preds[known_mask].cpu())
             self.log('test_loss', loss, prog_bar=False, logger=True)
             self.log('test_acc_known', acc, prog_bar=False, logger=True)

        return {
            'preds': preds,           # Predicted class index (0..N-1)
            'labels': labels,         # Original labels
            'logits': logits,         # Raw one-vs-rest logits
            'sigmoid_scores': sigmoid_scores, # Sigmoid scores (crucial for DOC eval)
            'embeddings': embeddings  # CLS embeddings
        }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        if self.total_steps <= 0: self.total_steps = 10000 # Fallback
        actual_warmup_steps = min(self.warmup_steps, self.total_steps)
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
        )
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}

class RobertaADB(pl.LightningModule):
    """RoBERTa model with Adaptive Decision Boundary (ADB) components."""
    def __init__( self, model_name: str = "roberta-base", num_classes: int = 20, learning_rate: float = 1e-3,
                  weight_decay: float = 0.0, warmup_steps: int = 0, total_steps: int = 0,
                  delta: float = 0.1, alpha: float = 0.1, freeze_backbone: bool = True ):
        super().__init__()
        self.save_hyperparameters(ignore=['model_name'])
        self.config = RobertaConfig.from_pretrained(model_name)
        self.roberta = RobertaModel.from_pretrained(model_name, config=self.config)
        self.feat_dim = self.config.hidden_size

        # Learnable centers
        self.centers = nn.Parameter(torch.empty(num_classes, self.feat_dim), requires_grad=True)
        nn.init.normal_(self.centers, std=0.05)

        # --- 수정: Radii 대신 Logits for Radii (Delta' in paper) 학습 ---
        # Softplus(delta_prime) = Delta (Radius) -> ensures Radius > 0
        # Initialize delta_prime such that initial radius is around 0.1-0.3
        # softplus(x) = log(1+exp(x)). If we want softplus(x) ~ 0.2, exp(x) ~ exp(0.2)-1 ~ 0.22. x ~ log(0.22) ~ -1.5
        initial_delta_prime = -1.5
        self.delta_prime = nn.Parameter(torch.full((num_classes,), initial_delta_prime), requires_grad=True)
        # --- ---

        self.learning_rate = learning_rate # LR for centers and delta_prime
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.delta = delta # Margin for ADB loss
        self.alpha = alpha # Weight for ADB loss
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        if self.freeze_backbone:
            print("[RobertaADB Init] Freezing RoBERTa backbone parameters.")
            for param in self.roberta.parameters(): param.requires_grad = False
        else: print("[RobertaADB Init] RoBERTa backbone parameters will be fine-tuned.")

    # --- 수정: get_radii 메서드 추가 ---
    def get_radii(self):
        """Calculate actual radii using Softplus."""
        return F.softplus(self.delta_prime)
    # --- ---

    @staticmethod
    def _cosine_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Calculates pairwise cosine distance (1 - similarity). Input x assumed normalized."""
        y_norm = F.normalize(y, p=2, dim=-1)
        similarity = torch.matmul(x, y_norm.t())
        # Clamp similarity to avoid numerical issues with acos or 1.0 - sim
        similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7)
        distance = 1.0 - similarity
        return distance

    def adb_margin_loss(self, feat_norm: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Calculates ADB margin loss: max(0, d(feat, c_y) - r_y + delta)."""
        distances = self._cosine_distance(feat_norm, self.centers) # feat_norm is normalized
        d_y = distances[torch.arange(feat_norm.size(0), device=self.device), labels]

        # --- 수정: get_radii() 사용 ---
        radii = self.get_radii() # Get positive radii via Softplus
        r_y = radii[labels]
        # --- ---

        # Loss = max(0, distance_to_correct_center - radius_of_correct_center + margin)
        loss_per_sample = torch.relu(d_y - r_y + self.delta)
        loss = loss_per_sample.mean()
        return loss

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        """Forward pass: get CLS embedding, normalize it, calculate similarity logits."""
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        feat = outputs.last_hidden_state[:, 0, :]
        feat_norm = F.normalize(feat, p=2, dim=-1)
        logits = 1.0 - self._cosine_distance(feat_norm, self.centers)
        return logits, feat_norm

    def training_step(self, batch, batch_idx):
        labels = batch["label"]
        logits, feat_norm = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        ce_loss = F.cross_entropy(logits, labels)
        adb_loss = self.adb_margin_loss(feat_norm, labels)
        loss = ce_loss + self.alpha * adb_loss
        self.log_dict({'train_loss': loss, 'train_ce_loss': ce_loss, 'train_adb_loss': adb_loss},
                      prog_bar=True, on_step=True, on_epoch=True, logger=True)
        # --- 수정: Log actual radius ---
        avg_radius = self.get_radii().mean().item()
        self.log("avg_radius", avg_radius, on_step=False, on_epoch=True, logger=True)
        # --- ---
        return loss

    def validation_step(self, batch, batch_idx):
        labels = batch["label"]
        logits, feat_norm = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        ce_loss = F.cross_entropy(logits, labels)
        adb_loss = self.adb_margin_loss(feat_norm, labels)
        loss = ce_loss + self.alpha * adb_loss
        preds = torch.argmax(logits, dim=1)
        acc = accuracy_score(labels.cpu(), preds.cpu())
        self.log_dict({'val_loss': loss, 'val_ce_loss': ce_loss, 'val_adb_loss': adb_loss, 'val_acc': acc},
                      prog_bar=True, on_step=False, on_epoch=True, logger=True)
        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch, batch_idx):
        labels = batch["label"] # Original labels (-1 possible)
        logits, feat_norm = self(batch["input_ids"], batch["attention_mask"], batch.get("token_type_ids"))
        preds = torch.argmax(logits, dim=1) # Predictions based on highest similarity

        # Calculate loss/acc on knowns only
        known_mask = labels >= 0
        loss = torch.tensor(0.0, device=self.device)
        acc = torch.tensor(0.0, device=self.device)
        if known_mask.any():
             ce_loss = F.cross_entropy(logits[known_mask], labels[known_mask])
             adb_loss = self.adb_margin_loss(feat_norm[known_mask], labels[known_mask])
             loss = ce_loss + self.alpha * adb_loss
             acc = accuracy_score(labels[known_mask].cpu(), preds[known_mask].cpu())
             self.log('test_loss', loss, prog_bar=False, logger=True)
             self.log('test_acc_known', acc, prog_bar=False, logger=True)

        return {
            'preds': preds,        # Predicted class index (0..N-1) based on similarity
            'labels': labels,      # Original labels
            'logits': logits,      # Similarity-based logits
            'features': feat_norm # Normalized features (crucial for ADB eval)
        }
        
    def configure_optimizers(self):
        """Configure optimizer."""
        params_to_optimize = []
        if not self.freeze_backbone:
             backbone_lr = getattr(self.hparams, 'lr', 2e-5) # General LR for backbone
             params_to_optimize.append({'params': self.roberta.parameters(), 'lr': backbone_lr})
             print(f"[ADB Optim] Fine-tuning RoBERTa with LR: {backbone_lr}")

        # Optimize centers and delta_prime (for radii) with the specific ADB LR
        params_to_optimize.append({'params': self.centers, 'lr': self.learning_rate})
        # --- 수정: radii 대신 delta_prime 최적화 ---
        params_to_optimize.append({'params': self.delta_prime, 'lr': self.learning_rate})
        # --- ---
        print(f"[ADB Optim] Optimizing centers/delta_prime with LR: {self.learning_rate}")

        optimizer = AdamW(params_to_optimize, weight_decay=self.weight_decay)

        # Scheduler logic remains the same
        if not self.freeze_backbone and self.total_steps > 0:
             actual_warmup_steps = min(self.warmup_steps, self.total_steps)
             scheduler = get_linear_schedule_with_warmup(
                 optimizer, num_warmup_steps=actual_warmup_steps, num_training_steps=self.total_steps
             )
             print("[ADB Optim] Using learning rate scheduler.")
             return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'interval': 'step'}}
        else:
             print("[ADB Optim] No learning rate scheduler used.")
             return optimizer

# =============================================================================
# OSR Algorithms (Base Class, ThresholdOSR, OpenMaxOSR, CROSROSR, DOCOSR, ADBOSR)
# Keep implementations as provided previously, ensuring they use the correct model outputs
# and handle label mapping / unknown determination correctly via datamodule.
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
              return {i: f"Known_{i}" for i in range(self.num_known_classes)}

         # Map original seen indices back to names
         seen_names = {}
         for original_idx in self.datamodule.original_seen_indices:
              if 0 <= original_idx < len(self.datamodule.class_names):
                   # The key should be the original index for CM labels
                   seen_names[original_idx] = self.datamodule.class_names[original_idx]
              else:
                   seen_names[original_idx] = f"Class_{original_idx}" # Fallback name
         return seen_names

    def _get_cm_labels(self):
         """Gets integer labels and string names for confusion matrix axes."""
         seen_class_names_map = self._get_seen_class_names()
         # CM labels should include -1 for Unknown and the *original* indices of seen classes
         cm_axis_labels_int = [-1] + sorted(list(self.datamodule.original_seen_indices))
         cm_axis_labels_names = ["Unknown"] + [seen_class_names_map.get(lbl, str(lbl)) for lbl in cm_axis_labels_int if lbl != -1]
         return cm_axis_labels_int, cm_axis_labels_names

class ThresholdOSR(OSRAlgorithm):
    """OSR using a simple threshold on the maximum softmax probability."""
    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        # Ensure model is the standard classifier
        if not isinstance(model, RobertaClassifier):
             print(f"Warning: ThresholdOSR typically uses RobertaClassifier, but received {type(model)}. Proceeding anyway.")
        # Get threshold from args, fallback handled by get_default_best_params or argparse default
        self.threshold = getattr(args, 'param_threshold', 0.5) # Use default 0.5 if not set
        print(f"[ThresholdOSR Init] Using softmax threshold: {self.threshold:.4f}")

    def predict(self, dataloader):
        self.model.eval().to(self.device)
        all_max_probs = []
        all_probs = []
        all_preds_final = [] # Final predictions: original seen indices or -1
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (Threshold OSR)"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None: token_type_ids = token_type_ids.to(self.device)
                labels_orig = batch['label'].cpu().numpy() # Keep original labels from loader

                # Get logits from the base classifier model (output size = num_seen_classes)
                logits, _ = self.model(input_ids, attention_mask, token_type_ids)
                probs = F.softmax(logits, dim=1)
                max_probs, pred_indices = torch.max(probs, dim=1) # pred_indices are 0..N-1

                # Apply threshold
                pred_indices_cpu = pred_indices.cpu().numpy()
                max_probs_cpu = max_probs.cpu().numpy()
                final_batch_preds = np.full_like(pred_indices_cpu, -1) # Initialize with -1

                # Map valid predictions back to original indices
                reject_mask = max_probs_cpu < self.threshold
                accept_mask = ~reject_mask

                if np.any(accept_mask):
                     accepted_indices = pred_indices_cpu[accept_mask]
                     # Map 0..N-1 back to original seen indices
                     original_indices = self.datamodule.original_seen_indices[accepted_indices]
                     final_batch_preds[accept_mask] = original_indices

                all_max_probs.append(max_probs_cpu)
                all_probs.append(probs.cpu().numpy())
                all_preds_final.extend(final_batch_preds)
                all_labels.extend(labels_orig)

        all_max_probs = np.concatenate(all_max_probs)
        all_probs = np.concatenate(all_probs)
        all_preds_final = np.array(all_preds_final)
        all_labels = np.array(all_labels)

        return all_probs, all_preds_final, all_labels, all_max_probs

    def evaluate(self, dataloader):
        all_probs, all_preds, all_labels, all_max_probs = self.predict(dataloader)
        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1)
        known_mask = ~unknown_labels_mask

        # Metrics
        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        auroc = roc_auc_score(unknown_labels_mask, -all_max_probs) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        # CM and F1
        labels_mapped_for_cm = all_labels.copy()
        labels_mapped_for_cm[unknown_labels_mask] = -1
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        conf_matrix = confusion_matrix(labels_mapped_for_cm, all_preds, labels=cm_axis_labels_int)
        precision, recall, f1_by_class, _ = precision_recall_fscore_support(
            labels_mapped_for_cm, all_preds, labels=cm_axis_labels_int, average=None, zero_division=0
        )
        macro_f1 = np.mean(f1_by_class)

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
        labels_np = results['labels']
        max_probs = results['max_probs']
        unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)

        # ROC Curve
        if len(np.unique(unknown_labels_mask)) > 1:
            fpr, tpr, _ = roc_curve(unknown_labels_mask, -max_probs) # Low score = unknown
            roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1], 'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (Threshold)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close()
            print(f"  ROC curve saved.")
        else: print("  Skipping ROC (only one class).")

        # Confidence Distribution
        plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': max_probs, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False); plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold={self.threshold:.2f}'); plt.title('Confidence Distribution (Threshold)'); plt.xlabel('Max Softmax Probability'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_confidence.png"); plt.close()
        print(f"  Confidence distribution saved.")

        # Confusion Matrix
        plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (Threshold)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close()
        print(f"  Confusion matrix saved.")
        print("[ThresholdOSR Visualize] Finished.")


# --- OpenMaxOSR, CROSROSR, DOCOSR, ADBOSR classes ---
# Keep implementations largely as provided before, ensuring:
# 1. They inherit from OSRAlgorithm.
# 2. `fit_*` methods use `datamodule.train_dataloader()`.
# 3. `predict` methods handle label mapping correctly (map 0..N-1 back to original seen indices or -1).
# 4. `evaluate` methods use `datamodule._determine_unknown_labels` and `_get_cm_labels`.
# 5. `visualize` methods save plots correctly.

# (Include the full class implementations for OpenMaxOSR, CROSROSR, DOCOSR, ADBOSR here -
#  they are quite long, so reusing the previously reviewed versions is efficient.
#  Make sure the predict methods correctly map internal indices back to original seen indices
#  before returning `all_preds_final`.)

# Example Snippet for OpenMaxOSR predict mapping:
# --- START OF studio.py (OpenMaxOSR fix only) ---
# ... (다른 임포트 및 클래스 정의) ...

class OpenMaxOSR(OSRAlgorithm):
    """
    OpenMax OSR algorithm implementation (Bendale & Boult, 2015).
    Uses penultimate layer features (embeddings) for EVT fitting.
    """
    def __init__(self, model, datamodule, args):
        super().__init__(model, datamodule, args)
        # Ensure model is the standard classifier
        if not isinstance(model, RobertaClassifier):
             print(f"Warning: OpenMaxOSR typically uses RobertaClassifier, but received {type(model)}. Proceeding anyway.")

        # OpenMax specific parameters from args
        self.tail_size = getattr(args, 'param_openmax_tailsize', 50) # Default 50
        self.alpha = getattr(args, 'param_openmax_alpha', 10) # Default 10
        # num_known_classes is inherited from OSRAlgorithm init

        print(f"[OpenMaxOSR Init] Tail size: {self.tail_size}, Alpha: {self.alpha}, Known Classes: {self.num_known_classes}")

        # --- 수정된 부분: mav와 weibull_models 초기화 ---
        # Stores Mean Activation Vectors (MAVs) per known class (internal index 0..N-1)
        self.mav: dict[int, np.ndarray] = {}
        # Stores fitted Weibull models (shape, loc, scale) per known class (internal index 0..N-1)
        self.weibull_models: dict[int, tuple[float, float, float]] = {}
        # ---------------------------------------------

        # Feature dimension from the model config (ensure model has config)
        self.feat_dim = model.config.hidden_size if hasattr(model, 'config') and hasattr(model.config, 'hidden_size') else 768 # Fallback

    def fit_weibull(self, dataloader):
        """
        Fits Weibull models based on distances between correctly classified
        training sample embeddings and their class MAVs.
        """
        print("[OpenMaxOSR Fit] Fitting Weibull models...")
        self.model.eval().to(self.device)
        # Store embeddings (AVs) per class {class_idx: [embedding1, embedding2, ...]}
        # Uses model's internal class indices (0 to num_known_classes-1)
        av_per_class = {c: [] for c in range(self.num_known_classes)}

        with torch.no_grad():
            # Ensure train dataloader provides mapped labels (0..N-1)
            for batch in tqdm(dataloader, desc="OpenMax Fit: Collecting embeddings"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_mapped = batch["label"].to(self.device) # Expecting 0..N-1 labels

                # Get logits and embeddings (penultimate features, AVs)
                logits, embeddings = self.model(ids, attn, tok)
                preds = torch.argmax(logits, dim=1) # Predicted class index (0..N-1)

                for i, true_mapped_idx in enumerate(labels_mapped):
                    true_idx = true_mapped_idx.item()
                    pred_idx = preds[i].item()
                    # Only use samples that are correctly classified for the current class
                    if pred_idx == true_idx and 0 <= true_idx < self.num_known_classes:
                        av_per_class[true_idx].append(embeddings[i].cpu().numpy())

        # Calculate MAVs and fit Weibull models
        self.mav.clear() # Clear previous results if any
        self.weibull_models.clear()
        print("[OpenMaxOSR Fit] Calculating MAVs and fitting Weibull models...")
        for c_idx, av_list in tqdm(av_per_class.items(), desc="OpenMax Fit: Weibull Fitting"):
            if not av_list:
                print(f"  Warning: No correctly classified samples found for internal class index {c_idx}. Skipping Weibull fitting.")
                continue

            avs = np.stack(av_list)
            self.mav[c_idx] = np.mean(avs, axis=0) # Calculate MAV

            # Calculate distances between each AV and the class MAV
            distances = np.linalg.norm(avs - self.mav[c_idx], axis=1)
            distances_sorted = np.sort(distances) # Sort distances ascendingly

            # Select the tail for Weibull fitting
            current_tail_size = min(self.tail_size, len(distances_sorted))
            if current_tail_size < 2:
                print(f"  Warning: Insufficient tail points ({current_tail_size}) for class index {c_idx}. Using default Weibull.")
                mean_dist = np.mean(distances_sorted) if len(distances_sorted) > 0 else 1.0
                shape, loc, scale = 1.0, 0.0, mean_dist
            else:
                tail_distances = distances_sorted[-current_tail_size:]
                try:
                    shape, loc, scale = weibull_min.fit(tail_distances, floc=0) # Fit Weibull (fix location to 0)
                    if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9:
                        print(f"  Warning: Weibull fit failed for class {c_idx} (invalid params). Using default.")
                        shape, loc, scale = 1.0, 0.0, np.mean(tail_distances)
                except Exception as e:
                    print(f"  Warning: Weibull fit exception for class {c_idx}: {e}. Using default.")
                    shape, loc, scale = 1.0, 0.0, np.mean(tail_distances)

            self.weibull_models[c_idx] = (shape, loc, scale)
            # print(f"  Class {c_idx}: MAV shape {self.mav[c_idx].shape}, Weibull (shape={shape:.2f}, scale={scale:.2f})")
        print("[OpenMaxOSR Fit] Weibull fitting complete.")


    def openmax_probability(self, embedding_av: np.ndarray, logits: np.ndarray) -> np.ndarray:
        """Recalibrates logits based on Weibull CDF scores."""
        if not self.mav or not self.weibull_models:
            # This check should ideally be done before calling this function
            raise RuntimeError("MAV or Weibull models not calculated. Call fit_weibull first.")

        num_known = len(logits)
        if num_known != self.num_known_classes:
             print(f"Warning: Logits dim ({num_known}) != expected known classes ({self.num_known_classes}).")
             # Adjust alpha if needed? Or proceed with caution.
             current_alpha = min(self.alpha, num_known)
        else:
             current_alpha = self.alpha

        distances = np.full(num_known, np.inf) # Default to infinite distance if MAV missing
        for c_idx in range(num_known):
             if c_idx in self.mav:
                 distances[c_idx] = np.linalg.norm(embedding_av - self.mav[c_idx])

        cdf_scores = np.ones(num_known) # Default to 1 (max rejection) if model missing
        for c_idx in range(num_known):
            if c_idx in self.weibull_models and np.isfinite(distances[c_idx]):
                shape, loc, scale = self.weibull_models[c_idx]
                cdf_scores[c_idx] = weibull_min.cdf(distances[c_idx], shape, loc=loc, scale=scale)

        revised_logits = logits.copy() # Start with original logits
        sorted_indices = np.argsort(logits)[::-1] # Indices sorted by logit value descending

        # Apply revision weight based on CDF score (1 - CDF) only to top alpha classes
        for rank, c_idx in enumerate(sorted_indices):
            if rank < current_alpha:
                weight = 1.0 - cdf_scores[c_idx] # Prob(Unknown based on distance)
                revised_logits[c_idx] *= weight # Attenuate logit based on distance

        # Calculate unknown score: sum of attenuated parts (original_logit * (1 - weight))
        unknown_logit_score = np.sum(logits[sorted_indices[:current_alpha]] * (1.0 - (1.0 - cdf_scores[sorted_indices[:current_alpha]])))
        # Simpler: unknown_logit_score = np.sum(logits[sorted_indices[:current_alpha]] * cdf_scores[sorted_indices[:current_alpha]])

        # Combine revised known logits and the unknown logit
        final_logits = np.append(revised_logits, unknown_logit_score)

        # Compute final OpenMax probabilities using softmax
        exp_logits = np.exp(final_logits - np.max(final_logits)) # Stability trick
        openmax_probs = exp_logits / np.sum(exp_logits)

        return openmax_probs

    # --- predict, evaluate, visualize methods remain as previously defined ---
    # Ensure they call fit_weibull if self.mav is empty
    def predict(self, dataloader):
        # --- 수정된 부분: fit_weibull 호출 전 mav 존재 여부 확인 ---
        if not hasattr(self, 'mav') or not self.mav or not self.weibull_models:
            print("Warning: OpenMax needs fitting. Fitting now using training data...")
            # Ensure train dataloader is available and provides correct labels (0..N-1)
            try:
                 train_loader = self.datamodule.train_dataloader()
                 self.fit_weibull(train_loader)
            except Exception as e:
                 print(f"Error during automatic OpenMax fitting: {e}")
                 raise RuntimeError("Automatic OpenMax fitting failed.")
            if not self.mav or not self.weibull_models:
                raise RuntimeError("Fit failed even after automatic attempt.")
        # ---------------------------------------------------------

        self.model.eval().to(self.device)
        openmax_probs_list = []
        preds_final_list = [] # Use original seen indices or -1
        labels_list = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (OpenMax OSR)"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch["label"].cpu().numpy() # Original labels from test loader (-1 possible)
                logits_batch, embeddings_batch = self.model(ids, attn, tok)

                for i in range(len(labels_orig)):
                    # Ensure correct dimensions before calling openmax_probability
                    embedding_np = embeddings_batch[i].cpu().numpy()
                    logits_np = logits_batch[i].cpu().numpy()
                    if embedding_np.ndim == 0 or logits_np.ndim == 0: continue # Skip if something went wrong

                    om_probs = self.openmax_probability(embedding_np, logits_np)
                    openmax_probs_list.append(om_probs)
                    pred_idx_with_unknown = np.argmax(om_probs)

                    if pred_idx_with_unknown == self.num_known_classes: # Last index is unknown
                        pred_final = -1
                    else:
                        # Map internal index (0..N-1) back to original dataset label
                        # Ensure original_seen_indices exists and index is valid
                        if self.datamodule.original_seen_indices is not None and \
                           0 <= pred_idx_with_unknown < len(self.datamodule.original_seen_indices):
                             pred_final = self.datamodule.original_seen_indices[pred_idx_with_unknown]
                        else:
                             print(f"Warning: Index mapping error in OpenMax predict. Index: {pred_idx_with_unknown}, Seen indices: {self.datamodule.original_seen_indices}")
                             pred_final = -1 # Fallback to unknown

                    preds_final_list.append(pred_final)
                    labels_list.append(labels_orig[i])

        # Handle case where no predictions were made
        if not openmax_probs_list:
             print("Warning: No OpenMax predictions generated.")
             return np.array([]), np.array([]), np.array([]), np.array([])

        all_openmax_probs = np.vstack(openmax_probs_list)
        all_preds_final = np.array(preds_final_list)
        all_labels = np.array(labels_list)
        all_unknown_probs = all_openmax_probs[:, -1] if all_openmax_probs.shape[1] > self.num_known_classes else np.zeros(len(all_labels)) # Safety check

        return all_openmax_probs, all_preds_final, all_labels, all_unknown_probs

    # evaluate and visualize methods remain the same...
    def evaluate(self, dataloader):
        # ... (previous evaluate implementation) ...
        all_probs, all_preds, all_labels, all_unknown_probs = self.predict(dataloader)
        if len(all_labels) == 0: print("Warning: No data to evaluate for OpenMax."); return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0} # Handle empty case

        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1)
        known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        auroc = roc_auc_score(unknown_labels_mask, all_unknown_probs) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        labels_mapped_for_cm = all_labels.copy(); labels_mapped_for_cm[unknown_labels_mask] = -1
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        # Ensure labels for confusion_matrix are within the expected range
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
        # ... (previous visualize implementation) ...
        print("[OpenMaxOSR Visualize] Generating result plots...")
        os.makedirs("results", exist_ok=True)
        base_filename = f"results/openmax_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np); unknown_probs = results["unknown_probs"]

        if len(np.unique(unknown_labels_mask)) > 1: # ROC
            fpr, tpr, _ = roc_curve(unknown_labels_mask, unknown_probs); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (OpenMax)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close()
            print("  ROC curve saved.")
        else: print("  Skipping ROC (only one class).")

        # Unknown Prob Dist
        plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': unknown_probs, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False); plt.title('Unknown Probability Distribution (OpenMax)'); plt.xlabel('OpenMax Unknown Probability'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_confidence.png"); plt.close()
        print("  Unknown probability distribution saved.")

        # Confusion Matrix
        plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results["confusion_matrix"], annot=True, fmt="d", cmap="Blues", xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (OpenMax)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close()
        print("  Confusion matrix saved.")
        print("[OpenMaxOSR Visualize] Finished.")


# ... (Rest of the file: CROSROSR, DOCOSR, ADBOSR, training/evaluation functions, main block) ...
# --- END OF studio.py (OpenMaxOSR fix only) ---
# --- Repeat similar structure for CROSROSR, DOCOSR, ADBOSR ---
# Make sure their predict methods also map indices back correctly
# and evaluate/visualize methods use the appropriate scores.

class CROSROSR(OSRAlgorithm):
    def __init__(self, model, datamodule, args):
        if not isinstance(model, RobertaAutoencoder): raise TypeError("CROSR needs RobertaAutoencoder.")
        super().__init__(model, datamodule, args)
        self.threshold = getattr(args, 'param_crosr_reconstruction_threshold', 0.9)
        self.tail_size = getattr(args, 'param_crosr_tailsize', 100)
        print(f"[CROSROSR Init] Threshold: {self.threshold:.4f}, Tail Size: {self.tail_size:.4f}")
        self.weibull_model = None

    def fit_evt_model(self, dataloader):
        print("[CROSROSR Fit] Fitting EVT model on reconstruction errors...")
        self.model.eval().to(self.device)
        errors = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="CROSR Fit: Collecting errors"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                _, cls_output, _, reconstructed = self.model(ids, attn, tok)
                errors.extend(torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy())
        if not errors: print("Warning: No errors collected for EVT fit."); self.weibull_model = (1.0, 0.0, 1.0); return

        errors = np.sort(np.array(errors))
        tail = errors[-min(self.tail_size, len(errors)):]
        if len(tail) < 2: print(f"Warning: Insufficient tail ({len(tail)}) for EVT fit. Using default."); self.weibull_model = (1.0, 0.0, np.mean(errors) if errors else 1.0); return
        try:
            shape, loc, scale = weibull_min.fit(tail, floc=0)
            if not np.isfinite([shape, scale]).all() or scale <= 1e-9 or shape <= 1e-9: raise ValueError("Invalid fit params")
            self.weibull_model = (shape, loc, scale)
            print(f"  CROSR Fitted Weibull: shape={shape:.4f}, scale={scale:.4f}")
        except Exception as e: print(f"Warning: CROSR Weibull fit exception: {e}. Using default."); self.weibull_model = (1.0, 0.0, np.mean(tail))
        print("[CROSROSR Fit] Complete.")

    def predict(self, dataloader):
        if self.weibull_model is None:
             print("Warning: CROSR EVT model not fitted. Fitting now..."); self.fit_evt_model(self.datamodule.train_dataloader())
             if self.weibull_model is None: raise RuntimeError("Fit failed.")

        self.model.eval().to(self.device)
        all_recon_errors = [] ; all_unknown_probs = [] ; all_preds_final = [] ; all_labels = []
        shape, loc, scale = self.weibull_model

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (CROSR)"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch['label'].cpu().numpy()
                logits, cls_output, _, reconstructed = self.model(ids, attn, tok)

                recon_errors_batch = torch.norm(reconstructed - cls_output, p=2, dim=1).cpu().numpy()
                unknown_probs_batch = weibull_min.cdf(recon_errors_batch, shape, loc=loc, scale=scale)
                pred_indices_classifier = torch.argmax(logits, dim=1).cpu().numpy() # 0..N-1

                batch_preds_final = np.full_like(pred_indices_classifier, -1)
                accept_mask = unknown_probs_batch <= self.threshold
                if np.any(accept_mask):
                     accepted_indices = pred_indices_classifier[accept_mask]
                     original_indices = self.datamodule.original_seen_indices[accepted_indices]
                     batch_preds_final[accept_mask] = original_indices

                all_recon_errors.extend(recon_errors_batch)
                all_unknown_probs.extend(unknown_probs_batch)
                all_preds_final.extend(batch_preds_final)
                all_labels.extend(labels_orig)

        return np.array(all_recon_errors), np.array(all_unknown_probs), np.array(all_preds_final), np.array(all_labels)

    def evaluate(self, dataloader):
        all_errors, all_unknown_probs, all_preds, all_labels = self.predict(dataloader)
        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1); known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        auroc = roc_auc_score(unknown_labels_mask, all_unknown_probs) if len(np.unique(unknown_labels_mask)) > 1 else float('nan') # Higher score -> more unknown

        labels_mapped_for_cm = all_labels.copy(); labels_mapped_for_cm[unknown_labels_mask] = -1
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        conf_matrix = confusion_matrix(labels_mapped_for_cm, all_preds, labels=cm_axis_labels_int)
        _, _, f1_by_class, _ = precision_recall_fscore_support(labels_mapped_for_cm, all_preds, labels=cm_axis_labels_int, average=None, zero_division=0)
        macro_f1 = np.mean(f1_by_class)

        print("\nCROSR OSR Evaluation Summary:")
        print(f"  Threshold: {self.threshold:.4f}, Tail size: {self.tail_size}")
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

        # --- ROC Curve (Keep as before) ---
        if len(np.unique(unknown_labels_mask)) > 1: # ROC
            fpr, tpr, _ = roc_curve(unknown_labels_mask, unknown_probs); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (CROSR)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close(); print("  ROC saved.")
        else: print("  Skipping ROC.")

        # --- Reconstruction Error Distribution (Add bins parameter) ---
        plt.figure(figsize=(7, 5))
        # 수정된 부분: bins=50 추가 (또는 다른 적절한 값)
        sns.histplot(data=pd.DataFrame({'error': recon_errors, 'Known': ~unknown_labels_mask}),
                     x='error', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
        plt.title('Reconstruction Error Distribution (CROSR)'); plt.xlabel('L2 Reconstruction Error'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_error.png"); plt.close(); print("  Error dist saved.")

        # --- Unknown Probability (CDF) Distribution (Add bins parameter) ---
        plt.figure(figsize=(7, 5))
        # 수정된 부분: bins=50 추가 (또는 다른 적절한 값)
        sns.histplot(data=pd.DataFrame({'score': unknown_probs, 'Known': ~unknown_labels_mask}),
                     x='score', hue='Known', kde=True, stat='density', common_norm=False, bins=50)
        plt.axvline(self.threshold, color='r', linestyle='--', label=f'Threshold={self.threshold:.2f}'); plt.title('Unknown Probability Distribution (CROSR)'); plt.xlabel('Weibull CDF of Recon Error'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_prob.png"); plt.close(); print("  Prob dist saved.")

        # --- Confusion Matrix (Keep as before) ---
        plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (CROSR)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close(); print("  CM saved.")
        print("[CROSROSR Visualize] Finished.")


class DOCOSR(OSRAlgorithm):
    def __init__(self, model, datamodule, args):
        if not isinstance(model, DOCRobertaClassifier): raise TypeError("DOC needs DOCRobertaClassifier.")
        super().__init__(model, datamodule, args)
        self.k_sigma = getattr(args, 'param_doc_k', 3.0)
        print(f"[DOCOSR Init] k-sigma: {self.k_sigma}")
        self.gaussian_params: dict[int, tuple[float, float]] = {} # key = internal class index (0..N-1)
        self.class_thresholds: dict[int, float] = {} # key = internal class index (0..N-1)

    def fit_gaussian(self, dataloader):
        print("[DOCOSR Fit] Fitting Gaussian models to sigmoid scores...")
        self.model.eval().to(self.device)
        scores_per_class = {c: [] for c in range(self.num_known_classes)} # Use 0..N-1 indices

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="DOC Fit: Collecting scores"):
                # Dataloader provides mapped labels (0..N-1) for training set
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_mapped = batch['label'].to(self.device) # These are 0..N-1

                logits, _ = self.model(ids, attn, tok)
                sigmoid_scores = torch.sigmoid(logits)

                for i, true_mapped_idx in enumerate(labels_mapped):
                     idx = true_mapped_idx.item()
                     if 0 <= idx < self.num_known_classes: # Ensure index is valid
                         scores_per_class[idx].append(sigmoid_scores[i, idx].item())

        self.gaussian_params.clear(); self.class_thresholds.clear()
        print("[DOCOSR Fit] Calculating Gaussian parameters and thresholds...")
        for c_idx, scores in tqdm(scores_per_class.items(), desc="DOC Fit: Fitting"):
            if len(scores) >= 2:
                mean, std = norm.fit(scores) # Fit normal distribution
                std = max(std, 1e-6) # Avoid zero std dev
                self.gaussian_params[c_idx] = (mean, std)
                # Threshold τ_c = μ_c - k * σ_c (as per DOC paper fig 2 logic, though they plot P(y=l|d))
                # We use score directly, so threshold is on score: max_score < threshold -> reject
                # But the paper uses max(0.5, 1-alpha*sigma), implying a threshold near 1.
                # Let's re-read DOC Section 2.3: "probability threshold t_i = max(0.5, 1 - α*σ_i)"
                # where sigma is std dev of *probabilities* mirrored around 1.
                # Let's implement the mirroring approach from the paper.

                # 1. Filter scores (optional, e.g., > 0.5)
                scores_np = np.array([s for s in scores if s > 0.1]) # Use scores > 0.1 for fitting
                if len(scores_np) < 2:
                     print(f"Warning: Insufficient valid scores ({len(scores_np)}) for class {c_idx} after filtering. Using default threshold.");
                     self.gaussian_params[c_idx] = (0.5, 0.5); self.class_thresholds[c_idx] = 0.5 # Default low threshold
                     continue

                # 2. Create mirrored points around 1.0
                mirrored_scores = 1.0 + (1.0 - scores_np)
                combined_scores = np.concatenate([scores_np, mirrored_scores])

                # 3. Estimate std dev from combined scores
                _, std_combined = norm.fit(combined_scores) # Fit normal distribution to combined points
                std_combined = max(std_combined, 1e-6)

                # 4. Calculate threshold t_i = max(0.5, 1 - k * sigma_combined)
                threshold = max(0.5, 1.0 - self.k_sigma * std_combined)

                self.gaussian_params[c_idx] = (np.mean(scores_np), std_combined) # Store original mean, combined std
                self.class_thresholds[c_idx] = threshold
                # print(f"  Class {c_idx}: Mean(orig)={np.mean(scores_np):.4f}, Std(mirrored)={std_combined:.4f}, Threshold={threshold:.4f}")

            else:
                print(f"Warning: Insufficient samples ({len(scores)}) for class {c_idx}. Using default threshold.")
                self.gaussian_params[c_idx] = (0.5, 0.5); self.class_thresholds[c_idx] = 0.5

        print("[DOCOSR Fit] Complete.")

    def predict(self, dataloader):
        if not self.class_thresholds:
            print("Warning: DOC thresholds not fitted. Fitting now..."); self.fit_gaussian(self.datamodule.train_dataloader())
            if not self.class_thresholds: raise RuntimeError("Fit failed.")

        self.model.eval().to(self.device)
        all_sigmoid_scores = []; all_max_scores = []; all_preds_final = []; all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (DOC OSR)"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch['label'].cpu().numpy()

                logits, _ = self.model(ids, attn, tok)
                sigmoid_scores_batch = torch.sigmoid(logits)
                max_scores_batch, pred_indices_batch = torch.max(sigmoid_scores_batch, dim=1) # Indices are 0..N-1

                pred_indices_np = pred_indices_batch.cpu().numpy()
                max_scores_np = max_scores_batch.cpu().numpy()
                batch_preds_final = np.full_like(pred_indices_np, -1)
                accept_mask = np.zeros_like(pred_indices_np, dtype=bool)

                for i in range(len(labels_orig)):
                     pred_mapped_idx = pred_indices_np[i]
                     threshold = self.class_thresholds.get(pred_mapped_idx, 0.5) # Use default if missing
                     if max_scores_np[i] >= threshold:
                         accept_mask[i] = True

                if np.any(accept_mask):
                    accepted_indices = pred_indices_np[accept_mask]
                    original_indices = self.datamodule.original_seen_indices[accepted_indices]
                    batch_preds_final[accept_mask] = original_indices

                all_sigmoid_scores.append(sigmoid_scores_batch.cpu().numpy())
                all_max_scores.extend(max_scores_np)
                all_preds_final.extend(batch_preds_final)
                all_labels.extend(labels_orig)

        # Calculate Z-scores (optional, using original mean and combined std)
        all_z_scores = np.full(len(all_max_scores), -np.inf)
        pred_indices_all = np.argmax(np.vstack(all_sigmoid_scores), axis=1) if all_sigmoid_scores else np.array([])
        for i in range(len(all_max_scores)):
            pred_idx = pred_indices_all[i]
            if pred_idx in self.gaussian_params:
                 mean_orig, std_combined = self.gaussian_params[pred_idx]
                 all_z_scores[i] = (all_max_scores[i] - mean_orig) / std_combined if std_combined > 1e-6 else 0

        return (np.vstack(all_sigmoid_scores) if all_sigmoid_scores else np.array([])), \
               np.array(all_max_scores), np.array(all_preds_final), np.array(all_labels), np.array(all_z_scores)

    def evaluate(self, dataloader):
        all_scores, all_max_scores, all_preds, all_labels, all_z_scores = self.predict(dataloader)
        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1); known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        # Use negative max score for AUROC (lower score -> more unknown, consistent with threshold logic)
        auroc = roc_auc_score(unknown_labels_mask, -all_max_scores) if len(np.unique(unknown_labels_mask)) > 1 else float('nan')

        labels_mapped_for_cm = all_labels.copy(); labels_mapped_for_cm[unknown_labels_mask] = -1
        cm_axis_labels_int, cm_axis_labels_names = self._get_cm_labels()
        conf_matrix = confusion_matrix(labels_mapped_for_cm, all_preds, labels=cm_axis_labels_int)
        _, _, f1_by_class, _ = precision_recall_fscore_support(labels_mapped_for_cm, all_preds, labels=cm_axis_labels_int, average=None, zero_division=0)
        macro_f1 = np.mean(f1_by_class)

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
        labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
        max_scores = results['max_scores']; z_scores = results['z_scores']

        if len(np.unique(unknown_labels_mask)) > 1: # ROC
            fpr, tpr, _ = roc_curve(unknown_labels_mask, -max_scores); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (DOC)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close(); print("  ROC saved.")
        else: print("  Skipping ROC.")

        # Max Score Dist
        plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': max_scores, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False)
        avg_threshold = np.mean(list(self.class_thresholds.values())) if self.class_thresholds else 0.5
        plt.axvline(avg_threshold, color='g', linestyle=':', label=f'Avg Thresh~{avg_threshold:.2f}')
        plt.title('Max Sigmoid Score Distribution (DOC)'); plt.xlabel('Max Sigmoid Score'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_score.png"); plt.close(); print("  Score dist saved.")

        # Z-Score Dist
        z_scores_clipped = np.clip(z_scores, -10, 10) # Clip for visualization
        plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'score': z_scores_clipped, 'Known': ~unknown_labels_mask}), x='score', hue='Known', kde=True, stat='density', common_norm=False); plt.axvline(-self.k_sigma, color='r', linestyle='--', label=f'Z=-k ({-self.k_sigma:.1f})'); plt.title('Z-Score Distribution (DOC)'); plt.xlabel('Z-Score (Approx)'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_zscore.png"); plt.close(); print("  Z-score dist saved.")

        # CM
        plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (DOC)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close(); print("  CM saved.")
        print("[DOCOSR Visualize] Finished.")


# --- START OF studio.py (ADBOSR fix only) ---
# ... (imports and other classes) ...

# --- START OF studio.py (ADBOSR fix only) ---
# ... (imports and other classes) ...

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
            similarity = torch.clamp(similarity, -1.0 + 1e-7, 1.0 - 1e-7) # Clamp for stability
            return 1.0 - similarity
        elif self.distance_metric == 'euclidean':
             # Ensure centers are also normalized if comparing normalized features
             centers_norm = F.normalize(centers, p=2, dim=-1)
             return torch.cdist(features_norm, centers_norm, p=2)
        else: raise ValueError(f"Unknown distance: {self.distance_metric}")

    def predict(self, dataloader) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        self.model.eval().to(self.device)
        all_features = []; all_distances = []; all_preds_final = []; all_labels = []; all_min_distances = []

        centers = self.model.centers.detach()
        # --- 수정: get_radii() 사용 ---
        # Calculate positive radii using the same method as in training
        radii = self.model.get_radii().detach()
        # --- ---

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Predicting (ADB OSR)"):
                ids, attn = batch['input_ids'].to(self.device), batch['attention_mask'].to(self.device)
                tok = batch.get('token_type_ids'); tok = tok.to(self.device) if tok is not None else None
                labels_orig = batch['label'].cpu().numpy()
                _, features_norm = self.model(ids, attn, tok) # Get normalized features

                distances_batch = self.compute_distances(features_norm, centers) # B x C
                min_distances_batch, closest_indices_batch = torch.min(distances_batch, dim=1) # Indices 0..N-1
                closest_radii_batch = radii[closest_indices_batch] # Use calculated positive radii

                pred_indices_np = closest_indices_batch.cpu().numpy()
                min_distances_np = min_distances_batch.cpu().numpy()
                closest_radii_np = closest_radii_batch.cpu().numpy()

                batch_preds_final = np.full_like(pred_indices_np, -1)
                # Accept if distance <= radius
                accept_mask = min_distances_np <= closest_radii_np

                if np.any(accept_mask):
                    accepted_indices = pred_indices_np[accept_mask]
                    # Map 0..N-1 back to original seen indices
                    if self.datamodule.original_seen_indices is not None:
                         original_indices = self.datamodule.original_seen_indices[accepted_indices]
                         batch_preds_final[accept_mask] = original_indices
                    else: # Fallback if original indices not found
                         print("Warning: original_seen_indices not found in datamodule for ADB mapping.")
                         batch_preds_final[accept_mask] = accepted_indices # Use internal index as fallback

                all_features.append(features_norm.cpu().numpy())
                all_distances.append(distances_batch.cpu().numpy())
                all_preds_final.extend(batch_preds_final)
                all_labels.extend(labels_orig)
                all_min_distances.extend(min_distances_np)

        return (np.concatenate(all_features) if all_features else np.array([])), \
               (np.concatenate(all_distances) if all_distances else np.array([])), \
               np.array(all_preds_final), np.array(all_labels), np.array(all_min_distances)

    # evaluate and visualize methods remain the same...
    def evaluate(self, dataloader) -> dict:
        # ... (previous evaluate implementation) ...
        all_features, all_distances, all_preds, all_labels, all_min_distances = self.predict(dataloader)
        if len(all_labels) == 0: print("Warning: No data to evaluate for ADB."); return {'accuracy': 0, 'auroc': float('nan'), 'f1_score': 0, 'unknown_detection_rate': 0}

        unknown_labels_mask = self.datamodule._determine_unknown_labels(all_labels)
        unknown_preds_mask = (all_preds == -1); known_mask = ~unknown_labels_mask

        accuracy = accuracy_score(all_labels[known_mask], all_preds[known_mask]) if known_mask.any() else 0.0
        unknown_correct = np.sum(unknown_preds_mask & unknown_labels_mask)
        unknown_total = np.sum(unknown_labels_mask)
        unknown_detection_rate = unknown_correct / unknown_total if unknown_total > 0 else 0.0
        auroc = roc_auc_score(unknown_labels_mask, all_min_distances) if len(np.unique(unknown_labels_mask)) > 1 else float('nan') # Higher distance -> more unknown

        labels_mapped_for_cm = all_labels.copy(); labels_mapped_for_cm[unknown_labels_mask] = -1
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
        # ... (previous visualize implementation, ensure it uses self.model.get_radii() for avg radius plot) ...
        print("[ADBOSR Visualize] Generating plots..."); os.makedirs("results", exist_ok=True)
        base_filename = f"results/adb_osr_{self.args.dataset}_{self.args.seen_class_ratio}"
        if 'labels' not in results or len(results['labels']) == 0: print("No data to visualize."); return

        labels_np = results['labels']; unknown_labels_mask = self.datamodule._determine_unknown_labels(labels_np)
        min_distances = results['min_distances']

        if len(np.unique(unknown_labels_mask)) > 1: # ROC
            fpr, tpr, _ = roc_curve(unknown_labels_mask, min_distances); roc_auc_val = auc(fpr, tpr)
            plt.figure(figsize=(7, 6)); plt.plot(fpr, tpr, lw=2, label=f'AUC={roc_auc_val:.3f}'); plt.plot([0,1],[0,1],'k--'); plt.xlabel('FPR'); plt.ylabel('TPR'); plt.title('ROC Curve (ADB)'); plt.legend(); plt.grid(); plt.tight_layout(); plt.savefig(f"{base_filename}_roc.png"); plt.close(); print("  ROC saved.")
        else: print("  Skipping ROC.")

        # Min Distance Dist
        plt.figure(figsize=(7, 5)); sns.histplot(data=pd.DataFrame({'dist': min_distances, 'Known': ~unknown_labels_mask}), x='dist', hue='Known', kde=True, stat='density', common_norm=False)
        # --- 수정: get_radii() 사용 ---
        mean_radius = self.model.get_radii().detach().mean().item()
        plt.axvline(mean_radius, color='g', linestyle=':', label=f'Avg Radius~{mean_radius:.3f}')
        # --- ---
        plt.title(f'Min Distance Distribution (ADB - {self.distance_metric})'); plt.xlabel(f'Min {self.distance_metric.capitalize()} Distance'); plt.legend(); plt.grid(alpha=0.5); plt.tight_layout(); plt.savefig(f"{base_filename}_distance.png"); plt.close(); print("  Distance dist saved.")

        # t-SNE (Keep as before)
        if 'features' in results and len(results['features']) > 100 and results['features'].shape[1] > 2:
             try:
                 from sklearn.manifold import TSNE
                 print("  Generating t-SNE plot (this may take a while)...")
                 features = results['features']; centers = self.model.centers.detach().cpu().numpy()
                 n_samples = features.shape[0]; max_tsne = 5000
                 indices = np.random.choice(n_samples, min(n_samples, max_tsne), replace=False)
                 features_sub = features[indices]; unknown_sub = unknown_labels_mask[indices]
                 # Normalize centers for t-SNE if using cosine distance
                 centers_norm = F.normalize(torch.from_numpy(centers), p=2, dim=-1).numpy()

                 combined = np.vstack([features_sub, centers_norm]) # Embed normalized centers
                 tsne = TSNE(n_components=2, random_state=self.args.random_seed, perplexity=min(30, combined.shape[0]-1), n_iter=300, init='pca', learning_rate='auto')
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

        # CM (Keep as before)
        plt.figure(figsize=(max(6, len(results['confusion_matrix_labels'])*0.6), max(5, len(results['confusion_matrix_labels'])*0.5))); sns.heatmap(results['confusion_matrix'], annot=True, fmt='d', cmap='Blues', xticklabels=results['confusion_matrix_names'], yticklabels=results['confusion_matrix_names'], annot_kws={"size": 8}); plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix (ADB)'); plt.xticks(rotation=45, ha='right', fontsize=9); plt.yticks(rotation=0, fontsize=9); plt.tight_layout(); plt.savefig(f"{base_filename}_confusion.png"); plt.close(); print("  CM saved.")
        print("[ADBOSR Visualize] Finished.")


# --- END OF studio.py (ADBOSR fix only) ---

# --- END OF studio.py (ADBOSR fix only) ---

# =============================================================================
# Training and Evaluation Functions (Updated)
# =============================================================================

def train_model(model, datamodule, args):
    """Trains a PyTorch Lightning model."""
    print(f"\n--- Training Model: {model.__class__.__name__} ---")
    output_dir = f"checkpoints/{args.dataset}_{args.osr_method}" # Maybe method-specific folder?
    log_dir = "logs"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Ensure data is setup for finding total steps
    datamodule.setup(stage='fit')
    try:
        train_batches = len(datamodule.train_dataloader())
        total_steps = train_batches * args.epochs
        warmup_steps = min(int(args.warmup_ratio * total_steps), args.max_warmup_steps)
    except Exception:
         print("Warning: Could not determine dataloader length. Using defaults.")
         total_steps = 10000; warmup_steps = 500

    # Set scheduler steps in the model if attributes exist
    if hasattr(model, 'total_steps'): model.total_steps = total_steps
    if hasattr(model, 'warmup_steps'): model.warmup_steps = warmup_steps
    print(f"Scheduler: Total steps={total_steps}, Warmup steps={warmup_steps}")

    # Callbacks
    monitor_metric = "val_loss" #"val_acc" if args.dataset == 'trec' else "val_loss" # Example: Use val_acc for TREC? Or stick to val_loss
    monitor_mode = "min" #"max" if monitor_metric == "val_acc" else "min"
    checkpoint_callback = ModelCheckpoint(
        dirpath=output_dir,
        filename=f"{model.__class__.__name__}-{{epoch:02d}}-{{{monitor_metric}:.4f}}-{timestamp}",
        save_top_k=1, verbose=True, monitor=monitor_metric, mode=monitor_mode
    )
    early_stopping_callback = EarlyStopping(
        monitor=monitor_metric, patience=args.early_stopping_patience,
        min_delta=args.early_stopping_delta, verbose=True, mode=monitor_mode
    )

    # Logger
    logger_name = f"{args.model.replace('/','_')}_{args.dataset}_{args.osr_method}_{args.seen_class_ratio}"
    try:
        logger = TensorBoardLogger(save_dir=log_dir, name=logger_name, version=timestamp)
        print("Using TensorBoardLogger.")
    except ImportError:
        print("TensorBoard not available. Using CSVLogger.")
        logger = CSVLogger(save_dir=log_dir, name=logger_name, version=timestamp)

    # Trainer Config
    use_gpu = args.force_gpu or torch.cuda.is_available()
    trainer_kwargs = {
        "max_epochs": args.epochs,
        "callbacks": [checkpoint_callback, early_stopping_callback],
        "logger": logger,
        "log_every_n_steps": max(1, train_batches // 10) if 'train_batches' in locals() else 50,
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
        # No accelerator needed for CPU

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
         print(f"Warning: No valid checkpoint found after training.")
         return None


# --- OSR Evaluation Wrappers (Updated with Tuning Integration) ---

MODEL_CLASS_MAP = {
    'standard': RobertaClassifier,
    'crosr': RobertaAutoencoder,
    'doc': DOCRobertaClassifier,
    'adb': RobertaADB
}
MODEL_NEEDS_SPECIAL_TRAINING = ['crosr', 'doc', 'adb']



def _prepare_evaluation(method_name, current_model, datamodule, args, osr_algorithm_class):
    """
    Handles model checking, potential retraining (outside tuning), and parameter setup
    (tuning with retraining OR loading defaults/saved).
    """
    print(f"\n--- Preparing for {method_name.upper()} OSR Evaluation ---")
    target_model_class = MODEL_CLASS_MAP.get(method_name if method_name in MODEL_NEEDS_SPECIAL_TRAINING else 'standard')
    model_to_evaluate_finally = None # 최종 평가에 사용할 모델

    # --- Hyperparameter Tuning Logic (with Retraining) ---
    if args.parameter_search and method_name == args.osr_method: # 해당 메소드만 튜닝
        print(f"Starting Optuna hyperparameter search for {method_name.upper()} (with retraining)...")
        tuner = OptunaHyperparameterTuner(method_name, datamodule, args)

        # Define the function that trains AND evaluates for one trial
        def train_and_evaluate_trial(trial_args):
            print(f"\n  Starting Training for Trial with LR={trial_args.lr_adb:.5f}, Alpha={trial_args.param_adb_alpha:.4f}, Delta={trial_args.param_adb_delta:.4f}, Freeze={trial_args.adb_freeze_backbone}")
            # 1. Initialize the target model with trial hyperparameters
            num_classes = datamodule.num_seen_classes
            init_kwargs = {'model_name': trial_args.model, 'num_classes': num_classes, 'weight_decay': trial_args.weight_decay}
            # Add method-specific init args based on trial_args
            if target_model_class == RobertaADB:
                 init_kwargs.update({'learning_rate': trial_args.lr_adb, 'delta': trial_args.param_adb_delta, 'alpha': trial_args.param_adb_alpha, 'freeze_backbone': trial_args.adb_freeze_backbone})
            elif target_model_class == RobertaAutoencoder: # For CROSR
                 init_kwargs.update({'learning_rate': trial_args.lr, 'reconstruction_weight': trial_args.param_crosr_recon_weight})
            elif target_model_class == DOCRobertaClassifier: # For DOC
                 init_kwargs.update({'learning_rate': trial_args.lr})
            else: # Standard Classifier
                 init_kwargs.update({'learning_rate': trial_args.lr})

            trial_model = target_model_class(**init_kwargs)

            # 2. Train the model for this trial
            # Use fewer epochs during tuning for speed? Or full epochs? Let's use full for now.
            # Consider creating a temporary checkpoint dir per trial?
            checkpoint_path = train_model(trial_model, datamodule, trial_args)
            if checkpoint_path is None or not os.path.exists(checkpoint_path):
                 print("  Trial Training Failed. Returning failure score.")
                 return {}, -1e9 # Return empty results and failure score

            # 3. Load the best model from training
            trained_trial_model = target_model_class.load_from_checkpoint(checkpoint_path)

            # 4. Evaluate the trained model
            evaluator = osr_algorithm_class(trained_trial_model, datamodule, trial_args)
            # Use a smaller subset for faster tuning evaluation? Or full test set? Full for now.
            results = evaluator.evaluate(datamodule.test_dataloader())

            # 5. Extract score and return
            score = results.get(args.tuning_metric)
            if score is None or not np.isfinite(score):
                 print(f"  Warning: Metric '{args.tuning_metric}' invalid ({score}) for trial.")
                 return results, -1e9
            return results, float(score)
        # --- End of train_and_evaluate_trial function ---

        # Pass the combined function to the tuner
        best_params, best_trial_metrics = tuner.tune(train_and_evaluate_trial)

        # Apply best parameters for the *final* run (which might involve one last retraining)
        print(f"Applying best tuned parameters for final {method_name.upper()} evaluation:")
        for name, value in best_params.items():
            setattr(args, name, value); print(f"  {name}: {value}")
        # Set flag to retrain one last time with best params
        needs_final_training = True

    else:
        # --- Logic for No Tuning or Other Methods ---
        needs_final_training = False # Assume no final retraining needed unless type mismatch
        # Load existing best parameters or use defaults
        best_params = load_best_params(method_name, args.dataset, args.seen_class_ratio)
        param_source = "loaded from previous tuning"
        if not best_params:
            best_params = get_default_best_params(method_name)
            param_source = "defaults"
        print(f"Applying parameters ({param_source}) for final {method_name.upper()} evaluation:")
        for name, value in best_params.items():
            setattr(args, name, value); print(f"  {name}: {value}")

        # Check model type mismatch even when not tuning
        if not isinstance(current_model, target_model_class) and method_name in MODEL_NEEDS_SPECIAL_TRAINING:
             print(f"Warning: Model type mismatch ({type(current_model).__name__} vs {target_model_class.__name__}). Retraining required.")
             needs_final_training = True


    # --- Final Model Preparation ---
    if needs_final_training:
         print(f"\nTraining final {target_model_class.__name__} model with best/default parameters...")
         num_classes = datamodule.num_seen_classes
         init_kwargs = {'model_name': args.model, 'num_classes': num_classes, 'weight_decay': args.weight_decay}
         if target_model_class == RobertaADB: init_kwargs.update({'learning_rate': args.lr_adb, 'delta': args.param_adb_delta, 'alpha': args.param_adb_alpha, 'freeze_backbone': args.adb_freeze_backbone})
         elif target_model_class == RobertaAutoencoder: init_kwargs.update({'learning_rate': args.lr, 'reconstruction_weight': args.param_crosr_recon_weight})
         elif target_model_class == DOCRobertaClassifier: init_kwargs.update({'learning_rate': args.lr})
         else: init_kwargs.update({'learning_rate': args.lr})

         final_model_instance = target_model_class(**init_kwargs)
         final_checkpoint_path = train_model(final_model_instance, datamodule, args)
         if final_checkpoint_path is None or not os.path.exists(final_checkpoint_path):
              raise RuntimeError(f"Failed to train final model for {method_name.upper()}.")
         print(f"Loading final trained model from: {final_checkpoint_path}")
         model_to_evaluate_finally = target_model_class.load_from_checkpoint(final_checkpoint_path)
    else:
         print("Using the initially provided/loaded model for final evaluation.")
         model_to_evaluate_finally = current_model # Use the one passed in

    return model_to_evaluate_finally # Return the model ready for final evaluation

# --- Main Evaluation Functions per Method (Simplified) ---
def evaluate_threshold_osr(base_model, datamodule, args, all_results):
    model = _prepare_evaluation('threshold', base_model, datamodule, args, ThresholdOSR)
    evaluator = ThresholdOSR(model, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results["threshold"] = results
    return results

def evaluate_openmax_osr(base_model, datamodule, args, all_results):
    model = _prepare_evaluation('openmax', base_model, datamodule, args, OpenMaxOSR)
    evaluator = OpenMaxOSR(model, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results["openmax"] = results
    return results

def evaluate_crosr_osr(base_model, datamodule, args, all_results):
    model = _prepare_evaluation('crosr', base_model, datamodule, args, CROSROSR)
    evaluator = CROSROSR(model, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results["crosr"] = results
    return results

def evaluate_doc_osr(base_model, datamodule, args, all_results):
    model = _prepare_evaluation('doc', base_model, datamodule, args, DOCOSR)
    evaluator = DOCOSR(model, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results["doc"] = results
    return results

def evaluate_adb_osr(base_model, datamodule, args, all_results):
    model = _prepare_evaluation('adb', base_model, datamodule, args, ADBOSR)
    evaluator = ADBOSR(model, datamodule, args)
    results = evaluator.evaluate(datamodule.test_dataloader())
    evaluator.visualize(results)
    all_results["adb"] = results
    return results


# --- OSCR Curve Calculation and Visualization ---
def calculate_oscr_curve(results, datamodule):
    """Calculates CCR vs FPR for the OSCR curve."""
    if 'predictions' not in results or 'labels' not in results: return np.array([0,1]), np.array([0,0]) # Handle missing data
    preds = np.array(results['predictions'])
    labels = np.array(results['labels'])
    if len(preds) != len(labels): return np.array([0,1]), np.array([0,0]) # Mismatch

    # Determine score for ranking (lower score -> more likely unknown)
    if 'max_probs' in results: scores_for_ranking = -np.array(results['max_probs']) # Threshold
    elif 'unknown_probs' in results: scores_for_ranking = np.array(results['unknown_probs']) # OpenMax, CROSR
    elif 'max_scores' in results: scores_for_ranking = -np.array(results['max_scores']) # DOC
    elif 'min_distances' in results: scores_for_ranking = np.array(results['min_distances']) # ADB
    else: print("Warning: No suitable score found for OSCR."); return np.array([0,1]), np.array([0,0])

    unknown_labels_mask = datamodule._determine_unknown_labels(labels)
    known_mask = ~unknown_labels_mask
    is_correct_known = (preds == labels) & known_mask
    is_false_positive = (preds != -1) & unknown_labels_mask # Predicted known, but is unknown

    sorted_indices = np.argsort(scores_for_ranking) # Sorts ascendingly (more unknown first)
    sorted_correct_known = is_correct_known[sorted_indices]
    sorted_false_positive = is_false_positive[sorted_indices]

    n_known = np.sum(known_mask)
    n_unknown = np.sum(unknown_labels_mask)

    ccr = np.cumsum(sorted_correct_known) / n_known if n_known > 0 else np.zeros(len(scores_for_ranking))
    fpr = np.cumsum(sorted_false_positive) / n_unknown if n_unknown > 0 else np.zeros(len(scores_for_ranking))

    return np.insert(fpr, 0, 0.0), np.insert(ccr, 0, 0.0) # Add (0,0) point


def visualize_oscr_curves(all_results, datamodule, args):
    """Plots OSCR curves for comparing multiple OSR methods."""
    print("\nGenerating OSCR Comparison Curve...")
    plt.figure(figsize=(8, 7))
    method_found = False
    for method, results in all_results.items():
        if results and isinstance(results, dict):
            try:
                fpr, ccr = calculate_oscr_curve(results, datamodule)
                oscr_auc = np.trapz(ccr, fpr) if len(fpr) > 1 else 0.0
                plt.plot(fpr, ccr, lw=2.5, label=f'{method.upper()} (AUC = {oscr_auc:.3f})', alpha=0.8)
                method_found = True
            except Exception as e: print(f"  Error calculating OSCR for {method}: {e}")
        else: print(f"Skipping OSCR for {method} (no results).")

    if not method_found: print("No valid results found to plot OSCR."); plt.close(); return

    plt.plot([0, 1], [1, 0], color='grey', lw=1.5, linestyle='--', label='Ideal Closed-Set (AUC = 0.5)')
    plt.xlim([-0.02, 1.02]); plt.ylim([-0.02, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('Correct Classification Rate (CCR)', fontsize=12)
    plt.title(f'OSCR Curves Comparison ({args.dataset}, Seen Ratio: {args.seen_class_ratio*100:.0f}%)', fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.6)
    save_path = f"results/oscr_comparison_{args.dataset}_{args.seen_class_ratio}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"OSCR comparison curve saved to: {save_path}")


# --- Main Evaluation Orchestrator ---
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
    methods_to_run = method_map.keys() if args.osr_method == "all" else [args.osr_method]

    for method in methods_to_run:
        if method in method_map:
            try:
                print(f"\n>>> Starting evaluation for: {method.upper()} <<<")
                method_map[method](base_model, datamodule, args, all_results)
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
        try: return obj.__dict__ # For simple objects
        except AttributeError: raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    try:
        # Remove potentially huge data before saving summary JSON
        summary_results = {}
        for method, res in all_results.items():
             if isinstance(res, dict):
                 summary_results[method] = {k: v for k, v in res.items() if k not in ['predictions', 'labels', 'probs', 'scores', 'features', 'distances', 'max_probs', 'unknown_probs', 'max_scores', 'min_distances', 'reconstruction_errors', 'z_scores', 'confusion_matrix']}
                 # Keep CM labels/names if they exist
                 if 'confusion_matrix_labels' in res: summary_results[method]['confusion_matrix_labels'] = res['confusion_matrix_labels']
                 if 'confusion_matrix_names' in res: summary_results[method]['confusion_matrix_names'] = res['confusion_matrix_names']
             else:
                 summary_results[method] = res # Store error message etc.


        with open(results_filename, 'w', encoding='utf-8') as f:
            # Use default=json_converter for broader type handling
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
    methods_evaluated = [m for m in all_results if isinstance(all_results[m], dict) and 'error' not in all_results[m]]

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
# Argument Parser and Main Execution Block (Keep as provided previously)
# =============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='Open-Set Recognition Experiments with RoBERTa')
    # --- Core Arguments ---
    parser.add_argument('-dataset', type=str, default='acm', choices=['newsgroup20', 'bbc_news', 'trec', 'custom_syslog', 'reuters8', 'acm', 'chemprot'], help='Dataset to use.')
    parser.add_argument('-model', type=str, default='roberta-base', help='Pre-trained RoBERTa model name (e.g., roberta-base, roberta-large).')
    parser.add_argument('-osr_method', type=str, default='all', choices=['threshold', 'openmax', 'crosr', 'doc', 'adb', 'all'], help='OSR method(s) to evaluate.')
    parser.add_argument('-seen_class_ratio', type=float, default=0.5, help='Ratio of classes used as known/seen (0.0 to 1.0).')
    parser.add_argument('-random_seed', type=int, default=42, help='Random seed for reproducibility.')
    # --- Training Arguments ---
    parser.add_argument('-epochs', type=int, default=5, help='Number of training epochs.') # Reduced default
    parser.add_argument('-batch_size', type=int, default=64, help='Batch size.') # Reduced default
    parser.add_argument('-lr', type=float, default=2e-5, help='Learning rate for backbone/standard classifier.')
    parser.add_argument('-lr_adb', type=float, default=5e-4, help='Specific learning rate for ADB centers/radii.') # Adjusted default
    parser.add_argument('-weight_decay', type=float, default=0.01, help='Weight decay for optimizer.')
    parser.add_argument('-warmup_ratio', type=float, default=0.1, help='Ratio of total steps for LR warmup.')
    parser.add_argument('-max_warmup_steps', type=int, default=500, help='Max warmup steps.') # Reduced default
    parser.add_argument('-gradient_clip_val', type=float, default=1.0, help='Gradient clipping value (0 to disable).')
    parser.add_argument('-early_stopping_patience', type=int, default=3, help='Patience for early stopping.') # Reduced default
    parser.add_argument('-early_stopping_delta', type=float, default=0.001, help='Min delta for early stopping improvement.')
    # --- Data Split Arguments ---
    parser.add_argument('-train_ratio', type=float, default=0.7, help='Proportion for training.')
    parser.add_argument('-val_ratio', type=float, default=0.15, help='Proportion for validation.')
    parser.add_argument('-test_ratio', type=float, default=0.15, help='Proportion for testing.')
    # --- Hardware Arguments ---
    parser.add_argument('-force_gpu', action='store_true', help='Force GPU usage.')
    parser.add_argument('-gpu_id', type=int, default=0, help='GPU ID to use.')
    # --- OSR Method Specific Parameters ---
    parser.add_argument('-param_threshold', type=float, default=None, help='Softmax threshold for ThresholdOSR.')
    parser.add_argument('-param_openmax_tailsize', type=int, default=None, help='Tail size for OpenMax.')
    parser.add_argument('-param_openmax_alpha', type=int, default=None, help='Alpha parameter for OpenMax.')
    parser.add_argument('-param_crosr_reconstruction_threshold', type=float, default=None, help='CDF threshold for CROSR.')
    parser.add_argument('-param_crosr_tailsize', type=int, default=None, help='Tail size for CROSR EVT.')
    parser.add_argument('-param_crosr_recon_weight', type=float, default=0.5, help='Reconstruction loss weight for CROSR model training.')
    parser.add_argument('-param_doc_k', type=float, default=None, help='k-sigma factor for DOC.')
    parser.add_argument('-param_adb_distance', type=str, default='cosine', choices=['cosine', 'euclidean'], help='Distance metric for ADB.')
    parser.add_argument('-param_adb_delta', type=float, default=0.1, help='Margin delta for ADB training loss.')
    parser.add_argument('-param_adb_alpha', type=float, default=0.1, help='Weight alpha for ADB training loss.')
    parser.add_argument('-adb_freeze_backbone', action=argparse.BooleanOptionalAction, default=True, help='Freeze backbone during ADB training.')
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
        print(f"Current Device: {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")
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
        max_length=384 # Adjust as needed
    )
    datamodule.prepare_data()
    datamodule.setup(stage=None) # Setup all splits
    num_model_classes = datamodule.num_seen_classes
    if num_model_classes is None or num_model_classes == 0: raise ValueError("Num seen classes not set.")
    print(f"Model training on {num_model_classes} known classes.")

    # --- Initialize Base Model ---
    # Decide initial model based on the *first* method to be run or default.
    # _prepare_evaluation will handle retraining if subsequent methods need different models.
    initial_method = args.osr_method if args.osr_method != 'all' else 'threshold' # Default initial if 'all'
    print(f"\nInitializing base model architecture (targeting initial method: {initial_method.upper()})...")
    initial_model_class = MODEL_CLASS_MAP.get(initial_method if initial_method in MODEL_NEEDS_SPECIAL_TRAINING else 'standard')
    init_kwargs = {'model_name': args.model, 'num_classes': num_model_classes, 'learning_rate': args.lr, 'weight_decay': args.weight_decay}
    if initial_method == 'crosr': init_kwargs['reconstruction_weight'] = args.param_crosr_recon_weight
    if initial_method == 'adb': init_kwargs.update({'learning_rate': args.lr_adb, 'delta': args.param_adb_delta, 'alpha': args.param_adb_alpha, 'freeze_backbone': args.adb_freeze_backbone})
    base_model = initial_model_class(**init_kwargs)

    # --- Train Initial Model ---
    print("\nStep 1: Training the initial model...")
    best_checkpoint_path = train_model(base_model, datamodule, args)
    if not best_checkpoint_path: print("Error: Training failed. Exiting."); sys.exit(1)

    # --- Load Best Model ---
    print(f"\nStep 2: Loading best model from checkpoint: {best_checkpoint_path}")
    try:
        # Important: Load using the class that was actually trained
        loaded_model = initial_model_class.load_from_checkpoint(best_checkpoint_path)
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model from {best_checkpoint_path}: {e}"); sys.exit(1)

    # --- Evaluate OSR Algorithms ---
    print("\nStep 3: Evaluating OSR algorithm(s)...")
    evaluate_osr_main(loaded_model, datamodule, args)

    print("\nExperiment finished.")

if __name__ == "__main__":
    main()

# --- END OF FILE studio.py ---