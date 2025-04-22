# --- START OF FILE hyperparameter_tuning.py ---

import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
# Ensure plotly is installed for visualization: pip install plotly kaleido
# Kaleido is needed for saving static images
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_contour, plot_slice
from sklearn.metrics import roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import joblib
import argparse # To use Namespace

class OptunaHyperparameterTuner:
    """
    Optuna 기반 하이퍼파라미터 튜닝을 위한 클래스
    (Class for hyperparameter tuning based on Optuna)
    """
    def __init__(self, method_name, datamodule, args):
        self.method_name = method_name
        self.datamodule = datamodule
        self.args = args # Store the main args namespace
        self.best_params = None
        self.best_trial_results = None # Store metrics from the best trial
        self.best_score = -float('inf') # Default for maximization
        self.metric = args.tuning_metric if hasattr(args, 'tuning_metric') else 'f1_score' # Default to f1_score
        self.n_trials = args.n_trials if hasattr(args, 'n_trials') else 20
        self.study = None
        self.evaluation_func = None # Function to call for evaluating a trial

        # Create directories for results
        self.results_dir = "tuning_results"
        self.plots_dir = os.path.join(self.results_dir, "plots")
        self.studies_dir = os.path.join(self.results_dir, "studies")
        os.makedirs(self.plots_dir, exist_ok=True)
        os.makedirs(self.studies_dir, exist_ok=True)
        print(f"[Optuna Tuner] Initialized for method '{self.method_name}', metric '{self.metric}', {self.n_trials} trials.")

    def _create_study(self):
        """Creates or loads an Optuna study object."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # Include seen_class_ratio in the study name for better organization
        study_name = f"{self.method_name}_{self.args.dataset}_{self.args.seen_class_ratio}_{timestamp}"
        storage_name = f"sqlite:///{self.studies_dir}/{study_name}.db"

        print(f"[Optuna Tuner] Creating/Loading study: '{study_name}' with storage: '{storage_name}'")
        return optuna.create_study(
            study_name=study_name,
            storage=storage_name,
            direction="maximize", # Assuming higher metric score is better
            load_if_exists=False # Start a new study each time by default
        )

    def _define_search_space(self, trial):
        """
        Defines the hyperparameter search space for each OSR method.
        """
        params = {}
        print(f"[Optuna Trial {trial.number}] Defining search space for '{self.method_name}'...")

        # --- Threshold OSR ---
        if self.method_name == 'threshold':
            # Check if param_threshold is defined in args, otherwise use a default range
            # This parameter comes directly from args, so we suggest based on its potential range
             params['param_threshold'] = trial.suggest_float('param_threshold', 0.05, 0.95, step=0.05)
             print(f"  Suggesting param_threshold: [{0.05}, {0.95}]")

        # --- OpenMax OSR ---
        elif self.method_name == 'openmax':
            params['param_openmax_tailsize'] = trial.suggest_int('param_openmax_tailsize', 10, 200, step=10)
            # Ensure alpha <= num_known_classes, retrieve from datamodule
            max_alpha = min(20, self.datamodule.num_seen_classes) if self.datamodule.num_seen_classes else 20
            if max_alpha < 1: max_alpha = 1 # Ensure at least 1
            params['param_openmax_alpha'] = trial.suggest_int('param_openmax_alpha', 1, max_alpha, step=1)
            print(f"  Suggesting param_openmax_tailsize: [{10}, {200}]")
            print(f"  Suggesting param_openmax_alpha: [{1}, {max_alpha}] (Max based on num_seen_classes)")

        # --- CROSR OSR ---
        elif self.method_name == 'crosr':
            params['param_crosr_reconstruction_threshold'] = trial.suggest_float('param_crosr_reconstruction_threshold', 0.1, 0.99, step=0.05)
            params['param_crosr_tailsize'] = trial.suggest_int('param_crosr_tailsize', 20, 200, step=10)
            # Optional: Tune recon_weight if retraining AE per trial (more complex)
            # params['param_crosr_recon_weight'] = trial.suggest_float('param_crosr_recon_weight', 0.1, 1.0, step=0.1)
            print(f"  Suggesting param_crosr_reconstruction_threshold: [{0.1}, {0.99}]")
            print(f"  Suggesting param_crosr_tailsize: [{20}, {200}]")

        # --- DOC OSR ---
        elif self.method_name == 'doc':
            # Based on DOC paper (Shu et al.), they use k*sigma, typically k=3. Let's search around it.
            params['param_doc_k'] = trial.suggest_float('param_doc_k', 1.0, 5.0, step=0.25)
            print(f"  Suggesting param_doc_k: [{1.0}, {5.0}]")

        # --- ADB OSR ---
        elif self.method_name == 'adb':
            # --- 평가 시점 파라미터 ---
            params['param_adb_distance'] = trial.suggest_categorical('param_adb_distance', ['cosine', 'euclidean'])
            print(f"  Suggesting param_adb_distance: ['cosine', 'euclidean']")

            # --- 학습 시점 파라미터 (모델 재학습 필요 시 사용) ---
            # 주석 해제 및 범위 설정
            params['lr_adb'] = trial.suggest_float('lr_adb', 1e-4, 5e-3, log=True)
            params['param_adb_delta'] = trial.suggest_float('param_adb_delta', 0.05, 0.4, step=0.05) # 범위 약간 조정
            params['param_adb_alpha'] = trial.suggest_float('param_adb_alpha', 0.01, 0.5, log=True)
            params['adb_freeze_backbone'] = trial.suggest_categorical('adb_freeze_backbone', [True, False])
            print(f"  Suggesting lr_adb: [1e-4, 5e-3]")
            print(f"  Suggesting param_adb_delta: [0.05, 0.4]")
            print(f"  Suggesting param_adb_alpha: [0.01, 0.5]")
            print(f"  Suggesting adb_freeze_backbone: [True, False]")
            # --- ---
        else:
            print(f"  Warning: No specific search space defined for method '{self.method_name}'.")

        return params

    def _objective(self, trial):
        """Optuna trial objective function."""
        # 1. Get suggested hyperparameters
        params = self._define_search_space(trial)
        trial_args = argparse.Namespace(**vars(self.args))
        for name, value in params.items():
            setattr(trial_args, name, value)

        param_str = ", ".join([f"{name.replace('param_', '').replace('lr_adb','LR').replace('adb_','')}"
                               f"={value:.4f}" if isinstance(value, float) else f"{name.replace('param_', '').replace('lr_adb','LR').replace('adb_','')}={value}"
                               for name, value in params.items()])
        print(f"\n--- Optuna Trial {trial.number + 1}/{self.n_trials} ---")
        print(f"Method: {self.method_name.upper()}, Params: {param_str}")

        try:
            # --- 수정: model_training_and_evaluation_func 호출 ---
            # 이 함수는 모델 학습 + 평가를 모두 수행하고 결과 딕셔너리와 점수를 반환해야 함
            results_dict, score_float = self.model_training_and_evaluation_func(trial_args)
            # --- ---

            if score_float is None or not np.isfinite(score_float):
                print(f"Warning: Invalid score ({score_float}). Failure.")
                # Store NaN/None in user attributes
                valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
                for metric_name in valid_metrics:
                     metric_value = results_dict.get(metric_name, float('nan'))
                     trial.set_user_attr(metric_name, float(metric_value) if pd.notna(metric_value) else None)
                return -1e9

            # Store metrics
            valid_metrics = ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate']
            print("Trial Results:")
            for metric_name in valid_metrics:
                metric_value = results_dict.get(metric_name, float('nan'))
                trial.set_user_attr(metric_name, float(metric_value) if pd.notna(metric_value) else None)
                print(f"  {metric_name}: {metric_value:.4f}")

            print(f"--> Trial {trial.number + 1} Score ({self.metric}): {score_float:.4f}")
            return score_float

        except Exception as e:
            print(f"Error during Optuna trial {trial.number + 1}: {e}")
            import traceback
            traceback.print_exc()
            return -1e9 # Report failure

    def tune(self, model_training_and_evaluation_func): # 인자 이름 변경
        """Performs hyperparameter optimization including model retraining per trial."""
        # --- 수정: evaluation_func 대신 model_training_and_evaluation_func 저장 ---
        self.model_training_and_evaluation_func = model_training_and_evaluation_func
        # --- ---

        print(f"\n[Hyperparameter Tuning] Starting Optuna for {self.method_name.upper()} (with retraining per trial)")
        print(f"Optimizing metric: {self.metric}")
        print(f"Number of trials: {self.n_trials}")

        self.study = self._create_study()

        try:
            self.study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)
        except KeyboardInterrupt: print("\nOptimization stopped by user.")
        except Exception as e: print(f"\nError during optimization: {e}")

        # --- 결과 처리 (기존과 유사) ---
        if not self.study.trials: print("No trials completed."); return {}, {}
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None and t.value > -1e8]
        if not completed_trials:
            print("Warning: No trials completed successfully."); self.best_params = get_default_best_params(self.method_name); self.best_trial_results = {}; self.best_score = -float('inf')
        else:
            best_trial = self.study.best_trial; self.best_params = best_trial.params; self.best_score = best_trial.value
            self.best_trial_results = { m: best_trial.user_attrs.get(m, None) for m in ['accuracy', 'auroc', 'f1_score', 'unknown_detection_rate'] }
            print(f"\n--- Optuna Tuning Finished for {self.method_name.upper()} ---")
            print(f"Best Trial: {best_trial.number + 1}, Best Score ({self.metric}): {self.best_score:.4f}")
            print("Best Parameters:"); [print(f"  {k.replace('param_', '')}: {v}") for k, v in self.best_params.items()]
            print("Metrics for Best Trial:"); [print(f"  {k}: {v if v is not None else 'N/A'}") for k, v in self.best_trial_results.items()]
            self._save_tuning_results(); self._visualize_tuning_results()
        return self.best_params, self.best_trial_results
    
    def _save_tuning_results(self):
        """Saves the tuning results to a JSON file and the study object."""
        if self.study is None or self.best_params is None:
            print("No tuning results to save.")
            return

        # Use seen_class_ratio in filename
        output_file = os.path.join(self.results_dir, f"{self.method_name}_tuning_{self.args.dataset}_{self.args.seen_class_ratio}.json")
        study_file = os.path.join(self.studies_dir, f"{self.method_name}_tuning_{self.args.dataset}_{self.args.seen_class_ratio}.pkl")

        output_data = {
            'method': self.method_name,
            'dataset': self.args.dataset,
            'seen_class_ratio': self.args.seen_class_ratio,
            'tuning_metric': self.metric,
            'n_trials_completed': len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            'total_trials_requested': self.n_trials,
            'best_params': self.best_params,
            'best_score': self.best_score,
            'best_trial_metrics': self.best_trial_results,
            'timestamp': datetime.datetime.now().isoformat()
        }

        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            print(f"Tuning summary saved to: {output_file}")
        except Exception as e:
            print(f"Error saving tuning summary JSON: {e}")

        try:
            joblib.dump(self.study, study_file)
            print(f"Optuna study object saved to: {study_file}")
        except Exception as e:
            print(f"Error saving Optuna study object: {e}")

    def _visualize_tuning_results(self):
        """Visualizes the Optuna tuning results using plotly."""
        if self.study is None or not self.study.trials:
            print("No study data available for visualization.")
            return

        if len([t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]) < 2:
            print("Need at least 2 completed trials for meaningful visualization.")
            return

        print("Generating Optuna visualization plots...")
        # Use seen_class_ratio in filenames
        base_filename = os.path.join(self.plots_dir, f"{self.method_name}_{self.args.dataset}_{self.args.seen_class_ratio}")

        try:
            # 1. Optimization History
            fig_hist = plot_optimization_history(self.study)
            fig_hist.write_image(f"{base_filename}_history.png")

            # 2. Parameter Importances (requires >1 parameter and enough trials)
            if len(self.study.best_params) > 1 and len(self.study.trials) >= 5:
                 try:
                     fig_imp = plot_param_importances(self.study)
                     fig_imp.write_image(f"{base_filename}_importance.png")
                 except Exception as e:
                     print(f"  Could not generate importance plot: {e}") # Sometimes fails if parameters are not comparable


            # 3. Slice Plot
            if self.study.best_params: # Check if there are parameters
                 fig_slice = plot_slice(self.study)
                 fig_slice.write_image(f"{base_filename}_slice.png")

            # 4. Contour Plot (for pairs of parameters)
            if len(self.study.best_params) >= 2:
                param_names = list(self.study.best_params.keys())
                # Plot first few pairs
                num_contour_plots = 0
                max_contour_plots = 3
                for i in range(len(param_names)):
                     for j in range(i + 1, len(param_names)):
                          if num_contour_plots >= max_contour_plots: break
                          try:
                              fig_contour = plot_contour(self.study, params=[param_names[i], param_names[j]])
                              fig_contour.write_image(f"{base_filename}_contour_{param_names[i]}_{param_names[j]}.png")
                              num_contour_plots += 1
                          except Exception as e:
                              # This can fail if a parameter has only one value tested etc.
                              print(f"  Could not generate contour plot for {param_names[i]} vs {param_names[j]}: {e}")
                     if num_contour_plots >= max_contour_plots: break


            print(f"Optuna visualization plots saved to: {self.plots_dir}")

        except ImportError:
             print("  Plotly or Kaleido not installed. Skipping visualization.")
             print("  Install using: pip install plotly kaleido")
        except Exception as e:
            print(f"Error during Optuna visualization: {e}")
            import traceback
            traceback.print_exc()


def load_best_params(method_name, dataset, seen_class_ratio):
    """Loads previously saved best hyperparameters from a JSON file."""
    file_path = f"tuning_results/{method_name}_tuning_{dataset}_{seen_class_ratio}.json"
    print(f"[Parameter Loading] Checking for previous tuning results: {file_path}")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if 'best_params' in data:
                print(f"  Found previous results. Loading best parameters.")
                return data['best_params']
            else:
                print("  Found previous results file, but 'best_params' key is missing.")
                return None
        except Exception as e:
            print(f"  Error loading parameters from {file_path}: {e}")
            return None
    else:
        print("  No previous tuning file found.")
        return None

def get_default_best_params(method_name):
    """Provides default hyperparameters for each method, based on papers or common practice."""
    print(f"[Parameter Loading] Getting default parameters for method '{method_name}'...")
    defaults = {}
    if method_name == 'threshold':
        defaults = {'param_threshold': 0.5}
    elif method_name == 'openmax':
        defaults = {'param_openmax_tailsize': 50, 'param_openmax_alpha': 10}
    elif method_name == 'crosr':
        defaults = {'param_crosr_reconstruction_threshold': 0.9, 'param_crosr_tailsize': 100}
    elif method_name == 'doc':
        defaults = {'param_doc_k': 3.0}
    elif method_name == 'adb':
        # --- ADB 기본값 설정 ---
        defaults = {
            'param_adb_distance': 'cosine', # 평가 시 사용
            # 아래는 학습 관련 파라미터의 기본값 (argparse와 일치시키거나 논문 기반 값 사용)
            'lr_adb': 5e-4, # argparse의 기본값과 일치시킴 (또는 실험적으로 찾은 값)
            'param_adb_delta': 0.1,
            'param_adb_alpha': 0.1,
            'adb_freeze_backbone': True
        }
        # --- ---
    else:
         print(f"  Warning: No defaults defined for method '{method_name}'.")

    print(f"  Defaults for {method_name}: {defaults}")
    return defaults
