import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy import stats as st
import torch
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from pytorch_tabnet.tab_model import TabNetClassifier
from tabicl import TabICLClassifier
from tabpfn import TabPFNClassifier
from tabpfn_extensions.interpretability.shap import get_shap_values
import shap
import matplotlib.pyplot as plt
plt.style.use('default')


scaled_cols = [
    "STAI_Baseline_STATE", "STAI_Baseline_TRAIT",
    "Promis_Total_Raw",
    "PSS_Total_Score",
    'Panas_Positive_Affect_Score',
    'Panas_Negative_Affect_Score',
    'Maas_Raw_Score',
    'Maas_Mean',
    'FFMQ_Total_RAW_Score_with_Observation_subscale_',
    'FFMQ_Total_Score_including_Observation_subscale_Mean',
    'FFMQ_Total_RAW_score_Excluding_Observation_Subscale_',
    'FFMQ_Total_Score_excluding_Observation_subscale_Mean',
    'FFMQ_Observation_subscale_Mean',
    'FFMQ_Description_Subscale_Mean',
    'FFMQ_Acting_with_Awareness_Subscale_Mean',
    'FFMQ_Non-judgemental_Mean',
    'FFMQ_Non-reactivity_Mean',
    "Age",
    "AQ_baseline", "TMB_base", "LC_baseline",
]

class SklearnWrapper(BaseEstimator, ClassifierMixin):
    def __init__(self, model_name, col_names=None, **kwargs):
        """
        model_name : the actual model name: tabstar, tabnet, tabpfn, tabicl
        kwargs      : hyperparameters for the model
        """
        
        if model_name == 'tabnet':
            model_class = TabNetClassifier
        elif model_name == 'tabpfn':
            model_class = TabPFNClassifier
        elif model_name == 'tabicl':
            model_class = TabICLClassifier
        else:
            raise ValueError(f"There is no model named: {model_name}")
        self.model_name = model_name
        self.model_class = model_class
        
        self.kwargs = kwargs
        self.model = None
        self.col_names = col_names
        self.preproccessing = None

    def fit(self, X, y):
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)
        columns = X.columns
        self.preproccessing = ColumnTransformer(
            transformers=[
                ('scaled', StandardScaler(), scaled_cols),  # scale_cols - scaling
            ],
            remainder='passthrough'
        )
        self.model = self.model_class(**self.kwargs)
        X = self.preproccessing.fit_transform(X)
        # X = X.to_numpy()
        if self.model_name == 'tabstar':
            X = pd.DataFrame(X, columns=columns)
        elif self.model_name == 'tabnet':
            X = X.astype(np.float32)
            y = y.to_numpy()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        if isinstance(X, np.ndarray):
            X = pd.DataFrame(X, columns=self.col_names)
        X = X.reset_index(drop=True)
        columns = X.columns
        X = self.preproccessing.transform(X)
        # X = X.to_numpy()
        if self.model_name == 'tabstar':
            X = pd.DataFrame(X, columns=columns)
        elif self.model_name == 'tabnet':
            X = X.astype(np.float32)
        return self.model.predict(X)

    def predict_proba(self, X):
        X = X.reset_index(drop=True)
        columns = X.columns
        X = self.preproccessing.transform(X)
        # X = X.to_numpy()
        if self.model_name == 'tabstar':
            X = pd.DataFrame(X, columns=columns)
        elif self.model_name == 'tabnet':
            X = X.astype(np.float32)
        preds = self.model.predict_proba(X)
        if self.model_name == 'tabstar':
            preds = np.stack([1 - preds, preds], axis=1)
            
        return preds

    def get_params(self, deep=True):
        params = {"model_name": self.model_name}
        params.update(self.kwargs)
        return params

    def set_params(self, **params):
        if "model_name" in params:
            self.model_name = params.pop("model_name")
        self.kwargs.update(params)
        return self



def run_gridsearch(model_name, model_param, X, y, param_grid, scoring="accuracy", n_splits=3, n_jobs=1, verbose=0, random_state=67):
    """
    model_name : the class of the model (e.g., TabNetClassifier)
    X, y        : dataset
    param_grid  : dict of hyperparameters for GridSearch
    scoring     : sklearn scoring metric
    n_splits          : cross validation folds
    n_jobs      : parallel jobs (set =1 for PyTorch models to avoid errors)
    """

    # cv = RepeatedStratifiedKFold(n_repeats=5, n_splits=n_splits, random_state=random_state)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    wrapper = SklearnWrapper(model_name, col_names=X.columns, **model_param)
    
    grid = GridSearchCV(
        estimator=wrapper,
        param_grid=param_grid,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        verbose=verbose,
        refit=True
    )
    grid.fit(X, y)
    return grid

def get_shap(best_model, X_train, X_test, random_state=67):
        

    transformer = best_model.preproccessing
    model = best_model.model
    
    prc_X_train = transformer.fit_transform(X_train).astype(np.float32)
    prc_X_test = transformer.transform(X_test).astype(np.float32)

    # explainer = shap.KernelExplainer(lambda X: model.predict_proba(X)[:, 1], prc_X_train, link="logit", seed=random_state)
    # explainer = shap.PermutationExplainer(lambda X: model.predict_proba(X)[:, 1], prc_X_train, seed=random_state)
    if model.__class__.__name__ == 'TabNetClassifier':
        train_shap_values, _ = model.explain(prc_X_train, normalize=True)
        test_shap_values, _ = model.explain(prc_X_test, normalize=True)
    elif model.__class__.__name__ == 'TabPFNClassifier':
        train_shap_values = get_shap_values(model, prc_X_train).values[:, :, 1]
        test_shap_values = get_shap_values(model, prc_X_test).values[:, :, 1]
    else:
        # explainer = shap.PartitionExplainer(lambda X: model.predict_proba(X)[:, 1], prc_X_train, seed=random_state)
        
        # train_shap_values = explainer(prc_X_train)#.shap_values(prc_X_train)
        # test_shap_values = explainer(prc_X_test)#.shap_values(prc_X_test)


        
        explainer = shap.PermutationExplainer(lambda X: model.predict_proba(X)[:, 1], prc_X_train, seed=random_state)
        train_shap_values = explainer.shap_values(prc_X_train)
        test_shap_values = explainer.shap_values(prc_X_test)
        
    
    train_shap_values = train_shap_values.squeeze()
    test_shap_values = test_shap_values.squeeze()
    
    return train_shap_values, test_shap_values

def save_shap_results(shap_results, data, save_path, mode='trainset'):
    
    
    shap.summary_plot(shap_results, data, show=False)
    
    plt.tight_layout()
    plt.title(f"SHAP Summary of {mode}")
    plt.savefig(save_path.replace('.EXP', '.pdf'), format='pdf', 
                  bbox_inches='tight', dpi=600)
    plt.savefig(save_path.replace('.EXP', '.svg'), format='svg', 
                  bbox_inches='tight', dpi=600)
    plt.close()

def save_comparing_plot(info, title, save_direc):

    models = list()
    means = list()
    ci_lower = list()
    ci_upper = list()
    for model_name, model_info in info.items():
        models.append(model_name)
        means.append(model_info['best_auc'])
        ci_lower.append(model_info['ci_lower'])
        ci_upper.append(model_info['ci_upper'])

    # Compute error bars
    error_lower = np.array(means) - np.array(ci_lower)
    error_upper = np.array(ci_upper) - np.array(means)
    error = [error_lower, error_upper]

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(means, models, xerr=error, fmt='o', color='navy',
                ecolor='black', elinewidth=1.5, capsize=4, markersize=8)

    # Add value labels ABOVE the points with more spacing
    for mean, y_pos in zip(means, range(len(models))):
        ax.text(mean, y_pos + 0.05, f"{mean:.4f}",   # 🔹 y축에서 0.25 위로 올림
                va='bottom', ha='center', fontsize=10, color="black")

    # Hide spines except right
    for spine in ["top", "left", "bottom"]:
        ax.spines[spine].set_visible(False)

    # Make right spine gray
    ax.spines["right"].set_color("gray")
    ax.spines["right"].set_alpha(0.5)

    # Vertical reference lines
    ax.xaxis.grid(True, linestyle='-', alpha=0.5, color='gray')

    # Labels and title
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models)
    ax.set_xlabel("Confidence Interval")
    ax.set_ylabel("Models", labelpad=14, fontsize=12)
    ax.set_title(f"Confidence Intervals AUCs for {title}", loc='center', pad=30)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(os.path.join(save_direc, 'shap.png'))
    plt.close()

def NestedCV(X, y, n_splits=5, random_state=67, save_path=None, **kwargs):
    # Outer CV: Validation with KFold
    if save_path is None:
        repeated_n = 1
    else:
        repeated_n = 5
    train_fold_shap_values = np.zeros_like(X, dtype=np.float32)
    test_fold_shap_values = np.zeros_like(X, dtype=np.float32)
    best_est = []
    for i in range(1, repeated_n+1):
        cv_outer = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state+30*i)
        print(f"No.{i} NCV")
        # Nested CV to achieve best estimators for each loop
        for fold, (train_index, test_index) in enumerate(cv_outer.split(X, y)):
          
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            
            search = run_gridsearch(
                X=X_train, y=y_train,
                random_state=random_state,
                n_splits=n_splits,
                **kwargs
            )
            
            
            best_est.append(search.best_estimator_)

            pred = search.best_estimator_.predict_proba(X_test)[:, 1]
            auc_score = roc_auc_score(y_test, pred)
            # outer_results.append(auc_score)

            print(f"Fold {fold+1}: Best Param: {search.best_estimator_}, AUC Score: {auc_score:.4f}")
            
                
            if save_path is not None:
              train_shap, test_shap = get_shap(search.best_estimator_, X_train, X_test, random_state=random_state)
              train_fold_shap_values[train_index] += train_shap
              test_fold_shap_values[test_index] += test_shap
            
            
      

    if save_path is not None:
        train_fold_shap_values /= (repeated_n*(n_splits-1))
        test_fold_shap_values /= repeated_n
        save_shap_results(train_fold_shap_values, X, save_path.format('train'), mode='trainset')
        save_shap_results(test_fold_shap_values, X, save_path.format('test'), mode='testset')
        pd.DataFrame(train_fold_shap_values, columns=X.columns).to_csv(save_path.format('train').replace('.EXP', '.csv'))
        pd.DataFrame(test_fold_shap_values, columns=X.columns).to_csv(save_path.format('test').replace('.EXP', '.csv'))
    print('--'*10)
    auc_list = list()
    ci_list = list()
    stdev_list = list()
    for i, est in enumerate(best_est):
        aucs = list()

        for train_index, test_index in cv_outer.split(X, y):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y[train_index], y[test_index]
            est.fit(X_train, y_train)
            pred = est.predict(X_test)
            pred_proba = est.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, pred_proba)
            aucs.append(auc)


        avg_auc = sum(aucs) / len(aucs)
        stdev = np.std(aucs)

        # 95% confidence interval using t-distribution
        ci_low, ci_high = st.t.interval(0.95,
            df=n_splits - 1,
            loc=avg_auc,
            scale=stdev / np.sqrt(n_splits)
        )
        # print(f"Params = {model}")
        auc_list.append(avg_auc)
        stdev_list.append(stdev)
        ci_list.append((ci_low, ci_high))
        print(f"MODEL-{i}: {est.get_params()}")
        print(f"AUC = {avg_auc:.4f} | Stdev = {stdev:.4f} | 95% CI = [{ci_low:.4f}, {ci_high:.4f}]")
    max_index = np.argmax(auc_list)
    return best_est, {'best_auc': auc_list[max_index], 'stdev':stdev_list[max_index], 'ci_lower':ci_list[max_index][0], 'ci_upper':ci_list[max_index][1]}


def get_param_grid(model_name, random_state=67):

    if model_name == 'tabnet':
        model_param = {
            "optimizer_fn": torch.optim.AdamW,
            "verbose": 0,
            "seed": random_state,
            "device_name": 'cpu',
        }
        param_grid = {
            "n_d": [8, 16, 32],
            "n_a": [8, 16, 32],
            "n_steps": [3, 5, 7],
            "gamma": [1.0, 1.3, 1.5, 2.0],
            "lambda_sparse": [1e-5, 1e-4, 1e-3],
            # "momentum": [0.02, 0.1, 0.3],
            # "clip_value": [1, 2, 5],
            # "mask_type": ["sparsemax", "entmax"],
        }
    elif model_name == 'tabicl':
        model_param = {
            "device": "cuda",
            "random_state": random_state,
            "verbose": False,
            "norm_methods": None,
            "n_jobs": 1
        }

        param_grid = {
            "n_estimators": [16, 32, 64],
            "feat_shuffle_method": ["latin", "random"],
            "class_shift": [True, False],
            "softmax_temperature": [0.7, 0.9, 1.1, 1.3],
            # "batch_size": [8, 16, 32],
        }

    elif model_name == 'tabstar':
        model_param = {
            "device": "cpu",
            "random_state": random_state,
            "verbose": True,
        }
        param_grid = {
            "lora_lr": [0.0001, 0.0005, 0.001, 0.005],
            "lora_r": [8, 16, 32, 64],
            "lora_batch": [32, 64, 128],
            "global_batch": [64, 128, 256],
            # "max_epochs": [30, 50, 80],
            # "patience": [3, 5, 8]
        }
    elif model_name == 'tabpfn':
        model_param = {
            "device": "cuda:0",
            "random_state": random_state,
            "n_jobs": 1,
        }
        param_grid = {
            "n_estimators": [4, 8, 16, 32],
            "softmax_temperature": [0.5, 0.9, 1.3],
            "balance_probabilities": [False, True],
            "average_before_softmax": [False, True],
            # "fit_mode": ["fit_preprocessors", "low_memory"],
        }
    return model_param, param_grid