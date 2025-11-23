from sklearn.model_selection import KFold, RepeatedKFold, GridSearchCV
from sklearn.linear_model import ElasticNet
from scipy.stats import pearsonr, ConstantInputWarning, NearConstantInputWarning
from sklearn.exceptions import ConvergenceWarning
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.metrics import root_mean_squared_error
import torch
import shap
from alibi.explainers import ALE
from tabpfn_extensions.interpretability.shap import get_shap_values

class RepeatedCV(object):

    def __init__(self, model_class=ElasticNet, scaled_cols=None, model_param={'max_iter': 10000},
                    param_grid={"alpha": np.logspace(-3, 2, 10), "l1_ratio": np.linspace(0.05, 1.0, 20) },
                    n_splits=4, n_repeats=10, random_state=67):
        
        if scaled_cols is None:
            preprocessor = StandardScaler()
        else:
            # Features are selectively scaled.
            preprocessor = ColumnTransformer(
                transformers=[
                    ('scaled', StandardScaler(), scaled_cols),  # scale_cols - scaling
                ],
                remainder='passthrough'
            )
        model = model_class(
            **model_param
        )
        pipeline = Pipeline([
            ('preprocessing', preprocessor),
            ('model', model)
        ])

        cv_setup = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
        grid = GridSearchCV(
            pipeline,
            param_grid={'model__' + k:v for k, v in param_grid.items()},
            cv=cv_setup,
            scoring='neg_root_mean_squared_error',
            refit=True,
            verbose=0,
            n_jobs=1
        )
        self.grid = grid
        self.seed = random_state
    

    def train(self, X_train, y_train):
        self.grid.fit(X_train, y_train)

    def predict(self, X_test):
        return self.grid.predict(X_test)
    

    def _model_family(self, model):
        name = type(model).__name__
        if any(k in name for k in ["ElasticNet", "Lasso", "Ridge", "LinearRegression", "SGDRegressor",
                                "LogisticRegression", "LinearSVC", "LinearSVR"]):
            return "linear"
        if any(k in name for k in ["Forest", "RandomForest", "ExtraTrees",
                                "XGB", "LGBM", "CatBoost", "DecisionTree", "GradientBoosting"]):
            return "tree"
        return "other"

    def get_shap(self, X_train, X_test):
        processed_train_X = self.grid.best_estimator_['preprocessing'].transform(X_train)
        processed_test_X = self.grid.best_estimator_['preprocessing'].transform(X_test)
        model_type = self._model_family(self.grid.best_estimator_['model'])

        if model_type == 'linear':
            explainer = shap.LinearExplainer(self.grid.best_estimator_['model'], processed_train_X, seed=self.seed)
            shap_values = explainer(processed_test_X).values
        elif model_type == 'tree':
            explainer = shap.TreeExplainer(self.grid.best_estimator_['model'], model_output='raw', feature_perturbation="tree_path_dependent")
            shap_values = explainer(processed_test_X).values
        else:
            explainer = shap.KernelExplainer(self.grid.best_estimator_['model'].predict, processed_train_X, link="identity", seed=self.seed)
            shap_values = explainer.shap_values(processed_test_X)
            shap_values = shap_values.squeeze()
            
        return shap_values
    
    def get_ale(self, X, target_names=["Change in Anxiety"]):
        
            
        
        feature_names = X.columns

        def predictor(A):
            A_df = pd.DataFrame(A, columns=feature_names)
            return self.grid.best_estimator_.predict(A_df)
        
        explainer = ALE(
            predictor,
            feature_names=feature_names,
            target_names=target_names
        )
        X = X.values
        scores_std = []
        ale_curves = []
        x_axes = []
        feature_deciles = []

        for j in range(len(feature_names)):
            exp = explainer.explain(
                X,
                features=[j]
            )

            ale_vals = exp.ale_values[0] # shape (m,)
            xs = exp.feature_values[0]   # shape (m,)
            fd = exp.feature_deciles[0]

            std_score = float(np.nanstd(ale_vals))

            scores_std.append(std_score)
            ale_curves.append(ale_vals)
            x_axes.append(xs)
            feature_deciles.append(fd)


        order_std = np.argsort(scores_std)[::-1]

        results = {
            "feature_names": feature_names,
            "scores_std": np.array(scores_std),
            "rank_std": order_std,
            "ale_curves": ale_curves,
            "x_axes": x_axes,
            "feature_deciles":feature_deciles 
        }
        return results


class CustomTabNetCV:
    def __init__(self, model_class, model_param, param_grid, scaled_cols=None, n_splits=4, n_repeats=5, max_epochs=200, patience=20, batch_size=32, random_state=67, **kwargs):
        self.model_class = model_class
        self.model_param = model_param
        self.param_grid = param_grid
        self.scaled_cols = scaled_cols
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.max_epochs = max_epochs
        self.patience = patience
        self.batch_size = batch_size
        self.random_state = random_state
        self.best_model = None
        self.best_score = np.inf
        self.best_params = {}
        self.preprocessor = None

    def _generate_param_combinations(self):
        from itertools import product
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())
        
        for comb in product(*values):
            yield dict(zip(keys, comb))

    def model_fit_tabnet(self, model, X_train, y_train, X_val, y_val, columns):
        if self.model_class.__name__ == 'TabSTARRegressor':
            X_train = pd.DataFrame(X_train, columns=columns)
            y_train = pd.Series(y_train.reshape(-1))
            X_val = pd.DataFrame(X_val, columns=columns)
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
        elif self.model_class.__name__ == 'TabNetRegressor':
            model.fit(
                X_train=X_train, y_train=y_train,
                eval_set=[(X_val, y_val)],
                eval_metric=['rmse'],
                max_epochs=self.max_epochs,
                patience=self.patience,
                batch_size=self.batch_size,
                virtual_batch_size=self.batch_size // 4,
                pin_memory=False
            )
            preds = model.predict(X_val)
        elif self.model_class.__name__ == 'TabPFNRegressor':
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
        return model, preds
    
    def train(self, X, y):
        rkf = RepeatedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats, random_state=self.random_state)
        
        assert self.preprocessor is None
        if self.scaled_cols is None:
            self.preprocessor = StandardScaler()
        else:
            # Features are selectively scaled.
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('scaled', StandardScaler(), self.scaled_cols),  # scale_cols - scaling
                ],
                remainder='passthrough'
            )
        
        y = y.copy().reshape(-1, 1) if len(y.shape) == 1 else y
        y = y.astype(np.float32)
        
        for param_set in self._generate_param_combinations():
            scores = []

            for train_idx, val_idx in rkf.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                self.preprocessor.fit(X_train)
                X_train_scaled = self.preprocessor.transform(X_train).astype(np.float32)
                X_val_scaled = self.preprocessor.transform(X_val).astype(np.float32)

                model = self.model_class(
                    **param_set,
                    **self.model_param,
                )
                
                model, preds = self.model_fit_tabnet(model, X_train_scaled, y_train, X_val_scaled, y_val, X_train.columns)

                rmse = root_mean_squared_error(y_val, preds)
                scores.append(rmse)

            avg_score = np.mean(scores)

            if avg_score < self.best_score:
                self.best_score = avg_score
                self.best_params = param_set
        
        final_model = self.model_class(
            **self.best_params,
            **self.model_param
        )
        
        
        X_scaled = self.preprocessor.fit_transform(X).astype(np.float32)
        final_model, _ = self.model_fit_tabnet(final_model, X_scaled, y, X_scaled, y, X.columns)

        self.best_model = final_model


    def predict(self, X):
        assert self.preprocessor is not None
        X_scaled = self.preprocessor.transform(X).astype(np.float32)
        return self.best_model.predict(X_scaled).reshape(-1)

    def get_best_params(self):
        return self.best_params

    def get_best_score(self):
        return self.best_score
    
    def get_shap(self, X_train, X_test):
        processed_train_X = self.preprocessor.transform(X_train).astype(np.float32)
        processed_test_X = self.preprocessor.transform(X_test).astype(np.float32)

        # explainer = shap.Explainer(self.grid.best_estimator_['model'], processed_train_X)
        # shap_values = explainer(processed_test_X).values
        if self.model_class.__name__ == 'TabPFNRegressor':
            print("Permutation Explainer")
            shap_values = get_shap_values(self.best_model, processed_test_X).values
        else:
            explainer = shap.KernelExplainer(self.best_model.predict, processed_train_X, link="identity", seed=self.random_state)
            shap_values = explainer.shap_values(processed_test_X)
        shap_values = shap_values.squeeze()
            
        return shap_values
    
    def get_ale(self, X, target_names=["Change in Anxiety"]):
        
        feature_names = X.columns

        def predictor(A):
            A_df = pd.DataFrame(A, columns=feature_names)
            processed_A_df = self.preprocessor.transform(A_df).astype(np.float32)
            return self.best_model.predict(processed_A_df)
        
        explainer = ALE(
            predictor,
            feature_names=feature_names,
            target_names=target_names
        )
        X = X.values
        scores_std = []
        ale_curves = []
        x_axes = []
        feature_deciles = []

        for j in range(len(feature_names)):
            exp = explainer.explain(
                X,
                features=[j]
            )

            ale_vals = exp.ale_values[0] # shape (m,)
            xs = exp.feature_values[0]   # shape (m,)
            fd = exp.feature_deciles[0]

            std_score = float(np.nanstd(ale_vals))

            scores_std.append(std_score)
            ale_curves.append(ale_vals)
            x_axes.append(xs)
            feature_deciles.append(fd)


        order_std = np.argsort(scores_std)[::-1]

        results = {
            "feature_names": feature_names,
            "scores_std": np.array(scores_std),
            "rank_std": order_std,
            "ale_curves": ale_curves,
            "x_axes": x_axes,
            "feature_deciles":feature_deciles 
        }
        return results