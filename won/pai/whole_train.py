import pandas as pd
from preprocessing import *
from data_load import *
from training import *
from display import *
import torch
from sklearn.metrics import root_mean_squared_error
# Linear
from sklearn.linear_model import ElasticNet, SGDRegressor
# Ensemble
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
# Neural
from pytorch_tabnet.tab_model import TabNetRegressor
from tabpfn import TabPFNRegressor
# from tab_transformer_pytorch import TabTransformer
import os
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use('default')
random_state = 67



def whole_training(baseline_score_cols, data_direc, model_name='elastic', vif_filtering=False, pdf=None, save_direc=None):

    # -------------- Data Load Process --------------

    # Smart Data Load
    df_smart = load_smart_data(data_direc, baseline_score_cols)
    # Mapping
    df_smart = custom_mapping(df_smart)
    # Reverse Scoring
    df_smart = reverse_scoring(df_smart)

    # Qualtrics Data Load
    df_Qualtrics = pd.read_excel(os.path.join(data_direc, "Qualtrics_Both.xlsx"))
    # delete minutes columns and the posture columns
    drop_col = [
        'required_mindfulness_min',
        'required_learn_min',
        'all_mindfulness_min',
        'all_required_min',
        'total_min',
        'active_posture_pct',
    ]
    df_Qualtrics = df_Qualtrics.drop(columns=drop_col)

    # Mapping
    aq_string = ["Definitely Agree", "Slightly Agree", "Slightly disagree", "Definitely Disagree"]
    df_Qualtrics = map_likert_responses(df_Qualtrics, "AQ", 1, 50, aq_string)
    # Reverse Scoring
    df_Qualtrics = reverse_scoring_per_score(df_Qualtrics, "AQ", [2, 4, 5, 6, 7, 9, 12, 13, 16, 18, 19, 20, 21, 22, 23, 26, 33, 35, 39, 41, 42, 43, 45, 46])
    # Diagnosed Feature Mapping
    df_Qualtrics = diagnosed_mapping(df_Qualtrics)
    # Encoding for remaining categorical Data
    # Centered binary Encoding
    df_Qualtrics = centered_binary_encoding(df_Qualtrics, "Birth_Sex")
    df_Qualtrics = centered_binary_encoding(df_Qualtrics, "Prior_Meditate")
    # Effect Encoding for multiple class
    df_Qualtrics = effect_encoding(df_Qualtrics, "Ethnicity")
    df_Qualtrics = df_Qualtrics.drop(columns=['group'])
    
    dataset_total = pd.merge(df_smart, df_Qualtrics, on='ID', how='inner')

    # Delete columns based on VIF to deal with multicollinearity
    if vif_filtering:
        print("VIF Filtering")
        dataset_total = remove_high_vif_with_correlation(orig_X = dataset_total, exception_cols=['t1_anxiety', 'group', 'ID', 'y'], threshold=10)
    dataset_hmp = dataset_total[dataset_total.group == 0.5].reset_index(drop=True).drop(columns=['ID', 'group'])
    dataset_wait = dataset_total[dataset_total.group == -0.5].reset_index(drop=True).drop(columns=['ID', 'group'])

    # -------------- Training Process ---------------

    # training settings of each model.

    CV_type = RepeatedCV
    standardize_all = False # whether to stardardize features selectively or every feature.
    fold_num = 3
    
    if model_name == 'tabpfn':
        model = TabPFNRegressor
        model_param = {
            "device": "cuda:0",
            "random_state": random_state,
            "n_jobs": 1,
        }
        param_grid = {
            "n_estimators": [4, 8, 16, 32],
            "softmax_temperature": [0.5, 0.9, 1.3],
            # "balance_probabilities": [False, True],
            # "average_before_softmax": [False, True],
            # "fit_mode": ["fit_preprocessors", "low_memory"],
        }
        CV_type = CustomTabNetCV
    elif model_name == 'tabnet':
        model = TabNetRegressor
        model_param = {
            "optimizer_fn": torch.optim.AdamW,
            "verbose": 0,
            "seed": random_state,
            "device_name": 'cpu',
        }
        param_grid = {
            "n_d": [8, 16],                 # decoder feature dimension
            "n_steps": [3, 5],              # the num of decision step
            "gamma": [1.0, 1.5],          # feature reuse penalty
            # "lambda_sparse": [1e-4, 1e-3],     # sparsity regularization
            # "momentum": [0.02, 0.05],          # batch norm momentum
        }

        CV_type = CustomTabNetCV
    elif model_name == 'random forest':
        model = RandomForestRegressor
        model_param = {
            'random_state': random_state
        }
        param_grid = {
            'n_estimators': [100, 200, 500],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            #'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
            'bootstrap': [True, False]
        }
    elif model_name == 'xgb':
        model = XGBRegressor
        model_param = {
            'random_state': random_state
        }
        param_grid = {
            'n_estimators': [100, 300],
            'max_depth': [3, 5, 7],
            'learning_rate': [0.01, 0.1, 0.3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            # 'reg_alpha': [0, 0.1, 1],
            # 'reg_lambda': [1, 5, 10]
        }
    elif model_name == 'decision-tree':
        model = DecisionTreeRegressor
        model_param = {
            'random_state': random_state,
            'criterion': 'squared_error'
        }
        param_grid = {
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10],
            # 'min_samples_leaf': [1, 2, 4],
            # 'max_features': ['auto', 'sqrt', 'log2', None],
            'ccp_alpha': [0.0, 0.001, 0.01, 0.1]
        }
    elif model_name == 'catboost':
        model = CatBoostRegressor
        model_param = {
            'random_state': random_state,
            'verbose': 0
        }
        param_grid = {
            'iterations': [100, 300],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.1],
            # 'l2_leaf_reg': [1, 3, 5],
            # 'border_count': [32, 64, 128]
        }
    elif model_name == 'sgd':
        model = SGDRegressor
        model_param = {
            'random_state': random_state,
            'loss': 'huber' # 'squared_error', 'huber', 'epsilon_insensitive'
        }

        param_grid = {
            # 'penalty': ['l2', 'l1', 'elasticnet'],
            'alpha': [1e-4, 1e-3, 1e-2],
            # 'learning_rate': ['constant', 'optimal', 'invscaling'],
            'eta0': [0.01, 0.1, 1]
        }
        standardize_all = True
    elif model_name == 'elastic':
        model = ElasticNet
        model_param = {
            'random_state': random_state,
            'max_iter': 10000
        }
        param_grid = {
            "alpha": np.logspace(-3, 2, 20),
            "l1_ratio": np.linspace(0.05, 1.0, 10)
        }
        standardize_all = True
    else:
        raise RuntimeError(f"There is no information about the model named {model_name}.")
    

    os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'
    scaled_cols = [
        "STAI Baseline STATE", "STAI Baseline TRAIT",
        "Promis Total Raw",
        "PSS Total Score",
        "Panas Positive Affect Score", "Panas Negative Affect Score",
        "Maas Raw Score", "Maas Mean",
        "FFMQ Total RAW Score with Observation subscale", "FFMQ Total Score including Observation subscale Mean",
        "FFMQ Total RAW score Excluding Observation Subscale", "FFMQ Total Score excluding Observation subscale Mean",
        "FFMQ Observation subscale Mean", "FFMQ Description Subscale Mean",
        "FFMQ Acting with Awareness Subscale Mean",
        "FFMQ Non-judgemental Mean", "FFMQ Non-reactivity Mean",
        "Age",
        "AQ_baseline", "TMB_base", "LC_baseline",
        "t1_anxiety"
    ]
    if standardize_all:
        scaled_cols = None
    else:
        scaled_cols = list((set(scaled_cols) & set(dataset_hmp.columns))- set(baseline_score_cols))

    n_repeats = 10
    
    
    
    X_hmp = dataset_hmp.drop(columns=["y"])
    y_hmp = dataset_hmp["y"].values

    X_wait = dataset_wait.drop(columns=["y"])
    y_wait = dataset_wait["y"].values

    # To save Holdout prediction
    within_predictions_hmp = np.ones(len(dataset_hmp)) * 100
    within_predictions_wait = np.ones(len(dataset_wait)) * 100

    within_RMSE_hmp, within_RMSE_train_hmp = [], []
    within_RMSE_wait, within_RMSE_train_wait = [], []

    kf_hmp = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)
    kf_wait = KFold(n_splits=fold_num, shuffle=True, random_state=random_state)

    within_hmp_shap = list()
    within_wait_shap = list()
    within_test_indices_hmp = list()
    within_test_indices_wait = list()

    # Within Training Process
    print("-WITHIN TRAINING-")
    for (fold_idx_hmp, (train_idx_hmp, test_idx_hmp)), (fold_idx_wait, (train_idx_wait, test_idx_wait)) in zip(
    enumerate(kf_hmp.split(X_hmp)), enumerate(kf_wait.split(X_wait))
    ):
        print(f"\n=== Fold {fold_idx_hmp+1}/{fold_num} ===")

        # --------- Achieve Prediction of HMP from HMP subjects ---------
        X_train_hmp, X_test_hmp = X_hmp.iloc[train_idx_hmp], X_hmp.iloc[test_idx_hmp]
        y_train_hmp, y_test_hmp = y_hmp[train_idx_hmp], y_hmp[test_idx_hmp]

        cv_module_hmp = CV_type(
            model_class=model,
            model_param=model_param,
            param_grid=param_grid,
            scaled_cols=scaled_cols,
            n_repeats=n_repeats,
            n_splits=fold_num,
            random_state=random_state
        )
        
        cv_module_hmp.train(X_train_hmp, y_train_hmp)

        # Inner train RMSE
        y_pred_train_hmp = cv_module_hmp.predict(X_train_hmp)
        train_rmse_hmp = root_mean_squared_error(y_train_hmp, y_pred_train_hmp)
        within_RMSE_train_hmp.append(train_rmse_hmp)


        # Outer holdout prediction
        y_pred_holdout_hmp = cv_module_hmp.predict(X_test_hmp)
        within_hmp_shap.append(
            cv_module_hmp.get_shap(X_train_hmp, X_test_hmp)
        )
        within_test_indices_hmp.append(
            test_idx_hmp
        )
        holdout_rmse_hmp = root_mean_squared_error(y_test_hmp, y_pred_holdout_hmp)

        within_RMSE_hmp.append(holdout_rmse_hmp)
        within_predictions_hmp[test_idx_hmp] = y_pred_holdout_hmp

        # print(f"[HMP] Best Params: {cv_module_hmp.grid.best_params_}")
        # print(f"[HMP] Train RMSE: {train_rmse_hmp:.4f} | Holdout RMSE: {holdout_rmse_hmp:.4f} | Cor: {holdout_cor_hmp:.4f}")

        # --------- Achieve Prediction of Wait from Wait subjects ---------
        X_train_wait, X_test_wait = X_wait.iloc[train_idx_wait], X_wait.iloc[test_idx_wait]
        y_train_wait, y_test_wait = y_wait[train_idx_wait], y_wait[test_idx_wait]

        cv_module_wait = CV_type(
            model_class=model,
            model_param=model_param,
            param_grid=param_grid,
            scaled_cols=scaled_cols,
            n_repeats=n_repeats,
            n_splits=fold_num,
            random_state=random_state
        )

        cv_module_wait.train(X_train_wait, y_train_wait)

        # Inner train RMSE
        y_pred_train_wait = cv_module_wait.predict(X_train_wait)
        train_rmse_wait = root_mean_squared_error(y_train_wait, y_pred_train_wait)
        within_RMSE_train_wait.append(train_rmse_wait)

        # Outer holdout prediction
        y_pred_holdout_wait = cv_module_wait.predict(X_test_wait)
        within_wait_shap.append(
            cv_module_wait.get_shap(X_train_wait, X_test_wait)
        )
        within_test_indices_wait.append(
            test_idx_wait
        )
        holdout_rmse_wait = root_mean_squared_error(y_test_wait, y_pred_holdout_wait)

        within_RMSE_wait.append(holdout_rmse_wait)
        within_predictions_wait[test_idx_wait] = y_pred_holdout_wait

        # print(f"[WAIT] Best Params: {cv_module_wait.grid.best_params_}")
        # print(f"[WAIT] Train RMSE: {train_rmse_wait:.4f} | Holdout RMSE: {holdout_rmse_wait:.4f} | Cor: {holdout_cor_wait:.4f}")
    
    # Between Training Process
    X_hmp = dataset_hmp.drop(columns=["y"])
    y_hmp = dataset_hmp["y"].values

    X_wait = dataset_wait.drop(columns=["y"])
    y_wait = dataset_wait["y"].values

    # --------- Acchieve HMP predictions from Wait subjects ----------
    print("-Between TRAINING-")
    cv_module_hmp = CV_type(
        model_class=model,
        model_param=model_param,
        param_grid=param_grid,
        scaled_cols=scaled_cols,
        n_repeats=n_repeats,
        n_splits=fold_num,
        random_state=random_state
    )
    cv_module_hmp.train(X_hmp, y_hmp)
    hmp_ale_result = cv_module_hmp.get_ale(X_hmp)
    hmp_ale_std = pd.Series([np.nanstd(ale).item() for ale in hmp_ale_result['ale_curves']], index=hmp_ale_result['feature_names'])
    hmp_ale_std['type'] = "Between HMP"
    hmp_prediction = cv_module_hmp.predict(X_hmp)


    between_wait_hmpPrediction = cv_module_hmp.predict(X_wait)
    total_control = pd.DataFrame(np.stack([within_predictions_wait, between_wait_hmpPrediction]).T, columns=['pred_control', 'pred_mindful'])
    total_control['y'] = y_wait
    total_control['Received'] = 'Control'
    total_control['t1_anxiety'] = X_wait['t1_anxiety']
    
    # --------- Achieve Wait predictions from Hmp subjects ----------
    cv_module_wait = CV_type(
        model_class=model,
        model_param=model_param,
        param_grid=param_grid,
        scaled_cols=scaled_cols,
        n_repeats=n_repeats,
        n_splits=fold_num,
        random_state=random_state
    )
    cv_module_wait.train(X_wait, y_wait)
    wait_ale_result = cv_module_wait.get_ale(X_wait)
    wait_ale_std = pd.Series([np.nanstd(ale).item() for ale in wait_ale_result['ale_curves']], index=wait_ale_result['feature_names'])
    wait_ale_std['type'] = "Between HMP"
    wait_prediction = cv_module_wait.predict(X_wait)


    between_hmp_waitPrediction = cv_module_wait.predict(X_hmp)
    total_mindful = pd.DataFrame(np.stack([between_hmp_waitPrediction, within_predictions_hmp]).T, columns=['pred_control', 'pred_mindful'])
    total_mindful['y'] = y_hmp
    total_mindful['Received'] = 'Mindful'
    total_mindful['t1_anxiety'] = X_hmp['t1_anxiety']
    
    # Calculate PAI based on Predictions
    total_merged = pd.concat([total_mindful, total_control], axis=0, ignore_index=True)
    total_merged['Received'] = total_merged['Received'].astype('category')
    total_merged['PAI'] = total_merged['pred_mindful'] - total_merged['pred_control']

    group_mean = total_merged.groupby("Received")["y"].mean()


    total_merged["Indicated"] = np.where(total_merged["PAI"] < 0, "Mindful", "Control")


    # -------------- Display Process ---------------
    fig, axes = plt.subplots(3, 3)
    plt.suptitle(', '.join(baseline_score_cols) + '\n\n', fontsize=24)


    # Display performance of predictions

    within_data = {
        "Group": ["Hmp", "Wait"],
        "Mean Inner RMSE": [round(np.mean(within_RMSE_train_hmp), 4), round(np.mean(within_RMSE_train_wait), 4)],
        "Mean Outer RMSE": [round(np.mean(within_RMSE_hmp), 4), round(np.mean(within_RMSE_wait), 4)]
    }

    between_data = {
        "Prediction": ["Hmp", "Wait"],
        "Train RMSE": [round(root_mean_squared_error(y_hmp, hmp_prediction), 4), round(root_mean_squared_error(y_wait, wait_prediction), 4)]
    }

    df_within = pd.DataFrame(within_data)
    df_between = pd.DataFrame(between_data)


    display_table(df_within, "==== Within Results ====", axes[0, 0])
    display_table(df_between, "==== Between Results ====", axes[0, 1])

    # Display Indicated (PAI < 0) figures for additional analysis
    tot_num = len(total_merged)
    mindful_indicated = sum(total_merged.Indicated == 'Mindful')
    summary_data = [
        ["Average of Mindful y", f"{group_mean.Mindful.item():.4f}"],
        ["Average of Control y", f"{group_mean.Control.item():.4f}"],
        ["Indicated counts of Mindful", f"{mindful_indicated}"],
        ["Indicated counts of Control", f"{tot_num - mindful_indicated}"],
        ["Indicated proportion of Mindful", f"{mindful_indicated / tot_num: .2f}"],
        ["Indicated proportion of Control", f"{1 - mindful_indicated/tot_num:.2f}"],
    ]

    summary_df = pd.DataFrame(summary_data, columns=["Metric", "Value"])

    display_table(summary_df, "Indicated = Mindful if PAI < 0 else Control", axes[1, 0])

    # Display PAI figure
    display_PAI_interaction(total_merged.copy(), axes[1, 1])
    
    # Display box plot and scatter plot of every predictions.
    validation_data = [within_predictions_wait, between_wait_hmpPrediction, between_hmp_waitPrediction, within_predictions_hmp]
    validation_src = [dataset_hmp.y, dataset_wait.y]
    title = ['wait_waitP', 'wait_hmpP', 'hmp_waitP', 'hmp_hmpP']
    
    axes[0, 2].set_title("Y")
    bplot = axes[0, 2].boxplot(
        validation_src,
        tick_labels=['hmp', 'wait'],
        patch_artist=True,
        widths=0.4,
        showfliers=False
    )
    colors = ["lightblue", "lightgreen"]
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for j, group_vals in enumerate(validation_src, start=1):
        x_jitter = np.random.normal(j, 0.03, size=len(group_vals))
        axes[0, 2].scatter(x_jitter, group_vals, alpha=0.9, edgecolor='k', s=10)

    axes[1, 2].set_title("Prediction")

    # Add everage value of y to the boxplot
    axes[1, 2].axhline(dataset_hmp.y.mean(), color='blue', label='avg(y) of HMP')
    axes[1, 2].axhline(dataset_wait.y.mean(), color='red', label='avg(y) of Wait')
    bplot = axes[1, 2].boxplot(
        validation_data,
        tick_labels=title,
        patch_artist=True,
        widths=0.4,
        showfliers=False
    )
    
    colors = ["lightblue", "lightgreen", 'pink', 'green']
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    for j, group_vals in enumerate(validation_data, start=1):
        x_jitter = np.random.normal(j, 0.03, size=len(group_vals))
        axes[1, 2].scatter(x_jitter, group_vals, alpha=0.9, edgecolor='k', s=10)

    # Display SHAP values
    within_hmp_shap = np.concatenate(within_hmp_shap, axis=0)
    avg_hmp_shap = pd.Series(np.mean(np.abs(within_hmp_shap), axis=0), index=X_hmp.columns)
    avg_hmp_shap['type'] = 'Within HMP'
    within_wait_shap = np.concatenate(within_wait_shap, axis=0)
    avg_wait_shap = pd.Series(np.mean(np.abs(within_wait_shap), axis=0), index=X_hmp.columns)
    avg_wait_shap['type'] = 'Within Wait'

    # Display SHAP values calculated by the model in "HMP within training process"
    plt.sca(axes[2, 0])
    shap.summary_plot(within_hmp_shap, X_hmp.reindex(np.concatenate(within_test_indices_hmp)), show=False)
    axes[2, 0].set_title("SHAP Summary Plot for HMP")

    # Display SHAP values calculated by the model in "Wait within training process"
    plt.sca(axes[2, 1])
    shap.summary_plot(within_wait_shap, X_wait.reindex(np.concatenate(within_test_indices_wait)), show=False)
    axes[2, 1].set_title("SHAP Summary Plot for Wait")

    # Display SHAP values based on merged SHAP results above.
    plt.sca(axes[2, 2])
    shap.summary_plot(
        np.concatenate([within_hmp_shap, within_wait_shap], axis=0),
        pd.concat([X_hmp.reindex(np.concatenate(within_test_indices_hmp)), X_wait.reindex(np.concatenate(within_test_indices_wait))], axis=0),
        show=False,
        plot_size=(30, 20)
    )
    axes[2, 2].set_title("SHAP Summary Plot for Total Data")

    final_shap_result = pd.concat(
        [avg_hmp_shap, avg_wait_shap],
        axis=1
    ).T
    final_ale_result = pd.concat(
        [hmp_ale_std, wait_ale_std],
        axis=1
    ).T

    final_ale_result['FA type'] = 'ALE'
    final_shap_result['FA type'] = 'SHAP'
    final_feature_importance_result = pd.concat([final_shap_result, final_ale_result], axis=0)
    final_feature_importance_result['base scores'] = ', '.join(baseline_score_cols)


    if pdf is not None:
        pdf.savefig(fig)
        plt.close()
    
    display_topk_ale(hmp_ale_result, k=20, cols=5, suptitle='ALE Summary Plot for HMP', pdf=pdf)
    display_topk_ale(wait_ale_result, k=20, cols=5, suptitle='ALE Summary Plot for WAIT', pdf=pdf)

    # save figures
    pai_fig, pai_ax = plt.subplots()

    plt.tight_layout()
    display_PAI_interaction(total_merged.copy(), pai_ax)
    if save_direc is not None:
      plt.savefig(os.path.join(save_direc, f'PAI_{"_".join(baseline_score_cols)}.svg'), format='svg', 
                  bbox_inches='tight', dpi=600)
      plt.savefig(os.path.join(save_direc, f'PAI_{"_".join(baseline_score_cols)}.pdf'), format='pdf', 
                  bbox_inches='tight', dpi=600)
    if pdf is not None:
        plt.close()
    

    

    return final_feature_importance_result

if __name__ == '__main__':
    
    from scipy.stats import ConstantInputWarning, NearConstantInputWarning
    from sklearn.exceptions import ConvergenceWarning
    # whole_training is a function conducting every process above.


    import pandas as pd
    from matplotlib.backends.backend_pdf import PdfPages
    import itertools
    import warnings
    import random
    import numpy as np
    random.seed(random_state)
    np.random.seed(random_state)
    os.environ["PYTHONHASHSEED"] = str(random_state)
    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)


    plt.rcParams['font.family'] = 'Arial'
    plt.rcParams['font.size'] = 16
    mpl.rcParams['pdf.fonttype'] = 42
    mpl.rcParams['ps.fonttype'] = 42
    mpl.rcParams['svg.fonttype'] = 'none'
    
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    warnings.filterwarnings("ignore", category=ConstantInputWarning)
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=NearConstantInputWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    root_path = os.path.dirname(__file__)
    data_direc = os.path.join(root_path, "data/aixliang")
    save_direc = os.path.join(root_path, 'results_fold3_1_tabpfn')
    os.makedirs(save_direc, exist_ok=True)

    # 'elastic', 'random forest', 'xgb', 'decision-tree', 'catboost', 'sgd', 'tabnet', 'tabpfn'
    for model_name in ['elastic', 'random forest', 'xgb', 'decision-tree', 'catboost', 'sgd', 'tabnet', 'tabpfn']:
        for score_num in [1, 2, 3]:
            for vif_filtering in [True]:
                print(f"[{model_name}-{score_num}]")
                save_model_direc = os.path.join(save_direc, model_name, f'{score_num}-scores')
                os.makedirs(save_model_direc, exist_ok=True)
                final_csv = pd.DataFrame([])
                with PdfPages(os.path.join(save_model_direc, f'{"VIF" if vif_filtering else "Raw"}.pdf')) as pdf:
                    for i in list(itertools.combinations([
                                "STAI Baseline STATE",
                                "STAI Baseline TRAIT",
                                "Promis Total Raw",
                                # "PSS Total Score"
                            ], score_num)):
                        final_feature_importance_result  = whole_training(
                            baseline_score_cols=list(i), data_direc=data_direc, vif_filtering=vif_filtering,
                            model_name=model_name, pdf=pdf, save_direc=save_model_direc
                        )
                        final_csv = pd.concat([final_csv, final_feature_importance_result], axis=0, ignore_index=True)
                        
                final_csv.to_csv(os.path.join(save_model_direc, 'feature_importance.csv'))