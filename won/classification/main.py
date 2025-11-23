from utils import NestedCV, get_param_grid
import pandas as pd
import os
import torch
import numpy as np
import random
import warnings
import json
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    # tabnet, tabicl, tabstar, tabpfn
    random_state = 67
    random.seed(random_state)
    np.random.seed(random_state)
    torch.random.manual_seed(random_state)


    root_path = os.path.dirname(__file__)
    save_path = os.path.join(root_path, 'cls_results')
    os.makedirs(save_path, exist_ok=True)
    y_state = pd.read_csv(os.path.join(root_path, "y_state.csv")).squeeze()
    y_trait = pd.read_csv(os.path.join(root_path, "y_trait.csv")).squeeze()
    X = pd.read_csv(os.path.join(root_path, "X.csv"))

    print("--------------------Comparing--------------------")
    print("--------------------STATE--------------------")

    state_log = dict()
    for model_name in ['tabnet', 'tabicl', 'tabpfn']:
        model_param, param_grid = get_param_grid(model_name=model_name)
        print(f"[{model_name} Training]")
        ests, best_auc_info = NestedCV(
            X=X,
            y=y_state,
            n_splits=5,
            random_state=random_state,
            model_name=model_name,
            model_param=model_param,
            param_grid=param_grid,
            scoring='f1'
        )

        state_log[model_name] = best_auc_info
        print('--'*20)
    with open(os.path.join(save_path, 'comparing_state.json'), 'w') as f:
        json.dump(state_log, f)
        
    print("--------------------TRAIT--------------------")
    trait_log = dict()
    for model_name in ['tabnet', 'tabicl', 'tabpfn']:
        model_param, param_grid = get_param_grid(model_name=model_name)
        print(f"[{model_name} Training]")
        ests, best_auc_info = NestedCV(
            X=X,
            y=y_trait,
            n_splits=5,
            random_state=random_state,
            model_name=model_name,
            model_param=model_param,
            param_grid=param_grid,
            scoring='f1'
        )

        trait_log[model_name] = best_auc_info
        print('--'*20)
    with open(os.path.join(save_path, 'comparing_trait.json'), 'w') as f:
        json.dump(trait_log, f)


    print("--------------------SHAP--------------------")
    print("--------------------STATE--------------------")

    state_log = dict()
    for model_name in ['tabnet', 'tabicl', 'tabpfn']:
        model_param, param_grid = get_param_grid(model_name=model_name)
        print(f"[{model_name} Training]")
        ests, best_auc_info = NestedCV(
            X=X,
            y=y_state,
            n_splits=5,
            random_state=random_state,
            model_name=model_name,
            model_param=model_param,
            param_grid=param_grid,
            scoring='f1',
            save_path=os.path.join(save_path, f'{model_name}' + '_shap_state_{}.EXP')
        )

        state_log[model_name] = best_auc_info
        print('--'*20)
        with open(os.path.join(save_path, f'shap_comparing_state_{model_name}.json'), 'w') as f:
            json.dump(state_log, f)

    print("--------------------TRAIT--------------------")
    trait_log = dict()
    for model_name in ['tabnet', 'tabicl', 'tabpfn']:
        model_param, param_grid = get_param_grid(model_name=model_name)
        print(f"[{model_name} Training]")
        ests, best_auc_info = NestedCV(
            X=X,
            y=y_trait,
            n_splits=5,
            random_state=random_state,
            model_name=model_name,
            model_param=model_param,
            param_grid=param_grid,
            scoring='f1',
            save_path=os.path.join(save_path, f'{model_name}' + '_shap_trait_{}.EXP')
        )

        trait_log[model_name] = best_auc_info
        print('--'*20)
        with open(os.path.join(save_path, f'shap_comparing_trait_{model_name}.json'), 'w') as f:
            json.dump(trait_log, f)