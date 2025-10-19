import os
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from sklearn.linear_model import LinearRegression

def load_smart_data(data_direc: str, score_cols: list) -> pd.DataFrame:

    # Baseline Dataset

    score_types = [
        'SRS', 'CATI', 'STAI', 'PROMIS', 'PSS', 'PANAS', 'MAAS', 'FFMQ', 
    ]
    new_col = list()
    # Baseline Score Load
    df_base_score = pd.read_excel(os.path.join(data_direc, 'SMART_Baseline_REDCap.xlsx'))
    for col in df_base_score.columns:
        try:
            int(col[0])
            for score_type in score_types:
                if score_type in col:
                    new_col.append(f"{score_type}_{col.split(' ')[0].split('.')[0]}")
        except:
            new_col.append(col)
    df_base_score.columns = new_col
    df_baseline_score_hmp = pd.read_excel(os.path.join(data_direc, "ARM1_Pre.xlsx"))
    # Extract Baseline scores corresponing ID that exists in ARM1_Pre.xlsx
    df_baseline_score_hmp = pd.merge(df_baseline_score_hmp[['ID']], df_base_score[df_baseline_score_hmp.columns], on='ID', how='inner')
    df_baseline_score_hmp["group"] = 0.5
    # Extract Baseline scores corresponing ID that exists in ARM2_Pre.xlsx
    df_baseline_score_wait = pd.read_excel(os.path.join(data_direc, "ARM2_Pre.xlsx"))
    df_baseline_score_wait = pd.merge(df_baseline_score_wait[['ID']], df_base_score[df_baseline_score_wait.columns], on='ID', how='inner')
    df_baseline_score_wait["group"] = -0.5

    # Main Score Load
    baseline_score_cols = ["ID"] + score_cols
    main_score_cols = [col.replace("Baseline ", "") for col in baseline_score_cols]

    df_main_score = pd.read_excel(os.path.join(data_direc, 'SMART_Main_REDCap.xlsx'))[list(main_score_cols) + ["Event Name"]]

    # Extract Main scores corresponing ID that exists in Baseline df for each group
    df_mid_score_hmp = df_main_score.merge(df_baseline_score_hmp[["ID"]], on="ID", how="inner")
    df_mid_score_hmp = df_mid_score_hmp[df_mid_score_hmp["Event Name"] == "Midpoint B (Arm 1: Intervention Group)"].drop(columns=['Event Name'])
    df_main_score_hmp = df_main_score.merge(df_baseline_score_hmp[["ID"]], on="ID", how="inner")
    df_main_score_hmp = df_main_score_hmp[df_main_score_hmp["Event Name"] == "Post-test (Arm 1: Intervention Group)"].drop(columns=['Event Name'])
    
    df_mid_score_wait = df_main_score.merge(df_baseline_score_wait[["ID"]], on="ID", how="inner")
    df_mid_score_wait = df_mid_score_wait[df_mid_score_wait["Event Name"] == "Midpoint A (Arm 2: Waitlist Control)"].drop(columns=['Event Name'])
    df_main_score_wait = df_main_score.merge(df_baseline_score_wait[["ID"]], on="ID", how="inner")
    df_main_score_wait = df_main_score_wait[df_main_score_wait["Event Name"] == "Pre-test  (Arm 2: Waitlist Control)"].drop(columns=['Event Name'])


    # merging scores for each group so that baseline & main df can be utilized.
    df_main_score_total = pd.concat([df_main_score_hmp, df_main_score_wait], axis=0, ignore_index=True)
    df_midscore_total = pd.concat([df_mid_score_hmp, df_mid_score_wait], axis=0, ignore_index=True)
    df_baseline_score_total = pd.concat([df_baseline_score_hmp, df_baseline_score_wait], axis=0, ignore_index=True)
    standardized_baseline = pd.DataFrame([])
    standardized_mid = pd.DataFrame([])
    standardized_main = pd.DataFrame([])

    # Standardizing each score using mean and std of baseline.
    for col in score_cols:
        base_mean = df_baseline_score_total[col].mean()
        base_std = df_baseline_score_total[col].std()
        standardized_baseline[f"standardized {col}"] = (df_baseline_score_total[col] - base_mean) / base_std
        standardized_mid[f"standardized {col}"] = (df_midscore_total[col.replace('Baseline ', "")] - base_mean) / base_std
        standardized_main[f"standardized {col}"] = (df_main_score_total[col.replace('Baseline ', "")] - base_mean) / base_std

    # Anxiety Calculation
    df_main_score_total["final_anxiety"] = standardized_main.mean(axis=1)
    df_main_score_total["mid_anxiety"] = standardized_mid.mean(axis=1)
    df_baseline_score_total["t1_anxiety"] = standardized_baseline.mean(axis=1)
    df_score_total = df_baseline_score_total[["ID", "t1_anxiety"]].merge(df_main_score_total[["ID", "mid_anxiety", "final_anxiety"]], on="ID", how="inner")
    
    df_long = df_score_total.melt(
        id_vars="ID",
        value_vars=["t1_anxiety", "mid_anxiety", "final_anxiety"],
        var_name="time",
        value_name="anxiety"
    )
    # time 순서 정렬
    time_map = {"t1_anxiety":0, "mid_anxiety":0.5, "final_anxiety":1}
    df_long["time"] = df_long["time"].map(time_map)


    # Mixed-Model
    model = smf.mixedlm("anxiety ~ time", df_long, groups=df_long["ID"], re_formula="~time")
    result = model.fit()
    random_slopes = pd.DataFrame([[id, v.time]for id, v in result.random_effects.items()], columns=['ID', 'y'])
    # Linear Model
    # random_slopes = list()
    # for subj, grp in df_long.groupby("ID"):
    #     X = grp["time"].values.reshape(-1, 1)
    #     y = grp["anxiety"].values
    #     model = LinearRegression().fit(X, y)
    #     slope = model.coef_[0]  # 선형회귀의 기울기
    #     random_slopes.append({"ID": subj, "y": slope})

    # random_slopes = pd.DataFrame(random_slopes)
    
    df_score_total = df_score_total.merge(random_slopes, on='ID', how='inner')

    # df_score_total['y'] = df_score_total['final_anxiety'] - df_score_total['t1_anxiety']
    # Final dataset merge
    dataset_total = pd.concat([df_baseline_score_hmp, df_baseline_score_wait], axis=0, ignore_index=True)
    dataset_total = dataset_total.merge(df_score_total[["ID", "y", "t1_anxiety"]], on="ID", how="inner")

    # Delete score_cols so that t1_anxiety and y exists at dataset_total.
    dataset_total = dataset_total.drop(columns=score_cols)

    dataset_total.columns = dataset_total.columns.str.strip()

    return dataset_total