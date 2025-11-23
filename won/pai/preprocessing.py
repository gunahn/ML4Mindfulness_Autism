import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

__all__ = [
    "custom_mapping",
    "map_likert_responses",
    "reverse_scoring",
    "reverse_scoring_per_score",
    "effect_encoding",
    "centered_binary_encoding",
    "diagnosed_mapping",
    "remove_high_vif_with_correlation"
]

def custom_mapping(df: pd.DataFrame) -> pd.DataFrame:
    # STAI_state
    stai_state_strings = ["Not at all", "Somewhat", "Moderately so", "Very much so"]
    df = map_likert_responses(df, "STAI_", 1, 20, stai_state_strings)

    # STAI_trait
    stai_trait_strings = ["Almost never", "Sometimes", "Often", "Almost always"]
    df = map_likert_responses(df, "STAI_", 21, 40, stai_trait_strings)

    # PROMIS
    promis_strings = ["Never", "Rarely", "Sometimes", "Often", "Always"]
    df = map_likert_responses(df, "PROMIS_", 1, 8, promis_strings)

    # PSS
    pss_strings = ["Never", "Almost never", "Sometimes", "Fairly often", "Very often"]
    df = map_likert_responses(df, "PSS_", 1, 10, pss_strings)

    # PANAS
    panas_strings = ["Very slightly or not at all", "A little", "Moderately", "Quite a bit", "Extremely"]
    df = map_likert_responses(df, "PANAS_", 1, 20, panas_strings)

    # MAAS
    maas_strings = ["Almost always", "Very frequently", "Somewhat frequently", 
                    "Somewhat infrequently", "Very infrequently", "Almost never"]
    df = map_likert_responses(df, "MAAS_", 1, 15, maas_strings)

    # FFMQ
    ffmq_strings = ["Never or very rarely true", "Rarely true", "Sometimes true", "Often true", "Very often or always true"]
    df = map_likert_responses(df, "FFMQ_", 1, 15, ffmq_strings)

    return df


def map_likert_responses(df: pd.DataFrame, score_name: str, start_index: int, end_index: int, mapping_strings: list) -> pd.DataFrame:
    """Likert [0, 1] normalized mapping"""
    mapping_normalized = {
        s.lower(): v for s, v in zip(mapping_strings, np.linspace(0, 1, num=len(mapping_strings)))
    }
    cols = [f"{score_name}{i}" for i in range(start_index, end_index + 1)]

    for col in cols:
        df[col] = (
            df[col]
            .astype(str)
            .str.lower()
            .map(mapping_normalized)
        )
    return df


def reverse_scoring(df: pd.DataFrame) -> pd.DataFrame:
    # STAI
    df = reverse_scoring_per_score(df, "STAI_", [1, 2, 5, 8, 10, 11, 15, 16, 19, 20, 21, 23, 26, 27, 30, 33, 34, 36, 39])
    # PSS
    df = reverse_scoring_per_score(df, "PSS_", [4, 5, 7, 8])
    # FFMQ
    df = reverse_scoring_per_score(df, "FFMQ_", [3, 4, 7, 8, 9, 13, 14])
    return df


def reverse_scoring_per_score(df: pd.DataFrame, score_name: str, indices: list) -> pd.DataFrame:
    cols = [f"{score_name}{i}" for i in indices]
    for col in cols:
        df[col] = 1 - df[col]
    return df


def effect_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Effect coding (sum-to-zero)"""
    df[col] = df[col].astype("category")
    categories = df[col].cat.categories

    dummies = pd.get_dummies(df[col]).astype('float')
    # dummies = dummies - dummies.mean()  # 중심화

    # original columns are deleted.
    df = df.drop(columns=[col])
    df = pd.concat([df, dummies.iloc[:, :-1]], axis=1)
    return df


def centered_binary_encoding(df: pd.DataFrame, col: str) -> pd.DataFrame:
    labels = df[col].unique()
    if len(labels) != 2:
        raise ValueError("Column must have exactly two unique values.")
    # [-0.5, 0.5]
    mapping = {labels[0]: -0.5, labels[1]: 0.5}
    df[f"{col}_centered"] = df[col].map(mapping)
    df = df.drop(columns=[col])
    return df


def diagnosed_mapping(df: pd.DataFrame) -> pd.DataFrame:
    # Diagnosed_Anxiety
    all_anxieties = set(sum([re.split(r",\s*", str(x)) for x in df["Diagnosed_Anxiety"]], []))
    all_anxieties = [x.strip() for x in all_anxieties if x.strip()]

    for anx in all_anxieties:
        clean_anx = re.sub(r"[()]", "", anx.replace(" ", "_"))
        df[f"DA_{clean_anx}"] = df["Diagnosed_Anxiety"].apply(
            lambda x: int(anx in str(x))
        )

    # Diagnosed_Disorders
    all_disorders = set(sum([re.split(r",\s*", str(x)) for x in df["Diagnosed_Disorders"]], []))
    all_disorders = [x.strip() for x in all_disorders if x.strip()]

    for dis in all_disorders:
        clean_dis = re.sub(r"[()]", "", dis.replace(" ", "_"))
        df[f"DD_{clean_dis}"] = df["Diagnosed_Disorders"].apply(
            lambda x: int(dis in str(x))
        )

    df = df.drop(columns=["Diagnosed_Anxiety", "Diagnosed_Disorders", "DD_No", "DA_No"], errors="ignore")
    return df


def calculate_vif(X: pd.DataFrame) -> pd.DataFrame:
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i)
                    for i in range(X.shape[1])]
    return vif_data

def remove_high_vif_with_correlation(orig_X: pd.DataFrame, exception_cols: list, threshold: int = 5.0) -> pd.DataFrame:
    X = orig_X.copy()
    X = X.drop(columns=exception_cols)
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    removed_features = []

    while True:
        vif = calculate_vif(X_scaled)
        high_vif = vif[vif["VIF"] > threshold]

        if high_vif.empty:
            break

        # Feature candidates whose VIF is greater than threshold value
        candidates = high_vif["feature"].tolist()

        # pairwise correlation matrix
        corr_matrix = X_scaled[candidates].corr().abs()
        np.fill_diagonal(corr_matrix.values, 0)

        # Find the pair whose correlation is the higest.
        max_corr = corr_matrix.max().max()
        if max_corr == 0:
            # if the correlation value is low, the highest VIF feature is deleted.
            feature_to_drop = high_vif.sort_values("VIF", ascending=False)["feature"].iloc[0]
        else:
            # Extract every pairs whose correlation is the highest correlation value.
            feature_pairs = np.argwhere(corr_matrix.values == max_corr)
            candidates_to_drop = set()
            for i, j in feature_pairs:
                if i >= j: continue  # 중복 제거
                f1, f2 = corr_matrix.index[i], corr_matrix.columns[j]
                candidates_to_drop.update([f1, f2])
            
            # priority prefix to be deleted
            priority_prefixes = ("STAI_", "PROMIS_", "PSS_", "PANAS_", "MAAS_", "FFMQ_", "AQ")
            prioritized = [f for f in candidates_to_drop if f.startswith(priority_prefixes)]

            if prioritized:
                # Among prioritized features, delete the highest VIF feature
                vif_subset = vif[vif["feature"].isin(prioritized)]
            else:
                # delete the highest VIF feature
                vif_subset = vif[vif["feature"].isin(candidates_to_drop)]

            feature_to_drop = vif_subset.sort_values("VIF", ascending=False)["feature"].iloc[0]


        removed_features.append(feature_to_drop)
        X_scaled.drop(columns=[feature_to_drop], inplace=True)
        X.drop(columns=[feature_to_drop], inplace=True)

    X = pd.concat([X, orig_X[exception_cols]], axis=1)
    return X