import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

__all__ = [
    "display_PAI_interaction",
    "display_table",
    "display_topk_ale",
]
def display_PAI_interaction(df, ax):
    model = smf.ols('y ~ t1_anxiety + PAI * C(Received)', data=df).fit()

    PAI_vals = np.linspace(df['PAI'].min(), df['PAI'].max(), 100)
    received_levels = ['Mindful', 'Control']

    pred_df = pd.DataFrame({
        'PAI': np.tile(PAI_vals, 2),
        'Received': np.repeat(received_levels, len(PAI_vals)),
    })
    pred_df['Received'] = pd.Categorical(pred_df['Received'], categories=received_levels)
    pred_df['t1_anxiety'] = df['t1_anxiety'].mean()


    pred_result = model.get_prediction(pred_df)
    pred_summary = pred_result.summary_frame(alpha=0.05)  # 95% CI


    pred_df['y_hat'] = pred_summary['mean']
    pred_df['ci_lower'] = pred_summary['mean_ci_lower']
    pred_df['ci_upper'] = pred_summary['mean_ci_upper']
    # ymax = max(pred_df['ci_upper'])
    # ymin = min(pred_df['ci_lower'])
    # dist = (ymax - ymin)*0.3



    colors = {"Mindful": "orange", "Control": "blue"}

    for group in received_levels:
        subset = pred_df[pred_df['Received'] == group]
        ax.plot(subset['PAI'], subset['y_hat'], label=group, color=colors[group])
        ax.fill_between(subset['PAI'], subset['ci_lower'], subset['ci_upper'], alpha=0.2, color=colors[group])


    for group in received_levels:
        subset_real = df[df["Received"] == group]
        ax.scatter(
            subset_real["PAI"],
            subset_real["y"],
            color=colors[group],
            alpha=0.6,
            # label=f"{group} (data)",
            edgecolors="k", linewidth=0.2
        )


    ax.set_xlabel("PAI")
    ax.set_ylabel("Change in Anxiety(slope)")
    ax.legend(title="Group")
    # ax.set_ylim(ymin-dist, ymax+dist)
    ax.grid(True)
    # ax.tight_layout()



def display_table(df, title, ax):
    
    ax.axis("off")

    ax.set_title(title, fontsize=24, weight="bold", pad=50)
    
    table = ax.table(cellText=df.values,
                     colLabels=df.columns,
                     cellLoc="center",
                     loc="center")
    
    table.auto_set_font_size(False)
    table.set_fontsize(15)
    table.scale(1.2, 1.0)



def display_topk_ale(results, k=12, cols=4, suptitle='', pdf=None):


    order = results["rank_std"]
        
    topk = order[:k]
    assert k % cols == 0
    rows = int(np.ceil(k / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(6*cols, 6*rows), squeeze=True, sharey=True)
    fig.suptitle(suptitle + "\nTop ALE (std)", fontsize=24, weight="bold")
    axes = axes.ravel()
    min_ale = results["ale_curves"][topk[0]].min()
    max_ale = results["ale_curves"][topk[0]].max()
    for idx in range(k):
        i = topk[idx]
        min_ale = min(min_ale, results["ale_curves"][i].min())
        max_ale = max(max_ale, results["ale_curves"][i].max())

    for idx, ax in enumerate(axes[:k]):
        j = topk[idx]
        xs = results["x_axes"][j]
        ys = results["ale_curves"][j]
        feature_deciles = results['feature_deciles'][j]


        min_x = xs.min()
        max_x = xs.max()
        
        fname = results["feature_names"][j]
        ax.plot(xs, ys)
        ax.scatter(xs, ys)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{fname}")
        ax.set_xlabel("feature value")
        ax.set_ylabel("ALE")
        ax.grid(True, alpha=0.3)
        
        line_ratio = 0.1
        ax.vlines(feature_deciles, ymin=min_ale, ymax= min_ale + line_ratio * (max_ale - min_ale), color='black')
        ax.hlines(0, xmin=min_x, xmax=max_x, ls='--')
        if idx == 0:
            ax.set_ylabel('ale values')
            ax.set_xlabel('feature values')


    for ax in axes[k:]:
        ax.axis("off")

    #fig.suptitle(title, fontsize=14)
    if pdf is not None:
        pdf.savefig(fig)
        plt.close()
