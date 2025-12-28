import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from anndata import AnnData

def marker_correlation_plots(
    adata: AnnData,
    markers: list = None,
    use_clustermap: bool = False,
    figsize: tuple = (10, 8),
    cmap: str = "coolwarm"
):
    all_pos_cols = [c for c in adata.obs.columns if c.endswith("_pos")]

    pos_cols = all_pos_cols if markers is None else [
        f"{m}_pos" for m in markers if f"{m}_pos" in adata.obs
    ]

    if len(pos_cols) < 2:
        raise ValueError("Need at least two valid markers.")

    # Correlation matrix
    df = adata.obs[pos_cols].astype(float)
    corr_df = df.corr()

    # Standard
    if not use_clustermap:
        plt.figure(figsize=figsize)
        sns.heatmap(
            corr_df,
            cmap=cmap,
            vmin=-1, vmax=1,
            square=True
        )
        plt.title("Correlation Heatmap")
        plt.tight_layout()
        plt.show()

    # Clustermap (I think it uses UPGMA)
    else:
        g = sns.clustermap(
            corr_df,
            cmap=cmap,
            vmin=-1, vmax=1,
            figsize=figsize
        )
        g.fig.suptitle("Correlation Clustermap")
        plt.show()

    return corr_df
