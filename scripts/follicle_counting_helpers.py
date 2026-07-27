import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN

# Force white-background plots regardless of the local matplotlib/OS theme.
# Applies to every figure drawn after this module is imported, including
# the barplots/boxplots built directly in the notebook.
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.edgecolor": "black",
})


def detect_follicles(
        adata,
        eps=55,
        min_samples=20,
        min_cluster_size=50,
):
    mask = (
            (adata.obs["type_B"] == True) &
            (adata.obs["B_follicle"] == True)
    )

    subset_idx = np.where(mask)[0]

    if len(subset_idx) == 0:
        adata.obs["follicle_cluster"] = -1
        return adata, 0

    coords = adata.obsm["spatial"][subset_idx]

    clustering = DBSCAN(
        eps=eps,
        min_samples=min_samples
    ).fit(coords)

    labels = clustering.labels_

    valid_labels = []
    for label in np.unique(labels):
        if label == -1:
            continue

        n_cells = np.sum(labels == label)
        if n_cells >= min_cluster_size:
            valid_labels.append(label)

    final_labels = np.full(len(adata), -1)

    for i, idx in enumerate(subset_idx):
        if labels[i] in valid_labels:
            final_labels[idx] = labels[i]

    adata.obs["follicle_cluster"] = final_labels

    return adata, len(valid_labels)


def _draw_follicle_scatter(adata):
    """
    Shared base plot: grey background of all cells, colored overlay of
    valid follicle clusters, on a white figure/axes background regardless
    of the local matplotlib theme. Returns coords and the follicle boolean
    mask so callers can add labels, legends, etc on top.
    """
    coords = adata.obsm["spatial"]

    fig = plt.figure(figsize=(10, 10), facecolor="white")
    ax = fig.gca()
    ax.set_facecolor("white")
    plt.grid(False)
    ax.set_aspect("equal", adjustable="box")

    sns.scatterplot(
        x=coords[:, 0],
        y=coords[:, 1],
        color="lightgrey",
        s=3,
        alpha=0.4
    )

    follicle_mask = adata.obs["follicle_cluster"] >= 0

    if follicle_mask.sum() > 0:
        sns.scatterplot(
            x=coords[follicle_mask, 0],
            y=coords[follicle_mask, 1],
            hue=adata.obs.loc[follicle_mask, "follicle_cluster"].astype(str),
            palette="tab20",
            s=8,
            linewidth=0
        )

    return coords, follicle_mask


def plot_follicles(adata, sample_name, out_dir):
    """
    Clean, final version of the follicle plot (no cluster ID labels).
    Saved as {sample_name}_follicles.png in out_dir.
    """
    _draw_follicle_scatter(adata)

    plt.title(sample_name, color="black")
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend([], [], frameon=False)
    plt.tight_layout()

    plt.savefig(
        out_dir / f"{sample_name}_follicles.png",
        dpi=300,
        facecolor="white"
    )

    plt.show()


def plot_follicles_numbered(adata, sample_name, out_dir):
    """
    Diagnostic version of the follicle plot with each cluster's DBSCAN
    label printed at its centroid, so individual clusters can be identified
    for manual exclusion. Saved as {sample_name}_follicles_numbered.png
    in out_dir.
    """
    coords, follicle_mask = _draw_follicle_scatter(adata)

    if follicle_mask.sum() > 0:
        labels = adata.obs.loc[follicle_mask, "follicle_cluster"]
        follicle_coords = coords[follicle_mask]

        for label in sorted(labels.unique()):
            label_mask = (labels == label).to_numpy()
            centroid = follicle_coords[label_mask].mean(axis=0)

            plt.text(
                centroid[0],
                centroid[1],
                str(label),
                fontsize=9,
                fontweight="bold",
                color="white",
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6, linewidth=0)
            )

    plt.title(f"{sample_name} (cluster IDs)", color="black")
    plt.gca().invert_yaxis()
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.legend([], [], frameon=False)
    plt.tight_layout()

    plt.savefig(
        out_dir / f"{sample_name}_follicles_numbered.png",
        dpi=300,
        facecolor="white"
    )

    plt.show()


def apply_cluster_exclusions(adata, excluded_labels):
    """
    Manually drop specified DBSCAN cluster labels from the follicle_cluster
    assignment (set back to -1), for clusters identified as false positives
    (e.g. capsule/edge artifacts) during visual review of the numbered plots.

    Parameters
    ----------
    adata : AnnData
        Must already have a 'follicle_cluster' column in .obs.
    excluded_labels : list of int
        Cluster labels to remove for this sample.

    Returns
    -------
    adata : AnnData
        Updated in place, also returned for convenience.
    n_removed : int
        Number of cells whose cluster assignment was cleared.
    n_follicles : int
        Number of valid follicle clusters remaining after exclusion.
    """
    labels = adata.obs["follicle_cluster"].copy()

    excluded_mask = labels.isin(excluded_labels)
    n_removed = int(excluded_mask.sum())

    labels[excluded_mask] = -1
    adata.obs["follicle_cluster"] = labels

    n_follicles = int(labels[labels >= 0].nunique())

    return adata, n_removed, n_follicles


def normalize_follicle_counts(df, count_col="total_immune_cells", per_n_cells=10000):
    """
    Add a column with follicle counts normalized to a fixed number of cells.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'n_follicles' and the column named by count_col.
    count_col : str
        Column to normalize against (default 'total_immune_cells'). Use
        'total_cells' to normalize to all cells instead.
    per_n_cells : int
        The cell count to normalize to (default 10000).

    Returns
    -------
    df : pandas.DataFrame
        Same dataframe with a new column named
        'follicles_per_{per_n_cells}_{count_col}'.
    """
    df = df.copy()
    label = count_col[len("total_"):] if count_col.startswith("total_") else count_col
    out_col = f"follicles_per_{per_n_cells}_{label}"
    df[out_col] = (
        df["n_follicles"] / df[count_col] * per_n_cells
    ).round(2)
    return df
