import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from anndata import AnnData
import numpy as np

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


def plot_marker_pair_by_leiden(
        adata,
        marker_x,
        marker_y,
        cluster_key='leiden',
        max_points=50000,
        point_size=4,
        alpha=0.6,
        cmap='tab20',
        random_state=0
):
    """
    Simple 2D scatter of two markers colored by Leiden clusters.

    Parameters
    ----------
    adata : AnnData
        AnnData object containing marker data.
    marker_x, marker_y : str
        Names of markers in adata.var_names.
    cluster_key : str
        Column in adata.obs containing cluster labels.
    max_points : int or None
        Maximum number of points to plot (downsampling for visualization only).
    point_size : float
        Marker size.
    alpha : float
        Point transparency.
    cmap : str
        Matplotlib categorical colormap.
    random_state : int
        Random seed for reproducible downsampling.
    """

    # --- Extract marker values ---
    x = adata[:, marker_x].X
    y = adata[:, marker_y].X

    if hasattr(x, "toarray"):
        x = x.toarray().ravel()
        y = y.toarray().ravel()

    # --- Extract cluster labels ---
    clusters = adata.obs[cluster_key].astype(int).values
    unique_clusters = np.unique(clusters)

    # --- Downsample for visualization ---
    if max_points is not None and len(x) > max_points:
        rng = np.random.default_rng(random_state)
        keep = rng.choice(len(x), size=max_points, replace=False)
        x = x[keep]
        y = y[keep]
        clusters = clusters[keep]

    # --- Colormap for discrete clusters ---
    cmap_obj = plt.get_cmap(cmap, len(unique_clusters))
    cluster_to_color = {cl: cmap_obj(i) for i, cl in enumerate(unique_clusters)}

    # --- Plot ---
    plt.figure(figsize=(6, 6))
    for cl in unique_clusters:
        idx = clusters == cl
        plt.scatter(
            x[idx],
            y[idx],
            s=point_size,
            alpha=alpha,
            color=cluster_to_color[cl],
            label=cl
        )

    plt.xlabel(marker_x)
    plt.ylabel(marker_y)
    plt.title(f'{marker_x} vs {marker_y} colored by {cluster_key}')
    plt.legend(title=cluster_key, bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
    plt.tight_layout()
    plt.show()


def plot_cluster_marker_heatmap(
    adata,
    cluster_key='leiden',
    use_raw_X=True,
    agg='mean',
    scale='row',          # 'row', 'col', or None
    cmap='coolwarm',
    figsize=(15, 10),
    cluster_rows=True,
    cluster_cols=True,
    show_values=False,
    value_fmt=".2f",
    title=None,
    return_data=False
):
    """
    Plot a cluster × marker heatmap summarizing marker expression per cluster.

    Parameters
    ----------
    adata : AnnData
        AnnData object.
    cluster_key : str
        Column in adata.obs with cluster labels.
    use_raw_X : bool
        Whether to use adata.X directly (assumed already processed).
    agg : {'mean', 'median'}
        Aggregation method per cluster.
    scale : {'row', 'col', None}
        Standard scaling for visualization (row = per cluster).
    cmap : str
        Colormap for heatmap.
    figsize : tuple
        Figure size.
    cluster_rows : bool
        Whether to hierarchically cluster clusters.
    cluster_cols : bool
        Whether to hierarchically cluster markers.
    show_values : bool
        Whether to annotate cells with numeric values (not recommended for many markers).
    value_fmt : str
        Format for numeric annotations.
    title : str or None
        Figure title.
    return_data : bool
        If True, return the reordered DataFrame shown in the heatmap.

    Returns
    -------
    pd.DataFrame or None
        Reordered (and scaled) data if return_data=True.
    """

    if cluster_key not in adata.obs:
        raise ValueError(f"'{cluster_key}' not found in adata.obs")

    # --- Extract expression matrix ---
    X = adata.X if use_raw_X else adata.layers[use_raw_X]
    expr_df = pd.DataFrame(
        np.asarray(X),
        columns=adata.var_names,
        index=adata.obs_names
    )

    # --- Add cluster labels ---
    clusters = adata.obs[cluster_key].values
    expr_df[cluster_key] = clusters

    # --- Aggregate per cluster ---
    if agg == 'mean':
        summary_df = expr_df.groupby(cluster_key).mean()
    elif agg == 'median':
        summary_df = expr_df.groupby(cluster_key).median()
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    # --- Sort clusters numerically if possible ---
    try:
        summary_df = summary_df.sort_index(key=lambda x: pd.to_numeric(x))
    except Exception:
        pass

    # --- Scaling choice ---
    if scale == 'row':
        standard_scale = 1
    elif scale == 'col':
        standard_scale = 0
    else:
        standard_scale = None

    # --- Plot ---
    g = sns.clustermap(
        summary_df,
        cmap=cmap,
        figsize=figsize,
        standard_scale=standard_scale,
        linewidths=0.5,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        annot=show_values,
        fmt=value_fmt,
        dendrogram_ratio=(0.15, 0.15)
    )

    if title is None:
        title = f'{agg.capitalize()} marker expression per {cluster_key} cluster'

    g.fig.suptitle(title, y=1.02)
    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90)
    plt.show()

    if return_data:
        # Extract reordered data as shown
        row_order = g.dendrogram_row.reordered_ind if cluster_rows else range(summary_df.shape[0])
        col_order = g.dendrogram_col.reordered_ind if cluster_cols else range(summary_df.shape[1])

        reordered_df = summary_df.iloc[row_order, col_order]
        return reordered_df

    return None

# This one was made to match the IHOPE version
def clustered_marker_heatmap(
    adata,
    celltype_cols,
    marker_pos_suffix="_pos",
    min_cells=15,
    base_fig_width=8,
    fig_height=10,
    cmap="viridis",
):
    """
    Clustered heatmap of marker positivity across cell types.

    - Supports overlapping cell-type definitions (type / state / subtype)
    - No averaging across definitions
    - One heatmap row per boolean column
    """

    # Marker columns
    marker_cols = [c for c in adata.obs.columns if c.endswith(marker_pos_suffix)]
    n_markers = len(marker_cols)
    fig_width = max(base_fig_width, n_markers * 0.4)

    records = []

    for ct in celltype_cols:
        if ct not in adata.obs:
            continue

        mask = adata.obs[ct].astype(bool)
        n = int(mask.sum())

        if n < min_cells:
            continue

        # infer level from prefix
        if ct.startswith("type_"):
            level = "type"
            name = ct.replace("type_", "")
        elif ct.startswith("state_"):
            level = "state"
            name = ct.replace("state_", "")
        elif ct.startswith("subtype_"):
            level = "subtype"
            name = ct.replace("subtype_", "")
        else:
            level = "other"
            name = ct

        row_id = f"{level}::{name}"

        for marker in marker_cols:
            frac_pos = adata.obs.loc[mask, marker].mean()
            records.append({
                "row_id": row_id,
                "marker": marker.replace(marker_pos_suffix, ""),
                "fraction_positive": frac_pos,
            })

    df = pd.DataFrame(records)

    if df.empty:
        raise ValueError("No cell types meet the min_cells threshold.")

    # Pivot is now safe because row_id is unique
    matrix = (
        df.pivot(index="row_id", columns="marker", values="fraction_positive")
        .fillna(0.0)
    )

    sns.clustermap(
        matrix,
        cmap=cmap,
        figsize=(fig_width, fig_height),
        metric="euclidean",
        method="average",
        cbar_kws={"label": "Fraction marker-positive"},
    )

    plt.show()

    return matrix