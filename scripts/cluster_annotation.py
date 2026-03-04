import pandas as pd


def summarize_single_cluster(adata, cluster_id, top_n=10, marker_subset=None):
    """
    Summarize top marker expression for a single cluster.

    Parameters
    ----------
    adata : AnnData
        Annotated data matrix.
    cluster_id : str
        Cluster ID from adata.obs['leiden'] to summarize.
    top_n : int
        Number of top markers to display.
    marker_subset : list of str, optional
        Subset of markers to consider. If None, all markers in adata.var_names are used.

    Returns
    -------
    pd.Series
        Top markers for the specified cluster.
    """
    markers = marker_subset if marker_subset is not None else adata.var_names
    expr_df = adata[:, markers].to_df()
    expr_df['leiden'] = adata.obs['leiden'].astype(str)

    # Select cells in the specified cluster
    if cluster_id not in expr_df['leiden'].unique():
        raise ValueError(f"Cluster {cluster_id} not found in adata.obs['leiden']")

    cluster_data = expr_df[expr_df['leiden'] == cluster_id].drop(columns='leiden')
    mean_markers = cluster_data.mean().sort_values(ascending=False)
    top_markers = mean_markers.head(top_n)

    return top_markers
