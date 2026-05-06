import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from anndata import AnnData
import numpy as np

# Made to match the IHOPE version
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