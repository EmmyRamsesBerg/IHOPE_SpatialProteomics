import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
import os
import seaborn as sns

def spatial_celltype_plot(
    adata,
    celltype_cols,
    x_coord="x",
    y_coord="y",
    min_cells=0,
    figsize=(8, 8),
    alpha=0.6,
    size=10,
    palette=None,
    title: str = "",
):
    """
    Plot spatial positions of cells colored by cell type.

    Only includes cell types with at least `min_cells`.

    Parameters
    ----------
    adata : AnnData
    celltype_cols : list of str
        Boolean columns defining cell types (e.g. subtype_* or state_*)
    x_coord, y_coord : str
        Columns in adata.obs for spatial coordinates
    min_cells : int
        Minimum number of cells required for a cell type to be included
    palette : dict or None
        Optional {cell_type: color} mapping. If None, will generate colors automatically.
    title: Str or None
        Optional plot title
    """
    # Filter cell types by min_cells
    valid_cts = [ct for ct in celltype_cols if adata.obs[ct].sum() >= min_cells]
    if not valid_cts:
        raise ValueError("No cell types meet min_cells threshold.")

    # Count cells for each valid cell type
    ct_counts = {ct: adata.obs[ct].sum() for ct in valid_cts}

    # Ensure unassigned types are plotted first
    unassigned = [ct for ct in valid_cts if ct.endswith("unassigned")]
    assigned = [ct for ct in valid_cts if not ct.endswith("unassigned")]
    assigned = sorted(assigned, key=lambda ct: ct_counts[ct], reverse=True) #Order from most abundant
    valid_cts = unassigned + assigned

    # Assign colors
    if palette is None:
        colors = sns.color_palette("tab20", len(valid_cts))
        palette = dict(zip(valid_cts, colors))

        # Override unassigned colors
        for ct in valid_cts:
            if ct.endswith("unassigned"):
                palette[ct] = (0.85, 0.85, 0.85)  # light grey

    plt.figure(figsize=figsize, facecolor="white")
    ax = plt.gca()
    ax.set_facecolor("white")
    ax.grid(False)
    ax.axis("off")

    # Plot all cells faint grey once (background)
    plt.scatter(
        adata.obs[x_coord],
        adata.obs[y_coord],
        c="lightgrey",
        s=size * 0.6,
        alpha=0.3,
        linewidths=0
    )

    # Overlay each cell type in color
    for ct in valid_cts:
        mask = adata.obs[ct].astype(bool)
        plt.scatter(
            adata.obs.loc[mask, x_coord],
            adata.obs.loc[mask, y_coord],
            c=[palette[ct]],
            s=size,
            alpha=alpha,
            label=ct,
            linewidths=0,
        )

    plt.gca().invert_yaxis()  # match image orientation if needed
    plt.legend(markerscale=2, bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.title(title)
    plt.show()

    return palette