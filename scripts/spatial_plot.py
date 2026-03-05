import matplotlib.pyplot as plt
import numpy as np
from anndata import AnnData
import os
import seaborn as sns

def plot_spatial(
    adata: AnnData,
    celltypes: list,
    size: float = 5,
    alpha: float = 0.7,
    colors: dict = None,
    exclude_overlaps: bool = False,
    background_color: str = "lightgrey",
    background_alpha: float = 0.2,
    invert_y: bool = True,
    title_prefix: str = "",
    save: bool = False,
    save_path: str = None
):
    """
    Plot spatial locations of multiple boolean cell types.

    Parameters
    ----------
    adata : AnnData
        AnnData object with boolean columns for each cell type
    celltypes : list of str
        Column names in adata.obs representing cell types (boolean)
    size : float
        Point size
    alpha : float
        Transparency
    colors : dict
        Optional dictionary mapping cell type names to colors
    exclude_overlaps : bool
        If True, cells positive for more than one selected type are not plotted
    background_color : str
        Color for non-selected cells
    background_alpha : float
        Transparency for background cells
    invert_y : bool
        Whether to invert y-axis
    title_prefix : str
        Prefix for the plot title
    save : bool
        Whether to save figure
    save_path : str
        Full path or filename to save
    """

    coords = adata.obsm['spatial']
    if coords.shape[1] != 2:
        raise ValueError("Expected spatial coordinates with shape (n_cells, 2).")

    x = coords[:, 0]
    y = coords[:, 1]

    # Check all celltypes exist
    for ct in celltypes:
        if ct not in adata.obs.columns:
            raise ValueError(f"{ct} not found in adata.obs.columns")

    # Build a mask for overlapping cells if requested
    combined_mask = adata.obs[celltypes].sum(axis=1)
    if exclude_overlaps:
        masks = {ct: (adata.obs[ct] & (combined_mask == 1)) for ct in celltypes}
    else:
        masks = {ct: adata.obs[ct] for ct in celltypes}

    # Background: cells not in any of the selected types
    bg_mask = combined_mask == 0

    fig, ax = plt.subplots(figsize=(6,6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(False)

    # Plot background cells first
    plt.scatter(
        x[bg_mask], y[bg_mask],
        c=background_color,
        s=size,
        alpha=background_alpha
    )

    # Plot each cell type
    for ct in celltypes:
        mask = masks[ct]
        color = colors.get(ct) if colors else None
        plt.scatter(
            x[mask], y[mask],
            c=color,
            s=size,
            alpha=alpha,
            label=ct
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{title_prefix}Spatial Plot")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
    plt.tight_layout()

    # Save if requested
    if save:
        if save_path is None:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.abspath(os.path.join(script_dir, ".."))
            fig_dir = os.path.join(project_root, "results", "figures")
            os.makedirs(fig_dir, exist_ok=True)
            save_path = os.path.join(fig_dir, "spatial_plot.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()

# This one was made to match the IHOPE version
def spatial_celltype_plot(
    adata,
    celltype_cols,
    x_coord="x",
    y_coord="y",
    min_cells=15,
    figsize=(8, 8),
    alpha=0.6,
    size=10,
    palette=None,
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
    """
    # Filter cell types by min_cells
    valid_cts = [ct for ct in celltype_cols if adata.obs[ct].sum() >= min_cells]
    if not valid_cts:
        raise ValueError("No cell types meet min_cells threshold.")

    # Ensure unassigned types are plotted first
    unassigned = [ct for ct in valid_cts if ct.endswith("unassigned")]
    assigned = [ct for ct in valid_cts if not ct.endswith("unassigned")]
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
    plt.show()

    return palette