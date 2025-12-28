import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from anndata import AnnData
import os

#TODO: add a color mapping here so that cell types always have consistent color codes.

def plot_spatial_marker(adata, marker,
                        thresholded=True,
                        title_prefix="",
                        color_hi="red",
                        color_lo="lightgrey",
                        size=2,
                        invert_y=True):

    coords = adata.obsm['spatial']
    if coords.shape[1] != 2:
        raise ValueError(f"Expected spatial coordinates with shape (n_cells, 2), got {coords.shape}")

    x = coords[:, 0]
    y = coords[:, 1]

    if thresholded:
        colname = f"{marker}_pos"
        if colname not in adata.obs.columns:
            raise ValueError(f"{colname} not found in adata.obs")
        colors = adata.obs[colname].map({True: color_hi, False: color_lo})
    else:
        colors = adata.obs[marker]

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c=colors, s=size, alpha=0.7)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{title_prefix}{marker} spatial mapping")
    if invert_y:
        plt.gca().invert_yaxis()
    plt.show()

def plot_spatial_cell_types(
    adata: AnnData,
    level: str = "type",  #branch/type/subtype
    size: float = 5,
    alpha: float = 0.6,
    invert_y: bool = True,
    title_prefix: str = "",
    save: bool = False,
    save_path: str = None
):
    if level not in adata.obs.columns:
        raise ValueError(f"{level} not found in adata.obs. Choose one of: {list(adata.obs.columns)}")

    coords = adata.obsm['spatial']
    if coords.shape[1] != 2:
        raise ValueError("Expected spatial coordinates with shape (n_cells, 2).")

    x = coords[:, 0]
    y = coords[:, 1]
    cell_vals = adata.obs[level].astype(str)
    unique_vals = cell_vals.unique()

    # Choose color palette
    n_colors = len(unique_vals)
    if n_colors <= 20:
        palette = sns.color_palette("tab20", n_colors=n_colors)
    else:
        palette = sns.color_palette("hls", n_colors=n_colors)
    color_map = dict(zip(unique_vals, palette))
    colors = cell_vals.map(color_map)

    # Plot
    fig, ax = plt.subplots(figsize=(6,6))

    # TODO Set background ???
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    plt.grid(False)

    ax.scatter(x, y, c=colors, s=size, alpha=alpha)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(f"{title_prefix}Spatial Mapping ({level})")
    if invert_y:
        ax.invert_yaxis()

    # Legend
    handles = [plt.Line2D([0],[0], marker='o', color='w', markerfacecolor=color_map[val], markersize=6)
               for val in unique_vals]
    ax.legend(handles, unique_vals, bbox_to_anchor=(1.05,1), loc='upper left', fontsize=8)

    plt.tight_layout()

    # TODO fix save path
    if save:
        # Absolute path to project root (folder containing this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))  # go up one level

        # Final results/figures directory
        fig_dir = os.path.join(project_root, "results", "figures")
        os.makedirs(fig_dir, exist_ok=True)

        # Default filename if none provided
        if save_path is None:
            save_path = os.path.join(fig_dir, f"spatial_celltype_{level}.png")
        else:
            save_path = os.path.join(fig_dir, os.path.basename(save_path))

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
    return fig

# To select which cell type to plot and in what color
def plot_spatial_subset(
    adata: AnnData,
    level: str,         # branch/type/subtype
    include: list,
    colors: dict = None,
    size: float = 4,
    alpha: float = 0.8,
    background_color: str = "lightgrey",
    background_alpha: float = 0.2,
    invert_y: bool = False,
    title_prefix: str = "",
    save: bool = False,
    save_path: str = None,
):

    if level not in adata.obs.columns:
        raise ValueError(f"{level} not found in adata.obs")

    coords = adata.obsm["spatial"]
    x = coords[:, 0]
    y = coords[:, 1]

    vals = adata.obs[level].astype(str)

    bg_mask = ~vals.isin(include)
    fig, ax = plt.subplots(figsize=(6, 6)) #was replaced, check it works

    # White backgrounds
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Remove gridlines
    ax.grid(False)

    plt.scatter(
        x[bg_mask], y[bg_mask],
        c=background_color,
        s=size,
        alpha=background_alpha
    )

    for label in include:
        mask = vals == label

        color = None
        if colors is not None and label in colors:
            color = colors[label]

        plt.scatter(
            x[mask], y[mask],
            c=color,
            s=size,
            alpha=alpha,
            label=label
        )

    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{title_prefix}Spatial mapping: {include}")

    if invert_y:
        plt.gca().invert_yaxis()

    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')

    if save:
        # Absolute path to project root (folder containing this script)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.abspath(os.path.join(script_dir, ".."))  # go up one level

        # Final results/figures directory
        fig_dir = os.path.join(project_root, "results", "figures")
        os.makedirs(fig_dir, exist_ok=True)

        # Default filename if none provided
        if save_path is None:
            save_path = os.path.join(fig_dir, f"spatial_subset_{level}.png")
        else:
            save_path = os.path.join(fig_dir, os.path.basename(save_path))

        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    plt.show()
