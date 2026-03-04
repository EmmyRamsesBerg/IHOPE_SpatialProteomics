import numpy as np
import matplotlib.pyplot as plt
from anndata import AnnData


def plot_marker_axes(
    adata: AnnData,
    x_marker: str,
    y_marker: str,
    base_mask: str,
    show_thresholds: bool = True,
    size: float = 3,
    alpha: float = 0.3,
    title: str | None = None,
):
    """
    Simple marker-vs-marker scatter plot using continuous intensities.

    - Intensities taken from adata.X
    - Marker names resolved via adata.var_names (arcsinh_cf5.0_<marker>)
    - Subsetting via boolean mask in adata.obs
    - Threshold lines from adata.uns["intensity_thresholds"]
    """

    # Resolve variable names
    x_var = f"arcsinh_cf5.0_{x_marker}"
    y_var = f"arcsinh_cf5.0_{y_marker}"

    if x_var not in adata.var_names or y_var not in adata.var_names:
        raise ValueError("Marker not found in adata.var_names")

    if base_mask not in adata.obs:
        raise ValueError("Base mask not found in adata.obs")

    # Get indices
    x_idx = adata.var_names.get_loc(x_var)
    y_idx = adata.var_names.get_loc(y_var)

    mask = adata.obs[base_mask].values

    x = adata.X[mask, x_idx]
    y = adata.X[mask, y_idx]

    # Plot
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, s=size, alpha=alpha)

    # Thresholds
    if show_thresholds:
        thr = adata.uns.get("intensity_thresholds", {})

        if x_marker in thr:
            plt.axvline(thr[x_marker]["low_thr"], linestyle="--")
            plt.axvline(thr[x_marker]["high_thr"], linestyle="--")

        if y_marker in thr:
            plt.axhline(thr[y_marker]["low_thr"], linestyle="--")
            plt.axhline(thr[y_marker]["high_thr"], linestyle="--")

    plt.xlabel(x_marker)
    plt.ylabel(y_marker)
    if title is not None:
        plt.title(title)

    plt.tight_layout()
    plt.show()

