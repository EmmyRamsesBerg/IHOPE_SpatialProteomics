"""
spatial_plotting.py

Spatial figures for the IHOPE project.

Two entry points
    plot_marker      spatial map of one marker, positivity or continuous intensity
    plot_celltypes   spatial map of boolean one hot cell type columns

Both accept either a path to an h5ad file or an already loaded AnnData, so a
large file can be read once and reused across several plots.

Both return the Matplotlib figure and axes so saving and further tweaking stay
in the caller. Saving is optional and uses an explicit output directory.

Data assumptions, matching the IHOPE h5ad files
    coordinates          adata.obsm['spatial'], falling back to obs x and y
    marker positivity    boolean columns named f"{marker}_pos" in adata.obs
    marker intensity     adata.X with var_names prefixed, for example z_CD20
    cell types           boolean one hot columns grouped by a level prefix,
                         for example type_B, intermediate_CD4_T, subtype_B_GC
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import seaborn as sns
import scanpy as sc
from anndata import AnnData

Source = Union[str, Path, AnnData]

# level prefixes that get stripped from legend labels
_LEVEL_PREFIXES = ("subtype_", "intermediate_", "type_", "state_", "branch_")


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _load_adata(source: Source) -> AnnData:
    """Return an AnnData from either an AnnData or a path to an h5ad."""
    if isinstance(source, AnnData):
        return source
    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"h5ad not found at {path}")
    return sc.read_h5ad(path)


def _get_coords(adata: AnnData, x_key: str = "x", y_key: str = "y"):
    """Return raw x and y arrays. Prefer obsm['spatial'], fall back to obs."""
    if "spatial" in adata.obsm:
        coords = np.asarray(adata.obsm["spatial"])
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(
                f"Expected obsm['spatial'] of shape (n_cells, 2), got {coords.shape}"
            )
        return coords[:, 0].astype(float), coords[:, 1].astype(float)
    if x_key in adata.obs and y_key in adata.obs:
        return (
            adata.obs[x_key].to_numpy(dtype=float),
            adata.obs[y_key].to_numpy(dtype=float),
        )
    raise ValueError(
        "No coordinates found. Expected obsm['spatial'] or "
        f"obs['{x_key}'] and obs['{y_key}']."
    )


def _get_marker_values(adata: AnnData, marker: str, marker_prefix: str = "z_") -> np.ndarray:
    """
    Return continuous values for a marker.

    Tries, in order, the prefixed name in var_names (for example z_CD20), the
    bare name in var_names, and the bare name in obs.
    """
    for name in (f"{marker_prefix}{marker}", marker):
        if name in adata.var_names:
            return np.asarray(adata[:, name].X, dtype=float).ravel()
    if marker in adata.obs.columns:
        return np.asarray(adata.obs[marker], dtype=float).ravel()
    raise ValueError(
        f"Could not find values for {marker}. Tried var_names "
        f"'{marker_prefix}{marker}' and '{marker}', and obs['{marker}']."
    )


def _clean_label(col: str) -> str:
    """Strip a known level prefix so legends read B_GC rather than subtype_B_GC."""
    for pre in _LEVEL_PREFIXES:
        if col.startswith(pre):
            return col[len(pre):]
    return col


def _resolve_celltype_columns(
    adata: AnnData,
    columns: Optional[Sequence[str]],
    level: Optional[str],
) -> list:
    """Return the list of boolean columns to plot from either columns or level."""
    if (columns is None) == (level is None):
        raise ValueError("Provide exactly one of columns or level.")
    if columns is not None:
        missing = [c for c in columns if c not in adata.obs.columns]
        if missing:
            raise ValueError(f"Columns not found in adata.obs: {missing}")
        return list(columns)
    cols = [
        c for c in adata.obs.columns
        if c.startswith(f"{level}_") and adata.obs[c].dtype == bool
    ]
    if not cols:
        raise ValueError(f"No boolean columns start with '{level}_'.")
    return cols


def _new_axes(figsize, facecolor="white"):
    """Figure and axes with equal aspect and no grid. facecolor sets both the
    figure and the axes background, defaulting to white so existing callers are
    unchanged."""
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(facecolor)
    ax.set_facecolor(facecolor)
    ax.set_aspect("equal")          # keep tissue geometry undistorted
    ax.grid(False)
    ax.axis("off")                  # hide coordinate axes and ticks
    return fig, ax


def _save_fig(fig, save_dir: Union[str, Path], filename: str, dpi: int) -> Path:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    out = save_dir / filename
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    print(f"Saved {out}")
    return out


# ----------------------------------------------------------------------
# marker plotting
# ----------------------------------------------------------------------

def plot_marker(
    source: Source,
    marker: str,
    base_name: str = "",
    thresholded: bool = True,
    pos_suffix: str = "_pos",
    marker_prefix: str = "z_",
    color_hi: str = "#d62728",
    color_lo: str = "#d9d9d9",
    cmap: str = "viridis",
    size: float = 4,
    alpha: float = 0.7,
    figsize=(7, 7),
    invert_y: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    dpi: int = 300,
):
    """
    Spatial map of a single marker.

    thresholded True  colours cells by the boolean column f"{marker}{pos_suffix}"
    thresholded False colours cells by continuous value from X, with a colorbar.
                      The value is looked up as f"{marker_prefix}{marker}".

    Returns
    -------
    (fig, ax)
    """
    adata = _load_adata(source)
    x, y = _get_coords(adata)
    fig, ax = _new_axes(figsize)

    if thresholded:
        col = f"{marker}{pos_suffix}"
        if col not in adata.obs.columns:
            raise ValueError(f"{col} not found in adata.obs")
        pos = adata.obs[col].to_numpy(dtype=bool)
        # negatives underneath, positives on top so positives stay visible
        ax.scatter(x[~pos], y[~pos], c=color_lo, s=size, alpha=alpha,
                   linewidths=0, rasterized=True)
        ax.scatter(x[pos], y[pos], c=color_hi, s=size, alpha=alpha,
                   linewidths=0, rasterized=True, label=f"{marker}+")
        ax.legend(markerscale=2, bbox_to_anchor=(1.02, 1), loc="upper left",
                  frameon=False)
        mode = "positivity"
    else:
        vals = _get_marker_values(adata, marker, marker_prefix)
        sctr = ax.scatter(x, y, c=vals, cmap=cmap, s=size, alpha=alpha,
                          linewidths=0, rasterized=True)
        # tighten the axes box to the data so the colorbar hugs the tissue
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(y.min(), y.max())
        cax = make_axes_locatable(ax).append_axes("right", size="4%", pad=0.05)
        fig.colorbar(sctr, cax=cax, label=marker)
        mode = "intensity"

    ax.set_title(f"{base_name} {marker} {mode}".strip())
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout()

    if save_dir is not None:
        fname = save_name or f"{base_name}_{marker}_{mode}.png".lstrip("_")
        _save_fig(fig, save_dir, fname, dpi)

    return fig, ax


# ----------------------------------------------------------------------
# cell type plotting
# ----------------------------------------------------------------------

def plot_celltypes(
    source: Source,
    columns: Optional[Sequence[str]] = None,
    level: Optional[str] = None,
    base_name: str = "",
    include: Optional[Sequence[str]] = None,
    palette: Optional[dict] = None,
    labels: Optional[dict] = None,
    drop_unassigned: bool = True,
    background_color: str = "#d9d9d9",
    background_alpha: float = 0.25,
    bg: str = "white",
    text_color: str = "black",
    size: float = 5,
    alpha: float = 0.7,
    figsize=(7, 7),
    invert_y: bool = True,
    min_cells: int = 1,
    save_dir: Optional[Union[str, Path]] = None,
    save_name: Optional[str] = None,
    dpi: int = 300,
):
    """
    Spatial map of boolean one hot cell type columns.

    Provide exactly one of
        columns   explicit list of boolean obs columns, for example
                  ["subtype_B_GC", "subtype_B_Plasmablast"]
        level     a prefix, one of type, intermediate, subtype. Every boolean
                  column at that level is collected.

    include            optional labels to keep, matched against either the raw
                       column name or the cleaned label (B_GC as well as
                       subtype_B_GC). Everything else falls to background.
    palette            optional colour map keyed by raw column name or cleaned
                       label. Pass the same dict across samples to keep colours
                       stable for comparison. Missing types get a fallback.
    labels             optional display-name map keyed by raw column name or
                       cleaned label, for example {"T": "T cells"}. Only the
                       legend text changes. Unset falls back to cleaned labels.
    bg                 figure and axes background colour. Default white. Set to
                       black for a fluorescence-style plot.
    text_color         colour for the title and legend text. Default black. Use
                       white on a dark background.
    drop_unassigned    columns ending in _unassigned are sent to background.
    min_cells          columns with fewer positive cells than this are dropped.

    Cells positive for none of the plotted columns are drawn as faint
    background. Where columns overlap, abundant types are drawn first so rare
    ones stay visible on top.

    Returns
    -------
    (fig, ax, palette)  palette keyed by raw column name, for reuse.
    """
    adata = _load_adata(source)
    x, y = _get_coords(adata)

    cols = _resolve_celltype_columns(adata, columns, level)

    if drop_unassigned:
        cols = [c for c in cols if not c.endswith(("_unassigned", "_unclassified"))]

    if include is not None:
        include = set(include)
        cols = [c for c in cols if c in include or _clean_label(c) in include]

    counts = {c: int(adata.obs[c].to_numpy(dtype=bool).sum()) for c in cols}
    cols = [c for c in cols if counts[c] >= min_cells]
    if not cols:
        raise ValueError("No cell type columns left to plot after filtering.")

    # abundant first, so the legend reads top down and rare cells draw last
    cols = sorted(cols, key=lambda c: counts[c], reverse=True)

    # resolve a colour for each column, honouring a supplied palette keyed by
    # either raw column name or cleaned label
    palette = dict(palette) if palette else {}
    resolved = {}
    needs_color = []
    for c in cols:
        if c in palette:
            resolved[c] = palette[c]
        elif _clean_label(c) in palette:
            resolved[c] = palette[_clean_label(c)]
        else:
            needs_color.append(c)
    if needs_color:
        fallback = sns.color_palette("tab20", len(needs_color))
        for c, col in zip(needs_color, fallback):
            resolved[c] = col

    fig, ax = _new_axes(figsize, facecolor=bg)

    # background is every cell not positive for any plotted column
    any_pos = np.zeros(adata.n_obs, dtype=bool)
    for c in cols:
        any_pos |= adata.obs[c].to_numpy(dtype=bool)
    ax.scatter(x[~any_pos], y[~any_pos], c=background_color, s=size,
               alpha=background_alpha, linewidths=0, rasterized=True)

    # draw abundant first, rare last so rare populations sit on top
    for c in cols:
        mask = adata.obs[c].to_numpy(dtype=bool)
        ax.scatter(x[mask], y[mask], c=[resolved[c]], s=size, alpha=alpha,
                   linewidths=0, rasterized=True)

    def _display_label_ct(c):
        # honour an explicit labels map keyed by raw column name or cleaned
        # label, otherwise fall back to the cleaned label
        if labels:
            if c in labels:
                return labels[c]
            cleaned = _clean_label(c)
            if cleaned in labels:
                return labels[cleaned]
        return _clean_label(c)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markersize=6,
               markerfacecolor=resolved[c], markeredgewidth=0)
        for c in cols
    ]
    legend = ax.legend(handles, [_display_label_ct(c) for c in cols],
                       bbox_to_anchor=(1.02, 1), loc="upper left",
                       frameon=False, fontsize=8)
    # colour the legend text so it stays readable on a dark background
    for text in legend.get_texts():
        text.set_color(text_color)

    tag = level if level is not None else "celltypes"
    ax.set_title(base_name, color=text_color)
    if invert_y:
        ax.invert_yaxis()
    fig.tight_layout()

    if save_dir is not None:
        fname = save_name or f"{base_name}_{tag}.png".lstrip("_")
        _save_fig(fig, save_dir, fname, dpi)

    return fig, ax, resolved
