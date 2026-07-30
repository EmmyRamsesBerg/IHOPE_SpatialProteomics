from itertools import combinations
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import mannwhitneyu

# Label formatting

def _display_label(cell_type):
    """Convert a cell_type string to a display-friendly label (underscores to spaces)."""
    return cell_type.replace("_", " ")


# Colormap utility

def truncate_cmap(cmap_name, minval=0.15, maxval=1.0, n=256):
    """
    Return a truncated version of a matplotlib colormap.

    Cuts off the pale low end so that near-zero values render as light
    lavender rather than white.

    Parameters
    ----------
    cmap_name : str
        Name of any registered matplotlib colormap (e.g. "Purples").
    minval : float
        Lower bound of the colormap range to keep (0–1). Default 0.15.
    maxval : float
        Upper bound. Default 1.0 (keep the dark end intact).
    n : int
        Number of colour steps in the output map.
    """
    base = plt.colormaps[cmap_name]
    return mcolors.LinearSegmentedColormap.from_list(
        f"trunc({cmap_name},{minval:.2f},{maxval:.2f})",
        base(np.linspace(minval, maxval, n)),
    )

def _step(value):
    """
    Pick an evenly spaced tick step for a 0-to-value colour scale.
    Defaults to 10, which suits the 0-100 percentage scales used
    throughout this project, and falls back to a smaller step only
    when the range itself is small.
    """
    if value <= 0:
        return 1.0
    if value <= 10:
        return 2.0
    elif value <= 20:
        return 5.0
    else:
        return 10.0


def _nice_step(vmax, max_ticks):
    """
    Pick a round tick step for a 0-to-vmax scale that yields at most
    max_ticks ticks (excluding 0). Chooses the smallest round step (1, 2,
    2.5 or 5 times a power of ten) whose tick count fits, which keeps as
    many clean ticks as the width allows rather than rounding up so far
    that only a single tick remains.
    """
    if vmax <= 0 or max_ticks < 1:
        return 1.0
    bases = [1, 2, 2.5, 5]
    steps = sorted({b * (10.0 ** e) for e in range(-3, 7) for b in bases})
    for s in steps:
        n_ticks = int(np.floor(vmax / s + 1e-9))
        if n_ticks <= max_ticks:
            return s
    return steps[-1]

# Data loading

def load_celltype_summaries(
    file_map,
    levels=None,
    include_cell_types=None,
):
    """
    Load and combine cell type summaries across samples.

    Parameters
    ----------
    file_map : dict
        {sample_name: filepath}
    levels : list[str] or None
        Classification levels to include.
        None = include all levels found in the file.
        Example: ['type', 'state']
    include_cell_types : list[str] or dict or None
        - None: include all cell types
        - list[str]: include these cell types across all levels
        - dict: {level: [cell_types]} for level-specific filtering

    Returns
    -------
    DataFrame with columns:
        ['sample', 'level', 'cell_type', 'pct_total', 'n_cells']
    """
    dfs = []

    for sample, fn in file_map.items():
        fn = Path(fn)
        if not fn.exists():
            raise FileNotFoundError(fn)

        df = pd.read_csv(fn)
        required = {"level", "cell_type", "pct_total", "n_cells"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"{fn} (sample={sample}) is missing required columns: {missing}"
            )
        df["sample"] = sample

        if levels is not None:
            df = df[df["level"].isin(levels)]

        if include_cell_types is not None:
            if isinstance(include_cell_types, dict):
                keep = []
                for lvl, cts in include_cell_types.items():
                    keep.append(
                        df[(df["level"] == lvl) & (df["cell_type"].isin(cts))]
                    )
                df = pd.concat(keep, ignore_index=True)
            else:
                df = df[df["cell_type"].isin(include_cell_types)]

        dfs.append(df)

    return pd.concat(dfs, ignore_index=True)


# Pivot helpers

def _apply_exclusions(df, drop_unassigned, drop_unclassified, renormalise,
                      immune_only=False, structural_cell_types=None):
    """
    Shared filtering and optional renormalisation used by both pivot functions.

    Parameters
    ----------
    drop_unassigned : bool
        Drop any cell_type whose name ends with '_unassigned'
        (e.g. T_unassigned, B_unassigned).
    drop_unclassified : bool
        Drop any cell_type whose name is exactly 'unclassified'.
    renormalise : bool
        If True AND at least one category was dropped, recalculate
        pct_total so values sum to 100 within each (sample, level) group.
        Ignored when nothing was dropped.
    immune_only : bool
        If True, restrict to the subtype level and drop the structural cell
        types (given by structural_cell_types) and unclassified. Pair with
        renormalise=True to make remaining percentages sum to 100 within
        each (sample, level) group (i.e. percentage of immune cells).
    structural_cell_types : list[str] or None
        Cell type names to treat as structural and drop when immune_only is
        True. Required when immune_only is True. Defined in the notebook so
        the biology stays out of this script.

    Returns
    -------
    Filtered (and optionally renormalised) copy of df.
    """
    df = df.copy()
    dropped_anything = False

    if immune_only:
        if structural_cell_types is None:
            raise ValueError(
                "immune_only=True requires structural_cell_types to be "
                "provided (the list of structural cell types to drop)."
            )
        df = df[df["level"] == "subtype"]
        struct_mask = df["cell_type"].isin(structural_cell_types)
        unclass_mask = df["cell_type"] == "unclassified"
        mask = struct_mask | unclass_mask
        if mask.any():
            df = df[~mask]
            dropped_anything = True

    if drop_unassigned:
        mask = df["cell_type"].str.endswith("_unassigned")
        if mask.any():
            df = df[~mask]
            dropped_anything = True

    if drop_unclassified:
        mask = df["cell_type"] == "unclassified"
        if mask.any():
            df = df[~mask]
            dropped_anything = True

    if renormalise and dropped_anything:
        group_col = "sample" if "sample" in df.columns else "tissue"
        group_sums = df.groupby([group_col, "level"])["pct_total"].transform("sum")
        df["pct_total"] = df["pct_total"] / group_sums * 100

    return df


def pivot_for_heatmap(
    df,
    drop_unassigned=True,
    drop_unclassified=False,
    renormalise=False,
    immune_only=False,
    structural_cell_types=None,
    celltype_order=None,
    tissue_order=None,
):
    """
    Pivot combined cell-type summaries into a heatmap-ready matrix.

    Rows    : cell_type (ordered type → intermediate → subtype, then by
              celltype_order within level, then alpha)
    Columns : sample
    Values  : pct_total

    Parameters
    ----------
    drop_unassigned : bool
        Drop cell types ending in '_unassigned'. Default True (preserves
        previous behaviour).
    drop_unclassified : bool
        Drop the 'unclassified' category entirely. Default False.
    renormalise : bool
        Recalculate percentages after dropping so each (sample, level)
        group sums to 100. Default False.
    immune_only : bool
        Restrict to the subtype level and drop structural and unclassified
        cells. Requires structural_cell_types. Pair with renormalise=True
        for percentage of immune cells. Default False.
    structural_cell_types : list[str] or None
        Structural cell types to drop when immune_only is True. Defined in
        the notebook.
    celltype_order : list[str] or None
        Manual display order for grouping related cell types as heatmap
        rows. Used as a secondary sort key within each level. When None
        (default), rows fall back to alphabetical order within level.
    tissue_order : list[str] or None
        Manual display order for grouping sample columns by tissue (e.g.
        ["MedLN", "MesLN", "Spleen"]). Requires a "tissue" column in df.
        Samples are sorted by tissue group first, then alphabetically
        within each group. When None (default), columns fall back to
        alphabetical order (no tissue grouping).
    """
    required = {"sample", "level", "cell_type", "pct_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _apply_exclusions(
        df, drop_unassigned, drop_unclassified, renormalise,
        immune_only, structural_cell_types,
    )

    level_order = ["type", "intermediate", "subtype"]
    df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)
    df = df.sort_values(["level", "cell_type"])

    matrix = df.pivot_table(
        index="cell_type",
        columns="sample",
        values="pct_total",
        fill_value=0.0,
    )

    level_rank = {"type": 0, "intermediate": 1, "subtype": 2}
    level_lookup = df.drop_duplicates("cell_type").set_index("cell_type")["level"]

    matrix = matrix.loc[
        sorted(
            matrix.index,
            key=lambda x: (
                level_rank.get(level_lookup.loc[x], 99),
                _manual_order_position(x, celltype_order),
                x,
            ),
        )
    ]

    if tissue_order and "tissue" in df.columns:
        sample_tissue = df.drop_duplicates("sample").set_index("sample")["tissue"]
        matrix = matrix[
            sorted(
                matrix.columns,
                key=lambda s: (
                    _manual_order_position(sample_tissue.get(s), tissue_order),
                    s,
                ),
            )
        ]

    return matrix


def pivot_for_tissue_heatmap(
    df,
    drop_unassigned=True,
    drop_unclassified=False,
    renormalise=False,
    immune_only=False,
    structural_cell_types=None,
    celltype_order=None,
    tissue_order=None,
):
    """
    Pivot tissue-aggregated summaries into a heatmap-ready matrix.

    Rows    : cell_type (ordered type → intermediate → subtype, then by
              celltype_order within level, then alpha)
    Columns : tissue
    Values  : pct_total (already averaged over samples upstream)

    Parameters
    ----------
    drop_unassigned : bool
        Drop cell types ending in '_unassigned'. Default True.
    drop_unclassified : bool
        Drop the 'unclassified' category entirely. Default False.
    renormalise : bool
        Recalculate percentages after dropping so each (tissue, level)
        group sums to 100. Default False.
    immune_only : bool
        Restrict to the subtype level and drop structural and unclassified
        cells. Requires structural_cell_types. Pair with renormalise=True
        for percentage of immune cells. Default False.
    structural_cell_types : list[str] or None
        Structural cell types to drop when immune_only is True. Defined in
        the notebook.
    celltype_order : list[str] or None
        Manual display order for grouping related cell types as heatmap
        rows. Used as a secondary sort key within each level. When None
        (default), rows fall back to alphabetical order within level.
    """
    required = {"tissue", "level", "cell_type", "pct_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = _apply_exclusions(
        df, drop_unassigned, drop_unclassified, renormalise,
        immune_only, structural_cell_types,
    )

    level_order = ["type", "intermediate", "subtype"]
    df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)
    df = df.sort_values(["level", "cell_type"])

    matrix = df.pivot_table(
        index="cell_type",
        columns="tissue",
        values="pct_total",
        fill_value=0.0,
    )

    level_rank = {"type": 0, "intermediate": 1, "subtype": 2}
    level_lookup = df.drop_duplicates("cell_type").set_index("cell_type")["level"]

    matrix = matrix.loc[
        sorted(
            matrix.index,
            key=lambda x: (
                level_rank.get(level_lookup.loc[x], 99),
                _manual_order_position(x, celltype_order),
                x,
            ),
        )
    ]

    return matrix


# Parent-relative normalisation

def normalise_to_parent(matrix, parent_of, root_children):
    """
    Express each row as a percentage of its parent population.

    Every value in `matrix` is on a common base (percentage of all cells),
    so a child divided by its parent gives the child as a percentage of
    that parent. Denominators are read from the original matrix, so
    overwriting a row never corrupts a denominator used by another row.

    Parameters
    ----------
    matrix : DataFrame
        cell_type x sample (or cell_type x tissue) matrix of pct_total.
        Must contain the parent rows named in parent_of and the rows in
        root_children.
    parent_of : dict[str, str]
        Mapping of child cell_type to the parent cell_type whose own total
        is the denominator. Biology lives in the notebook, not here.
    root_children : list[str]
        Top-level cell types (e.g. the immune types) that have no single
        parent row. Each is divided by the per-column sum over the
        root_children that are present, which is their combined parent
        population.

    Returns
    -------
    DataFrame with only the displayed rows (root_children plus the keys of
    parent_of), in the same row order as the input matrix. Rows not named
    in either (structural cells, *_unassigned, unclassified) are dropped.
    Division by an absent parent yields NaN, left for the caller to handle.
    """
    out = matrix.copy().astype(float)

    present_roots = [r for r in root_children if r in matrix.index]
    root_denom = matrix.loc[present_roots].sum(axis=0).replace(0, np.nan)
    for ct in present_roots:
        out.loc[ct] = matrix.loc[ct] / root_denom * 100

    for child, parent in parent_of.items():
        if child in matrix.index and parent in matrix.index:
            denom = matrix.loc[parent].replace(0, np.nan)
            out.loc[child] = matrix.loc[child] / denom * 100

    displayed = set(root_children) | set(parent_of)
    keep = [r for r in matrix.index if r in displayed]
    return out.loc[keep]


def parent_ratio_labels(
    matrix,
    parent_of,
    root_children,
    display_names=None,
    root_label="immune cells",
):
    """
    Relabel rows as 'child / parent' for parent-relative heatmaps.

    Each row name becomes the child display name, a slash, and the display
    name of the population it was divided by. Root children use root_label
    as the denominator name. Only the row labels change, the values are
    untouched, and the numeric matrix passed in is left alone.

    Parameters
    ----------
    matrix : DataFrame
        Output of normalise_to_parent, indexed by cell_type.
    parent_of : dict[str, str]
        Same child to parent mapping used for the normalisation.
    root_children : list[str]
        Top-level cell types whose denominator has no single row.
    display_names : dict[str, str] or None
        Optional cell_type to display-name map. Cell types not listed fall
        back to underscores replaced by spaces.
    root_label : str
        Display name for the combined root denominator (e.g. immune cells).
    """
    def disp(ct):
        if display_names and ct in display_names:
            return display_names[ct]
        return ct.replace("_", " ")

    new_index = []
    for ct in matrix.index:
        if ct in root_children:
            new_index.append(f"{disp(ct)} / {root_label}")
        elif ct in parent_of:
            new_index.append(f"{disp(ct)} / {disp(parent_of[ct])}")
        else:
            new_index.append(disp(ct))

    out = matrix.copy()
    out.index = new_index
    return out


# Heatmap

def plot_celltype_heatmap(
    matrix,
    cmap="Purples",
    cmap_minval=0.15,
    figsize=None,
    title=None,
    colorbar_legend=None,
    scale="linear",
    vmax=None,
    narrow=False,
):
    """
    Static heatmap of cell types × samples (or tissues).

    Parameters
    ----------
    matrix : DataFrame
        cell_type × sample matrix of percentages.
    cmap : str
        Base matplotlib colormap name. Default "Purples".
    cmap_minval : float
        Lower truncation point for the colormap (0–1). Raises the colour
        floor so near-zero values are not white. Default 0.15.
    figsize : tuple or None
        Auto-computed from matrix size when None.
    title : str or None
    colorbar_legend : str or None
    scale : "linear" or "log"
    vmax : float or None
        Upper limit of the linear colour scale. When None (default), it is
        taken from the actual data maximum so the scale adapts to the range
        present (e.g. after rescaling to percentage of immune cells). Ignored
        for log scale. Pass a number to fix the scale across figures.
    narrow : bool
        Layout tweaks for heatmaps with very few columns, such as the
        tissue-grouped heatmap (three columns). When True, the auto figure
        width reserves extra room for long row labels and the linear
        colorbar uses fewer, rounder ticks so they do not crowd. Default
        False keeps the original sizing and ticks, so sample-level heatmaps
        are unchanged.
    """
    n_rows, n_cols = matrix.shape

    if figsize is None:
        tile_size = 0.35
        if narrow:
            # Row labels can be long (e.g. "CD4 T cells / T cells"), so
            # reserve horizontal room for them. Scaling width on column count
            # alone left no space for the labels and colorbar on a
            # few-column matrix, which made tight_layout fail.
            max_label_len = max((len(str(lbl)) for lbl in matrix.index), default=1)
            width = n_cols * tile_size + 0.07 * max_label_len
            height = max(n_rows * tile_size, 2.0)
            figsize = (width, height)
        else:
            figsize = (n_cols * tile_size, n_rows * tile_size)

    fig, ax = plt.subplots(figsize=figsize)

    # Truncated colormap = the low percentages are not shown as white
    used_cmap = truncate_cmap(cmap, minval=cmap_minval)

    if scale == "linear":
        vmin = 0
        if vmax is None:
            vmax = float(np.nanmax(matrix.values))
        if narrow:
            # The horizontal colorbar spans the heatmap columns, so its width
            # in inches is roughly the column count times the tile size. Allow
            # about one tick per 0.6 inch so a few-column bar is not crowded.
            cbar_width_estimate = n_cols * 0.35
            max_cbar_ticks = max(2, int(cbar_width_estimate / 0.6))
            # Allow one extra over the width estimate so a low-vmax bar keeps
            # at least two ticks (e.g. 10, 20, 30) instead of a single value.
            tick_step = _nice_step(vmax, max_cbar_ticks + 1)
        else:
            tick_step = _step(vmax)
    elif scale == "log":
        vmin = np.nanmin(matrix.values)

    im = ax.imshow(
        matrix.values,
        cmap=used_cmap,
        aspect="equal",
        interpolation="nearest",
        vmin=vmin,
        vmax=vmax,
    )

    ax.set_yticks(range(n_rows))
    ax.set_yticklabels([_display_label(ct) for ct in matrix.index], fontsize=8)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)

    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size=0.15, pad=0.3)
    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("bottom")
    cax.xaxis.set_label_position("bottom")

    if colorbar_legend is not None:
        cbar.ax.text(
            -0.02, 0.5,
            colorbar_legend,
            transform=cbar.ax.transAxes,
            va="center",
            ha="right",
            fontsize=10,
        )

    vmin_actual, vmax_actual = im.get_clim()
    if scale == "linear":
        ticks = np.arange(tick_step, vmax_actual + 1e-9, tick_step)
        ticks = ticks[ticks <= vmax_actual]
        if vmax_actual >= 10:
            labels = [f"{t:.0f}" for t in ticks]
        else:
            labels = [f"{t:.1f}" for t in ticks]
    else:
        ticks = np.linspace(vmin_actual, vmax_actual, 3)
        labels = [f"{t:.1f}" for t in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)
    cax.tick_params(axis="x", length=0)
    for t in ticks:
        cax.axvline(t, ymin=0.0, ymax=0.25, color="white", linewidth=0.8)
        cax.axvline(t, ymin=0.75, ymax=1.0, color="white", linewidth=0.8)

    if title is not None:
        ax.set_title(title, fontsize=12, pad=20)

    plt.tight_layout()
    plt.show()


# Clustermap

def plot_celltype_clustermap(
    matrix,
    cmap="Purples",
    cmap_minval=0.15,
    scale="linear",
    figsize=(8, 10),
    colorbar_legend=None,
    cluster_rows=True,
    cluster_cols=True,
    metric="euclidean",
    vmax=None,
):
    """
    Clustered heatmap of cell types × samples.

    Parameters
    ----------
    matrix : DataFrame
        cell_type × sample matrix.
    cmap : str
        Base matplotlib colormap name. Default "Purples".
    cmap_minval : float
        Lower truncation point for the colormap. Default 0.15.
    scale : "linear" or "log"
    figsize : tuple
    colorbar_legend : str or None
    cluster_rows : bool
    cluster_cols : bool
    metric : str
        Distance metric for clustering (e.g. "euclidean", "correlation").
    vmax : float or None
        Upper limit of the linear colour scale. When None (default), taken
        from the actual data maximum. Ignored for log scale.
    """
    data = matrix.copy()
    data.index = [_display_label(ct) for ct in data.index]

    used_cmap = truncate_cmap(cmap, minval=cmap_minval)

    if scale == "linear":
        vmin = 0
        if vmax is None:
            vmax = float(np.nanmax(data.values))
        tick_step = _step(vmax)
    else:
        vmin = np.nanmin(data.values)
        vmax = np.nanmax(data.values)

    row_ratio = 0.2 if cluster_rows else 0.02
    col_ratio = 0.2 if cluster_cols else 0.02

    g = sns.clustermap(
        data,
        cmap=used_cmap,
        figsize=figsize,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        dendrogram_ratio=(row_ratio, col_ratio),
        vmin=vmin,
        vmax=vmax,
        xticklabels=True,
        yticklabels=True,
        cbar_pos=(0.02, 0.2, 0.02, 0.4),
        metric=metric,
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.ax_heatmap.set_xlabel("")

    # Reposition the colorbar relative to the heatmap's actual bounding
    # box, rather than the fixed figure coordinates seaborn defaults to,
    # so it stays close to the plot and vertically centred on it instead
    # of trailing down to the x-axis label area.
    heatmap_pos = g.ax_heatmap.get_position()
    leftmost = (
        g.ax_row_dendrogram.get_position().x0 if cluster_rows else heatmap_pos.x0
    )
    cbar_width = 0.02
    cbar_height = heatmap_pos.height * 0.3
    cbar_left = leftmost - cbar_width - 0.01
    cbar_bottom = heatmap_pos.y0 + (heatmap_pos.height - cbar_height) / 2
    g.ax_cbar.set_position([cbar_left, cbar_bottom, cbar_width, cbar_height])

    cbar = g.ax_cbar
    cbar.yaxis.set_ticks_position("left")
    cbar.yaxis.set_label_position("left")

    if colorbar_legend is not None:
        cbar.set_ylabel(colorbar_legend, fontsize=10, rotation=90, labelpad=10)

    if scale == "linear":
        ticks = np.arange(tick_step, vmax + 1e-9, tick_step)
        ticks = ticks[ticks <= vmax]

        if vmax >= 10:
            labels = [f"{t:.0f}" for t in ticks]
        else:
            labels = [f"{t:.1f}" for t in ticks]
    else:
        ticks = np.linspace(vmin, vmax, 3)
        labels = [f"{t:.1f}" for t in ticks]

    cbar.set_yticks(ticks)
    cbar.set_yticklabels(labels)
    cbar.tick_params(axis="y", length=0)
    for t in ticks:
        if np.isclose(t, vmax):
            continue
        cbar.axhline(t, xmin=0.0, xmax=0.25, color="white", linewidth=0.8)
        cbar.axhline(t, xmin=0.75, xmax=1.0, color="white", linewidth=0.8)

    plt.show()

# Numeric summary

def print_numeric_summary(df):
    """Print a clean numeric comparison table."""
    summary = (
        df.pivot_table(
            index=["level", "cell_type"],
            columns="sample",
            values="pct_total",
        )
        .fillna(0.0)
        .round(2)
    )
    print("\nCell type percentage summary (% of total cells):\n")
    print(summary)


# Stacked barplot

def _manual_order_position(cell_type, celltype_order):
    """
    Return the index of cell_type in celltype_order, or a large number if
    not listed (so unlisted types sort after the listed ones). When
    celltype_order is None or empty, returns 0 for everything, so callers
    fall back to their alphabetical tiebreaker (no manual grouping).
    """
    if not celltype_order:
        return 0
    try:
        return celltype_order.index(cell_type)
    except ValueError:
        return len(celltype_order)



def _rescale_to_100(df, group_cols):
    """
    Recalculate pct_total so it sums to 100 within each group.

    Always rescales per sample when a 'sample' column is present, even if
    the caller's x-axis grouping is something else (e.g. "tissue"). This
    matters when a per-sample dataframe is passed with x="tissue" and the
    tissue-level averaging is meant to happen afterwards in the pivot, not
    here. Falls back to group_cols when there is no 'sample' column (e.g.
    an already tissue-aggregated dataframe).
    """
    df = df.copy()
    if "sample" in df.columns:
        group_cols = ["sample"]
    group_sums = df.groupby(group_cols)["pct_total"].transform("sum")
    df["pct_total"] = df["pct_total"] / group_sums * 100
    return df


def plot_celltype_stacked_barplot(
    df,
    levels=None,
    x="sample",
    figsize=(10, 6),
    title=None,
    palette=None,
    ylim=(0, 100),
    immune_only=False,
    lineage_subset=None,
    structural_cell_types=None,
    lineage_subsets=None,
    drop_unclassified=False,
    ylabel=None,
    tissue_order=None,
    celltype_order=None,
):
    """
    Stacked barplot of cell type composition.

    Parameters
    ----------
    df : DataFrame
        Must contain ['level', 'cell_type', 'pct_total'] plus the column
        named by `x`.
    levels : list[str] or None
        Hierarchy levels to include (e.g. ['type'], ['intermediate']).
    x : str
        Column to use as the x-axis ("sample" or "tissue").
    figsize : tuple
    title : str or None
    palette : dict or None
        Optional predefined {cell_type: color}. Returned so it can be
        reused across calls.
    ylim : tuple or None
        Y-axis limits. A warning is printed if any bar exceeds the upper
        limit.
    immune_only : bool
        If True, drop structural cell types (given by structural_cell_types)
        and unclassified cells, then rescale remaining percentages to sum to
        100 within each x-group. Y-axis label switches to "Percentage of
        immune cells (CD45+)". Requires structural_cell_types. Default False
        keeps current behaviour (all cell types, no rescaling). Ignored if
        lineage_subset is set.
    lineage_subset : str or None
        A key into lineage_subsets (e.g. "T", "B", "Myeloid"). If set, the
        plot is restricted to that lineage's subtypes only and rescaled so
        they sum to 100 within each x-group. Requires lineage_subsets.
        Y-axis label becomes "Percentage of {lineage} cells".
    structural_cell_types : list[str] or None
        Structural cell types to drop when immune_only is True. Defined in
        the notebook.
    lineage_subsets : dict[str, list[str]] or None
        Mapping of lineage name to its subtype cell_type names. Required when
        lineage_subset is set. Defined in the notebook.
    drop_unclassified : bool
        If True, drop the 'unclassified' category and rescale the remaining
        percentages to sum to 100 within each x-group. Ignored if
        immune_only or lineage_subset is set, since both already exclude
        unclassified as part of their own rescaling. Default False.
    ylabel : str or None
        If provided, overrides the automatically generated y-axis label
        (including the immune_only/lineage_subset/drop_unclassified
        defaults). Default None keeps the automatic label.
    tissue_order : list[str] or None
        Manual display order for the x-axis groups, grouped by tissue
        (e.g. ["MedLN", "MesLN", "Spleen"]). If x="sample", requires a
        "tissue" column in df; samples are sorted by tissue group first,
        then alphabetically within each group. If x="tissue", the tissue
        labels themselves are sorted directly by this order. When None
        (default), falls back to alphabetical order. Same convention as
        tissue_order in pivot_for_heatmap.
    celltype_order : list[str] or None
        Manual display order for the stacked segments and legend (e.g.
        ["TEMRA_CD4", "TN_CD4", "TEM_CD4", "TCM_CD4"]). Cell types not
        listed are appended after the listed ones, in their existing
        order. When None (default), falls back to the current behaviour
        of sorting segments by mean fraction across x-groups, largest
        first.
    """
    required = {"level", "cell_type", "pct_total", x}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    manual_ylabel = ylabel

    df = df.copy()

    df = df[~df["cell_type"].str.endswith("_unassigned")]
    df = df[df["cell_type"] != "B_plasmablast"]

    if levels is not None:
        df = df[df["level"].isin(levels)]

    ylabel = "Percentage of cells"

    if lineage_subset is not None:
        if lineage_subsets is None:
            raise ValueError(
                "lineage_subset requires lineage_subsets (the mapping of "
                "lineage name to its subtype cell types) to be provided."
            )
        if lineage_subset not in lineage_subsets:
            raise ValueError(
                f"lineage_subset must be one of {list(lineage_subsets)}"
            )
        df = df[df["cell_type"].isin(lineage_subsets[lineage_subset])]
        ylabel = f"Percentage of {lineage_subset} cells"
        df = _rescale_to_100(df, group_cols=[x])

    elif immune_only:
        if structural_cell_types is None:
            raise ValueError(
                "immune_only=True requires structural_cell_types (the list "
                "of structural cell types to drop) to be provided."
            )
        df = df[~df["cell_type"].isin(structural_cell_types)]
        df = df[df["cell_type"] != "unclassified"]
        ylabel = "Percentage of immune cells (CD45+)"
        df = _rescale_to_100(df, group_cols=[x])

    elif drop_unclassified:
        df = df[df["cell_type"] != "unclassified"]
        ylabel = "Percentage of cells (unclassified excluded)"
        df = _rescale_to_100(df, group_cols=[x])

    matrix = df.pivot_table(
        index=x,
        columns="cell_type",
        values="pct_total",
        fill_value=0.0,
    )

    if ylim is not None and (matrix.sum(axis=1) > ylim[1] + 0.5).any():
        print("Warning: some stacked bars exceed the specified ylim upper bound.")

    if tissue_order:
        if x == "tissue":
            matrix = matrix.loc[
                sorted(matrix.index, key=lambda t: _manual_order_position(t, tissue_order))
            ]
        elif x == "sample" and "tissue" in df.columns:
            sample_tissue = df.drop_duplicates("sample").set_index("sample")["tissue"]
            matrix = matrix.loc[
                sorted(
                    matrix.index,
                    key=lambda s: (
                        _manual_order_position(sample_tissue.get(s), tissue_order),
                        s,
                    ),
                )
            ]

    if celltype_order:
        order = pd.Index(
            sorted(
                matrix.columns,
                key=lambda ct: (_manual_order_position(ct, celltype_order), ct),
            )
        )
    else:
        order = matrix.mean(axis=0).sort_values(ascending=False).index
    matrix = matrix[order]

    if palette is None:
        colors = sns.color_palette("tab20", n_colors=len(order))
        palette = dict(zip(order, colors))
    else:
        missing_types = [ct for ct in order if ct not in palette]
        if missing_types:
            new_colors = sns.color_palette("tab20", n_colors=len(missing_types))
            for ct, col in zip(missing_types, new_colors):
                palette[ct] = col

    fig, ax = plt.subplots(figsize=figsize)
    bottom = np.zeros(len(matrix))

    for ct in matrix.columns:
        values = matrix[ct].values
        ax.bar(
            range(len(matrix.index)),
            values,
            bottom=bottom,
            color=palette[ct],
            label=ct,
            edgecolor="none",
        )
        bottom += values

    ax.set_xticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.index, rotation=45, ha="right")

    if manual_ylabel is not None:
        ylabel = manual_ylabel
    ax.set_ylabel(ylabel)

    if ylim is not None:
        ax.set_ylim(*ylim)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    handles = [plt.Rectangle((0, 0), 1, 1, color=palette[ct]) for ct in order[::-1]]
    ax.legend(
        handles,
        [_display_label(ct) for ct in order[::-1]],
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        frameon=False,
        title="Cell type",
    )

    if title:
        ax.set_title(title)

    plt.tight_layout()
    plt.show()

    return palette

def plot_celltype_stripplot(
    df,
    level="type",
    x="tissue",
    tissue_order=None,
    donor_colors=None,
    summary="bar",
    immune_only=False,
    structural_cell_types=None,
    drop_unclassified=False,
    celltype_order=None,
    cell_types=None,
    denominator_cell_type=None,
    parent_of=None,
    root_children=None,
    display_names=None,
    root_label="immune cells",
    ylim=None,
    ylabel=None,
    ncols=3,
    figsize=None,
    jitter=0.08,
    point_size=30,
    mannwhitney=False,
):
    """
    Facet grid with one subplot per cell type, showing individual sample
    points grouped by tissue, colored by donor.

    Unlike plot_celltype_stacked_barplot, nothing is averaged across ROIs
    or donors. Every sample is its own point, so a donor with multiple
    ROIs in the same tissue shows multiple points there.

    Parameters
    ----------
    df : DataFrame
        Full combined summary dataframe, all levels and all samples.
        Must contain ['level', 'cell_type', 'pct_total', 'sample', 'donor', x].
        Pass the unfiltered dataframe even when using cell_types to
        restrict the facets, since denominator_cell_type (when set) needs
        to look up values at the intermediate level that would otherwise
        be filtered out.
    level : str
        Single hierarchy level to facet over (e.g. "type", "subtype").
    x : str
        Column to use as the x-axis within each subplot. Default "tissue".
    tissue_order : list[str] or None
        Manual display order for the x-axis groups. Falls back to
        alphabetical order when None.
    donor_colors : dict[str, color]
        Mapping of donor id to a matplotlib color. Required.
    summary : "bar", "box", or None
        What to draw alongside the points for each x-group.
    immune_only : bool
        If True, drop structural_cell_types and unclassified, then
        rescale remaining percentages to sum to 100 within each sample.
        Requires structural_cell_types. Mutually exclusive with
        denominator_cell_type, since both define what pct_total means.
    structural_cell_types : list[str] or None
        Structural cell types to drop when immune_only is True.
    drop_unclassified : bool
        If True, drop 'unclassified' and rescale to 100 within each
        sample. Ignored if immune_only is True.
    celltype_order : list[str] or None
        Manual order for the facets. Falls back to alphabetical order
        when None.
    cell_types : list[str] or None
        Restrict facets to this list of cell_type names within the given
        level. When None (default), all cell types found at that level
        are faceted, same as before this parameter existed.
    denominator_cell_type : str or None
        Name of a cell_type at the intermediate level (e.g. "CD4_T") to
        use as the denominator. When set, each facet's pct_total is
        divided by that sample's percentage for denominator_cell_type
        and multiplied by 100, so values become percentage of that
        lineage rather than percentage of total cells. Looked up from
        the full df passed in, before any level or cell_types filtering,
        so pass the unfiltered dataframe when using this. Mutually
        exclusive with immune_only, drop_unclassified and parent_of.
    parent_of : dict[str, str] or None
        Per-facet denominator mapping of child cell_type to the parent
        cell_type whose own total is the denominator, same object used by
        normalise_to_parent. When set, each facet is divided by its own
        parent (looked up across all levels in the full df passed in) and
        multiplied by 100, so every facet is a percentage of its parent.
        Facets are restricted to the keys of parent_of plus any
        root_children present at the faceted level, so parentless cell
        types (structural, unclassified) drop out on their own. Biology
        lives in the notebook, not here. Mutually exclusive with
        denominator_cell_type, immune_only and drop_unclassified.
    root_children : list[str] or None
        Top-level cell types (e.g. the immune types) that have no single
        parent row. When one is faceted, it is divided by the per-sample
        sum over the root_children that are present, which is their
        combined parent population. Same convention as normalise_to_parent.
    display_names : dict[str, str] or None
        Optional cell_type to display-name map used for the "child /
        parent" facet titles when parent_of is set. Cell types not listed
        fall back to underscores replaced by spaces.
    root_label : str
        Display name for the combined root denominator in facet titles
        (e.g. "immune cells"). Only used when parent_of is set.
    ylim : tuple or None
        Shared y-axis limit across all facets. Default None, meaning
        each facet's y-axis is scaled by matplotlib to fit its own data.
    ylabel : str or None
        Shared y-axis label. Overrides the automatic label, including
        the denominator_cell_type default, when provided.
    ncols : int
        Number of facet columns requested. Automatically capped to the
        number of facets actually present, so a single-facet plot uses
        a single column rather than reserving empty space.
    figsize : tuple or None
        Auto-computed from the number of facets when None.
    jitter : float
        Horizontal jitter applied to points within each x-group, in axis
        units. Set to 0 to disable.
    point_size : float
        Marker size for the individual points.
    mannwhitney : bool
        If True, run a Mann-Whitney U test (scipy.stats.mannwhitneyu,
        two-sided) between every pair of x-groups present in each facet,
        and annotate the facet with a bracket and the raw p-value (e.g.
        "p = 0.31") for each pair. This is a formality, not a claim of
        statistical significance, especially at the sample sizes typical
        here, so no significance stars or thresholds are applied. Pairs
        are stacked bottom-to-top by increasing span (adjacent x-groups
        first) so brackets don't overlap, and the axis is expanded as
        needed to fit them, overriding ylim's upper bound if it would
        otherwise clip a bracket. Default False (no test, current
        behaviour unchanged).
    """
    required = {"level", "cell_type", "pct_total", "sample", "donor", x}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if donor_colors is None:
        raise ValueError("donor_colors is required (mapping of donor id to color).")
    if summary not in ("bar", "box", None):
        raise ValueError('summary must be "bar", "box", or None')
    if immune_only and denominator_cell_type is not None:
        raise ValueError(
            "immune_only and denominator_cell_type are mutually exclusive."
        )
    if drop_unclassified and denominator_cell_type is not None:
        raise ValueError(
            "drop_unclassified and denominator_cell_type are mutually exclusive."
        )
    if parent_of is not None:
        if denominator_cell_type is not None:
            raise ValueError(
                "parent_of and denominator_cell_type are mutually exclusive."
            )
        if immune_only:
            raise ValueError("parent_of and immune_only are mutually exclusive.")
        if drop_unclassified:
            raise ValueError(
                "parent_of and drop_unclassified are mutually exclusive."
            )

    df_full = df.copy()

    def _disp(ct):
        if display_names and ct in display_names:
            return display_names[ct]
        return ct.replace("_", " ")

    denom_lookup = None
    ylabel_default = "Percentage of cells"

    if denominator_cell_type is not None:
        denom_rows = df_full[
            (df_full["level"] == "intermediate")
            & (df_full["cell_type"] == denominator_cell_type)
        ]
        if denom_rows.empty:
            raise ValueError(
                f"denominator_cell_type '{denominator_cell_type}' not found "
                f"at level 'intermediate' in df."
            )
        denom_lookup = denom_rows.set_index("sample")["pct_total"]
        ylabel_default = f"Percentage of {denominator_cell_type} cells"

    df = df_full[df_full["level"] == level].copy()
    df = df[~df["cell_type"].str.endswith("_unassigned")]

    if cell_types is not None:
        df = df[df["cell_type"].isin(cell_types)]

    if parent_of is not None:
        allowed = set(parent_of.keys()) | set(root_children or [])
        df = df[df["cell_type"].isin(allowed)]

    if immune_only:
        if structural_cell_types is None:
            raise ValueError(
                "immune_only=True requires structural_cell_types (the list "
                "of structural cell types to drop) to be provided."
            )
        df = df[~df["cell_type"].isin(structural_cell_types)]
        df = df[df["cell_type"] != "unclassified"]
        ylabel_default = "Percentage of immune cells (CD45+)"
        df = _rescale_to_100(df, group_cols=["sample"])
    elif drop_unclassified:
        df = df[df["cell_type"] != "unclassified"]
        ylabel_default = "Percentage of cells"
        df = _rescale_to_100(df, group_cols=["sample"])
    elif denominator_cell_type is not None:
        df["pct_total"] = df.apply(
            lambda row: row["pct_total"] / denom_lookup.get(row["sample"], np.nan) * 100,
            axis=1,
        )
    elif parent_of is not None:
        present_at_level = set(df["cell_type"].unique())
        roots_present = [r for r in (root_children or []) if r in present_at_level]

        # Combined root denominator, the per-sample sum over the root
        # children that are present, matching normalise_to_parent.
        root_denom = (
            df_full[df_full["cell_type"].isin(roots_present)]
            .groupby("sample")["pct_total"].sum()
            if roots_present else None
        )

        # One denominator Series per faceted cell type, indexed by sample.
        # Parent rows are read from the unfiltered df_full, so a parent at
        # any level (e.g. CD4_T at intermediate, B at type) resolves.
        parent_series = {}
        for ct in present_at_level:
            if ct in parent_of:
                parent = parent_of[ct]
                rows = df_full[df_full["cell_type"] == parent]
                parent_series[ct] = rows.set_index("sample")["pct_total"]
            elif ct in roots_present and root_denom is not None:
                parent_series[ct] = root_denom

        def _parent_value(row):
            s = parent_series.get(row["cell_type"])
            if s is None:
                return np.nan
            d = s.get(row["sample"], np.nan)
            if not d or np.isnan(d):
                return np.nan
            return row["pct_total"] / d * 100

        df["pct_total"] = df.apply(_parent_value, axis=1)
        ylabel_default = "Percentage"

    if ylabel is None:
        ylabel = ylabel_default

    cell_type_order = sorted(
        df["cell_type"].unique(),
        key=lambda ct: (_manual_order_position(ct, celltype_order), ct),
    )
    x_groups = sorted(
        df[x].unique(),
        key=lambda v: (_manual_order_position(v, tissue_order), v),
    )
    x_positions = {val: i for i, val in enumerate(x_groups)}

    n_facets = len(cell_type_order)
    ncols = min(ncols, n_facets)
    nrows = int(np.ceil(n_facets / ncols))
    if figsize is None:
        figsize = (ncols * 3.2, nrows * 3.0)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes_flat = axes.flatten()

    rng = np.random.default_rng(0)  # fixed seed so jitter is reproducible across reruns

    for ax, ct in zip(axes_flat, cell_type_order):
        sub = df[df["cell_type"] == ct]

        for val in x_groups:
            group = sub[sub[x] == val]
            if group.empty:
                continue
            xpos = x_positions[val]

            if summary == "bar":
                mean_val = group["pct_total"].mean()
                ax.plot(
                    [xpos - 0.3, xpos + 0.3],
                    [mean_val, mean_val],
                    color="black",
                    linewidth=1.5,
                    linestyle="--",
                    zorder=2,
                )
            elif summary == "box":
                ax.boxplot(
                    group["pct_total"].values,
                    positions=[xpos],
                    widths=0.5,
                    showfliers=False,
                    zorder=2,
                )

            x_jitter = rng.uniform(-jitter, jitter, size=len(group)) if jitter else 0
            point_colors = [donor_colors.get(d, "gray") for d in group["donor"]]
            ax.scatter(
                xpos + x_jitter,
                group["pct_total"].values,
                c=point_colors,
                s=point_size,
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

        ax.set_xticks(range(len(x_groups)))
        ax.set_xticklabels(x_groups, rotation=45, ha="right")
        if parent_of is not None and ct in parent_of:
            facet_title = f"{_disp(ct)}\n/ {_disp(parent_of[ct])}"
        elif parent_of is not None and root_children and ct in root_children:
            facet_title = f"{_disp(ct)}\n/ {root_label}"
        else:
            facet_title = _display_label(ct)
        ax.set_title(facet_title, fontsize=10)
        if ylim is not None:
            ax.set_ylim(*ylim)

        if mannwhitney:
            present_groups = [
                val for val in x_groups if not sub[sub[x] == val].empty
            ]
            pairs = list(combinations(present_groups, 2))
            # Adjacent x-groups get the lowest brackets, wider spans stack
            # above them, so brackets don't overlap.
            pairs.sort(key=lambda p: abs(x_positions[p[1]] - x_positions[p[0]]))

            if pairs:
                data_min = sub["pct_total"].min()
                data_max = sub["pct_total"].max()
                data_range = (data_max - data_min) or (abs(data_max) or 1.0)
                step = data_range * 0.12
                base = data_max + step

                for i, (g1, g2) in enumerate(pairs):
                    vals1 = sub.loc[sub[x] == g1, "pct_total"].values
                    vals2 = sub.loc[sub[x] == g2, "pct_total"].values
                    if len(vals1) == 0 or len(vals2) == 0:
                        continue
                    try:
                        _, p_value = mannwhitneyu(vals1, vals2, alternative="two-sided")
                    except ValueError:
                        # e.g. all values identical or a group too small
                        continue

                    y = base + i * step
                    bracket_top = y + step * 0.15
                    x1, x2 = x_positions[g1], x_positions[g2]
                    ax.plot(
                        [x1, x1, x2, x2],
                        [y, bracket_top, bracket_top, y],
                        color="black", linewidth=0.8, zorder=4,
                    )
                    ax.text(
                        (x1 + x2) / 2, bracket_top + step * 0.05,
                        f"p = {p_value:.2f}",
                        ha="center", va="bottom", fontsize=7, zorder=4,
                    )

                # Grow the axis to fit the brackets, overriding ylim's
                # upper bound (set just above) if it would clip them.
                top_needed = base + len(pairs) * step + step
                current_bottom, current_top = ax.get_ylim()
                ax.set_ylim(current_bottom, max(current_top, top_needed))

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    for ax in axes_flat[n_facets:]:
        ax.axis("off")

    for row in range(nrows):
        axes[row, 0].set_ylabel(ylabel)

    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="white", markerfacecolor=color,
            markeredgecolor="white", markersize=8, label=donor,
        )
        for donor, color in donor_colors.items()
    ]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(1.0, 1.0),
        frameon=False,
        title="Donor",
    )

    plt.tight_layout()
    plt.show()