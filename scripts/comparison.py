from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable

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

def _nice_step(value):
    """
    Pick an evenly spaced tick step appropriate for a 0-to-value colour
    scale, so ticks land on clean numbers (5s, 10s, 20s, ...) rather
    than arbitrary fractions.
    """
    if value <= 0:
        return 1.0
    if value <= 20:
        return 5.0
    elif value <= 100:
        return 20.0
    elif value <= 500:
        return 100.0
    else:
        return float(10 ** np.floor(np.log10(value)))


def _round_up_to_step(value, step):
    """Round value up to the nearest multiple of step."""
    return float(np.ceil(value / step) * step)

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

    return matrix


def pivot_for_tissue_heatmap(
    df,
    drop_unassigned=True,
    drop_unclassified=False,
    renormalise=False,
    immune_only=False,
    structural_cell_types=None,
    celltype_order=None,
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
    """
    n_rows, n_cols = matrix.shape

    if figsize is None:
        tile_size = 0.35
        figsize = (n_cols * tile_size, n_rows * tile_size)

    fig, ax = plt.subplots(figsize=figsize)

    # Truncated colormap = the low percentages are not shown as white
    used_cmap = truncate_cmap(cmap, minval=cmap_minval)

    if scale == "linear":
        vmin = 0
        if vmax is None:
            raw_max = float(np.nanmax(matrix.values))
            tick_step = _nice_step(raw_max)
            vmax = _round_up_to_step(raw_max, tick_step)
        else:
            tick_step = _nice_step(vmax)
    elif scale == "log":
        vmin = np.nanmin(matrix.values)
        vmax = np.nanmax(matrix.values)
    else:
        vmin = np.nanmin(matrix.values)
        vmax = np.nanmax(matrix.values)

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
    cax = divider.append_axes("top", size="3%", pad=0.3)
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
        ticks = np.arange(tick_step, vmax_actual + tick_step / 2, tick_step)
        # Whole numbers when the range is wide, one decimal when small
        if vmax_actual >= 10:
            labels = [f"{t:.0f}" for t in ticks]
        else:
            labels = [f"{t:.1f}" for t in ticks]

    cbar.set_ticks(ticks)
    cbar.set_ticklabels(labels)

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
        vmax = float(np.nanmax(data.values)) if vmax is None else vmax
    else:
        vmin = np.nanmin(data.values)
        vmax = np.nanmax(data.values)

    g = sns.clustermap(
        data,
        cmap=used_cmap,
        figsize=figsize,
        row_cluster=cluster_rows,
        col_cluster=cluster_cols,
        vmin=vmin,
        vmax=vmax,
        xticklabels=True,
        yticklabels=True,
        cbar_pos=(0.02, 0.2, 0.02, 0.4),
        metric=metric,
    )

    plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.ax_heatmap.set_xlabel("")

    cbar = g.ax_cbar
    cbar.yaxis.set_ticks_position("left")
    cbar.yaxis.set_label_position("left")

    if colorbar_legend is not None:
        cbar.set_ylabel(colorbar_legend, fontsize=10, rotation=90, labelpad=10)

    if scale == "linear":
        mid = vmax / 2
        ticks = [0, mid, vmax]
        if vmax >= 10:
            labels = [f"{t:.0f}" for t in ticks]
        else:
            labels = [f"{t:.1f}" for t in ticks]
    else:
        ticks = np.linspace(vmin, vmax, 3)
        labels = [f"{t:.1f}" for t in ticks]

    cbar.set_yticks(ticks)
    cbar.set_yticklabels(labels)

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
    """
    required = {"level", "cell_type", "pct_total", x}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

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

#TODO add violin plot function with dot colors by donor