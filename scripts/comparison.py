from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


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
        None = include all levels found in the file
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

        # Filter levels if requested
        if levels is not None:
            df = df[df["level"].isin(levels)]

        # Filter cell types
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


def pivot_for_heatmap(df):
    """
    Pivot combined cell-type summaries into a heatmap-ready matrix.

    Rows: cell_type (ordered by level)
    Columns: sample
    Values: pct_total
    """
    required = {"sample", "level", "cell_type", "pct_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df = df[~df["cell_type"].str.endswith("_Unassigned")] #Drop unassigned

    # Ordering by type-intermediate-subtype
    level_order = ["type", "intermediate", "subtype"]
    df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)

    # sort properly
    df = df.sort_values(["level", "cell_type"])

    # pivot
    matrix = df.pivot_table(
        index="cell_type",
        columns="sample",
        values="pct_total",
        fill_value=0.0,
    )

    return matrix


def plot_celltype_heatmap(
    matrix,
    cmap="Purples",
    figsize=None,
    title=None,
    colorbar_legend=None,
):
    """
    Parameters:
    matrix : DataFrame
        cell_type × sample matrix of percentages
    cmap : str
        Matplotlib colormap
    figsize : tuple or None
        If None, automatically computed for square tiles
    colorbar_legend: str or None
    title: str or None
    """

    n_rows, n_cols = matrix.shape

    # --- figure sizing ---
    if figsize is None:
        tile_size = 0.35
        figsize = (n_cols * tile_size, n_rows * tile_size)

    fig, ax = plt.subplots(figsize=figsize)

    # --- heatmap ---
    im = ax.imshow(
        matrix.values,
        cmap=cmap,
        aspect="equal",
        interpolation="nearest",
    )

    # --- axis labels ---
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index, fontsize=8)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right", fontsize=9)

    # --- clean axes ---
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # --- better colorbar placement (IMPORTANT FIX) ---
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("top", size="3%", pad=0.3)

    cbar = fig.colorbar(im, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top")
    cax.xaxis.set_label_position("top")

    if colorbar_legend is not None: #Attempt to move text to left
        cbar.ax.text(
            -0.02, 0.5,
            colorbar_legend,
            transform=cbar.ax.transAxes,
            va="center",
            ha="right",
            fontsize=10,
        )

    # --- title styling ---
    if title is not None:
        ax.set_title(title, fontsize=12, pad=20)

    plt.tight_layout()
    plt.show()


def print_numeric_summary(df):
    """
    Print a clean numeric comparison table.
    """
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

def pivot_for_tissue_heatmap(df):
    df = df.copy()
    df = df[~df["cell_type"].str.endswith("_Unassigned")]

    # enforce biological order
    level_order = ["type", "intermediate", "subtype"]
    df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)

    # sort within biological structure
    df = df.sort_values(["level", "cell_type"])

    # build matrix (cell_type only on axis)
    matrix = df.pivot_table(
        index="cell_type",
        columns="tissue",
        values="pct_total",
        fill_value=0.0,
    )

    # Enforce correct row order
    ordered_rows = (
        df[["level", "cell_type"]]
        .drop_duplicates()
        .sort_values(["level", "cell_type"])
        ["cell_type"]
        .tolist()
    )

    matrix = matrix.reindex(ordered_rows)

    return matrix