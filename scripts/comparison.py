from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

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
):
    """
    Plot a clean heatmap with square tiles and no gridlines.

    Parameters
    ----------
    matrix : DataFrame
        cell_type × sample matrix of percentages
    cmap : str
        Matplotlib colormap
    figsize : tuple or None
        If None, automatically computed for square tiles
    """

    n_rows, n_cols = matrix.shape

    # Auto-size figure to keep tiles square
    if figsize is None:
        tile_size = 0.35  # inches per tile; adjust if needed
        figsize = (n_cols * tile_size, n_rows * tile_size)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        matrix.values,
        cmap=cmap,
        aspect="equal",     # <-- square tiles
        interpolation="nearest",
    )

    # Axis ticks & labels
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(matrix.index)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(matrix.columns, rotation=60, ha="right")

    # Remove gridlines, spines, and tick marks
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)

    # Colorbar
    cbar = fig.colorbar(
    im,
    ax=ax,
    orientation="horizontal",
    fraction=0.05,  # thickness of the colorbar
    pad=0.15,       # space between heatmap and colorbar
    location="bottom"  # place on top instead of bottom
)
    cbar.set_label("log10(% of cells + 0.1)") # Log scale +0.1 for better contrast

    if title is not None:
        ax.set_title(title)

    fig.tight_layout()
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

    # --- enforce correct row order manually ---
    ordered_rows = (
        df[["level", "cell_type"]]
        .drop_duplicates()
        .sort_values(["level", "cell_type"])
        ["cell_type"]
        .tolist()
    )

    matrix = matrix.reindex(ordered_rows)

    return matrix