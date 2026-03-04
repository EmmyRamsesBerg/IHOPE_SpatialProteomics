import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

'''
- Loads a raw CSV containing imaging or single-cell intensities.
- Selects marker channels defined in TARGET_COLUMNS.
- Applies a normalization method (arcsinh, zscore, or none).
- Outputs a cleaned + transformed CSV with both metadata + transformed columns.
- Plots distributions of the transformed markers (optional).
- Returns all components for use in notebook and downstream AnnData construction.

Arguments:

Returns: 

'''



# Markers to always transform
TARGET_COLUMNS = [
    "CCR6: Mean", "CCR7: Mean", "CD107a: Mean", "CD11c: Mean", "CD14: Mean", "CD141: Mean",
    "CD163: Mean", "CD1c: Mean", "CD20: Mean", "CD21: Mean", "CD27: Mean", "CD31: Mean",
    "CD34: Mean", "CD38: Mean", "CD3e: Mean", "CD4: Mean", "CD40: Mean", "CD45: Mean",
    "CD45RA: Mean", "CD45RO: Mean", "CD57: Mean", "CD68: Mean", "CD69: Mean", "CD79a: Mean",
    "CD8: Mean", "Collagen IV: Mean", "CXCL13: Mean", "DAPI: Mean", "FOXP3: Mean",
    "Granzyme B: Mean", "HLA-DR: Mean", "ICOS: Mean", "IFNG: Mean", "Ki67: Mean", "LYVE1: Mean",
    "PD-1: Mean", "TCF-1: Mean", "Vimentin: Mean"
]


def _plot_distributions(df, columns, title_prefix="", save_plot=True, output_path=None, view=True):
    if not columns:
        return None

    n_cols = min(4, len(columns))
    n_rows = math.ceil(len(columns) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5), squeeze=False)
    axes = axes.flatten()

    for i, col in enumerate(columns):
        if col in df.columns:
            sns.histplot(df[col], kde=True, ax=axes[i], color="steelblue")
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel("Value", fontsize=8)
            axes[i].set_ylabel("Frequency", fontsize=8)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"{title_prefix}Marker Distributions", fontsize=16, y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])

    if save_plot:
        if output_path is not None:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fig.savefig(output_path, dpi=300, bbox_inches="tight")
            print(f"Plot saved to: {output_path}")
        else:
            fig.savefig("plot_normalized.png", dpi=300, bbox_inches="tight")
            print("Plot saved to: plot_normalized.png")
    if view:
        plt.show()

    return fig


def _arcsinh(df, columns, cofactor=5.0):
    df_trans = df.copy()
    new_cols = []
    for col in columns:
        if col in df_trans.columns:
            marker_name = col.replace(': Mean', '')
            new_col = f"arcsinh_cf{cofactor}_{marker_name}"
            df_trans[new_col] = np.arcsinh(df_trans[col] / cofactor)
            new_cols.append(new_col)
    return df_trans, new_cols

def _zscore(df, columns):
    df_trans = df.copy()
    new_cols = []
    for col in columns:
        if col in df_trans.columns:
            marker_name = col.replace(': Mean', '')
            new_col = f"z_{marker_name}"
            df_trans[new_col] = zscore(df_trans[col])
            new_cols.append(new_col)
    return df_trans, new_cols


def apply_transform(input_file: str,
                    method: str = "arcsinh",
                    cofactor: float = 5.0,
                    output_file: str | None = None,
                    save_plot: bool = True):
    """
    Load CSV, apply normalization to marker columns, save CSV, and plot distributions.

    Args:
        input_file: path to filtered CSV
        method: "arcsinh", "zscore", or "none"
        cofactor: used if method="arcsinh"
        output_file: optional output CSV path
        save_plot: whether to save plot

    Returns:
        df_out: DataFrame with metadata + transformed columns
        marker_cols: list of transformed marker columns
        metadata_cols: list of metadata columns
        fig: matplotlib figure of marker distributions
    """
    df = pd.read_csv(input_file)

    marker_cols = [col for col in TARGET_COLUMNS if col in df.columns]
    metadata_cols = [col for col in df.columns if col not in marker_cols]

    # Apply transformation
    if method.lower() == "arcsinh":
        df_out, transformed_cols = _arcsinh(df, marker_cols, cofactor)
        label = f"arcsinh_cf{cofactor}"
    elif method.lower() == "zscore":
        df_out, transformed_cols = _zscore(df, marker_cols)
        label = "zscore"
    elif method.lower() == "none":
        df_out = df.copy()
        transformed_cols = []
        for col in marker_cols:
            marker_name = col.replace(': Mean', '')
            new_col = f"raw_{marker_name}"
            df_out[new_col] = df_out[col]
            transformed_cols.append(new_col)
        label = "raw"
    else:
        raise ValueError(f"Unknown method: {method}")

    # Determine output file name
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_{label}.csv"

    # Save CSV
    df_out[metadata_cols + transformed_cols].to_csv(output_file, index=False)
    print(f"Saved CSV to: {output_file}")

    if save_plot:
        fig_dir = "../results/figures"
        os.makedirs(fig_dir, exist_ok=True)

        base_name = os.path.basename(output_file).replace(".csv", "_distributions.png")
        plot_path = os.path.join(fig_dir, base_name)
    else:
        plot_path = None

    fig = _plot_distributions(
        df_out,
        transformed_cols,
        title_prefix=f"{label} ",
        save_plot=save_plot,
        output_path=plot_path
    )

    return df_out, transformed_cols, metadata_cols, fig
