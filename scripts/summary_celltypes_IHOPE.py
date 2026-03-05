import os
import pandas as pd
from anndata import AnnData

def summarize_celltypes_IHOPE(
    adata: AnnData,
    filename: str,
    prefixes=("type_", "intermediate_", "subtype_"),
    output_dir: str = os.path.abspath("../results/reports")
):
    """
    Summarize boolean cell-type annotations into counts and percentages.

    Supports hierarchical annotation layers:
        type
        intermediate
        subtype

    Assumptions
    ----------
    - Cell types are encoded as boolean columns in adata.obs
    - Relevant columns start with prefixes (default: type_, intermediate_, subtype_)
    - Annotation logic has already been applied

    Output
    ------
    CSV written to ../results/reports/
    Filename: celltype_summary_{filename}.csv
    """

    if not isinstance(adata, AnnData):
        raise ValueError("adata must be an AnnData object")

    base_name = os.path.basename(filename)
    base_name = os.path.splitext(base_name)[0]

    os.makedirs(output_dir, exist_ok=True)

    total_cells = adata.n_obs
    records = []

    for col in adata.obs.columns:

        if not col.startswith(prefixes):
            continue

        if adata.obs[col].dtype != bool:
            continue

        n_cells = int(adata.obs[col].sum())
        pct_total = round(100 * n_cells / total_cells, 1) if total_cells > 0 else 0

        level = col.split("_", 1)[0]
        cell_type = col[len(level) + 1:]

        records.append({
            "level": level,
            "cell_type": cell_type,
            "column": col,
            "n_cells": n_cells,
            "pct_total": pct_total,
            "total_cells": total_cells
        })

    if not records:
        raise ValueError(
            f"No boolean columns starting with {prefixes} found in adata.obs"
        )

    df = pd.DataFrame(records)

    # enforce biological order of levels
    level_order = ["type", "intermediate", "subtype"]
    df["level"] = pd.Categorical(df["level"], categories=level_order, ordered=True)

    df = df.sort_values(
        ["level", "n_cells"],
        ascending=[True, False]
    )

    out_path = os.path.join(output_dir, f"celltype_summary_{base_name}.csv")
    df.to_csv(out_path, index=False)

    print("Cell type summary written to:")
    print(f"  {out_path}\n")

    print("Cell type summary:")
    print(df.to_string(index=False))

    return df
