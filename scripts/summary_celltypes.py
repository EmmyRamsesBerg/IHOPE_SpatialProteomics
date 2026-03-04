import os
import pandas as pd
from anndata import AnnData


def summarize_celltypes(
    adata: AnnData,
    filename: str,
    prefixes = ("branch_", "type_", "subtype_"),
    output_dir: str = os.path.abspath("../results/reports")
):
    """
    Summarize boolean cell-type annotations into counts and percentages.

    Assumptions:
    - Cell types are encoded as boolean columns in adata.obs
    - Relevant columns start with `prefix` (default: 'subtype_')
    - Annotation logic has already been applied

    Output:
    - CSV written to ../results/reports/
    - Filename: celltype_summary_{filename}.csv
    """

    if not isinstance(adata, AnnData):
        raise ValueError("adata must be an AnnData object")

    if "obs" not in dir(adata):
        raise ValueError("adata.obs not found")

    # Clean input filename (remove extension, keep stem)
    base_name = os.path.basename(filename)
    base_name = os.path.splitext(base_name)[0]

    os.makedirs(output_dir, exist_ok=True)

    total_cells = adata.n_obs

    records = []

    for col in adata.obs.columns:
        if col.startswith((prefixes)):
            if adata.obs[col].dtype != bool:
                continue

            n_cells = int(adata.obs[col].sum())
            pct_cells = 100 * n_cells / total_cells if total_cells > 0 else 0  # <- new line for percentage
            pct_cells = round(pct_cells, 1)

            level = col.split("_", 1)[0]  # branch / type / subtype
            cell_type = col.split("_", 1)[1]

            records.append({
                "level": level,
                "cell_type": cell_type,
                "column": col,
                "n_cells": n_cells,
                "pct_cells": pct_cells,  # <- include percentage in record
                "total_cells": total_cells
            })

    if len(records) == 0:
        raise ValueError(f"No boolean columns starting with '{prefixes}' found in adata.obs")

    df = pd.DataFrame(records).sort_values("n_cells", ascending=False)

    out_path = os.path.join(
        output_dir,
        f"celltype_summary_{base_name}.csv"
    )

    df.to_csv(out_path, index=False)

    print("Cell type summary written to:")
    print(f"  {out_path}")
    print()
    print("Cell type summary:")
    print(df.to_string(index=False))

    return df

