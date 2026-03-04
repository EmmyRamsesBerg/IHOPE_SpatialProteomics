# Rules adapted from FACS gating scheme

import pandas as pd
from anndata import AnnData

def assign_cell_types_bool_IHOPE(adata: AnnData):
    """
    Cell typing rules aligned to IHOPE FACS gating logic.
    Parallel assignment at all levels.
    """

    # BASIC LINEAGE DEFINITIONS
    adata.obs["type_B"] = (adata.obs["CD20_pos"] | adata.obs["CD79a_pos"]) & ~adata.obs["CD3e_pos"]
    adata.obs["type_T"] = adata.obs["CD3e_pos"] & ~adata.obs["type_B"]
    adata.obs["type_unclassified"] = ~(adata.obs["type_B"] | adata.obs["type_T"])

    # B CELL DEFINITIONS
    b = adata.obs["type_B"]

    # B cell state axes
    adata.obs["state_B_CD27neg_naive_like"] = b & ~adata.obs["CD27_pos"]
    adata.obs["state_B_CD27pos_memory_like"] = b & adata.obs["CD27_pos"]

    adata.obs["state_B_resting"] = (
            b
            & adata.obs["CD21_high"]
            & adata.obs["CD38_low"]
    )

    adata.obs["state_B_GC_like"] = (
            b
            & adata.obs["CD21_low"]
            & adata.obs["CD38_mid"]
    )

    adata.obs["state_B_plasmablast_like"] = (
            b
            & adata.obs["CD38_high"]
    )

    # B cell subtypes
    adata.obs["subtype_B_naive_like"] = (
            adata.obs["state_B_resting"]
            & adata.obs["state_B_CD27neg_naive_like"]
    )

    adata.obs["subtype_B_memory_like"] = (
            adata.obs["state_B_resting"]
            & adata.obs["state_B_CD27pos_memory_like"]
    )

    adata.obs["subtype_B_GC_like"] = (
            adata.obs["state_B_GC_like"]
            & adata.obs["state_B_CD27pos_memory_like"]
    )

    adata.obs["subtype_B_plasmablast"] = (
        adata.obs["state_B_plasmablast_like"]
    )


    # T CELL DEFINITIONS
    t = adata.obs["type_T"]

    adata.obs["type_CD4_T"] = t & adata.obs["CD4_pos"] & ~adata.obs["CD8_pos"]
    adata.obs["type_CD8_T"] = t & adata.obs["CD8_pos"] & ~adata.obs["CD4_pos"]

    naive = t & adata.obs["CCR7_pos"] & adata.obs["CD45RA_pos"]
    memory = t & ~naive

    # NAIVE T CELLS
    adata.obs["state_T_naive"] = (naive)
    adata.obs["state_T_memory"] = (memory)

    adata.obs["subtype_TN_CD4"] = (
        adata.obs["type_CD4_T"]
        & naive
    )

    adata.obs["subtype_TN_CD8"] = (
        adata.obs["type_CD8_T"]
        & naive
    )

    # CENTRAL MEMORY
    adata.obs["subtype_TCM_CD4"] = (
        adata.obs["type_CD4_T"]
        & adata.obs["CCR7_pos"]
        & ~adata.obs["CD45RA_pos"]
    )

    adata.obs["subtype_TCM_CD8"] = (
        adata.obs["type_CD8_T"]
        & adata.obs["CCR7_pos"]
        & ~adata.obs["CD45RA_pos"]
    )

    # EFFECTOR MEMORY
    adata.obs["subtype_TEM_CD4"] = (
        adata.obs["type_CD4_T"]
        & memory
        & ~adata.obs["CCR7_pos"]
        & ~adata.obs["CD45RA_pos"]
    )

    adata.obs["subtype_TEM_CD8"] = (
        adata.obs["type_CD8_T"]
        & memory
        & ~adata.obs["CCR7_pos"]
        & ~adata.obs["CD45RA_pos"]
    )

    # TEMRA
    adata.obs["subtype_TEMRA_CD4"] = (
        adata.obs["type_CD4_T"]
        & adata.obs["CD45RA_pos"]
        & ~adata.obs["CCR7_pos"]
    )

    adata.obs["subtype_TEMRA_CD8"] = (
        adata.obs["type_CD8_T"]
        & adata.obs["CD45RA_pos"]
        & ~adata.obs["CCR7_pos"]
    )

    # MEMORY CD4 FUNCTIONAL SUBSETS
    mem_cd4 = adata.obs["type_CD4_T"] & memory

    #TODO: change definition later, use FOXP3_high?
   # adata.obs["subtype_Treg"] = (
   #     mem_cd4
   #     & adata.obs["FOXP3_pos"]
   # )

    # TODO: add TfH-like

    # Capture cells unassigned for subtype:
    b_subtypes = [col for col in adata.obs.columns if col.startswith("subtype_B") and col != "subtype_B_unassigned"]
    t_subtypes = [col for col in adata.obs.columns if col.startswith("subtype_T") and col != "subtype_T_unassigned"]

    adata.obs["subtype_B_unassigned"] = b & ~adata.obs[b_subtypes].any(axis=1)
    adata.obs["subtype_T_unassigned"] = t & ~adata.obs[t_subtypes].any(axis=1)

    # PRINT SUMMARY
    total_cells = adata.n_obs
    print(f"Total cells: {total_cells}\n")

    print("Lineage assignment:")
    print(f"  B cells: {b.sum()} ({100 * b.sum() / total_cells:.1f}%)")
    print(f"  T cells: {t.sum()} ({100 * t.sum() / total_cells:.1f}%)")
    print(
        f"  Other: {adata.obs['type_unclassified'].sum()} "
        f"({100 * adata.obs['type_unclassified'].sum() / total_cells:.1f}%)\n"
    )

    print("B cells CD27 memory axis (state):")
    for col in ["state_B_CD27neg_naive_like", "state_B_CD27pos_memory_like"]:
        count = adata.obs[col].sum()
        pct_total = 100 * count / total_cells
        pct_B = 100 * count / b.sum() if b.sum() > 0 else 0
        print(f"  {col}: {count} ({pct_total:.1f}% of total, {pct_B:.1f}% of B cells)")
    print("")

    print("B cells activation/differentiation axis (state):")
    for col in ["state_B_resting", "state_B_GC_like", "state_B_plasmablast_like"]:
        count = adata.obs[col].sum()
        pct_total = 100 * count / total_cells
        pct_B = 100 * count / b.sum() if b.sum() > 0 else 0
        print(f"  {col}: {count} ({pct_total:.1f}% of total, {pct_B:.1f}% of B cells)")
    print("")

    print("B cell subtypes:")
    for col in ["subtype_B_naive_like", "subtype_B_memory_like", "subtype_B_GC_like", "subtype_B_plasmablast",
                "subtype_B_unassigned"]:
        count = adata.obs[col].sum()
        pct_total = 100 * count / total_cells
        pct_B = 100 * count / b.sum() if b.sum() > 0 else 0
        print(f"  {col}: {count} ({pct_total:.1f}% of total, {pct_B:.1f}% of B cells)")
    print("")

    print("T cell broad states:")
    for col in ["state_T_naive", "state_T_memory"]:
        count = adata.obs[col].sum()
        pct_total = 100 * count / total_cells
        pct_T = 100 * count / t.sum() if t.sum() > 0 else 0
        print(f"  {col}: {count} ({pct_total:.1f}% of total, {pct_T:.1f}% of T cells)")
    print("")

    print("T cell subtypes:")
    for col in t_subtypes + ["subtype_T_unassigned"]:
        count = adata.obs[col].sum()
        pct_total = 100 * count / total_cells
        pct_T = 100 * count / t.sum() if t.sum() > 0 else 0
        print(f"  {col}: {count} ({pct_total:.1f}% of total, {pct_T:.1f}% of T cells)")
    print("")

    adata.obs = adata.obs.copy()

    # To avoid fragmentation of the anndata object:
    adata.obs = adata.obs.copy()

    return adata

def add_TfH_like_cells(
    adata: AnnData,
    follicle_key: str = "B_follicle",
    output_key: str = "subtype_TfH_like",
    plot: bool = False,
    size: float = 1.0,
    sample_name = str,
):
    """
    Define TfH-like cells using spatial follicle context and markers.

    TfH-like =
        CD4 T cells
        AND located in B-cell follicle
        AND PD-1+ OR ICOS+
    """

    required_keys = [
        "type_CD4_T",
        follicle_key,
        "PD-1_pos",
        "ICOS_pos",
    ]

    for key in required_keys:
        if key not in adata.obs:
            raise ValueError(f"{key} not found in adata.obs")

    adata.obs[output_key] = (
        adata.obs["type_CD4_T"]
        & adata.obs[follicle_key]
        & (adata.obs["PD-1_pos"] | adata.obs["ICOS_pos"])
    )

    print(
        f"TfH-like cells: {adata.obs[output_key].sum()} "
        f"({100 * adata.obs[output_key].mean():.2f}% of all cells)"
    )

    if plot:
        import matplotlib.pyplot as plt

        x = adata.obsm["spatial"][:, 0]
        y = adata.obsm["spatial"][:, 1]

        plt.figure(figsize=(6, 6))

        # Background: all cells
        plt.scatter(
            x,
            y,
            s=size,
            c="lightgrey",
            alpha=0.3,
            label="All cells",
        )

        # B-cell follicles
        f = adata.obs[follicle_key]
        plt.scatter(
            x[f],
            y[f],
            s=size * 2,
            c="orange",
            alpha=0.5,
            label="B-cell follicles",
        )

        # TfH-like cells
        t = adata.obs[output_key]
        plt.scatter(
            x[t],
            y[t],
            s=size * 4,
            c="crimson",
            alpha=0.9,
            label="TfH-like cells",
        )

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")
        plt.legend(markerscale=3, frameon=False)
        plt.title(f"{sample_name}: TfH-like cells within B-cell follicles")
        plt.show()

    return adata
