# Rules adapted from FACS gating scheme

import pandas as pd
from anndata import AnnData

def assign_cell_types_bool_IHOPE(adata: AnnData):
    """
    Cell typing rules aligned strictly to the IHOPE table.
    Parallel assignment at all levels (type / intermediate / subtype).
    """

    # LEVEL 1: LINEAGE/TYPE

    # B cells: CD45+, (CD20 OR CD79a)+, CD3e-
    adata.obs["type_B"] = (
        adata.obs["CD45_pos"]
        & (adata.obs["CD20_pos"] | adata.obs["CD79a_pos"])
        & ~adata.obs["CD3e_pos"]
    )

    # T cells: CD45+, CD3e+, CD20-
    adata.obs["type_T"] = (
        adata.obs["CD45_pos"]
        & adata.obs["CD3e_pos"]
        & ~adata.obs["CD20_pos"]
    )

    # NK cells: CD57+, CD20-, CD3e-
    # COMMENT: added CD45
    adata.obs["type_NK"] = (
        adata.obs["CD45_pos"]
        & adata.obs["CD57_pos"]
        & ~adata.obs["CD3e_pos"]
        & ~adata.obs["CD20_pos"]
    )

    # Myeloid cells: CD45+ AND (HLA-DR+ OR CD11c+) excluding T/B
    adata.obs["type_Myeloid"] = (
            adata.obs["CD45_pos"]
            & (adata.obs["HLA-DR_pos"] | adata.obs["CD11c_pos"])
            & ~adata.obs["CD3e_pos"]
            & ~adata.obs["CD20_pos"]
            & ~adata.obs["CD79a_pos"]
    )

    # Stromal cells: Vimentin+ OR Collagen IV+, excluding CD45, LYVE1, CD31
    adata.obs["type_Stromal"] = (
            (adata.obs["Vimentin_pos"] | adata.obs["Collagen IV_pos"])
            & ~adata.obs["CD45_pos"]
            & ~adata.obs["LYVE1_pos"]
            & ~adata.obs["CD31_pos"]
    )

    # Endothelial cells: CD31+ OR CD34+ OR LYVE1+, excluding CD45
    adata.obs["type_Endothelial"] = (
            (adata.obs["CD31_pos"] | adata.obs["CD34_pos"] | adata.obs["LYVE1_pos"])
            & ~adata.obs["CD45_pos"]
    )

    adata.obs["type_unclassified"] = ~(
            adata.obs["type_B"]
            | adata.obs["type_T"]
            | adata.obs["type_NK"]
            | adata.obs["type_Myeloid"]
            | adata.obs["type_Stromal"]
            | adata.obs["type_Endothelial"]
    )

    # LEVEL 2: INTERMEDIATE STATES

    # T cells
    t = adata.obs["type_T"]

    # CD4 T cells: CD4+, CD8-
    adata.obs["intermediate_CD4_T"] = (
        t & adata.obs["CD4_pos"] & ~adata.obs["CD8_pos"]
    )

    # CD8 T cells: CD8+, CD4-
    adata.obs["intermediate_CD8_T"] = (
        t & adata.obs["CD8_pos"] & ~adata.obs["CD4_pos"]
    )

    # Naïve T cells: CCR7+, CD45RO-
    adata.obs["intermediate_T_naive"] = (
        t
        & adata.obs["CCR7_pos"]
        & ~adata.obs["CD45RO_pos"]
    )

    not_T_naive = ~adata.obs["intermediate_T_naive"]

    # Memory T cells: CD45RO+, CCR7+
    # NOTE: intermediate_T_memory corresponds to TCM-like
    adata.obs["intermediate_T_memory"] = (
        t
        & adata.obs["CCR7_pos"]
        & adata.obs["CD45RO_pos"]
    )

    # B cells
    b = adata.obs["type_B"]

    # Naïve B cells: CD21+, CD27-
    adata.obs["intermediate_B_naive"] = (
        b
        & adata.obs["CD21_pos"]
        & ~adata.obs["CD27_pos"]
    )

    # Memory B cells: CD21+, CD27+
    adata.obs["intermediate_B_memory"] = (
        b
        & adata.obs["CD21_pos"]
        & adata.obs["CD27_pos"]
    )

    not_B_naive = ~adata.obs["intermediate_B_naive"]

    # LEVEL 3: SUBTYPES

    # B cell subtypes

    # GC B cells: CD20+, CD27+, CD38+
    # TODO: add CD21 or not?
    adata.obs["subtype_B_GC"] = (
        b
        & not_B_naive
        & adata.obs["CD20_pos"]
        & adata.obs["CD27_pos"]
        & adata.obs["CD38_pos"]
    )

    # Plasmablasts:
    # CD45+, (CD20 OR CD79a)+, CD27+, CD38+, CD21-
    adata.obs["subtype_B_plasmablast"] = (
        b
        & not_B_naive
        & adata.obs["CD27_pos"]
        & adata.obs["CD38_pos"]
        & ~adata.obs["CD21_pos"]
    )

    # T cell naïve subtypes

    adata.obs["subtype_TN_CD4"] = (
        adata.obs["intermediate_CD4_T"]
        & adata.obs["intermediate_T_naive"]
    )

    adata.obs["subtype_TN_CD8"] = (
        adata.obs["intermediate_CD8_T"]
        & adata.obs["intermediate_T_naive"]
    )

    # Central memory (TCM)

    adata.obs["subtype_TCM_CD4"] = (
        adata.obs["intermediate_CD4_T"]
        & not_T_naive
        & adata.obs["CCR7_pos"]
        & adata.obs["CD45RO_pos"]
    )

    adata.obs["subtype_TCM_CD8"] = (
        adata.obs["intermediate_CD8_T"]
        & not_T_naive
        & adata.obs["CCR7_pos"]
        & adata.obs["CD45RO_pos"]
    )

    # Effector memory (TEM)

    adata.obs["subtype_TEM_CD4"] = (
        adata.obs["intermediate_CD4_T"]
        & not_T_naive
        & ~adata.obs["CCR7_pos"]
        & adata.obs["CD45RO_pos"]
    )

    adata.obs["subtype_TEM_CD8"] = (
        adata.obs["intermediate_CD8_T"]
        & not_T_naive
        & ~adata.obs["CCR7_pos"]
        & adata.obs["CD45RO_pos"]
    )

    # TEMRA

    adata.obs["subtype_TEMRA_CD4"] = (
        adata.obs["intermediate_CD4_T"]
        & not_T_naive
        & ~adata.obs["CCR7_pos"]
        & ~adata.obs["CD45RO_pos"]
    )

    adata.obs["subtype_TEMRA_CD8"] = (
        adata.obs["intermediate_CD8_T"]
        & not_T_naive
        & ~adata.obs["CCR7_pos"]
        & ~adata.obs["CD45RO_pos"]
    )

    # Activated T cells

    adata.obs["subtype_Activated_CD4"] = (
        adata.obs["intermediate_CD4_T"]
        & not_T_naive
        & adata.obs["CD69_pos"]
    )

    adata.obs["subtype_Activated_CD8"] = (
        adata.obs["intermediate_CD8_T"]
        & not_T_naive
        & adata.obs["CD69_pos"]
    )

    # Regulatory T cells

    adata.obs["subtype_Treg"] = (
        adata.obs["intermediate_CD4_T"]
        & adata.obs["FOXP3_pos"]
    )

    # Terminally differentiated T cells

    adata.obs["subtype_T_terminal"] = (
        t
        & not_T_naive
        & adata.obs["CD57_pos"]
    )

    # UNASSIGNED SUBTYPES

    b_subtypes = [
        c for c in adata.obs.columns
        if c.startswith("subtype_B_")
    ]
    t_subtypes = [
        c for c in adata.obs.columns
        if c.startswith("subtype_T")
    ]

    adata.obs["subtype_B_unassigned"] = (
        b & ~adata.obs[b_subtypes].any(axis=1)
    )

    adata.obs["subtype_T_unassigned"] = (
        t & ~adata.obs[t_subtypes].any(axis=1)
    )

    # -----------------------------
    # LEVEL 3: SUBTYPES – MYELOID
    # -----------------------------
    my = adata.obs["type_Myeloid"]

    # cDC1: CD141+ AND Myeloid, exclude T/B/CD1c/CD68/CD163
    adata.obs["subtype_cDC1"] = (
            my
            & adata.obs["CD141_pos"]
            & ~adata.obs["CD1c_pos"]
            & ~adata.obs["CD68_pos"]
            & ~adata.obs["CD163_pos"]
            & ~adata.obs["CD3e_pos"]
            & ~adata.obs["CD20_pos"]
    )

    # cDC2: CD1c+ AND Myeloid, exclude T/B/CD141/CD68/CD163
    adata.obs["subtype_cDC2"] = (
            my
            & adata.obs["CD1c_pos"]
            & ~adata.obs["CD141_pos"]
            & ~adata.obs["CD68_pos"]
            & ~adata.obs["CD163_pos"]
            & ~adata.obs["CD3e_pos"]
            & ~adata.obs["CD20_pos"]
    )

    # Monocyte/macrophage: CD14 OR CD68 OR CD163, exclude T/B
    adata.obs["subtype_Monocyte_Macrophage"] = (
            my
            & (adata.obs["CD14_pos"] | adata.obs["CD68_pos"] | adata.obs["CD163_pos"])
            & ~adata.obs["CD3e_pos"]
            & ~adata.obs["CD20_pos"]
    )

    # STROMAL
    # Fibroblast: Vimentin+, exclude LYVE1/CD45 (optionally CD31-/Collagen IV-)
    adata.obs["subtype_Fibroblast"] = (
            adata.obs["Vimentin_pos"]
            & ~adata.obs["LYVE1_pos"]
            & ~adata.obs["CD45_pos"]
    )

    # Basement membrane: Collagen IV+, exclude CD31/CD45
    adata.obs["subtype_Basement_Membrane"] = (
            adata.obs["Collagen IV_pos"]
            & ~adata.obs["CD31_pos"]
            & ~adata.obs["CD45_pos"]
    )

    # FDC: CD21+ AND CXCL13+, exclude CD45/T/B
    adata.obs["subtype_FDC"] = (
            adata.obs["CD21_pos"]
            & adata.obs["CXCL13_pos"]
            & ~adata.obs["CD45_pos"]
            & ~adata.obs["CD3e_pos"]
            & ~adata.obs["CD20_pos"]
    )

    # ENDOTHELIAL
    # Blood endothelial cells: CD31+ AND CD34+, exclude LYVE1/CD45/T/B
    adata.obs["subtype_Blood_Endothelial"] = (
            adata.obs["CD31_pos"]
            & adata.obs["CD34_pos"]
            & ~adata.obs["LYVE1_pos"]
            & ~adata.obs["CD45_pos"]
            & ~adata.obs["CD3e_pos"]
            & ~adata.obs["CD20_pos"]
    )

    # Lymphatic endothelial cells: LYVE1+, exclude T/B/CD45
    adata.obs["subtype_Lymphatic_Endothelial"] = (
            adata.obs["LYVE1_pos"]
            & ~adata.obs["CD3e_pos"]
            & ~adata.obs["CD20_pos"]
            & ~adata.obs["CD45_pos"]
    )

    # Avoid anndata fragmentation
    adata.obs = adata.obs.copy()

    # PRINT SUMMARY
    total_cells = adata.n_obs
    print(f"Total cells: {total_cells}\n")

    for level_prefix in ["type_", "intermediate_", "subtype_"]:
        level_cols = [c for c in adata.obs.columns if c.startswith(level_prefix)]
        if not level_cols:
            continue

        print(f"Level — {level_prefix[:-1]}:")
        for col in level_cols:
            count = int(adata.obs[col].sum())
            pct_total = 100 * count / total_cells if total_cells else 0
            print(f"  {col}: {count} ({pct_total:.1f}% of total)")
        print("")

    return adata

def add_TfH_like_cells(
    adata: AnnData,
    follicle_key: str = "B_follicle",
    output_key: str = "subtype_TfH_like",
    plot: bool = False,
    size: float = 1.0,
    sample_name: str = "",
):
    """
    Define TfH-like cells using spatial follicle context and markers.

    TfH-like =
        CD4 T cells
        AND located in B-cell follicle
        AND PD-1+ OR ICOS+
    """

    required_keys = [
        "intermediate_CD4_T",
        follicle_key,
        "PD-1_pos",
        "ICOS_pos",
    ]

    for key in required_keys:
        if key not in adata.obs:
            raise ValueError(f"{key} not found in adata.obs")

    not_naive = ~adata.obs["intermediate_T_naive"]

    adata.obs[output_key] = (
        adata.obs["intermediate_CD4_T"]
        & adata.obs[follicle_key]
        & not_naive
        & (adata.obs["PD-1_pos"] | adata.obs["ICOS_pos"])
    )

    # Overwrite unassigned with TfH
    adata.obs["subtype_T_unassigned"] &= ~adata.obs["subtype_TfH_like"]

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