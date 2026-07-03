# Rules adapted from FACS gating scheme
import pandas as pd
from anndata import AnnData

def assign_cell_types_bool_IHOPE(adata: AnnData):
    """
    Cell typing rules aligned strictly to the IHOPE table.
    Parallel assignment at all levels (type / intermediate / subtype).
    """

    # LEVEL 0: REMOVE OLD CELL TYPING
    old_columns = [c for c in adata.obs.columns if c.startswith(("type_", "intermediate_", "subtype_", "state_"))]
    if old_columns:
        adata.obs.drop(columns=old_columns, inplace=True)

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

    # --- ENFORCE MUTUAL EXCLUSIVITY VIA PRECEDENCE ---

    # Define priority (highest first)
    priority = [
        "type_T",
        "type_B",
        "type_Myeloid",
        "type_NK",
        "type_Endothelial",
        "type_Stromal"
    ]

    type_cols = priority.copy()

    # Apply precedence: higher priority "wins"
    for i, higher in enumerate(priority):
        for lower in priority[i + 1:]:
            adata.obs[lower] = adata.obs[lower] & ~adata.obs[higher]

    adata.obs["type_unclassified"] = ~(
            adata.obs["type_T"]
            | adata.obs["type_B"]
            | adata.obs["type_NK"]
            | adata.obs["type_Myeloid"]
            | adata.obs["type_Stromal"]
            | adata.obs["type_Endothelial"]
    )

    # --- VALIDATION: enforce strict single-label assignment ---

    type_cols = [
        "type_B",
        "type_T",
        "type_NK",
        "type_Myeloid",
        "type_Stromal",
        "type_Endothelial",
        "type_unclassified"
    ]

    type_sum = adata.obs[type_cols].sum(axis=1)

    print("\n[TYPE VALIDATION]")
    print("Counts of assignments per cell:")
    print(type_sum.value_counts().sort_index())

    if (type_sum != 1).any():
        n_bad = (type_sum != 1).sum()
        raise ValueError(f"TYPE assignment invalid: {n_bad} cells do not have exactly one label.")
    else:
        print("TYPE assignment is strictly exclusive (each cell has exactly one type)")

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

    # Naive B cells, copied from intermediate level so naive has a
    # subtype-level label alongside the GC and Plasmablast marker calls
    adata.obs["subtype_B_naive"] = adata.obs["intermediate_B_naive"]

    # GC B cells (marker call): CD20+, CD27+, CD38+
    adata.obs["subtype_B_GC"] = (
        b
        & not_B_naive
        & adata.obs["CD20_pos"]
        & adata.obs["CD27_pos"]
        & adata.obs["CD38_pos"]
    )

    # Plasmablast B cells (marker call):
    # CD45+, (CD20 OR CD79a)+, CD27+, CD38+, CD21-
    adata.obs["subtype_B_Plasmablast"] = (
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
        CD4 memory T cells
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

    # CD4 memory T cells inside a follicle that are PD-1+ or ICOS+
    adata.obs[output_key] = (
        adata.obs["intermediate_CD4_T"]
        & adata.obs[follicle_key]
        & ~adata.obs["intermediate_T_naive"]
        & (adata.obs["PD-1_pos"] | adata.obs["ICOS_pos"])
    )

    # Remove TfH-like cells from the T unassigned pool
    if "subtype_T_unassigned" in adata.obs:
        adata.obs["subtype_T_unassigned"] = (
            adata.obs["subtype_T_unassigned"] & ~adata.obs[output_key]
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
            label="TfH-like (in follicle)",
        )

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")
        plt.legend(markerscale=3, frameon=False)
        plt.title(f"{sample_name}: TfH-like cells within B-cell follicles")
        plt.show()

    return adata

def add_spatial_B_context(
    adata: AnnData,
    follicle_key: str = "B_follicle",
    gc_key: str = "subtype_B_GC",
    plasmablast_key: str = "subtype_B_Plasmablast",
    type_b_key: str = "type_B",
    recompute_unassigned: bool = True,
    plot: bool = False,
    size: float = 1.0,
    sample_name: str = "",
):
    """
    Refine B-cell subsets by follicle location, OVERWRITING the marker calls in place.

    subtype_B_GC becomes
        marker-based GC call AND inside follicle
    subtype_B_Plasmablast becomes
        marker-based plasmablast call AND outside follicle

    Because inside and outside the follicle are mutually exclusive, GC and
    Plasmablast are mutually exclusive after this step. Cells that were marker
    positive but on the wrong side of the follicle boundary lose their subtype
    label here. When recompute_unassigned is True, subtype_B_unassigned is
    recomputed from the final subtype_B_ columns so those cells are counted as
    unassigned rather than falling through the accounting.
    """

    required_keys = [
        follicle_key,
        gc_key,
        plasmablast_key,
    ]

    for key in required_keys:
        if key not in adata.obs:
            raise ValueError(f"{key} not found in adata.obs")

    in_follicle = adata.obs[follicle_key]

    # GC B cells kept only inside follicles, Plasmablast only outside
    adata.obs[gc_key] = adata.obs[gc_key] & in_follicle
    adata.obs[plasmablast_key] = adata.obs[plasmablast_key] & ~in_follicle

    # Recompute the B unassigned pool so marker-positive, wrong-location cells
    # are accounted for after the overwrite
    if recompute_unassigned and type_b_key in adata.obs:
        b_subtypes = [
            c for c in adata.obs.columns
            if c.startswith("subtype_B_") and c != "subtype_B_unassigned"
        ]
        adata.obs["subtype_B_unassigned"] = (
            adata.obs[type_b_key] & ~adata.obs[b_subtypes].any(axis=1)
        )

    # Summary
    print("Spatial B-cell annotations applied (overwrite in place):")
    for col in [gc_key, plasmablast_key]:
        count = int(adata.obs[col].sum())
        pct = 100 * adata.obs[col].mean()
        print(f"  {col}: {count} ({pct:.2f}%)")

    # Optional plot (same style as TfH)
    if plot:
        import matplotlib.pyplot as plt

        x = adata.obsm["spatial"][:, 0]
        y = adata.obsm["spatial"][:, 1]

        plt.figure(figsize=(6, 6))

        plt.scatter(x, y, s=size, c="lightgrey", alpha=0.3)

        f = adata.obs[follicle_key]
        plt.scatter(x[f], y[f], s=size * 2, c="orange", alpha=0.4, label="Follicle")

        gc = adata.obs[gc_key]
        plt.scatter(x[gc], y[gc], s=size * 4, c="blue", label="GC (follicular)")

        pb = adata.obs[plasmablast_key]
        plt.scatter(x[pb], y[pb], s=size * 4, c="red", label="Plasmablast (extrafollicular)")

        plt.gca().invert_yaxis()
        plt.axis("equal")
        plt.axis("off")
        plt.legend(markerscale=3, frameon=False)
        plt.title(f"{sample_name}: Spatial B-cell subsets")
        plt.show()

    return adata