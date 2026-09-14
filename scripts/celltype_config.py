"""
celltype_config.py

This module holds display and bookkeeping metadata shared across
the IHOPE notebooks: which sample is which donor/tissue, what order
things should be plotted in, what each cell type should be called
and colored.

Import only what a given notebook needs, for example

    from scripts.celltype_config import (
        NAME_MAP, DONOR_MAP, TISSUE_MAP, TISSUE_ORDER,
        STRUCTURAL_CELL_TYPES, LINEAGE_SUBSETS, CELLTYPE_ORDER,
        PARENT_OF, IMMUNE_TYPES, CELLTYPE_DISPLAY, CELLTYPE_COLORS,
    )

"""

# Samples

# (file_basename, input_suffix, out_basename) used by the batch cell typing
# and BANKSY notebooks. file_basename + input_suffix is the name used to
# find the input CSV. out_basename is the name used for all
# outputs from that sample onward.
SAMPLES = [
    ("IHOPE14_MedLN_BottomLeft",  "_cleaned_filtered",        "IHOPE14_MedLN_BottomLeft"),
    ("IHOPE14_MedLN_TopRight",    "_cleaned_filtered_looped", "IHOPE14_MedLN_TopRight"),
    ("IHOPE14_MedLN_BottomRight", "_cleaned_filtered_looped", "IHOPE14_MedLN_BottomRight"),
    ("IHOPE14_mesLN",             "_cleaned_filtered_looped", "IHOPE14_MesLN"),
    ("IHOPE20_LN",                "_cleaned_filtered",        "IHOPE20_MedLN"),
    ("IHOPE20_Spleen",            "_cleaned_filtered_looped", "IHOPE20_Spleen"),
    ("IHOPE26_LN",                "_cleaned_filtered",        "IHOPE26_MedLN"),
    ("IHOPE26_Spleen",            "_cleaned_filtered",        "IHOPE26_Spleen"),
    ("IHOPE27_LN",                "_cleaned_filtered",        "IHOPE27_MedLN"),
    ("IHOPE27_Spleen",            "_cleaned_filtered",        "IHOPE27_Spleen"),
    ("IHOPE39_LN",                "_cleaned_filtered",        "IHOPE39_MedLN"),
    ("IHOPE39_MesLN_A",           "_cleaned_filtered",        "IHOPE39_MesLN_A"),
    ("IHOPE39_MesLN_B",           "_cleaned_filtered",        "IHOPE39_MesLN_B"),
    ("IHOPE39_Spleen",            "_cleaned_filtered",        "IHOPE39_Spleen"),
]

# Manual display order for sample columns in the follicle-counting bar
# plots, and for the per-sample follicle-size strip/box plots.
SAMPLE_ORDER = [
    "IHOPE14_MedLN_BottomLeft",
    "IHOPE14_MedLN_BottomRight",
    "IHOPE14_MedLN_TopRight",
    "IHOPE20_MedLN",
    "IHOPE26_MedLN",
    "IHOPE27_MedLN",
    "IHOPE39_MedLN",
    "IHOPE14_MesLN",
    "IHOPE39_MesLN_A",
    "IHOPE39_MesLN_B",
    "IHOPE20_Spleen",
    "IHOPE26_Spleen",
    "IHOPE27_Spleen",
    "IHOPE39_Spleen",
]

# celltype_summary_<x>.csv filename -> display sample name. The sample
# comparison/SVG export notebooks base donor_map and tissue_map (below)
# on these display names.
NAME_MAP = {
    "celltype_summary_IHOPE14_MedLN_BottomLeft.csv": "IHOPE14 MedLN Bottom Left",
    "celltype_summary_IHOPE14_MedLN_BottomRight.csv": "IHOPE14 MedLN Bottom Right",
    "celltype_summary_IHOPE14_MedLN_TopRight.csv": "IHOPE14 MedLN Top Right",
    "celltype_summary_IHOPE14_MesLN.csv": "IHOPE14 MesLN",
    "celltype_summary_IHOPE20_MedLN.csv": "IHOPE20 MedLN",
    "celltype_summary_IHOPE20_Spleen.csv": "IHOPE20 Spleen",
    "celltype_summary_IHOPE26_MedLN.csv": "IHOPE26 MedLN",
    "celltype_summary_IHOPE26_Spleen.csv": "IHOPE26 Spleen",
    "celltype_summary_IHOPE27_MedLN.csv": "IHOPE27 MedLN",
    "celltype_summary_IHOPE27_Spleen.csv": "IHOPE27 Spleen",
    "celltype_summary_IHOPE39_MedLN.csv": "IHOPE39 MedLN",
    "celltype_summary_IHOPE39_MesLN_A.csv": "IHOPE39 MesLN A",
    "celltype_summary_IHOPE39_MesLN_B.csv": "IHOPE39 MesLN B",
    "celltype_summary_IHOPE39_Spleen.csv": "IHOPE39 Spleen",
}

# Display sample name -> donor id. Used wherever df["sample"] holds the
# NAME_MAP display names (comparison/SVG export notebooks).
DONOR_MAP = {
    "IHOPE14 MedLN Bottom Left": "IHOPE14",
    "IHOPE14 MedLN Bottom Right": "IHOPE14",
    "IHOPE14 MedLN Top Right": "IHOPE14",
    "IHOPE14 MesLN": "IHOPE14",
    "IHOPE20 MedLN": "IHOPE20",
    "IHOPE20 Spleen": "IHOPE20",
    "IHOPE26 MedLN": "IHOPE26",
    "IHOPE26 Spleen": "IHOPE26",
    "IHOPE27 MedLN": "IHOPE27",
    "IHOPE27 Spleen": "IHOPE27",
    "IHOPE39 MedLN": "IHOPE39",
    "IHOPE39 MesLN A": "IHOPE39",
    "IHOPE39 MesLN B": "IHOPE39",
    "IHOPE39 Spleen": "IHOPE39",
}

# Display sample name -> tissue.
TISSUE_MAP = {
    "IHOPE14 MedLN Bottom Left": "MedLN",
    "IHOPE14 MedLN Bottom Right": "MedLN",
    "IHOPE14 MedLN Top Right": "MedLN",
    "IHOPE14 MesLN": "MesLN",
    "IHOPE20 MedLN": "MedLN",
    "IHOPE20 Spleen": "Spleen",
    "IHOPE26 MedLN": "MedLN",
    "IHOPE26 Spleen": "Spleen",
    "IHOPE27 MedLN": "MedLN",
    "IHOPE27 Spleen": "Spleen",
    "IHOPE39 MedLN": "MedLN",
    "IHOPE39 MesLN A": "MesLN",
    "IHOPE39 MesLN B": "MesLN",
    "IHOPE39 Spleen": "Spleen",
}

# basename (as used for h5ad / follicle-counting files, e.g.
# "IHOPE14_MedLN_BottomLeft") -> donor id/tissue. Same information as
# DONOR_MAP/TISSUE_MAP above, but keyed by the underscore-joined
# basename rather than the display name, for the notebooks that work
# directly from file basenames (follicle counting, follicle-count section
# of the SVG export notebook).
BASENAME_DONOR_MAP = {
    "IHOPE14_MedLN_BottomLeft": "IHOPE14",
    "IHOPE14_MedLN_BottomRight": "IHOPE14",
    "IHOPE14_MedLN_TopRight": "IHOPE14",
    "IHOPE14_MesLN": "IHOPE14",
    "IHOPE20_MedLN": "IHOPE20",
    "IHOPE20_Spleen": "IHOPE20",
    "IHOPE26_MedLN": "IHOPE26",
    "IHOPE26_Spleen": "IHOPE26",
    "IHOPE27_MedLN": "IHOPE27",
    "IHOPE27_Spleen": "IHOPE27",
    "IHOPE39_MedLN": "IHOPE39",
    "IHOPE39_MesLN_A": "IHOPE39",
    "IHOPE39_MesLN_B": "IHOPE39",
    "IHOPE39_Spleen": "IHOPE39",
}

BASENAME_TISSUE_MAP = {
    "IHOPE14_MedLN_BottomLeft": "MedLN",
    "IHOPE14_MedLN_BottomRight": "MedLN",
    "IHOPE14_MedLN_TopRight": "MedLN",
    "IHOPE14_MesLN": "MesLN",
    "IHOPE20_MedLN": "MedLN",
    "IHOPE20_Spleen": "Spleen",
    "IHOPE26_MedLN": "MedLN",
    "IHOPE26_Spleen": "Spleen",
    "IHOPE27_MedLN": "MedLN",
    "IHOPE27_Spleen": "Spleen",
    "IHOPE39_MedLN": "MedLN",
    "IHOPE39_MesLN_A": "MesLN",
    "IHOPE39_MesLN_B": "MesLN",
    "IHOPE39_Spleen": "Spleen",
}

# Manual display order for tissue groups, used where a plot groups by tissue.
TISSUE_ORDER = ["MedLN", "MesLN", "Spleen"]


# Cell type metadata

# Structural/non-immune cell types, dropped when immune_only=True.
STRUCTURAL_CELL_TYPES = [
    "Blood_Endothelial",
    "Lymphatic_Endothelial",
    "Basement_Membrane",
    "Fibroblast",
    "Stromal",
    "Endothelial",
    "FDC",
]

# Lineage subsets for the per-lineage breakdown barplots. Some notebooks
# add "CD4_T"/"CD8_T" keys to this dict for the CD4/CD8 state split barplots.
LINEAGE_SUBSETS = {
    "T": [
        "Activated_CD4", "Activated_CD8", "TCM_CD4", "TCM_CD8",
        "TEM_CD4", "TEM_CD8", "TEMRA_CD4", "TEMRA_CD8",
        "TN_CD4", "TN_CD8", "Treg", "TfH_like", "T_terminal",
    ],
    "B": [
        "B_naive", "B_GC", "B_Plasmablast",
    ],
    "Myeloid": [
        "Monocyte_Macrophage", "cDC1", "cDC2",
    ],
}

# Manual row order for heatmaps, grouping related subtypes together.
CELLTYPE_ORDER = [
    # type level
    "T", "NK", "B", "Myeloid", "Stromal", "Endothelial", "unclassified",
    # intermediate level
    "T_naive", "T_memory", "CD4_T", "CD8_T", "B_memory",
    # subtype: CD4 T cells
    "TN_CD4", "TCM_CD4", "TEM_CD4", "TEMRA_CD4", "Activated_CD4", "Treg", "TfH_like",
    # subtype: CD8 T cells
    "TN_CD8", "TCM_CD8", "TEM_CD8", "TEMRA_CD8", "Activated_CD8",
    # subtype: other T
    "T_terminal",
    # subtype: B cells (GC and Plasmablast are spatially defined in place)
    "B_naive", "B_GC", "B_Plasmablast",
    # subtype: myeloid
    "Monocyte_Macrophage", "cDC1", "cDC2",
    # subtype: stromal/structural (only appear when immune_only=False)
    "FDC", "Fibroblast", "Basement_Membrane",
    "Blood_Endothelial", "Lymphatic_Endothelial",
]

# Parent population for each row in the parent-relative heatmaps/screens.
# Each cell type is divided by its parent's own total. The gating in
# celltype_rules_IHOPE.py is overlapping below the type level, so these
# rows are not usually expected to sum to 100 within a parent population.
PARENT_OF = {
    # percentage of total T
    "CD4_T": "T", "CD8_T": "T", "T_naive": "T", "T_memory": "T", "T_terminal": "T",
    # percentage of CD4 T
    "TN_CD4": "CD4_T", "TCM_CD4": "CD4_T", "TEM_CD4": "CD4_T", "TEMRA_CD4": "CD4_T",
    "Activated_CD4": "CD4_T", "Treg": "CD4_T", "TfH_like": "CD4_T",
    # percentage of CD8 T
    "TN_CD8": "CD8_T", "TCM_CD8": "CD8_T", "TEM_CD8": "CD8_T",
    "TEMRA_CD8": "CD8_T", "Activated_CD8": "CD8_T",
    # percentage of total B. GC and Plasmablast are gated from "B and not
    # naive", and Plasmablast is CD21- so it sits outside memory B, so
    # total B is the parent that contains all of them.
    "B_naive": "B", "B_memory": "B", "B_GC": "B", "B_Plasmablast": "B",
    # percentage of Myeloid
    "Monocyte_Macrophage": "Myeloid", "cDC1": "Myeloid", "cDC2": "Myeloid",
}

# Immune lineage cell types
IMMUNE_TYPES = ["T", "B", "NK", "Myeloid"]

# Display names for the parent-relative heatmap/facet row labels. Each row
# is labelled "child / parent" (e.g. "CD4 T cells / T cells").
CELLTYPE_DISPLAY = {
    "T": "T cells", "B": "B cells", "NK": "NK cells", "Myeloid": "Myeloid cells",
    "CD4_T": "CD4 T cells", "CD8_T": "CD8 T cells",
    "T_naive": "Naive T cells", "T_memory": "Memory T cells",
    "T_terminal": "Terminally differentiated T cells",
    "TN_CD4": "Naive CD4 T cells", "TCM_CD4": "TCM CD4 T cells",
    "TEM_CD4": "TEM CD4 T cells", "TEMRA_CD4": "TEMRA CD4 T cells",
    "Activated_CD4": "Activated CD4 T cells", "Treg": "Regulatory T cells",
    "TfH_like": "TfH-like cells",
    "TN_CD8": "Naive CD8 T cells", "TCM_CD8": "TCM CD8 T cells",
    "TEM_CD8": "TEM CD8 T cells", "TEMRA_CD8": "TEMRA CD8 T cells",
    "Activated_CD8": "Activated CD8 T cells",
    "B_naive": "Naive B cells", "B_memory": "Memory B cells",
    "B_GC": "GC B cells", "B_Plasmablast": "Plasmablasts",
    "Monocyte_Macrophage": "Monocytes and macrophages",
    "cDC1": "cDC1", "cDC2": "cDC2",
}

# Color palette for cell types, used by every heatmap, barplot and spatial
# plot that needs a consistent color per cell type.
CELLTYPE_COLORS = {
    # type level
    "T": "#d62728",
    "B": "#1f77b4",
    "NK": "#2ca02c",
    "Myeloid": "#e377c2",
    "Stromal": "#8c564b",
    "Endothelial": "#9467bd",
    "unclassified": "#d9d9d9",

    # intermediate level
    "CD4_T": "#d62728",
    "CD8_T": "#ff7f0e",
    "T_naive": "#9467bd",
    "T_memory": "#2ca02c",
    "B_memory": "#1f77b4",

    # subtype level, B cells
    "B_naive": "#aec7e8",
    "B_GC": "#1f77b4",
    "B_Plasmablast": "#17becf",

    # subtype level, CD4 T cells
    "TN_CD4": "#d62728",
    "TCM_CD4": "#ff7f0e",
    "TEM_CD4": "#ff9896",
    "TEMRA_CD4": "#c5b0d5",
    "Activated_CD4": "#ad494a",
    "Treg": "#843c39",
    "TfH_like": "#e7298a",

    # subtype level, CD8 T cells
    "TN_CD8": "#fdd0a2",
    "TCM_CD8": "#ffbb78",
    "TEM_CD8": "#bcbd22",
    "TEMRA_CD8": "#dbdb8d",
    "Activated_CD8": "#8c6d31",

    # subtype level, other T
    "T_terminal": "#7f7f7f",

    # subtype level, myeloid
    "cDC1": "#2ca02c",
    "cDC2": "#98df8a",
    "Monocyte_Macrophage": "#8c564b",

    # subtype level, stromal/structural
    "FDC": "#9467bd",
    "Fibroblast": "#c49c94",
    "Basement_Membrane": "#c7c7c7",
    "Blood_Endothelial": "#393b79",
    "Lymphatic_Endothelial": "#5254a3",
}
