import pandas as pd
from anndata import AnnData
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def assign_cell_types_bool(adata: AnnData):
    """
Branch assignment (CD45 split):
Immune branch: CD45+
Non-immune branch: CD45−

Type assignment within immune branch:
T-cell: CD3e+ AND CD20− AND CD79a−
B-cell: (CD20+ OR CD79a+) AND CD3e−
Myeloid/APC: CD11c+ AND CD3e− AND CD20−
Remaining immune cells: labeled Immune_Unclassified

Type assignment non-immune:
Endothelial: CD31+ AND (CD34+ OR LYVE1+) # TODO consider looser rule CD31+ AND Vimentin-
Stromal: Vimentin+ OR CollagenIV+ AND LYVE1-
Non-Immune_Unclassified: not meeting these criteria

Subtype assignment within T-cells:
Helper T: CD4+ AND CD8−
Cytotoxic T: CD8+ AND CD4−
Treg putative: CD4+ AND FOXP3+ #TODO discuss putative
TfH-like: CD4+ AND PD-1+ AND ICOS+ #TODO Discuss putative, PD-1 sliding scale, but if in Bc follicle it is TfH
Naïve T: CD45RA+ #TODO is this feasible or will the definitions get messed up
Tissue-resident T: CD45RO+ #TODO is this feasible or will the definitions get messed up
Stem-like memory T (exhausted?): TCF-1+, CD27+, CD45RA+, CD45RO- #TODO discuss
Th17-like: CD4+, CCR6+ #TODO discuss gut marker
TEMRA CD8 T: CD45RA+, CD8+, CD3+ CCR7- #TODO From Curtis & co
T_Unclassified: not meeting these criteria

Subtype assignment within B-cells:
Memory B: CD20+ AND CD27+
Plasma-like: CD20− AND CD79a+ #TODO make sure no "low" left
GC-like B: CD20+ AND CD38+ #TODO discuss CD27
Follicular-like B: CD20+ AND CD21+ #TODO discuss
Activated B: CD20+ OR CD79a+ AND CD40+ AND CD69+ OR HLA-DR #TODO include CD79a?
B_Unclassified: not meeting these criteria

Subtype assignment within Myeloid/APC:
Monocyte/Macrophage: CD14+ #TODO optional CD68+ or CD163+, include?
cDC1: CD141+ AND CD1c− AND CD68- AND CD163- #TODO optional HLA-DR
cDC2: CD1c+ AND CD141− AND CD68- AND CD163- #TODO optional HLA-DR
Myeloid_Unclassified: not meeting these criteria

Subtype assignment within non-immune branch:
Endothelial (blood): CD31+ AND CD34+
Lymphatic Endothelial: LYVE1+ #TODO discuss accuracy
Fibroblast/stromal: Vimentin+ AND CollagenIV+ AND LYVE1− #TODO discuss Collagen, move to separate Bas.Mem.?
# TODO Also consider using Vimentin+ AND CD31- as a simple rule to exclude endothelium?
FDC: CD21+ AND CXCL13+
Stromal_Unclassified: not meeting these criteria

Note:
Each new subtype on a given level is independent. This means that overlapping cases  will be assigned to
the first rule defined. Order matters where markers co-express.
Only cells meeting all criteria are labeled as a type or subtype. Otherwise saved as "unclassified" on that level.
#TODO double-check if we want Treg/TfH to override helper T etc
    """

    # -----------------------------
    # BRANCH SPLIT – IMMUNE/NON-IMMUNE
    # -----------------------------
    has_cd45 = adata.obs['CD45_pos'].notna()
    cd45_pos = adata.obs.loc[has_cd45, 'CD45_pos']

    adata.obs['branch_Immune'] = False
    adata.obs['branch_Non_Immune'] = False

    adata.obs.loc[cd45_pos.index[cd45_pos], 'branch_Immune'] = True
    adata.obs.loc[cd45_pos.index[~cd45_pos], 'branch_Non_Immune'] = True

    # -----------------------------
    # TYPE SPLIT – CELL TYPES
    # -----------------------------
    immune = adata.obs['branch_Immune']
    non_immune = adata.obs['branch_Non_Immune']

    # Immune types
    adata.obs['type_Tcell'] = immune & adata.obs['CD3e_pos'] & ~adata.obs['CD20_pos'] & ~adata.obs['CD79a_pos']
    adata.obs['type_Bcell'] = immune & (adata.obs['CD20_pos'] | adata.obs['CD79a_pos']) & ~adata.obs['CD3e_pos']
    adata.obs['type_Myeloid'] = immune & adata.obs['CD11c_pos'] & ~adata.obs['CD3e_pos'] & ~adata.obs['CD20_pos']
    adata.obs['type_Immune_Unclassified'] = immune & ~(adata.obs['type_Tcell'] | adata.obs['type_Bcell'] | adata.obs['type_Myeloid'])

    # Non-immune types
    adata.obs['type_Endothelial'] = non_immune & (adata.obs['CD31_pos'] | adata.obs['CD34_pos'] | adata.obs['LYVE1_pos'])
    adata.obs['type_Stromal'] = non_immune & (adata.obs['Vimentin_pos'] | adata.obs['Collagen IV_pos']) & ~adata.obs['LYVE1_pos'] & ~adata.obs['CD31_pos']
    adata.obs['type_Non_Immune_Unclassified'] = non_immune & ~(adata.obs['type_Endothelial'] | adata.obs['type_Stromal'])

    # -----------------------------
    # SUBTYPE SPLIT – GRANULAR CELL TYPES
    # -----------------------------
    # T-cell subtypes
    t = adata.obs['type_Tcell']
    adata.obs['subtype_Treg'] = t & adata.obs['CD4_pos'] & adata.obs['FOXP3_pos']
    adata.obs['subtype_TfH_like'] = t & adata.obs['CD4_pos'] & adata.obs['PD-1_pos'] & adata.obs['ICOS_pos']
    adata.obs['subtype_Th17_like'] = t & adata.obs['CD4_pos'] & adata.obs['CCR6_pos']
    adata.obs['subtype_Stem_like_memory_T'] = t & adata.obs['TCF-1_pos'] & adata.obs['CD27_pos'] & adata.obs['CD45RA_pos'] & ~adata.obs['CD45RO_pos']
    adata.obs['subtype_Naive_T'] = t & adata.obs['CD45RA_pos'] #TODO remove later?
    adata.obs['subtype_Tissue_Resident_T'] = t & adata.obs['CD45RO_pos']
    adata.obs['subtype_CD4_T'] = t & adata.obs['CD4_pos'] & ~adata.obs['CD8_pos'] & ~adata.obs['FOXP3_pos'] & ~(adata.obs['PD-1_pos'] & adata.obs['ICOS_pos']) & ~adata.obs['CCR6_pos']
    adata.obs['subtype_CD8_T'] = t & adata.obs['CD8_pos'] & ~adata.obs['CD4_pos']
    adata.obs['subtype_T_Unclassified'] = t & ~(adata.obs['subtype_Treg'] | adata.obs['subtype_TfH_like'] | adata.obs['subtype_Th17_like'] |
                                                adata.obs['subtype_Stem_like_memory_T'] | adata.obs['subtype_Naive_T'] |
                                                adata.obs['subtype_Tissue_Resident_T'] | adata.obs['subtype_CD4_T'] | adata.obs['subtype_CD8_T'])
    adata.obs['subtype_TEMRA_CD8_T'] = (adata.obs['CD3e_pos'] & adata.obs['CD8_pos'] & adata.obs['CD45RA_pos'] & ~adata.obs['CCR7_pos'])
    adata.obs['subtype_Naive_CD4_T'] = (adata.obs['CD4_pos'] & adata.obs['CD45RA_pos'] & adata.obs['CCR7_pos'])

    # B-cell subtypes
    b = adata.obs['type_Bcell']
    adata.obs['subtype_Activated_B'] = b & ((adata.obs['CD20_pos'] | adata.obs['CD79a_pos']) & (adata.obs['CD40_pos'] & adata.obs['CD69_pos'] | adata.obs['HLA-DR_pos']))
    adata.obs['subtype_GC_like_B'] = b & adata.obs['CD20_pos'] & adata.obs['CD38_pos']
    adata.obs['subtype_Follicular_like_B'] = b & adata.obs['CD20_pos'] & adata.obs['CD21_pos']
    adata.obs['subtype_Memory_B'] = b & adata.obs['CD20_pos'] & adata.obs['CD27_pos']
    adata.obs['subtype_Plasma_like'] = b & ~adata.obs['CD20_pos'] & adata.obs['CD79a_pos']
    adata.obs['subtype_B_Unclassified'] = b & ~(adata.obs['subtype_Activated_B'] | adata.obs['subtype_GC_like_B'] |
                                                adata.obs['subtype_Follicular_like_B'] | adata.obs['subtype_Memory_B'] | adata.obs['subtype_Plasma_like'])

    # Myeloid/APC subtypes
    my = adata.obs['type_Myeloid']
    adata.obs['subtype_cDC1'] = my & adata.obs['CD141_pos'] & ~adata.obs['CD1c_pos'] & ~adata.obs['CD68_pos'] & ~adata.obs['CD163_pos']
    adata.obs['subtype_cDC2'] = my & adata.obs['CD1c_pos'] & ~adata.obs['CD141_pos'] & ~adata.obs['CD68_pos'] & ~adata.obs['CD163_pos']
    adata.obs['subtype_Monocyte_Macrophage'] = my & (adata.obs['CD14_pos'] | adata.obs['CD68_pos'] | adata.obs['CD163_pos'])
    adata.obs['subtype_Myeloid_Unclassified'] = my & ~(adata.obs['subtype_cDC1'] | adata.obs['subtype_cDC2'] | adata.obs['subtype_Monocyte_Macrophage'])

    # Non-immune subtypes
    non = adata.obs['type_Endothelial'] | adata.obs['type_Stromal'] | adata.obs['type_Non_Immune_Unclassified']
    adata.obs['subtype_FDC'] = non & adata.obs['CD21_pos'] & adata.obs['CXCL13_pos']
    adata.obs['subtype_Fibroblast'] = non & adata.obs['Vimentin_pos'] & ~adata.obs['LYVE1_pos']
    adata.obs['subtype_Endothelial'] = non & adata.obs['CD31_pos'] & adata.obs['CD34_pos'] & (~adata.obs['Vimentin_pos'] | ~adata.obs['LYVE1_pos'])
    adata.obs['subtype_Lymphatic_Endothelial'] = non & adata.obs['LYVE1_pos']
    adata.obs['subtype_Basement_Membrane'] = non & adata.obs['Collagen IV_pos']
    adata.obs['subtype_Stromal_Unclassified'] = non & ~(adata.obs['subtype_FDC'] | adata.obs['subtype_Fibroblast'] |
                                                        adata.obs['subtype_Endothelial'] | adata.obs['subtype_Lymphatic_Endothelial'] |
                                                        adata.obs['subtype_Basement_Membrane'])
    # -------- Summary --------
    branch_cols = [c for c in adata.obs.columns if c.startswith("branch_")]

    print("Branch summary:")
    for c in sorted(branch_cols):
        print(f"  {c.replace('branch_', '')}: {adata.obs[c].sum()}")

    branch_true = adata.obs[branch_cols].sum(axis=1)
    print(f"  Cells with ≥2 branches: {(branch_true > 1).sum()}")
    print(f"  Cells with no branch: {(branch_true == 0).sum()}")
    print(f"  Total cells: {len(adata)}\n")

    # -------- Type summary --------
    type_cols = [c for c in adata.obs.columns if c.startswith("type_")]

    print("Type summary:")
    for c in sorted(type_cols):
        print(f"  {c.replace('type_', '')}: {adata.obs[c].sum()}")

    type_true = adata.obs[type_cols].sum(axis=1)
    print(f"  Cells with ≥2 types: {(type_true > 1).sum()}")
    print(f"  Cells with no type: {(type_true == 0).sum()}")
    print(f"  Total cells: {len(adata)}\n")

    # -------- Subtype summary --------
    subtype_cols = [c for c in adata.obs.columns if c.startswith("subtype_")]

    print("Subtype summary:")
    for c in sorted(subtype_cols):
        print(f"  {c.replace('subtype_', '')}: {adata.obs[c].sum()}")

    subtype_true = adata.obs[subtype_cols].sum(axis=1)
    print(f"  Cells with ≥2 subtypes: {(subtype_true > 1).sum()}")
    print(f"  Cells with no subtype: {(subtype_true == 0).sum()}")
    print(f"  Total cells: {len(adata)}\n")

    return adata

def check_celltype_overlap(adata, celltype, level_prefix="subtype_", plot=True):
    """
    Check overlap of a given cell type with other types at the same level.

    Parameters
    ----------
    adata : AnnData
        Annotated data object with boolean columns for cell types
    celltype : str
        Full column name (e.g. 'subtype_TEMRA_CD8_T') or just the suffix (e.g. 'TEMRA_CD8_T')
    level_prefix : str
        Prefix to filter columns at the desired annotation level ('branch_', 'type_', 'subtype_')
    plot : bool
        Whether to generate a bar plot of overlaps
    """
    # Handle user input (allow suffix only)
    if not celltype.startswith(level_prefix):
        celltype_col = f"{level_prefix}{celltype}"
    else:
        celltype_col = celltype

    if celltype_col not in adata.obs.columns:
        raise ValueError(f"{celltype_col} not found in adata.obs.columns")

    # Get relevant columns
    cols = [c for c in adata.obs.columns if c.startswith(level_prefix) and c != celltype_col]

    # Compute overlap counts
    overlaps = {}
    target_cells = adata.obs[celltype_col]
    n_target = int(target_cells.sum())

    for c in cols:
        overlap_count = int((target_cells & adata.obs[c]).sum())
        pct_of_target = 100 * overlap_count / n_target if n_target > 0 else 0

        overlaps[c.replace(level_prefix, '')] = {
            "overlap_count": overlap_count,
            "pct_of_target": pct_of_target
        }

    overlap_df = pd.DataFrame.from_dict(overlaps, orient='index')
    overlap_df = overlap_df.sort_values('overlap_count', ascending=False)
    # Exclude zero-overlap types
    overlap_df = overlap_df[overlap_df['overlap_count'] > 0]

    # Print top overlaps
    print(f"\nOverlaps for {celltype_col} ({target_cells.sum()} cells total):")
    print(overlap_df)

    # Optional visualization
    if plot:
        plt.figure(figsize=(8, max(4, len(overlap_df)/2)))
        sns.barplot(x='overlap_count', y=overlap_df.index, data=overlap_df, palette="viridis")
        plt.title(f"Overlap of {celltype_col} with other {level_prefix} types")
        plt.xlabel("Number of overlapping cells")
        plt.ylabel("Other cell types")
        plt.tight_layout()
        plt.show()

    return overlap_df