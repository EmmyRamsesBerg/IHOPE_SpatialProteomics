import pandas as pd
from anndata import AnnData

def assign_cell_types(adata: AnnData):
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
Endothelial: CD31+ AND (CD34+ OR LYVE1+) 
Stromal: Vimentin+ OR CollagenIV+ AND LYVE1-
Non-Immune_Unclassified: not meeting these criteria

Subtype assignment within T-cells:
Helper T: CD4+ AND CD8−
Cytotoxic T: CD8+ AND CD4−
Treg putative: CD4+ AND FOXP3+ 
TfH-like: CD4+ AND PD-1+ AND ICOS+ 
Naïve T: CD45RA+ 
Tissue-resident T: CD45RO+ 
Stem-like memory T (exhausted?): TCF-1+, CD27+, CD45RA+, CD45RO- 
Th17-like: CD4+, CCR6+ 
T_Unclassified: not meeting these criteria

Subtype assignment within B-cells:
Memory B: CD20+ AND CD27+
Plasma-like: CD20− AND CD79a+ 
GC-like B: CD20+ AND CD38+ 
Follicular-like B: CD20+ AND CD21+ 
Activated B: CD20+ OR CD79a+ AND CD40+ AND CD69+ OR HLA-DR 
B_Unclassified: not meeting these criteria

Subtype assignment within Myeloid/APC:
Monocyte/Macrophage: CD14+ 
cDC1: CD141+ AND CD1c− AND CD68- AND CD163- 
cDC2: CD1c+ AND CD141− AND CD68- AND CD163- 
Myeloid_Unclassified: not meeting these criteria

Subtype assignment within non-immune branch:
Endothelial (blood): CD31+ AND CD34+
Lymphatic Endothelial: LYVE1+ 
Fibroblast/stromal: Vimentin+ AND CollagenIV+ AND LYVE1− 
FDC: CD21+ AND CXCL13+
Stromal_Unclassified: not meeting these criteria

Note:
Each new subtype on a given level is independent. This means that overlapping cases  will be assigned to
the first rule defined. Order matters where markers co-express.
Only cells meeting all criteria are labeled as a type or subtype. Otherwise saved as "unclassified" on that level.
    """

    # Initialize as unclassified
    adata.obs['branch'] = 'Unclassified_branch'
    adata.obs['type'] = 'Unclassified_type'
    adata.obs['subtype'] = 'Unclassified_subtype'

    # 1: BRANCH SPLIT – IMMUNE/NON-IMMUNE
    has_cd45 = adata.obs['CD45_pos'].notna()
    cd45_pos = adata.obs.loc[has_cd45, 'CD45_pos']

    adata.obs.loc[cd45_pos.index[cd45_pos], 'branch'] = 'Immune'
    adata.obs.loc[cd45_pos.index[~cd45_pos], 'branch'] = 'Non-Immune'

    # Drop cells with missing CD45 info
    dropped = adata.obs['branch'] == 'Unclassified_branch'
    print(f"Dropped cells (no CD45 info): {dropped.sum()}")
    adata = adata[~dropped]

    print(f"CD45+ (Immune): {(adata.obs['branch'] == 'Immune').sum()}")
    print(f"CD45- (Non-Immune): {(adata.obs['branch'] == 'Non-Immune').sum()}")

    # 2. TYPE SPLIT – CELL TYPES
    # Immune T/B/Myeloid:
    immune = adata.obs['branch'] == 'Immune'

    tcell = immune & adata.obs['CD3e_pos'] & ~adata.obs['CD20_pos'] & ~adata.obs['CD79a_pos']
    bcell = immune & (adata.obs['CD20_pos'] | adata.obs['CD79a_pos']) & ~adata.obs['CD3e_pos']
    myeloid = immune & adata.obs['CD11c_pos'] & ~adata.obs['CD3e_pos'] & ~adata.obs['CD20_pos'] 

    immune_remaining = immune & ~(tcell | bcell | myeloid)

    adata.obs.loc[tcell, 'type'] = 'T-cell'
    adata.obs.loc[bcell, 'type'] = 'B-cell'
    adata.obs.loc[myeloid, 'type'] = 'Myeloid'
    adata.obs.loc[immune_remaining, 'type'] = 'Immune_Unclassified'

    print(f"T-cells: {tcell.sum()}")
    print(f"B-cells: {bcell.sum()}")
    print(f"Myeloid: {myeloid.sum()}")
    print(f"Immune_Unclassified: {immune_remaining.sum()}")

    # Non-immune endothelial/stromal:
    non_immune = adata.obs['branch'] == 'Non-Immune'

    endothelial = non_immune & (adata.obs['CD31_pos'] | adata.obs['CD34_pos'] | adata.obs['LYVE1_pos'])
    stromal = non_immune & (adata.obs['Vimentin_pos'] | adata.obs['Collagen IV_pos']) & ~adata.obs['LYVE1_pos'] & ~adata.obs['CD31_pos']
    non_immune_remaining = non_immune & ~(endothelial | stromal)

    adata.obs.loc[endothelial, 'type'] = 'Endothelial'
    adata.obs.loc[stromal, 'type'] = 'Stromal'
    adata.obs.loc[non_immune_remaining, 'type'] = 'Non-Immune_Unclassified'

    print(f"Endothelial: {endothelial.sum()}")
    print(f"Stromal: {stromal.sum()}")
    print(f"Non-Immune_Unclassified: {non_immune_remaining.sum()}")

    # 3. SUBTYPE SPLIT – GRANULAR CELL TYPES

    # T-cell subtypes
    t = adata.obs['type'] == 'T-cell'

    adata.obs.loc[t & adata.obs['CD4_pos'] & adata.obs['FOXP3_pos'], 'subtype'] = 'Treg'
    adata.obs.loc[t & adata.obs['CD4_pos'] & adata.obs['PD-1_pos'] & adata.obs['ICOS_pos'], 'subtype'] = 'TfH_like'
    adata.obs.loc[t & adata.obs['CD4_pos'] & adata.obs['CCR6_pos'], 'subtype'] = 'Th17_like'
    adata.obs.loc[t & adata.obs['TCF-1_pos'] & adata.obs['CD27_pos'] & adata.obs['CD45RA_pos'] & ~adata.obs[
        'CD45RO_pos'], 'subtype'] = 'Stem_like_memory_T'   #(exhausted?)
    adata.obs.loc[t & adata.obs['CD45RA_pos'], 'subtype'] = 'Naive_T'
    adata.obs.loc[t & adata.obs['CD45RO_pos'], 'subtype'] = 'Tissue_Resident_T'

    adata.obs.loc[t & adata.obs['CD4_pos'] & ~adata.obs['CD8_pos'] &   # Helper T (CD4+ only, excl Treg/TfH/Th17)?
                  ~adata.obs['FOXP3_pos'] & ~(adata.obs['PD-1_pos'] & adata.obs['ICOS_pos']) &
                  ~adata.obs['CCR6_pos'], 'subtype'] = 'CD4_T'
    adata.obs.loc[t & adata.obs['CD8_pos'] & ~adata.obs['CD4_pos'], 'subtype'] = 'CD8_T'

    adata.obs.loc[t & (adata.obs['subtype'] == 'Unclassified_subtype'), 'subtype'] = 'T_Unclassified'

    # Print summary
    print("T-cell subtypes summary:")
    for subtype in ['Treg', 'TfH_like', 'Th17_like', 'CD4_T', 'CD8_T', 'Stem_like_memory_T', 'Naive_T',
                    'Tissue_Resident_T', 'T_Unclassified']:
        print(f"    {subtype}: {(adata.obs['subtype'] == subtype).sum()}")

    # B-cell subtypes
    b = adata.obs['type'] == 'B-cell'

    adata.obs.loc[b & ((adata.obs['CD20_pos'] | adata.obs['CD79a_pos']) &
                       (adata.obs['CD40_pos'] & adata.obs['CD69_pos'] | adata.obs[
                           'HLA-DR_pos'])), 'subtype'] = 'Activated_B' 
    adata.obs.loc[b & adata.obs['CD20_pos'] & adata.obs['CD38_pos'], 'subtype'] = 'GC_like_B'
    adata.obs.loc[b & adata.obs['CD20_pos'] & adata.obs['CD21_pos'], 'subtype'] = 'Follicular_like_B'
    adata.obs.loc[b & adata.obs['CD20_pos'] & adata.obs['CD27_pos'], 'subtype'] = 'Memory_B'
    adata.obs.loc[b & ~adata.obs['CD20_pos'] & adata.obs['CD79a_pos'], 'subtype'] = 'Plasma_like'

    adata.obs.loc[b & (adata.obs['subtype'] == 'Unclassified_subtype'), 'subtype'] = 'B_Unclassified'

    # Print summary
    print("B-cell subtypes summary:")
    for subtype in ['Memory_B', 'Plasma_like', 'GC_like_B', 'Follicular_like_B', 'Activated_B', 'B_Unclassified']:
        print(f"    {subtype}: {(adata.obs['subtype'] == subtype).sum()}")

    # Myeloid/APC subtypes
    my = adata.obs['type'] == 'Myeloid'

    adata.obs.loc[my & adata.obs['CD141_pos'] & ~adata.obs['CD1c_pos'] & ~adata.obs['CD68_pos'] & ~adata.obs[
        'CD163_pos'], 'subtype'] = 'cDC1'
    adata.obs.loc[my & adata.obs['CD1c_pos'] & ~adata.obs['CD141_pos'] & ~adata.obs['CD68_pos'] & ~adata.obs[
        'CD163_pos'], 'subtype'] = 'cDC2'
    adata.obs.loc[my & (adata.obs['CD14_pos'] | adata.obs['CD68_pos'] | adata.obs[
        'CD163_pos']), 'subtype'] = 'Monocyte/Macrophage'
    adata.obs.loc[my & (adata.obs['subtype'] == 'Unclassified_subtype'), 'subtype'] = 'Myeloid_Unclassified'

    # Print summary
    print("Myeloid/APC subtypes summary:")
    for subtype in ['Monocyte/Macrophage', 'cDC1', 'cDC2', 'Myeloid_Unclassified']:
        print(f"    {subtype}: {(adata.obs['subtype'] == subtype).sum()}")

    # Non-immune subtypes
    non = adata.obs['branch'] == 'Non-Immune'

    adata.obs.loc[non & adata.obs['CD21_pos'] & adata.obs['CXCL13_pos'], 'subtype'] = 'FDC' #May contain Vimentin
    adata.obs.loc[non & adata.obs['Vimentin_pos'] & ~adata.obs['LYVE1_pos'], 'subtype'] = 'Fibroblast'
    adata.obs.loc[non & adata.obs['CD31_pos'] & adata.obs['CD34_pos'] & (~adata.obs[
        'Vimentin_pos'] | ~adata.obs['LYVE1_pos']), 'subtype'] = 'Endothelial'
    adata.obs.loc[non & adata.obs['LYVE1_pos'], 'subtype'] = 'Lymphatic_Endothelial'
    adata.obs.loc[non & adata.obs['Collagen IV_pos'], 'subtype'] = 'Basement Membrane'
    adata.obs.loc[non & (adata.obs['subtype'] == 'Unclassified_subtype'), 'subtype'] = 'Stromal_Unclassified'

    # Print summary
    print("Non-immune subtypes summary:")
    for subtype in ['Endothelial', 'Lymphatic_Endothelial', 'Fibroblast', 'FDC', 'Stromal_Unclassified', 'Basement Membrane']:
        print(f"    {subtype}: {(adata.obs['subtype'] == subtype).sum()}")

    return adata
