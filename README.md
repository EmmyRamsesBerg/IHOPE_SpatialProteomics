# IHOPE Spatial Proteomics Analysis

This is a workflow for spatial proteomics analysis of human lymphoid tissue under the IHOPE project. 
Here, a 37-marker plus DAPI panel was imaged on the PhenoCycler Fusion (Akoya Biosciences) platform.
There were 14 samples from 5 donors.
Tissues: mediastinal lymph node (MedLN), mesenteric lymph node (MesLN), spleen

## Pipeline overview

- Step 0 (before this workflow): segmentation with InstanSeg in QuPath, one CSV per sample 
- Step 1: preprocessing, drop unused columns, filter to 1st to 99th percentile DAPI intensity, keep cells 20 to 200 square micrometers
- Step 2: normalization, z-score after log2 transform (optional, can also use arcsinh or none)
- Step 3: build AnnData object
- Step 4: marker positivity thresholds, 2-component GMM or percentile cutoff, chosen per marker based on BIC
- Step 5: rule-based cell typing using cell type marker positivity/negativity from canonical immunology 
- Step 6: BANKSY spatial domain analysis, then manual annotation of B cell dense domains as follicles
- Step 7: spatially constrained cell types added (TfH and GC B inside follicle domains, plasmablasts outside)
- Step 8: DBSCAN follicle detection and counting, follicle counts normalized per sample
- Step 9: tissue-level comparison and figure generation, exploratory analysis

## Folder structure

Change as desired, may require additional changes in scripts/notebooks

```
IHOPE_SpatialProteomics/
├── notebooks/             
│   ├── batch_celltyping_IHOPE.ipynb         
│   ├── figures_svg_export.ipynb
│   ├── Follicle_counting.ipynb
│   ├── IHOPE_spatialplots.ipynb
│   ├── rulebased_celltyping_IHOPE.ipynb 
│   ├── rulebased_celltyping_with_banksy.ipynb 
│   └── Sample_comparison.ipynb                       
├── scripts/
│   ├── celltype_config.py              
│   ├── celltype_rules_IHOPE.py         
│   ├── preprocessing.py
│   ├── transforms.py
│   ├── anndata_helpers.py
│   ├── annotation.py
│   ├── banksy_domains.py
│   ├── follicle_counting_helpers.py
│   ├── summary_celltypes_IHOPE.py
│   ├── comparison.py
│   ├── differential_screen.py
│   └── spatial_plotting.py
├── data/                   
│   ├── raw/                # Segmented PhenoCycler data in CSV format
│   └── processed/          # Cleaned and filtered data  
│       └── anndata/        
│           └── zscore_log2/
│               └── celltyped/
│                   └── follicledomains/
└── results/
    ├── reports/
    │   ├── zscore_log2/
    │   └── follicle_counts/
    └── figures/
        └── vector/
```

## Notebooks

- rulebased_celltyping_IHOPE.ipynb, single sample, from segmented single cell CSV through cell typing
- batch_celltyping_IHOPE.ipynb, same steps looped over all samples in a folder (defined in celltype_config.py)
- rulebased_celltyping_with_banksy.ipynb, BANKSY domains, manual follicle annotation, spatially constrained cell types
- Sample_comparison_IHOPE.ipynb, main comparison notebook, heatmaps, barplots, strip plots, differential screen, follicle screen
- figures_svg_export.ipynb, trimmed notebook, exports a fixed set of figures as SVG to results/figures/vector
- Follicle_counting.ipynb, DBSCAN follicle detection and counting, manual cluster exclusion, size metrics
- IHOPE_spatialplots.ipynb, spatial plots for markers/cell types, one or two samples at a time, QC and figure panels

## Shared configuration

- scripts/celltype_config.py, single source for sample/donor/tissue metadata, cell type colors, display names, plot order
- scripts/celltype_rules_IHOPE.py, gating logic for cell typing
