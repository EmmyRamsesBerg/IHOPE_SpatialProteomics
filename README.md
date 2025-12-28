# IHOPE_SpatialProteomics
Workflow to load spatial proteomics data from .csv table, filter and normalize, and visualize cell types and marker intensities. 
The workflow contains the following steps:

1. Loading of raw data from segmented image in csv format, filtering by size and DAPI intensity
2. Transformation applied to cleaned .csv, method of choice with options none/z-score/arcsinh + choice of cofactor, returns df_out, transformed_cols, metadata_cols, fig. 
3. Build AnnData object using input file (from step 2),  optional ‘uns’ addition for notes. adata.var contains the transformed (raw/arcsinh/zscore) intensities. 
4. Marker positivity annotation using GMMs, boolean matrix and thresholds added to AnnData. Markers deemed bimodal get GMM thresholding, unimodal get a percentile cutoff (BIC-based modality)
5. Assign cell types based on crude definitions from marker positivity combinations. Levels ‘branch’, ‘type’ and ‘subtype’ will get a column each in adata, with different identities depending on rule fulfilment. 
6. Clustering with PCA/Leiden, followed by an example of cell type annotation of the resulting clusters.
7. Comparison between cell type annotations (contingency tables, Pearson correlation and Chi-squared tests).
8. A sketch for BANKSY analysis and analysis of the banksy domains. Gene overrepresentation analysis where adata.uns["rank_genes_groups"] stores, for each domain, the ranked marker names, associated test statistics, log fold changes, p-values, and test parameters from differential expression analysis as a dictionary. Summary of manually annotated cell types in each BANKSY domain. 

The notebook IHOPE_Workflow.ipynb should be sufficient to walk through the whole workflow, granted that all scripts are in place within the project repository. File paths/names may need to be changed. The user needs to use their own spatial proteomics data. 
