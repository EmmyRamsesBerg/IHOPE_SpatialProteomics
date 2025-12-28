import pandas as pd
import numpy as np
import anndata as ad

OBS_COLUMNS = [
    'Object ID',
    'Centroid X µm',
    'Centroid Y µm',
    'Area µm^2',
    'Nucleus/Cell area ratio'
]

def validate_dataframe(df: pd.DataFrame,
                       intensity_columns: list,
                       obs_columns: list) -> pd.DataFrame:
    """
    Validates that columns exist and contain finite values, drops invalid rows.
    """

    # Check missing columns
    missing_intensity = [c for c in intensity_columns if c not in df.columns]
    missing_obs = [c for c in obs_columns if c not in df.columns]

    if missing_intensity or missing_obs:
        raise ValueError(f"Missing columns: {missing_intensity}, {missing_obs}")

    # Validate intensities
    if not np.all(np.isfinite(df[intensity_columns].values)):
        raise ValueError("Invalid values detected in intensity columns (NaN/Inf).")

    # Validate numeric obs columns
    numeric_obs = df[obs_columns].select_dtypes(include=[np.number])
    if not np.all(np.isfinite(numeric_obs.values)):
        raise ValueError("Invalid values detected in observation columns (NaN/Inf).")

    # Drop rows with NaNs
    df_clean = df.dropna(subset=intensity_columns + obs_columns)

    return df_clean


def build_anndata(df: pd.DataFrame,
                  intensity_columns: list,
                  obs_columns: list) -> ad.AnnData:
    """
    Build an AnnData object identical to the original notebook behavior.
    """

    # X matrix
    X = df[intensity_columns].values

    # obs
    obs = df[obs_columns].copy()
    obs.index = df["Object ID"].astype(str)

    # var
    var_names = [col.replace(": Mean", "") for col in intensity_columns]
    var = pd.DataFrame(index=var_names)

    # Create AnnData
    adata = ad.AnnData(X=X, obs=obs, var=var)

    # Spatial coordinates
    adata.obsm["spatial"] = obs[["Centroid X µm", "Centroid Y µm"]].values
    adata.obs["x"] = adata.obsm["spatial"][:, 0]
    adata.obs["y"] = adata.obsm["spatial"][:, 1]

    return adata


def load_and_build_anndata(file_path: str) -> ad.AnnData:
    """
    User-facing function: read file → infer intensity columns → build AnnData.
    """

    df = pd.read_csv(file_path)

    prefixes = ['arcsinh_', 'z_', 'raw_']
    intensity_columns = [
        col for col in df.columns
        if any(col.startswith(p) for p in prefixes)
    ]

    df_clean = validate_dataframe(df, intensity_columns, OBS_COLUMNS)

    return build_anndata(df_clean, intensity_columns, OBS_COLUMNS)


def save_h5ad(adata: ad.AnnData, output_path: str):
    """Optional save function (user decides if they want it)."""
    adata.write_h5ad(output_path)
