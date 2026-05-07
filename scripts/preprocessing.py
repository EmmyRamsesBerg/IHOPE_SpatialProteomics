# New version 2026-03-11

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Shared robust CSV reader
def robust_read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8")
    except UnicodeDecodeError:
        return pd.read_csv(path, sep=None, engine="python", encoding="cp1252")


def normalize_colnames(col: str) -> str:
    """
    Normalize column names across different exports.
    Handles micro symbols, stray spaces, and QuPath prefixes.
    """
    return (
        col.strip()
        .lstrip("\ufeff")
        .replace("Cell: ", "")
        .replace("Âµ", "µ")
        .replace("μ", "µ")
        .replace(" um", " µm")
        .replace("u m", "µm")
        .replace("um", "µm")
    )


# - Cleaning step -

def clean_cell_columns(input_csv_path: str, output_csv_path: str, encoding: str = "utf-8"):
    """
    Clean a spatial proteomics CSV by:
    - Removing 'Image', 'Name', 'Classification' columns
    - Renaming columns starting with 'Cell: ' by stripping the prefix
    - Normalizing column names
    - Standardizing micro symbols
    Saves the cleaned CSV to the output path.
    """

    try:
        df = pd.read_csv(input_csv_path, sep=None, engine="python", encoding=encoding)
    except UnicodeDecodeError:
        print(f"Failed to read {input_csv_path} as {encoding}, trying cp1252...")
        df = pd.read_csv(input_csv_path, sep=None, engine="python", encoding="cp1252")

    df.columns = [normalize_colnames(c) for c in df.columns]

    cols_to_drop = ["Image", "Name", "Classification"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Cleaned CSV saved to: {output_csv_path}")


# Main preprocessing
def preprocess(input_file: str, output_file: str | None = None, plot: bool = True):
    """
    Preprocess CSV by filtering cells based on area and DAPI intensity.
    """

    df = pd.read_csv(input_file, encoding="utf-8")

    df.columns = [normalize_colnames(c) for c in df.columns]

    area_column = "Area µm^2"
    dapi_column = "DAPI: Mean"

    if area_column not in df.columns:
        raise ValueError(f"Missing required column: {area_column}")
    if dapi_column not in df.columns:
        raise ValueError(f"Missing required column: {dapi_column}")

    filtered_df = df[(df[area_column] >= 20) & (df[area_column] <= 200)]

    filtered_df = filtered_df.dropna(subset=[dapi_column])

    low, high = np.percentile(filtered_df[dapi_column], [1, 99])

    filtered_df = filtered_df[
        (filtered_df[dapi_column] >= low) &
        (filtered_df[dapi_column] <= high)
    ]

    if plot:
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.boxplot(
            df[dapi_column].values,
            vert=False,
            showfliers=False,
        )
        plt.xlim(
            np.percentile(df[dapi_column], 0.5),
            np.percentile(df[dapi_column], 99.5),
        )
        plt.title("Original DAPI")

        plt.subplot(1, 2, 2)
        plt.hist(df[dapi_column], bins=30)

        plt.tight_layout()
        plt.show()

    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filtered{ext}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    filtered_df.to_csv(output_file, index=False)

    print(f"Filtered data saved to: {output_file}")


# --- Large file processing ---

def extract_and_filter_columns_chunked(
        input_csv_path,
        output_csv_path,
        target_columns,
        obs_columns,
        chunksize=100000,
        encoding="utf-8"
):
    """
    Memory-safe processing for very large spatial proteomics CSV files.

    Steps:
    - Read CSV in chunks
    - Normalize column names
    - Remove 'Cell: ' prefixes
    - Keep only marker + observation columns
    - Filter cells by area and DAPI
    - Append filtered rows to output CSV
    """

    target_columns = [normalize_colnames(c) for c in target_columns]
    obs_columns = [normalize_colnames(c) for c in obs_columns]

    keep_columns = target_columns + obs_columns

    area_column = normalize_colnames("Area µm^2")
    dapi_column = "DAPI: Mean"

    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    first_chunk = True

    reader = pd.read_csv(
        input_csv_path,
        chunksize=chunksize,
        encoding=encoding
    )

    for chunk in reader:

        chunk.columns = [normalize_colnames(c) for c in chunk.columns]

        chunk = chunk[[c for c in keep_columns if c in chunk.columns]]

        if area_column not in chunk.columns:
            raise ValueError(f"Missing required column: {area_column}")

        if dapi_column not in chunk.columns:
            raise ValueError(f"Missing required column: {dapi_column}")

        filtered = chunk[
            (chunk[area_column] >= 20) &
            (chunk[area_column] <= 200)
        ]

        filtered = filtered.dropna(subset=[dapi_column])

        if len(filtered) == 0:
            continue

        low, high = np.percentile(filtered[dapi_column], [1, 99])

        filtered = filtered[
            (filtered[dapi_column] >= low) &
            (filtered[dapi_column] <= high)
        ]

        filtered.to_csv(
            output_csv_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print(f"Processed CSV saved to: {output_csv_path}")
