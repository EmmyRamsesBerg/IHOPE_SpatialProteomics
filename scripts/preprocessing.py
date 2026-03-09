# New version 2026-02-13

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


def normalize_micro(col: str) -> str:
    return (
        col
        .replace("Âµ", "µ")
        .replace(" um", " µm")
        .replace("u m", "µm")
        .replace("um", "µm")
    )


# - Cleaning step -

import os
import pandas as pd

def clean_cell_columns(input_csv_path: str, output_csv_path: str, encoding: str = "utf-8"):
    """
    Clean a spatial proteomics CSV by:
    - Removing 'Image', 'Name', 'Classification' columns
    - Renaming columns starting with 'Cell: ' by stripping the prefix
    - Normalizing column names
    - Standardizing micro symbols
    Saves the cleaned CSV to the output path.
    """

    #  Load CSV with encoding fallback
    try:
        df = pd.read_csv(input_csv_path, sep=None, engine="python", encoding=encoding)
    except UnicodeDecodeError:
        print(f"Failed to read {input_csv_path} as {encoding}, trying cp1252...")
        df = pd.read_csv(input_csv_path, sep=None, engine="python", encoding="cp1252")

    #  Normalize column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lstrip("\ufeff")
        .str.replace("Âµ", "µ", regex=False)
        .str.replace("μ", "µ", regex=False)
    )

    #  Drop unwanted columns
    cols_to_drop = ["Image", "Name", "Classification"]
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    #  Strip 'Cell:' prefix
    df.columns = df.columns.str.replace(r"^Cell:\s*", "", regex=True)

    #  Ensure output directory exists
    output_dir = os.path.dirname(output_csv_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created directory: {output_dir}")

    # Save as UTF-8
    df.to_csv(output_csv_path, index=False, encoding="utf-8")
    print(f"Cleaned CSV saved to: {output_csv_path}")



# Main preprocessing

def preprocess(input_file: str, output_file: str | None = None, plot: bool = True):
    """
    Preprocess CSV by filtering cells based on area and DAPI intensity.
    """
    df = pd.read_csv(input_file, encoding="utf-8")

    area_column = "Area µm^2"
    dapi_column = "DAPI: Mean"

    if area_column not in df.columns:
        raise ValueError(f"Missing required column: {area_column}")
    if dapi_column not in df.columns:
        raise ValueError(f"Missing required column: {dapi_column}")

    # Area filter
    filtered_df = df[(df[area_column] >= 20) & (df[area_column] <= 200)]

    # DAPI filter
    filtered_df = filtered_df.dropna(subset=[dapi_column])

    low, high = np.percentile(filtered_df[dapi_column], [1, 99])
    filtered_df = filtered_df[
        (filtered_df[dapi_column] >= low) &
        (filtered_df[dapi_column] <= high)
    ]

    # Plotting
    if plot:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        #plt.boxplot(df[dapi_column], vert=False)
        plt.boxplot( #start addition
            df[dapi_column].values,
            vert=False,
            showfliers=False,
        )
        plt.xlim(
            np.percentile(df[dapi_column], 0.5),
            np.percentile(df[dapi_column], 99.5),
        ) #end addition
        plt.title("Original DAPI")
        plt.subplot(1, 2, 2)
        plt.hist(df[dapi_column], bins=30)
        plt.tight_layout()
        plt.show()

    # Output path
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filtered{ext}"

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to: {output_file}")

def preprocess_chunked(input_file: str,
                       output_file: str,
                       chunksize: int = 100000):
    """
    Memory-safe filtering of large CSV files.
    Applies the same filtering as preprocess(), but in chunks.
    """

    import os
    import numpy as np
    import pandas as pd

    area_column = "Area µm^2"
    dapi_column = "DAPI: Mean"

    if os.path.exists(output_file):
        os.remove(output_file)

    first_chunk = True

    reader = pd.read_csv(input_file, encoding="utf-8", chunksize=chunksize)

    for chunk in reader:

        if area_column not in chunk.columns:
            raise ValueError(f"Missing required column: {area_column}")
        if dapi_column not in chunk.columns:
            raise ValueError(f"Missing required column: {dapi_column}")

        # Area filter
        filtered_df = chunk[
            (chunk[area_column] >= 20) &
            (chunk[area_column] <= 200)
        ]

        # DAPI filter
        filtered_df = filtered_df.dropna(subset=[dapi_column])

        if len(filtered_df) == 0:
            continue

        low, high = np.percentile(filtered_df[dapi_column], [1, 99])

        filtered_df = filtered_df[
            (filtered_df[dapi_column] >= low) &
            (filtered_df[dapi_column] <= high)
        ]

        filtered_df.to_csv(
            output_file,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False
        )

        first_chunk = False

    print(f"Filtered data saved to: {output_file}")


def clean_cell_columns_chunked(input_csv_path: str,
                               output_csv_path: str,
                               chunksize: int = 100000,
                               encoding: str = "utf-8"):
    """
    Memory-safe version of clean_cell_columns for very large CSV files.
    Processes the file in chunks.
    """

    import os
    import pandas as pd

    # Remove output file if it already exists
    if os.path.exists(output_csv_path):
        os.remove(output_csv_path)

    first_chunk = True

    try:
        reader = pd.read_csv(
            input_csv_path,
            sep=None,
            engine="python",
            encoding=encoding,
            chunksize=chunksize
        )
    except UnicodeDecodeError:
        reader = pd.read_csv(
            input_csv_path,
            sep=None,
            engine="python",
            encoding="cp1252",
            chunksize=chunksize
        )

    for chunk in reader:

        # Normalize column names
        chunk.columns = (
            chunk.columns
            .str.strip()
            .str.lstrip("\ufeff")
            .str.replace("Âµ", "µ", regex=False)
            .str.replace("μ", "µ", regex=False)
        )

        # Drop unwanted columns
        cols_to_drop = ["Image", "Name", "Classification"]
        chunk = chunk.drop(columns=[c for c in cols_to_drop if c in chunk.columns])

        # Strip 'Cell:' prefix
        chunk.columns = chunk.columns.str.replace(r"^Cell:\s*", "", regex=True)

        chunk.to_csv(
            output_csv_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8"
        )

        first_chunk = False

    print(f"Chunk-cleaned CSV saved to: {output_csv_path}")
