import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#TODO: check if this function is OK in the long run.
#def clean_cell_columns(input_csv_path: str, output_csv_path: str):
#    """
#    Clean a spatial proteomics CSV by:
#    - Removing 'Image', 'Name', 'Classification' columns
#    - Keeping all other columns
#    - Renaming columns starting with 'Cell: ' by stripping the prefix
#    Saves the cleaned CSV to the output path.
#    """
#    # Load CSV
#    df = pd.read_csv(input_csv_path, sep=None, engine='python') #TODO maybe not hardcoded but whatever
#
 #   # Drop columns we don't want
  #  cols_to_drop = ['Image', 'Name', 'Classification']
   # df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
#
 #   df.columns = df.columns.str.replace(r'^Cell: \s*', '', regex=True)  # remove 'Cell: ' from name
  #  print(df.head()) #TODO remove
#
 #   # Save cleaned CSV
  #  df.to_csv(output_csv_path, index=False)
   # print(f"Cleaned CSV saved to: {output_csv_path}")

#TODO new ver
import pandas as pd

def clean_cell_columns(input_csv_path: str, output_csv_path: str, encoding: str = "utf-8"):
    """
    Clean a spatial proteomics CSV by:
    - Removing 'Image', 'Name', 'Classification' columns
    - Keeping all other columns
    - Renaming columns starting with 'Cell: ' by stripping the prefix
    - Normalizing column names to remove extra whitespace
    Saves the cleaned CSV to the output path.
    """
    # Load CSV
    try:
        df = pd.read_csv(input_csv_path, sep=None, engine='python', encoding=encoding)
    except UnicodeDecodeError:
        # fallback for Windows Excel files
        print(f"Failed to read {input_csv_path} as {encoding}, trying cp1252...")
        df = pd.read_csv(input_csv_path, sep=None, engine='python', encoding="cp1252")

    # Normalize column names
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.lstrip("\ufeff") #this might be a bad idea

    # Drop columns we don't want
    cols_to_drop = ['Image', 'Name', 'Classification']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Remove 'Cell: ' prefix from relevant columns
    df.columns = df.columns.str.replace(r'^Cell:\s*', '', regex=True)

    # Save cleaned CSV
    df.to_csv(output_csv_path, index=False, encoding="utf-8")  # always UTF-8
    print(f"Cleaned CSV saved to: {output_csv_path}")


def preprocess(input_file: str, output_file: str | None = None, plot: bool = True):
    """
    Preprocess CSV by filtering cells based on area and DAPI intensity.
    Saves filtered data to CSV.

    Parameters:
        input_file: str, path to the input CSV file
        output_file: str, optional path for output; if None, defaults to inputname + '_filtered.csv'
        plot: bool, whether to show DAPI boxplots and histograms
    """

    # Load data
    df = pd.read_csv(input_file)

    # --- Filter by area ---
    area_column = "Area µm^2"
    filtered_df = df[(df[area_column] >= 20) & (df[area_column] <= 200)]

    # --- Filter by DAPI ---
    dapi_column = "DAPI: Mean"
    if dapi_column not in filtered_df.columns:
        raise ValueError(f"Column '{dapi_column}' not found in the data.")

    filtered_df = filtered_df.dropna(subset=[dapi_column])

    low_thresh = np.percentile(filtered_df[dapi_column], 1)
    high_thresh = np.percentile(filtered_df[dapi_column], 99)

    filtered_df = filtered_df[
        (filtered_df[dapi_column] >= low_thresh) &
        (filtered_df[dapi_column] <= high_thresh)
    ]

    # --- Plotting ---
    if plot:
        # Original DAPI
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.boxplot(df[dapi_column], vert=False)
        plt.title("Original DAPI Boxplot")
        plt.subplot(1, 2, 2)
        plt.hist(df[dapi_column], bins=30, edgecolor='k')
        plt.title("Original DAPI Histogram")
        plt.tight_layout()
        plt.show()

        # Filtered DAPI
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.boxplot(filtered_df[dapi_column], vert=False)
        plt.title("Filtered DAPI Boxplot")
        plt.subplot(1, 2, 2)
        plt.hist(filtered_df[dapi_column], bins=30, edgecolor='k')
        plt.title("Filtered DAPI Histogram")
        plt.tight_layout()
        plt.show()

    # --- Determine output file path ---
    if output_file is None:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_filtered{ext}"

    # --- Save filtered CSV ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    filtered_df.to_csv(output_file, index=False)
    print(f"Filtered data saved to: {output_file}")
