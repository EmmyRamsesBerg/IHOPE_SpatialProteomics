import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    print(f"✔ Filtered data saved to: {output_file}")
