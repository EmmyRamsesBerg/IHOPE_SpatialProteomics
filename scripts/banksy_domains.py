# bcell_follicle_analysis.py

import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt
import numpy as np


def compute_domain_bcell_stats(
    adata: AnnData,
    banksy_domain_key: str = "banksy_domain",
    bcell_key: str = "type_B",
    tcell_key: str = "type_T",
    min_cells_per_domain: int = 1,
) -> pd.DataFrame:
    """
    Compute B-cell enrichment statistics for BANKSY domains.

    Returns a dataframe with:
        domain, n_cells, frac_B, frac_T, B_T_ratio
    Sorted by frac_B descending.
    """

    if banksy_domain_key not in adata.obs:
        raise ValueError(f"{banksy_domain_key} not found in adata.obs")
    if bcell_key not in adata.obs:
        raise ValueError(f"{bcell_key} not found in adata.obs")

    stats = []
    for domain, idx in adata.obs.groupby(banksy_domain_key).groups.items():
        n_cells = len(idx)
        if n_cells < min_cells_per_domain:
            continue

        frac_b = adata.obs.loc[idx, bcell_key].mean()
        frac_t = adata.obs.loc[idx, tcell_key].mean() if tcell_key in adata.obs else None
        stats.append({
            "domain": domain,
            "n_cells": n_cells,
            "frac_B": frac_b,
            "frac_T": frac_t,
            "B_T_ratio": frac_b / frac_t if frac_t and frac_t > 0 else None,
        })

    stats_df = pd.DataFrame(stats).sort_values("frac_B", ascending=False)
    #print("BANKSY domains ranked by B-cell fraction:")
    #print(stats_df.to_string(index=False))
    return stats_df


def plot_domains_by_bcell_fraction(
    adata: AnnData,
    stats_df: pd.DataFrame,
    banksy_domain_key: str = "banksy_domain",
    size: float = 1,
    alpha: float = 0.8,
    cmap: str = "coolwarm",
    sample_name: str = ""
):
    """
    Plot cells colored by BANKSY domain.
    Domains are ranked by B-cell fraction and assigned discrete colors
    from red (high B) to blue (low B).
    Legend shows domain number and % B cells.
    """

    import numpy as np

    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    # Rank domains by B-cell fraction
    stats_df = stats_df.sort_values("frac_B", ascending=False).reset_index(drop=True)

    domains = stats_df["domain"].tolist()
    n_domains = len(domains)

    # Discrete colors sampled from colormap
    cmap_obj = plt.get_cmap(cmap)
    colors = cmap_obj(np.linspace(1, 0, n_domains))  # red -> blue

    # Map domain -> color
    domain_to_color = dict(zip(domains, colors))

    # Assign colors to cells
    domains_series = adata.obs[banksy_domain_key]

    # Ensure domains are simple hashable values
    domains_series = domains_series.astype(str)

    domain_to_color = {str(k): v for k, v in domain_to_color.items()}

    cell_colors = domains_series.map(domain_to_color)

    plt.figure(figsize=(6, 6))

    plt.scatter(
        x,
        y,
        c=list(cell_colors),
        s=size,
        alpha=alpha
    )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.title(f"{sample_name} BANKSY domains ranked by B-cell fraction")

    # Legend
    for domain, color, frac in zip(
        stats_df["domain"],
        colors,
        stats_df["frac_B"]
    ):
        label = f"Domain {domain} ({frac*100:.0f}% B-cells)"
        plt.scatter([], [], c=[color], label=label, s=30, alpha=0.9)

    plt.legend(
        markerscale=2,
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        frameon=False
    )

    plt.show()


def assign_bcell_follicles(
    adata: AnnData,
    follicle_domains: list,
    banksy_domain_key: str = "banksy_domain",
    output_key: str = "B_follicle",
) -> AnnData:

    domains = adata.obs[banksy_domain_key].astype(str)
    follicle_domains = [str(d) for d in follicle_domains]

    adata.obs[output_key] = domains.isin(follicle_domains)

    #Store metadata
    adata.uns["B_follicle_domains"] = follicle_domains

    n = adata.obs[output_key].sum()

    print(
        f"{output_key}: {n} cells "
        f"in {len(follicle_domains)} selected domains"
    )

    return adata



def plot_bcell_follicles(
    adata: AnnData,
    follicle_key: str = "B_follicle",
    bcell_key: str = "type_B",
    sample_name: str = "",
    size: float = 1,
):
    """
    Plot B-cell follicles:
    - Follicle B cells in red
    - Non-follicle B cells in blue
    - All other cells in light gray
    """

    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    plt.figure(figsize=(6, 6))

    # Background: all cells
    plt.scatter(x, y, s=size, c="lightgray", alpha=0.3, label="All cells")

    # B cells outside follicle
    b_non_follicle = adata.obs[bcell_key] & ~adata.obs[follicle_key]
    plt.scatter(x[b_non_follicle], y[b_non_follicle], s=size, c="royalblue", alpha=0.6, label="B cells (non-follicle)")

    # Follicle B cells
    b_follicle = adata.obs[bcell_key] & adata.obs[follicle_key]
    plt.scatter(x[b_follicle], y[b_follicle], s=size*3, c="crimson", alpha=0.9, label="B cells (follicle)")

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.legend(markerscale=3, frameon=False)
    plt.title(f"{sample_name}: B-cell follicles")
    plt.show()

def plot_domain_mask(adata, domain="4"):
    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    d = adata.obs["banksy_domain"].astype(str) == str(domain)

    plt.figure(figsize=(6,6))
    plt.scatter(x, y, c="lightgray", s=1, alpha=0.3)
    plt.scatter(x[d], y[d], c="black", s=2)

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.title(f"Domain {domain} (ALL cells)")
    plt.show()