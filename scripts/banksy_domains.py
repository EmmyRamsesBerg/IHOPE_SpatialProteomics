import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt


def compute_domain_bcell_stats(
    adata: AnnData,
    banksy_domain_key: str = "banksy_domain",
    bcell_key: str = "type_B",
    tcell_key: str = "type_T",
    min_cells_per_domain: int = 50,
):
    """
    Compute B-cell enrichment statistics for BANKSY domains.

    Returns a dataframe with:
        domain
        n_cells
        frac_B
        frac_T
        B_T_ratio
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

        stats.append(
            {
                "domain": domain,
                "n_cells": n_cells,
                "frac_B": frac_b,
                "frac_T": frac_t,
                "B_T_ratio": frac_b / frac_t if frac_t and frac_t > 0 else None,
            }
        )

    stats_df = pd.DataFrame(stats).sort_values("frac_B", ascending=False)

    print("BANKSY domains ranked by B-cell fraction:")
    print(stats_df.to_string(index=False))

    return stats_df


def plot_domains_by_bcell_fraction(
    adata: AnnData,
    stats_df: pd.DataFrame,
    banksy_domain_key: str = "banksy_domain",
    size: float = 1,
    alpha: float = 0.8,
    cmap: str = "coolwarm",
):
    """
    Plot spatial domains colored by B-cell fraction.
    """

    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    domain_to_frac = dict(zip(stats_df["domain"], stats_df["frac_B"]))
    colors = adata.obs[banksy_domain_key].map(domain_to_frac)

    plt.figure(figsize=(6, 6))

    sc = plt.scatter(
        x,
        y,
        c=colors,
        s=size,
        alpha=alpha,
        cmap=cmap,
        vmin=0,
        vmax=1,
    )

    plt.colorbar(sc, label="B-cell fraction")

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.title("BANKSY domains colored by B-cell fraction")

    plt.show()


def assign_bcell_follicles(
    adata: AnnData,
    follicle_domains: list,
    banksy_domain_key: str = "banksy_domain",
    output_key: str = "B_follicle",
):
    """
    Create a boolean follicle mask based on selected BANKSY domains.
    """

    adata.obs[output_key] = adata.obs[banksy_domain_key].isin(follicle_domains)

    print(
        f"{output_key}: {adata.obs[output_key].sum()} cells "
        f"in {len(follicle_domains)} domains"
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
    Visualize B-cell follicles after domain selection.
    """

    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    plt.figure(figsize=(6, 6))

    # background
    plt.scatter(
        x,
        y,
        s=size,
        c="lightgray",
        alpha=0.3,
        label="All cells",
    )

    # B cells
    b = adata.obs[bcell_key]
    plt.scatter(
        x[b],
        y[b],
        s=size,
        c="royalblue",
        alpha=0.6,
        label="B cells",
    )

    # follicles
    f = adata.obs[follicle_key]
    plt.scatter(
        x[f],
        y[f],
        s=size * 3,
        c="crimson",
        alpha=0.9,
        label="B-cell follicle",
    )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.legend(markerscale=4, frameon=False)

    plt.title(f"{sample_name}: B-cell follicles")

    plt.show()
