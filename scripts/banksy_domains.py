import pandas as pd
from anndata import AnnData
import matplotlib.pyplot as plt

def identify_bcell_follicles(
    adata: AnnData,
    banksy_domain_key: str = "banksy_domain",
    bcell_key: str = "type_B",
    min_fraction_bcells: float = 0.3,
    min_cells_per_domain: int = 50,
    output_key: str = "B_follicle",
):
    """
    Identify B-cell–enriched BANKSY domains (putative follicles).

    Adds a boolean column `output_key` to adata.obs.
    """

    if banksy_domain_key not in adata.obs:
        raise ValueError(f"{banksy_domain_key} not found in adata.obs")

    if bcell_key not in adata.obs:
        raise ValueError(f"{bcell_key} not found in adata.obs")

    domain_stats = []

    for domain, idx in adata.obs.groupby(banksy_domain_key).groups.items():
        n_cells = len(idx)
        if n_cells < min_cells_per_domain:
            continue

        frac_b = adata.obs.loc[idx, bcell_key].mean()

        domain_stats.append(
            {
                "domain": domain,
                "n_cells": n_cells,
                "frac_B": frac_b,
            }
        )

    stats_df = pd.DataFrame(domain_stats)

    follicle_domains = stats_df.loc[
        stats_df["frac_B"] >= min_fraction_bcells, "domain"
    ].tolist()

    adata.obs[output_key] = adata.obs[banksy_domain_key].isin(follicle_domains)

    print(
        f"Identified {len(follicle_domains)} B-cell–enriched domains "
        f"(≥{min_fraction_bcells:.0%} B cells)"
    )

    return adata, stats_df


def plot_banksy_domains(
    adata,
    banksy_domain_key="banksy_domain",
    size=1,
    alpha=0.8,
):
    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    domains = adata.obs[banksy_domain_key].astype("category")

    plt.figure(figsize=(6, 6))
    plt.scatter(
        x,
        y,
        c=domains.cat.codes,
        s=size,
        alpha=alpha,
        cmap="tab20",
    )
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.title("BANKSY spatial domains")
    plt.show()

def plot_bcell_follicles(
    adata,
    follicle_key="B_follicle",
    bcell_key="type_B",
    banksy_domain_key="banksy_domain",
    sample_name=str,
    size=1,
):
    x = adata.obsm["spatial"][:, 0]
    y = adata.obsm["spatial"][:, 1]

    plt.figure(figsize=(6, 6))

    # Background: all cells in light gray
    plt.scatter(
        x,
        y,
        s=size,
        c="lightgray",
        alpha=0.3,
        label="All cells",
    )

    # Overlay: B cells
    b = adata.obs[bcell_key]
    plt.scatter(
        x[b],
        y[b],
        s=size,
        c="royalblue",
        alpha=0.6,
        label="B cells",
    )

    # Overlay: follicle cells (thick outline)
    # Overlay: follicle cells, colored by BANKSY domain
    follicle_domains = (
        adata.obs.loc[adata.obs[follicle_key], banksy_domain_key]
        .astype("category")
        .cat.remove_unused_categories()
    )

    domain_ids = follicle_domains.cat.categories
    cmap = plt.get_cmap("tab10")

    for i, domain in enumerate(domain_ids):
        idx = (
                (adata.obs[banksy_domain_key] == domain)
                & adata.obs[follicle_key]
        )

        plt.scatter(
            x[idx],
            y[idx],
            s=size * 3,
            color=cmap(i % cmap.N),
            alpha=0.9,
            label=f"Follicle domain {domain}",
        )

    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.axis("off")
    plt.legend(markerscale=4, frameon=False)
    plt.title(f"{sample_name}: BANKSY-identified B-cell follicles")
    plt.show()
