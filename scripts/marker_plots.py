import matplotlib.pyplot as plt

def plot_marker_axes(
    adata: AnnData,
    x_marker: str,
    y_marker: str,
    base_mask: str,
    show_thresholds: bool = True,
    size: float = 3,
    alpha: float = 0.3,
    title: str | None = None,
    show_background: bool = True,
    clip_percentile: float = 99.5,
):
    x_var = f"arcsinh_cf5.0_{x_marker}"
    y_var = f"arcsinh_cf5.0_{y_marker}"

    if x_var not in adata.var_names or y_var not in adata.var_names:
        raise ValueError("Marker not found in adata.var_names")
    if base_mask not in adata.obs:
        raise ValueError("Base mask not found in adata.obs")

    x_idx = adata.var_names.get_loc(x_var)
    y_idx = adata.var_names.get_loc(y_var)

    mask = adata.obs[base_mask].values

    x_all = adata.X[:, x_idx]
    y_all = adata.X[:, y_idx]
    x = x_all[mask]
    y = y_all[mask]

    plt.figure(figsize=(5, 5))

    # Background
    if show_background:
        plt.scatter(
            x_all,
            y_all,
            s=1,
            alpha=0.05,
            color="lightgrey",
            label="All cells",
        )

    # Foreground
    plt.scatter(
        x,
        y,
        s=size,
        alpha=alpha,
        color="crimson",
        label=base_mask,
    )

    # Thresholds
    if show_thresholds:
        thr = adata.uns.get("thresholds", {})

        if show_thresholds:
            thr = adata.uns.get("thresholds", {})
            if x_marker in thr:
                plt.axvline(thr[x_marker], linestyle="--", color="blue", label=f"{x_marker} threshold")
            if y_marker in thr:
                plt.axhline(thr[y_marker], linestyle="--", color="yellow", label=f"{y_marker} threshold")

    # Robust axis limits
    xmax = np.percentile(x_all, clip_percentile)
    ymax = np.percentile(y_all, clip_percentile)
    plt.xlim(0, xmax)
    plt.ylim(0, ymax)

    n = mask.sum()
    total = adata.n_obs

    plt.xlabel(x_marker)
    plt.ylabel(y_marker)
    plt.title(
        title and f"{base_mask}: {n} / {total} cells"
    )
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()
