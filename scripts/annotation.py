# Canonical-marker-based GMM/positivity analysis for AnnData
import os
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq
from anndata import AnnData
import matplotlib.pyplot as plt
import seaborn as sns
import math
from collections import Counter

# Canonical marker names (without prefix)
CANONICAL_MARKERS = [
    'CCR6', 'CCR7', 'CD107a', 'CD11c', 'CD14', 'CD141', 'CD163', 'CD1c',
    'CD20', 'CD21', 'CD27', 'CD31', 'CD34', 'CD38', 'CD3e', 'CD4', 'CD40',
    'CD45', 'CD45RA', 'CD45RO', 'CD57', 'CD68', 'CD69', 'CD79a', 'CD8',
    'Collagen IV', 'CXCL13', 'DAPI', 'FOXP3', 'Granzyme B', 'HLA-DR',
    'ICOS', 'IFNG', 'Ki67', 'LYVE1', 'PD-1', 'TCF-1', 'Vimentin'
]

def fit_gmm_models(x, random_state=0):
    """Fit GMM for k=1 and k=2, return best model and component count."""
    x = np.asarray(x).flatten().reshape(-1,1)
    best_model = None
    best_bic = np.inf
    best_k = None

    for k in [1,2]:
        model = GaussianMixture(n_components=k, random_state=random_state)
        model.fit(x)
        bic = model.bic(x)
        if bic < best_bic:
            best_bic = bic
            best_model = model
            best_k = k
    return best_model, best_k

def compute_intersection(model):
    """Compute intersection point of two Gaussian components in a 2-component GMM."""
    means = model.means_.flatten()
    variances = np.array([np.atleast_1d(np.diag(cov))[0] for cov in model.covariances_])
    weights = model.weights_
    idx = np.argsort(means)
    m1, m2 = means[idx]
    v1, v2 = variances[idx]
    w1, w2 = weights[idx]

    def pdf1(x): return w1 * (1/np.sqrt(2*np.pi*v1)) * np.exp(-0.5*((x-m1)**2)/v1)
    def pdf2(x): return w2 * (1/np.sqrt(2*np.pi*v2)) * np.exp(-0.5*((x-m2)**2)/v2)
    def diff(x): return pdf1(x) - pdf2(x)

    try:
        return brentq(diff, m1, m2)
    except ValueError:
        return (m1 + m2)/2

def compute_positivity_matrix(adata: AnnData, quantile=0.95, random_state=0):
    """
    Compute boolean positivity columns for canonical markers in AnnData,
    automatically finding the correct variable column (raw_, arcsinh_cf*, or zscore_).
    Returns:
        adata (updated with *_pos columns)
        thresholds (dict, keyed by canonical marker)
        best_gmms (dict, keyed by canonical marker)
    """
    thresholds = {}
    best_gmms = {}
    bimodal_markers = []
    unimodal_markers = []

    for marker in CANONICAL_MARKERS:
        # Find the actual variable column in adata
        var_candidates = [v for v in adata.var_names if v.endswith(marker)]
        if len(var_candidates) != 1:
            print(f"Warning: marker {marker} not found uniquely in adata.var_names, skipping")
            continue
        var_name = var_candidates[0]

        x = np.asarray(adata[:, var_name].X).flatten()
        gmm, k = fit_gmm_models(x, random_state=random_state)
        best_gmms[marker] = gmm

        if k == 2:
            thr = compute_intersection(gmm)
            bimodal_markers.append(marker)
        else:
            thr = np.quantile(x, quantile)
            unimodal_markers.append(marker)

        thresholds[marker] = thr
        adata.obs[f"{marker}_pos"] = x > thr

    adata.uns['thresholds'] = thresholds
    print(f"Added {len(thresholds)} _pos columns to adata.obs")
    print(f"Unimodal markers: {unimodal_markers}")
    print(f"Bimodal markers: {bimodal_markers}")

    return adata, thresholds, best_gmms

def plot_marker_gmm_adata(adata: AnnData, thresholds: dict, gmms: dict = None,
                           title_prefix="", save=False, filename="marker_distributions_GMM.png"):
    """
    Plot histograms of canonical markers with optional GMM overlay.
    gmms: dict from compute_positivity_matrix or None (will compute automatically).
    """
    marker_names = [m for m in CANONICAL_MARKERS if m in adata.obs.columns or
                    any(v.endswith(m) for v in adata.var_names)]
    x_data = adata.X

    if gmms is None:
        gmms = {}
        for marker in marker_names:
            var_candidates = [v for v in adata.var_names if v.endswith(marker)]
            if len(var_candidates) != 1:
                continue
            var_name = var_candidates[0]
            x = np.asarray(adata[:, var_name].X).flatten()
            gmms[marker], _ = fit_gmm_models(x)

    n_cols = min(4, len(marker_names))
    n_rows = math.ceil(len(marker_names)/n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*4, n_rows*3.5), squeeze=False)
    axes = axes.flatten()

    for i, marker in enumerate(marker_names):
        var_candidates = [v for v in adata.var_names if v.endswith(marker)]
        if len(var_candidates) != 1:
            continue
        var_name = var_candidates[0]
        x = np.asarray(adata[:, var_name].X).flatten()

        sns.histplot(x, kde=False, bins=30, ax=axes[i], color="steelblue", stat="density")

        gmm = gmms[marker]
        x_vals = np.linspace(x.min(), x.max(), 1000)
        pdf_vals = np.zeros_like(x_vals)
        for w, mu, var in zip(gmm.weights_, gmm.means_.flatten(), gmm.covariances_.flatten()):
            pdf_vals += w * (1/np.sqrt(2*np.pi*var)) * np.exp(-0.5*((x_vals-mu)**2)/var)
        axes[i].plot(x_vals, pdf_vals, color="darkorange", lw=2)

        thr = thresholds.get(marker)
        if thr is not None:
            axes[i].axvline(thr, color="red", linestyle="--", lw=2)

        modality_label = 'bimodal' if gmm.n_components == 2 else 'unimodal'
        axes[i].text(0.95, 0.95, modality_label, transform=axes[i].transAxes,
                     ha='right', va='top', fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

        axes[i].set_title(f"{title_prefix}{marker}", fontsize=10)
        axes[i].set_xlabel("Value", fontsize=8)
        axes[i].set_ylabel("Density", fontsize=8)

    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    if save:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {filename}")

    return fig
