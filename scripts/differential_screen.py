"""
differential_screen.py

Descriptive screening for the IHOPE MesLN vs MedLN comparison.

Ranks cell types and markers by how differently they behave between two
tissue groups, so a clear candidate can be picked for a side by side spatial
display. The numbers here are a sorting key, not an inferential claim. With
two donors per tissue arm the exact p values are floored near 0.33 and are
reported only as an accessory, never as the sort key.

Aggregation matches the rest of the project. Replicate tissue pieces are
pooled at the cell level within each donor before any percentage is formed,
so a donor contributes exactly one value per feature regardless of how many
pieces it has.

Two feature sets run through the same screen
    cell type abundance   from the celltype_summary CSVs, at type,
                          intermediate and subtype level, optionally parent
                          relative
    marker positivity     from a skinny per sample positivity table extracted
                          from the h5ad files in backed mode
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D
from matplotlib.gridspec import GridSpec
from scipy.stats import mannwhitneyu


# ----------------------------------------------------------------------
# basename parsing
# ----------------------------------------------------------------------

def _donor_from_basename(basename):
    """Donor id is the token before the first underscore, for example IHOPE14."""
    return basename.split("_")[0]


def _tissue_from_basename(basename):
    """Tissue is whichever known tag appears in the basename."""
    for tag in ("MedLN", "MesLN", "Spleen"):
        if tag in basename:
            return tag
    raise ValueError(f"No known tissue tag found in basename '{basename}'.")


# ----------------------------------------------------------------------
# cell type abundance, loading and donor level pooling
# ----------------------------------------------------------------------

def load_celltype_long(reports_dir, basenames, prefix="celltype_summary_"):
    """
    Load the per sample cell type summary CSVs into one long dataframe.

    Reads one small CSV per basename and attaches sample, donor and tissue.
    Donor and tissue are parsed from the basename directly, so no external
    name map is needed and the spacing of the basename does not matter.

    Returns the original columns plus sample, donor, tissue.
    """
    reports_dir = Path(reports_dir)
    frames = []
    for basename in basenames:
        path = reports_dir / f"{prefix}{basename}.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        df = pd.read_csv(path)
        required = {"level", "cell_type", "column", "n_cells", "pct_total", "total_cells"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{path} is missing columns {missing}")
        df["sample"] = basename
        df["donor"] = _donor_from_basename(basename)
        df["tissue"] = _tissue_from_basename(basename)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def pool_to_donor_level(long_df):
    """
    Pool replicate pieces at the cell level within each donor and tissue.

    For every (donor, tissue, level, cell_type, column) the cell counts are
    summed across the donor's pieces, and the denominator is the sum of the
    per piece total_cells for that donor and tissue. The donor level
    percentage is pooled_n / pooled_total * 100, which is cell level pooling
    rather than an average of per piece percentages.

    Returns one row per (donor, tissue, level, cell_type, column) with
    n_cells (pooled), total_cells (pooled) and pct (donor level percentage).
    """
    # one total per file, summed across the pieces of each donor and tissue
    sample_totals = long_df.drop_duplicates("sample")[
        ["donor", "tissue", "sample", "total_cells"]
    ]
    dt_totals = (
        sample_totals.groupby(["donor", "tissue"])["total_cells"]
        .sum()
        .rename("pooled_total")
        .reset_index()
    )

    pooled_n = (
        long_df.groupby(["donor", "tissue", "level", "cell_type", "column"])["n_cells"]
        .sum()
        .rename("n_cells")
        .reset_index()
    )

    out = pooled_n.merge(dt_totals, on=["donor", "tissue"], how="left")
    out["total_cells"] = out["pooled_total"]
    out["pct"] = out["n_cells"] / out["pooled_total"] * 100
    return out.drop(columns="pooled_total")


def add_parent_relative(donor_level, parent_of):
    """
    Add a parent relative percentage column to the donor level table.

    For each row whose cell_type is a key in parent_of, pct_parent is the
    donor level child percentage divided by the same donor and tissue parent
    percentage, times 100. Rows with no parent entry, or whose parent is
    absent or zero for that donor and tissue, get NaN.

    parent_of maps a cleaned cell_type name to its parent cleaned cell_type
    name, for example {"CD8_T": "T", "TEM_CD8": "CD8_T"}. Biology lives in the
    notebook, this only applies the mapping. Parent names are assumed unique
    per donor and tissue, which holds because parents are type or intermediate
    level names.
    """
    parent_names = set(parent_of.values())
    denom_df = donor_level[donor_level["cell_type"].isin(parent_names)]
    pct_lookup = denom_df.set_index(["donor", "tissue", "cell_type"])["pct"]

    def _parent_pct(row):
        parent = parent_of.get(row["cell_type"])
        if parent is None:
            return np.nan
        denom = pct_lookup.get((row["donor"], row["tissue"], parent), np.nan)
        if denom is None or (isinstance(denom, float) and np.isnan(denom)) or denom == 0:
            return np.nan
        return row["pct"] / denom * 100

    out = donor_level.copy()
    out["pct_parent"] = out.apply(_parent_pct, axis=1)
    return out


def exclude_unresolved(donor_level, suffixes=("_unclassified", "_unassigned")):
    """
    Drop unclassified and unassigned rows and add a per level classified pct.

    Both are removed from the table. Unlike unclassified, which is a single
    type level partition, unassigned is level specific, a cell can be assigned
    at the type level yet unassigned at the subtype level, so there is no
    single base across levels. The denominator is therefore computed per donor,
    tissue and level as the summed n_cells over the kept rows at that level.
    Every level then sums to 100 over its kept rows, matching the per level
    rescaling in comparison.py.

    Note the intermediate level is not a partition, its calls overlap (a cell
    is both CD4_T and T_naive), so its pct_resolved denominator double counts
    across overlapping calls. Read intermediates from the parent relative
    screen rather than pct_resolved. Type and subtype are genuine partitions
    and rescale cleanly.

    Adds pct_resolved as a new column and leaves pct, n_cells and total_cells
    untouched so the percentage of all cells stays available. Parent relative
    values are unaffected since they read the untouched pct column.
    """
    drop_mask = donor_level["column"].str.endswith(suffixes)
    out = donor_level[~drop_mask].copy()

    level_base = (
        out.groupby(["donor", "tissue", "level"])["n_cells"]
        .transform("sum")
    )
    out["pct_resolved"] = out["n_cells"] / level_base * 100
    return out


# ----------------------------------------------------------------------
# marker positivity, extraction and donor level pooling
# ----------------------------------------------------------------------

def extract_marker_positivity(
    anndata_dir,
    basenames,
    out_csv,
    suffix="_celltypes_follicledomains.h5ad",
    pos_suffix="_pos",
):
    """
    Build a skinny per sample marker positivity table from the h5ad files.

    Each file is opened in backed mode so the expression matrix stays on disk
    and only obs is read into memory. Every boolean column ending in
    pos_suffix is treated as a marker positivity call, and its positive count
    and the file cell count are recorded. Fractions and pooling are left to
    the screening step so this file stays a raw count table.

    Writes a long CSV with columns sample, marker, n_pos, n_cells.
    """
    import scanpy as sc

    anndata_dir = Path(anndata_dir)
    out_csv = Path(out_csv)
    frames = []
    for basename in basenames:
        path = anndata_dir / f"{basename}{suffix}"
        if not path.exists():
            raise FileNotFoundError(path)
        adata = sc.read_h5ad(path, backed="r")
        obs = adata.obs
        n_cells = int(obs.shape[0])
        pos_cols = [
            c for c in obs.columns
            if c.endswith(pos_suffix) and obs[c].dtype == bool
        ]
        recs = [
            {
                "sample": basename,
                "marker": c[: -len(pos_suffix)],
                "n_pos": int(obs[c].to_numpy(dtype=bool).sum()),
                "n_cells": n_cells,
            }
            for c in pos_cols
        ]
        frames.append(pd.DataFrame(recs))
        # release the backed file handle before the next file
        if getattr(adata, "isbacked", False):
            adata.file.close()
        del adata
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(
        f"Wrote {out_csv} with {out.shape[0]} rows "
        f"({out['sample'].nunique()} samples, {out['marker'].nunique()} markers)"
    )
    return out


def marker_to_donor_level(positivity_csv):
    """
    Load the skinny positivity CSV and pool to donor level.

    Attaches donor and tissue from the sample name, sums positive and total
    counts across the pieces of each donor and tissue, and forms the donor
    level positive fraction as a percentage. Returns a frame shaped like the
    cell type donor level table (level, cell_type, column, pct) so the same
    screen runs on it, with level set to "marker".
    """
    df = pd.read_csv(positivity_csv)
    df["donor"] = df["sample"].map(_donor_from_basename)
    df["tissue"] = df["sample"].map(_tissue_from_basename)

    dt = (
        df.groupby(["donor", "tissue", "marker"])[["n_pos", "n_cells"]]
        .sum()
        .reset_index()
    )
    dt["pct"] = dt["n_pos"] / dt["n_cells"] * 100
    dt["level"] = "marker"
    dt["cell_type"] = dt["marker"]
    dt["column"] = dt["marker"] + "_pos"
    dt = dt.rename(columns={"n_cells": "total_cells"})
    return dt[
        ["donor", "tissue", "level", "cell_type", "column", "n_pos", "total_cells", "pct"]
    ]


# ----------------------------------------------------------------------
# the screen
# ----------------------------------------------------------------------

def _cliffs_delta(b, a):
    """
    Cliff's delta for group b relative to group a. Positive means b tends to
    sit above a. Ranges from minus one to one.
    """
    b = np.asarray(b, dtype=float)
    a = np.asarray(a, dtype=float)
    if b.size == 0 or a.size == 0:
        return np.nan
    gt = sum((bi > ai) for bi in b for ai in a)
    lt = sum((bi < ai) for bi in b for ai in a)
    return (gt - lt) / (b.size * a.size)


def screen(
    donor_level,
    group_a="MesLN",
    group_b="MedLN",
    value_col="pct",
    paired_donors=None,
    pseudocount=0.01,
    fill_absent_zero=None,
):
    """
    Rank features by fold change between two tissue groups.

    group_b is the numerator, so a positive log2 fold change and a positive
    difference mean higher in group_b. Only donors present in both tissues are
    used, so both sides rest on the same donors and each yields a within donor
    paired difference.

    Parameters
    ----------
    donor_level : DataFrame
        Donor level table with columns level, cell_type, column, donor,
        tissue and the value_col. Produced by pool_to_donor_level (optionally
        add_parent_relative) or marker_to_donor_level.
    group_a, group_b : str
        Tissue names. group_b is the fold change numerator.
    value_col : str
        Which column to screen, "pct" for percentage of all cells, "pct_parent"
        for percentage of parent (needs add_parent_relative first), or "pct"
        on a marker table for positivity.
    paired_donors : list[str] or None
        Restrict to these donors. None auto-detects donors present in both
        tissues.
    pseudocount : float
        Added to both group means before the log2 ratio so a zero denominator
        does not blow up. Read the plain difference alongside the fold change,
        since a rare type present in only one tissue still gives a large ratio.
    fill_absent_zero : bool or None
        Whether a feature absent for a donor and tissue counts as zero. None
        defaults to True for value_col "pct" (absence means no cells) and False
        for "pct_parent" (a missing parent is genuinely undefined, not zero).

    Returns
    -------
    DataFrame, one row per feature, sorted by absolute log2 fold change
    descending. Columns include the two group means, the difference, the log2
    fold change, one paired difference per donor, a flag for whether all paired
    differences agree in direction with the group difference, Cliff's delta and
    the exact Mann-Whitney p value.
    """
    if fill_absent_zero is None:
        fill_absent_zero = value_col == "pct"

    keys = ["level", "cell_type", "column"]
    d = donor_level[donor_level["tissue"].isin([group_a, group_b])].copy()

    have = d.groupby("donor")["tissue"].agg(set)
    both = [dn for dn, ts in have.items() if {group_a, group_b} <= ts]
    if paired_donors is None:
        paired_donors = sorted(both)
    else:
        paired_donors = [dn for dn in paired_donors if dn in both]
    if not paired_donors:
        raise ValueError(
            f"No donors have both {group_a} and {group_b}. Found tissue sets: "
            f"{dict(have)}"
        )

    d = d[d["donor"].isin(paired_donors)]

    a_tbl = d[d["tissue"] == group_a].pivot_table(
        index="donor", columns=keys, values=value_col
    )
    b_tbl = d[d["tissue"] == group_b].pivot_table(
        index="donor", columns=keys, values=value_col
    )

    feats = a_tbl.columns.union(b_tbl.columns)
    a_tbl = a_tbl.reindex(index=paired_donors, columns=feats)
    b_tbl = b_tbl.reindex(index=paired_donors, columns=feats)
    if fill_absent_zero:
        a_tbl = a_tbl.fillna(0.0)
        b_tbl = b_tbl.fillna(0.0)

    records = []
    for feat in feats:
        a_col = a_tbl[feat]
        b_col = b_tbl[feat]

        a_vals = a_col.dropna()
        b_vals = b_col.dropna()
        if a_vals.empty or b_vals.empty:
            continue

        a_mean = float(a_vals.mean())
        b_mean = float(b_vals.mean())
        diff = b_mean - a_mean
        log2fc = float(
            np.log2((b_mean + pseudocount) / (a_mean + pseudocount))
        )

        # per donor paired values where the donor has both tissues:
        # the plain difference and the log2 fold change, both group_b vs group_a
        paired = {}
        paired_log2fc = {}
        paired_abund = {}
        for dn in paired_donors:
            av, bv = a_col.get(dn), b_col.get(dn)
            if pd.notna(av) and pd.notna(bv):
                paired[dn] = float(bv - av)
                paired_log2fc[dn] = float(
                    np.log2((bv + pseudocount) / (av + pseudocount))
                )
                paired_abund[dn] = max(float(av), float(bv))

        group_sign = np.sign(diff)
        agree = bool(
            paired
            and group_sign != 0
            and all(np.sign(v) == group_sign for v in paired.values())
        )

        delta = _cliffs_delta(b_vals.to_numpy(), a_vals.to_numpy())

        try:
            _, p = mannwhitneyu(
                b_vals.to_numpy(), a_vals.to_numpy(),
                alternative="two-sided", method="exact",
            )
            p = float(p)
        except ValueError:
            p = np.nan

        level, cell_type, column = feat
        rec = {
            "level": level,
            "cell_type": cell_type,
            "column": column,
            f"{group_a}_mean": a_mean,
            f"{group_b}_mean": b_mean,
            "diff": diff,
            "log2fc": log2fc,
            "abs_log2fc": abs(log2fc),
            "paired_agree": agree,
            "cliffs_delta": delta,
            "mwu_p": p,
            "n_donors": len(paired),
        }
        for dn in paired_donors:
            rec[f"pdiff_{dn}"] = paired.get(dn, np.nan)
            rec[f"log2fc_{dn}"] = paired_log2fc.get(dn, np.nan)
            rec[f"abund_{dn}"] = paired_abund.get(dn, np.nan)
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    return out.sort_values("abs_log2fc", ascending=False).reset_index(drop=True)



# ----------------------------------------------------------------------
# plotting the screen output
# ----------------------------------------------------------------------

def _disp(name):
    """Cleaned cell type or marker name to a display label."""
    return str(name).replace("_", " ")


def _mean_cols(group_a, group_b):
    return f"{group_a}_mean", f"{group_b}_mean"


def _per_donor_cols(screen_df, metric):
    """Per donor column names present for a metric, and the donor ids."""
    prefix = "log2fc_" if metric == "log2fc" else "pdiff_"
    cols = [c for c in screen_df.columns
            if c.startswith(prefix) and c not in ("log2fc", "abs_log2fc")]
    donors = [c[len(prefix):] for c in cols]
    return cols, donors


def _per_donor_abund_cols(screen_df, donor_ids):
    """Per donor abundance column names, same order as donor_ids."""
    cols = [f"abund_{d}" for d in donor_ids]
    missing = [c for c in cols if c not in screen_df.columns]
    if missing:
        raise ValueError(
            "points='donor' with size_by_abundance=True needs per donor "
            f"abundance columns in the screen output: missing {missing}. "
            "Re-run screen() so it emits abund_<donor>."
        )
    return cols

def plot_fc_lollipop(
    screen_df,
    metric="log2fc",
    points="group",
    size_by_abundance=True,
    group_a="MesLN",
    group_b="MedLN",
    top_n=12,
    level_order=("type", "intermediate", "subtype", "marker"),
    agree_color="#2166ac",
    disagree_color="#cccccc",
    donor_colors=None,
    size_range=(25, 260),
    point_size=70,
    title=None,
    figsize=None,
):
    """
    Ranked lollipop of the screen output, one facet per level.

    Each row is a feature. Within a facet the strongest group_b-up sits at the
    top, the strongest group_a-up at the bottom, and each facet is capped to the
    top_n features by absolute group metric.

    metric      "log2fc" for the log2 fold change, group_b over group_a, or
                "diff" for the plain percentage point difference, group_b minus
                group_a. Positive means higher in group_b.
    points      "group" draws one dot per feature at the group value (the ratio
                or difference of the averaged donor values), coloured by whether
                both donors agree in direction.
                "donor" draws one dot per donor at that donor's own value, the
                two joined by a thin line so the spread between donors is
                visible. Colour then encodes donor, and the agreement colouring
                is dropped since agreement is read directly from whether the two
                dots share a side of zero.
    size_by_abundance
                When True, dot area scales to the larger of the two group means
                on a square root scale, so a big change on a rare population
                draws a small dot. When False, all dots use point_size.

    Guides at plus and minus one are drawn only for log2fc, where a unit is a
    doubling. Ranking is always by the absolute group metric, so "group" and
    "donor" show the same features in the same order.

    Returns (fig, axes).
    """
    if metric not in ("log2fc", "diff"):
        raise ValueError('metric must be "log2fc" or "diff"')
    if points not in ("group", "donor"):
        raise ValueError('points must be "group" or "donor"')

    a_mean, b_mean = _mean_cols(group_a, group_b)
    for col in (a_mean, b_mean, metric, "paired_agree", "level"):
        if col not in screen_df.columns:
            raise ValueError(f"screen_df is missing column '{col}'")

    per_donor_cols, donor_ids = ([], [])
    per_donor_abund_cols = []
    if points == "donor":
        per_donor_cols, donor_ids = _per_donor_cols(screen_df, metric)
        if not per_donor_cols:
            raise ValueError(
                "points='donor' needs per donor columns in the screen output. "
                "Re-run screen() so it emits log2fc_<donor> / pdiff_<donor>."
            )
        if donor_colors is None:
            pal = sns.color_palette("Set2", len(donor_ids))
            donor_colors = {d: pal[i] for i, d in enumerate(donor_ids)}
        if size_by_abundance:
            per_donor_abund_cols = _per_donor_abund_cols(screen_df, donor_ids)

    df = screen_df.copy()
    df["abundance"] = df[[a_mean, b_mean]].max(axis=1)

    levels_present = [lv for lv in level_order if lv in set(df["level"])]
    if not levels_present:
        levels_present = list(dict.fromkeys(df["level"]))

    facet = {}
    for lv in levels_present:
        sub = df[df["level"] == lv].copy()
        sub = sub.loc[sub[metric].abs().sort_values(ascending=False).index].head(top_n)
        sub = sub.sort_values(metric)  # ascending, so most positive lands at the top
        if not sub.empty:
            facet[lv] = sub
    levels_present = [lv for lv in levels_present if lv in facet]

    # global square root size scaling across every shown row (and, in donor
    # mode, every individual donor point, so the scale matches what's drawn)
    if points == "donor" and size_by_abundance:
        ab_all = np.concatenate([
            facet[lv][per_donor_abund_cols].to_numpy().ravel()
            for lv in levels_present
        ])
    else:
        ab_all = np.concatenate([facet[lv]["abundance"].to_numpy() for lv in levels_present])
    ab_all = ab_all[np.isfinite(ab_all)]

    amin = np.sqrt(max(ab_all.min(), 0)) if ab_all.size else 0.0
    amax = np.sqrt(ab_all.max()) if ab_all.size else 1.0

    def _size(a):
        if not size_by_abundance:
            return point_size
        if not np.isfinite(a) or amax <= amin:
            return np.mean(size_range)
        frac = (np.sqrt(max(a, 0)) - amin) / (amax - amin)
        return size_range[0] + frac * (size_range[1] - size_range[0])

    rows_per = [len(facet[lv]) for lv in levels_present]
    total_rows = sum(rows_per)
    min_h = 4.0 if len(levels_present) == 1 else 2.4
    if figsize is None:
        figsize = (10.0, max(min_h, 0.34 * total_rows + 0.9 * len(levels_present)))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        len(levels_present), 1, height_ratios=rows_per, hspace=0.45,
        figure=fig, left=0.20, right=0.74, top=0.90, bottom=0.14,
    )
    axes = []

    for i, lv in enumerate(levels_present):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        sub = facet[lv]
        yy = np.arange(len(sub))
        sizes = np.array([_size(a) for a in sub["abundance"].to_numpy()])
        if points == "donor" and size_by_abundance:
            sizes_by_donor = {
                d: np.array([_size(a) for a in sub[c].to_numpy()])
                for d, c in zip(donor_ids, per_donor_abund_cols)
            }

        ax.axvline(0, color="black", linewidth=0.8, zorder=1)
        if metric == "log2fc":
            for g in (-1, 1):
                ax.axvline(g, color="0.85", linewidth=0.8, linestyle="--", zorder=0)

        if points == "group":
            vals = sub[metric].to_numpy()
            colors = np.where(sub["paired_agree"].to_numpy(), agree_color, disagree_color)
            ax.hlines(yy, 0, vals, color=colors, linewidth=1.6, zorder=2)
            ax.scatter(vals, yy, s=sizes, c=colors, edgecolor="white",
                       linewidth=0.6, zorder=3)
        else:
            donor_vals = {d: sub[c].to_numpy()
                          for c, d in zip(per_donor_cols, donor_ids)}
            # thin connector spanning each feature's two donor values
            lo = np.nanmin(np.vstack(list(donor_vals.values())), axis=0)
            hi = np.nanmax(np.vstack(list(donor_vals.values())), axis=0)
            ax.hlines(yy, lo, hi, color="0.7", linewidth=1.0, zorder=2)
            for d in donor_ids:
                v = donor_vals[d]
                s = sizes_by_donor[d] if size_by_abundance else sizes
                ax.scatter(v, yy, s=s, c=[donor_colors[d]] * len(sub),
                           edgecolor="white", linewidth=0.6, zorder=3)

        ax.set_yticks(yy)
        ax.set_yticklabels([_disp(c) for c in sub["cell_type"]], fontsize=8)
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(lv, loc="left", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.tick_params(length=0)

    xlabel = (
        f"log2 fold change ({group_b} / {group_a})"
        if metric == "log2fc"
        else f"difference in percentage points ({group_b} - {group_a})"
    )
    bottom = axes[-1]
    bottom.set_xlabel(xlabel, fontsize=9)
    bottom.annotate(
        f"higher in {group_a}  <-", xy=(0, -0.32), xycoords=("data", "axes fraction"),
        ha="right", va="top", fontsize=8, color="0.35", annotation_clip=False,
    )
    bottom.annotate(
        f"->  higher in {group_b}", xy=(0, -0.32), xycoords=("data", "axes fraction"),
        ha="left", va="top", fontsize=8, color="0.35", annotation_clip=False,
    )

    # Legends live in the reserved right margin, anchored to each facet's top so
    # every panel gets a size key and nothing is clipped. The first legend (donor
    # or direction) sits at the top of the first facet, with the first facet's
    # size legend directly beneath it.
    legend_x = 0.76
    if points == "group":
        top_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=agree_color,
                   markeredgecolor="white", markersize=8, label="both donors agree"),
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=disagree_color,
                   markeredgecolor="white", markersize=8, label="donors disagree"),
        ]
        top_title = "direction"
    else:
        top_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor=donor_colors[d],
                   markeredgecolor="white", markersize=8, label=d)
            for d in donor_ids
        ]
        top_title = "donor"

    leg_top = fig.legend(
        handles=top_handles, loc="upper left",
        bbox_to_anchor=(legend_x, axes[0].get_position().y1),
        bbox_transform=fig.transFigure, frameon=False, fontsize=8,
        title=top_title, title_fontsize=8,
    )

    if size_by_abundance and ab_all.size:
        fig.canvas.draw()
        top_h = leg_top.get_window_extent(fig.canvas.get_renderer()).height / fig.bbox.height
        gap = 0.03

        def _size_handles(values):
            return [
                Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.5",
                       markeredgecolor="white", markersize=np.sqrt(_size(v)),
                       label=f"{v:g}%")
                for v in values
            ]

        for i, lv in enumerate(levels_present):
            if points == "donor":
                fab = facet[lv][per_donor_abund_cols].to_numpy().ravel()
            else:
                fab = facet[lv]["abundance"].to_numpy()
            fab = fab[np.isfinite(fab)]
            if not fab.size:
                continue
            ref = np.unique(np.quantile(fab, [0.1, 0.5, 0.9]).round(1))
            top = axes[i].get_position().y1
            y = top - top_h - gap if i == 0 else top
            fig.legend(
                handles=_size_handles(ref), loc="upper left",
                bbox_to_anchor=(legend_x, y),
                bbox_transform=fig.transFigure, frameon=False, fontsize=8,
                title="abundance", title_fontsize=8,
                labelspacing=1.2, borderpad=0.5,
            )

    if title:
        fig.suptitle(title, fontsize=12, x=0.05, ha="left")
    return fig, axes
