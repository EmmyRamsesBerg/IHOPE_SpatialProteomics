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

A third, spatial layer asks the same fold change question of location within
a tissue rather than of tissue itself: for each tissue, is a cell type or
marker enriched inside vs outside a follicle (obs["B_follicle"], already
computed upstream). follicle_screen answers that by reusing screen() with
domain (inside/outside) standing in for tissue. build_cross_tissue_effect then
takes each donor's inside-minus-outside difference and screens it again,
tissue vs tissue, so the follicle effect itself, not just raw abundance, can
be compared across MedLN, MesLN and Spleen. plot_effect_by_tissue offers a
descriptive side by side view of that same effect across tissues, one dot per
tissue per feature, as a complement to the pairwise screen.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def add_parent_relative(donor_level, parent_of, extra_keys=()):
    """
    Add a parent relative percentage column to the donor level table.

    For each row whose cell_type is a key in parent_of, pct_parent is the
    donor level child percentage divided by the same donor (and tissue, and
    any extra_keys) parent percentage, times 100. Rows with no parent entry,
    or whose parent is absent or zero for that grouping, get NaN.

    parent_of maps a cleaned cell_type name to its parent cleaned cell_type
    name, for example {"CD8_T": "T", "TEM_CD8": "CD8_T"}. Biology lives in the
    notebook, this only applies the mapping. Parent names are assumed unique
    within each donor/tissue/extra_keys group, which holds because parents are
    type or intermediate level names.

    extra_keys adds further grouping columns beyond donor and tissue, for
    example ("domain",) when donor_level also splits inside vs outside a
    follicle, so a child's parent percentage is looked up within the same
    domain rather than pooled across domains.
    """
    key_cols = ["donor", "tissue", *extra_keys, "cell_type"]
    parent_names = set(parent_of.values())
    denom_df = donor_level[donor_level["cell_type"].isin(parent_names)]
    pct_lookup = denom_df.set_index(key_cols)["pct"]

    def _parent_pct(row):
        parent = parent_of.get(row["cell_type"])
        if parent is None:
            return np.nan
        lookup_key = tuple(row[k] for k in key_cols[:-1]) + (parent,)
        denom = pct_lookup.get(lookup_key, np.nan)
        if denom is None or (isinstance(denom, float) and np.isnan(denom)) or denom == 0:
            return np.nan
        return row["pct"] / denom * 100

    out = donor_level.copy()
    out["pct_parent"] = out.apply(_parent_pct, axis=1)
    return out


def exclude_unresolved(
    donor_level,
    suffixes=("_unclassified", "_unassigned"),
    group_keys=("donor", "tissue", "level"),
):
    """
    Drop unclassified and unassigned rows and add a per level classified pct.

    Both are removed from the table. Unlike unclassified, which is a single
    type level partition, unassigned is level specific, a cell can be assigned
    at the type level yet unassigned at the subtype level, so there is no
    single base across levels. The denominator is therefore computed per
    group_keys (donor, tissue and level by default) as the summed n_cells over
    the kept rows at that level. Every level then sums to 100 over its kept
    rows, matching the per level rescaling in comparison.py.

    Note the intermediate level is not a partition, its calls overlap (a cell
    is both CD4_T and T_naive), so its pct_resolved denominator double counts
    across overlapping calls. Read intermediates from the parent relative
    screen rather than pct_resolved. Type and subtype are genuine partitions
    and rescale cleanly.

    group_keys can be extended, for example ("donor", "tissue", "domain",
    "level"), so donor_level tables that also split inside vs outside a
    follicle get a pct_resolved computed within each domain rather than
    pooled across domains.

    Adds pct_resolved as a new column and leaves pct, n_cells and total_cells
    untouched so the percentage of all cells stays available. Parent relative
    values are unaffected since they read the untouched pct column.
    """
    drop_mask = donor_level["column"].str.endswith(suffixes)
    out = donor_level[~drop_mask].copy()

    level_base = (
        out.groupby(list(group_keys))["n_cells"]
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
# follicle domain, extraction and donor level pooling
# ----------------------------------------------------------------------
#
# The h5ad files already carry a per cell boolean obs["B_follicle"] (computed
# upstream from banksy_domain membership in uns["B_follicle_domains"], see
# add_spatial_B_context / add_TfH_like_cells in celltype_rules_IHOPE.py), so
# inside vs outside a follicle is read directly rather than recomputed here.
#
# The two extraction functions below mirror extract_marker_positivity, but
# cross each boolean column (cell type calls or marker positivity) against
# that follicle flag, so every count is split into a "follicle" and a
# "nonfollicle" row. domain_to_donor_level then pools either table to donor
# level the same way pool_to_donor_level / marker_to_donor_level do, keeping
# domain as an extra grouping key instead of collapsing it away.

def _celltype_bool_columns(obs, level_prefixes=("type_", "intermediate_", "subtype_")):
    return [
        c for c in obs.columns
        if c.startswith(level_prefixes) and obs[c].dtype == bool
    ]


def _level_and_celltype(column, level_prefixes=("type_", "intermediate_", "subtype_")):
    for prefix in level_prefixes:
        if column.startswith(prefix):
            return prefix[:-1], column[len(prefix):]
    raise ValueError(f"Column '{column}' does not start with a known level prefix.")


def extract_celltype_domain_counts(
    anndata_dir,
    basenames,
    out_csv,
    follicle_key="B_follicle",
    suffix="_celltypes_follicledomains.h5ad",
    level_prefixes=("type_", "intermediate_", "subtype_"),
):
    """
    Build a skinny per sample, per domain cell type count table.

    Reads each h5ad in backed mode (obs only). For every boolean column at the
    type, intermediate or subtype level, counts how many of that call's cells
    fall inside vs outside a follicle, using the follicle_key boolean column.
    Also records the domain's own total cell count per sample, the
    denominator for the donor level percentage.

    Writes a long CSV with columns sample, level, cell_type, column, domain,
    n_cells, domain_total_cells. domain is "follicle" or "nonfollicle".
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
        if follicle_key not in obs.columns:
            raise ValueError(f"{path} has no '{follicle_key}' column in obs.")
        in_follicle = obs[follicle_key].to_numpy(dtype=bool)
        domain_totals = {
            "follicle": int(in_follicle.sum()),
            "nonfollicle": int((~in_follicle).sum()),
        }

        recs = []
        for col in _celltype_bool_columns(obs, level_prefixes):
            level, cell_type = _level_and_celltype(col, level_prefixes)
            vals = obs[col].to_numpy(dtype=bool)
            for domain, mask in (("follicle", in_follicle), ("nonfollicle", ~in_follicle)):
                recs.append({
                    "sample": basename,
                    "level": level,
                    "cell_type": cell_type,
                    "column": col,
                    "domain": domain,
                    "n_cells": int((vals & mask).sum()),
                    "domain_total_cells": domain_totals[domain],
                })
        frames.append(pd.DataFrame(recs))
        # release the backed file handle before the next file
        if getattr(adata, "isbacked", False):
            adata.file.close()
        del adata

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(
        f"Wrote {out_csv} with {out.shape[0]} rows "
        f"({out['sample'].nunique()} samples, {out['column'].nunique()} cell type columns)"
    )
    return out


def extract_marker_domain_positivity(
    anndata_dir,
    basenames,
    out_csv,
    follicle_key="B_follicle",
    suffix="_celltypes_follicledomains.h5ad",
    pos_suffix="_pos",
):
    """
    Marker positivity version of extract_celltype_domain_counts.

    Same idea, but for every boolean marker column ending in pos_suffix
    rather than the type/intermediate/subtype cell type calls. Writes a long
    CSV with columns sample, marker, domain, n_pos, domain_total_cells, shaped
    to feed domain_to_donor_level the same way the cell type table does.
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
        if follicle_key not in obs.columns:
            raise ValueError(f"{path} has no '{follicle_key}' column in obs.")
        in_follicle = obs[follicle_key].to_numpy(dtype=bool)
        domain_totals = {
            "follicle": int(in_follicle.sum()),
            "nonfollicle": int((~in_follicle).sum()),
        }

        pos_cols = [
            c for c in obs.columns
            if c.endswith(pos_suffix) and obs[c].dtype == bool
        ]
        recs = []
        for col in pos_cols:
            marker = col[: -len(pos_suffix)]
            vals = obs[col].to_numpy(dtype=bool)
            for domain, mask in (("follicle", in_follicle), ("nonfollicle", ~in_follicle)):
                recs.append({
                    "sample": basename,
                    "marker": marker,
                    "domain": domain,
                    "n_pos": int((vals & mask).sum()),
                    "domain_total_cells": domain_totals[domain],
                })
        frames.append(pd.DataFrame(recs))
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


def domain_to_donor_level(domain_csv, count_col="n_cells"):
    """
    Load a domain long CSV (cell type or marker) and pool to donor level.

    Mirrors pool_to_donor_level / marker_to_donor_level, but keeps domain
    (inside vs outside a follicle) as a grouping key instead of collapsing it
    away. count_col is "n_cells" for the cell type table and "n_pos" for the
    marker table; either way the pooled count column is named n_cells so the
    same screen() call works on both, and a marker table is reshaped to the
    level/cell_type/column shape the cell type table already has.

    Returns one row per (donor, tissue, domain, level, cell_type, column) with
    n_cells (pooled), domain_total_cells (pooled) and pct, the percentage of
    that domain's cells within that donor and tissue.
    """
    df = pd.read_csv(domain_csv)
    df["donor"] = df["sample"].map(_donor_from_basename)
    df["tissue"] = df["sample"].map(_tissue_from_basename)

    if "marker" in df.columns and "cell_type" not in df.columns:
        df = df.rename(columns={count_col: "n_cells"})
        df["level"] = "marker"
        df["cell_type"] = df["marker"]
        df["column"] = df["marker"] + "_pos"
    elif count_col != "n_cells":
        df = df.rename(columns={count_col: "n_cells"})

    totals = df.drop_duplicates(["sample", "domain"])[
        ["donor", "tissue", "sample", "domain", "domain_total_cells"]
    ]
    dt_totals = (
        totals.groupby(["donor", "tissue", "domain"])["domain_total_cells"]
        .sum()
        .rename("pooled_domain_total")
        .reset_index()
    )

    pooled_n = (
        df.groupby(["donor", "tissue", "domain", "level", "cell_type", "column"])["n_cells"]
        .sum()
        .rename("n_cells")
        .reset_index()
    )

    out = pooled_n.merge(dt_totals, on=["donor", "tissue", "domain"], how="left")
    out["domain_total_cells"] = out["pooled_domain_total"]
    out["pct"] = out["n_cells"] / out["pooled_domain_total"] * 100
    return out.drop(columns="pooled_domain_total")


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

        # per donor paired difference where the donor has both tissues
        paired = {}
        for dn in paired_donors:
            av, bv = a_col.get(dn), b_col.get(dn)
            if pd.notna(av) and pd.notna(bv):
                paired[dn] = float(bv - av)

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
        records.append(rec)

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out
    return out.sort_values("abs_log2fc", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# follicle screen: inside vs outside, then the effect across tissues
# ----------------------------------------------------------------------
#
# Both layers reuse screen() as-is rather than duplicating its logic.
#
# Layer 1, follicle_screen: within one tissue, donors are paired on
# themselves (inside vs outside follicle for the same donor and tissue), so
# domain stands in for the "tissue" argument screen() expects.
#
# Layer 2, build_cross_tissue_effect + a second screen() call: each donor's
# inside-minus-outside difference from layer 1 (already computed as
# pdiff_<donor> inside screen()) becomes one value per donor per tissue, and
# that value is what gets compared, tissue vs tissue. plot_effect_by_tissue is
# a descriptive alternative to that pairwise screen: it lays every tissue's
# effect side by side on the same row per feature, rather than testing one
# tissue pair at a time.

def follicle_screen(
    donor_domain_level,
    tissue,
    value_col="pct",
    group_out="nonfollicle",
    group_in="follicle",
    **screen_kwargs,
):
    """
    Inside vs outside follicle screen for a single tissue.

    Filters donor_domain_level (from domain_to_donor_level, optionally passed
    through exclude_unresolved / add_parent_relative with domain added to
    their grouping keys) down to one tissue, then calls screen() with the
    domain column standing in for tissue. group_in is the fold change
    numerator, so a positive log2fc or diff means higher inside the follicle.

    Returns the same shape as screen(), including one pdiff_<donor> column
    per donor: that donor's inside-minus-outside difference. Those columns
    are what build_cross_tissue_effect and plot_effect_by_tissue read.
    """
    sub = donor_domain_level[donor_domain_level["tissue"] == tissue].copy()
    sub["tissue"] = sub["domain"]
    return screen(sub, group_a=group_out, group_b=group_in, value_col=value_col, **screen_kwargs)


def build_cross_tissue_effect(follicle_screens):
    """
    Reshape per tissue in/out screens into a donor level table for a second
    screen, so the follicle effect itself can be compared between tissues.

    follicle_screens is a dict of tissue name to the screen_df returned by
    follicle_screen for that tissue (run with matching value_col, group_in and
    group_out so the effects are comparable). Melts each tissue's per donor
    paired difference columns (pdiff_<donor>, that donor's inside minus
    outside percentage) into long rows with columns level, cell_type, column,
    tissue, donor, value. Donors missing one of the two domains for a tissue
    have no pdiff there and are dropped for that tissue.

    Feed the result into screen(effect_df, group_a=tissue_x, group_b=tissue_y,
    value_col="value") to ask, for example, whether the follicle enrichment of
    a cell type is bigger in MesLN than in MedLN. fill_absent_zero defaults to
    False for a "value" column, which is correct here: a donor absent for a
    tissue is missing that comparison, not a zero effect.
    """
    frames = []
    for tissue, df in follicle_screens.items():
        if df is None or df.empty:
            continue
        pdiff_cols = [c for c in df.columns if c.startswith("pdiff_")]
        long = df.melt(
            id_vars=["level", "cell_type", "column"],
            value_vars=pdiff_cols,
            var_name="donor", value_name="value",
        )
        long["donor"] = long["donor"].str.removeprefix("pdiff_")
        long = long.dropna(subset=["value"])
        long["tissue"] = tissue
        frames.append(long)
    if not frames:
        return pd.DataFrame(columns=["level", "cell_type", "column", "donor", "value", "tissue"])
    return pd.concat(frames, ignore_index=True)


def build_effect_by_tissue_table(follicle_screens, metric="diff"):
    """
    Combine per tissue follicle_screen() outputs into one wide table, one row
    per feature (level, cell_type, column) and one column per tissue holding
    that tissue's inside vs outside effect (metric: "diff" or "log2fc").

    Purely a reshape for plot_effect_by_tissue, no statistics computed here;
    it complements build_cross_tissue_effect rather than replacing it. A
    feature missing for a tissue (for example absent at that level in that
    tissue's data) gets NaN in that tissue's column.
    """
    frames = []
    for tissue, df in follicle_screens.items():
        if df is None or df.empty:
            continue
        if metric not in df.columns:
            raise ValueError(f"'{metric}' not found in the {tissue} screen output.")
        sub = df[["level", "cell_type", "column", metric]].rename(columns={metric: tissue})
        frames.append(sub.set_index(["level", "cell_type", "column"]))
    if not frames:
        return pd.DataFrame()
    out = frames[0]
    for f in frames[1:]:
        out = out.join(f, how="outer")
    return out.reset_index()


# ----------------------------------------------------------------------
# plotting the screen output
# ----------------------------------------------------------------------

def _disp(name):
    """Cleaned cell type or marker name to a display label."""
    return str(name).replace("_", " ")


def _mean_cols(group_a, group_b):
    return f"{group_a}_mean", f"{group_b}_mean"


def plot_fc_lollipop(
    screen_df,
    metric="log2fc",
    group_a="MesLN",
    group_b="MedLN",
    top_n=12,
    level_order=("type", "intermediate", "subtype", "marker"),
    agree_color="#2166ac",
    disagree_color="#cccccc",
    size_range=(25, 260),
    title=None,
    figsize=None,
):
    """
    Ranked lollipop of the screen output, one facet per level.

    Each row is a feature. The stem runs from zero to its value, so length and
    side show how strongly and in which direction the feature differs. Within a
    facet the strongest MedLN-up sits at the top, the strongest MesLN-up at the
    bottom. Each facet is capped to the top_n features by absolute metric.

    metric      "log2fc" for the group log2 fold change, group_b over group_a,
                or "diff" for the plain percentage point difference,
                group_b minus group_a. Positive means higher in group_b.
    Dot size    scaled to the larger of the two group means, on a square root
                scale so area reads as abundance. A big fold change on a rare
                population draws a small dot, so fragile hits stay visually
                quiet. Sizes are comparable within a figure, not across figures.
    Dot colour  features where both donors move the same direction
                (paired_agree True) are drawn solid, disagreeing features are
                muted, so the credibility signal is visible at a glance.

    Guides at plus and minus one are drawn only for the log2fc metric, where a
    unit is a doubling. The x axis is annotated with the direction of each side.

    Takes the screen output directly, so the same call plots the abundance
    screen, the parent relative screen or the marker screen depending on which
    table is passed. group_a and group_b must match the ones the screen used,
    since the mean column names are derived from them.

    Returns (fig, axes).
    """
    if metric not in ("log2fc", "diff"):
        raise ValueError('metric must be "log2fc" or "diff"')

    a_mean, b_mean = _mean_cols(group_a, group_b)
    for col in (a_mean, b_mean, metric, "paired_agree", "level"):
        if col not in screen_df.columns:
            raise ValueError(f"screen_df is missing column '{col}'")

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

    # global square root size scaling across every shown row
    ab_all = np.concatenate([facet[lv]["abundance"].to_numpy() for lv in levels_present])
    ab_all = ab_all[np.isfinite(ab_all)]
    amin = np.sqrt(max(ab_all.min(), 0)) if ab_all.size else 0.0
    amax = np.sqrt(ab_all.max()) if ab_all.size else 1.0

    def _size(a):
        if not np.isfinite(a) or amax <= amin:
            return np.mean(size_range)
        frac = (np.sqrt(max(a, 0)) - amin) / (amax - amin)
        return size_range[0] + frac * (size_range[1] - size_range[0])

    rows_per = [len(facet[lv]) for lv in levels_present]
    total_rows = sum(rows_per)
    # single facet stacks both legends on the right, so it needs vertical room
    min_h = 4.0 if len(levels_present) == 1 else 2.4
    if figsize is None:
        figsize = (10.0, max(min_h, 0.34 * total_rows + 0.9 * len(levels_present)))

    fig = plt.figure(figsize=figsize)
    # reserve the right margin for the legends explicitly. tight_layout does not
    # account for legends anchored outside the axes, so without this they are
    # clipped by the figure edge.
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
        vals = sub[metric].to_numpy()
        colors = np.where(sub["paired_agree"].to_numpy(), agree_color, disagree_color)
        sizes = [_size(a) for a in sub["abundance"].to_numpy()]

        ax.axvline(0, color="black", linewidth=0.8, zorder=1)
        if metric == "log2fc":
            for g in (-1, 1):
                ax.axvline(g, color="0.85", linewidth=0.8, linestyle="--", zorder=0)

        ax.hlines(yy, 0, vals, color=colors, linewidth=1.6, zorder=2)
        ax.scatter(vals, yy, s=sizes, c=colors, edgecolor="white",
                   linewidth=0.6, zorder=3)

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

    # Both legends are attached to the figure, not to an axis, and placed in the
    # right margin reserved by the GridSpec above. Figure coordinates mean the
    # placement does not depend on how many facets there are, so the single
    # facet case cannot stack them on top of each other, and staying inside the
    # canvas means they cannot be clipped by the figure edge.
    color_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=agree_color,
               markeredgecolor="white", markersize=8, label="both donors agree"),
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=disagree_color,
               markeredgecolor="white", markersize=8, label="donors disagree"),
    ]
    fig.legend(
        handles=color_handles, loc="upper left", bbox_to_anchor=(0.76, 0.90),
        bbox_transform=fig.transFigure, frameon=False, fontsize=8,
        title="direction", title_fontsize=8,
    )

    if ab_all.size:
        ref = np.unique(np.quantile(ab_all, [0.1, 0.5, 0.9]).round(1))
        size_handles = [
            Line2D([0], [0], marker="o", linestyle="", markerfacecolor="0.5",
                   markeredgecolor="white", markersize=np.sqrt(_size(v)),
                   label=f"{v:g}%")
            for v in ref
        ]
        fig.legend(
            handles=size_handles, loc="upper left", bbox_to_anchor=(0.76, 0.62),
            bbox_transform=fig.transFigure, frameon=False, fontsize=8,
            title="abundance\n(larger group)", title_fontsize=8,
            labelspacing=1.4, borderpad=0.8,
        )

    if title:
        fig.suptitle(title, fontsize=12, x=0.05, ha="left")
    return fig, axes


# ----------------------------------------------------------------------
# side by side follicle effect across tissues
# ----------------------------------------------------------------------

DEFAULT_TISSUE_COLORS = {
    "MedLN": "#1f77b4",
    "MesLN": "#d62728",
    "Spleen": "#2ca02c",
}


def plot_effect_by_tissue(
    effect_table,
    tissue_order=("MedLN", "MesLN", "Spleen"),
    tissue_colors=None,
    level_order=("type", "intermediate", "subtype", "marker"),
    top_n=12,
    metric_label="inside minus outside follicle (percentage points)",
    title=None,
    figsize=None,
):
    """
    Dumbbell/dot plot comparing the follicle effect across tissues side by
    side, one panel per level. Takes the wide table from
    build_effect_by_tissue_table (one row per feature, one column per
    tissue).

    Each row is a feature. One dot per tissue sits at that tissue's value,
    coloured by tissue_colors, and a faint line spans the dots present on
    that row so the spread across tissues reads at a glance without having
    to read three separate lollipop panels. This is purely descriptive: no
    cross-tissue test is computed here. It complements
    build_cross_tissue_effect + screen(), which does test one tissue pair at
    a time, rather than replacing it.

    Rows are ranked within each level by the largest absolute value across
    the tissue columns present for that row, and capped to top_n per level.
    metric_label is the x axis label; override it if effect_table was built
    with metric="log2fc" rather than the default "diff".

    Returns (fig, axes).
    """
    tissues_present = [t for t in tissue_order if t in effect_table.columns]
    if not tissues_present:
        raise ValueError("None of tissue_order found as columns in effect_table.")
    if tissue_colors is None:
        tissue_colors = DEFAULT_TISSUE_COLORS

    df = effect_table.copy()
    df["_rank"] = df[tissues_present].abs().max(axis=1)

    levels_present = [lv for lv in level_order if lv in set(df["level"])]
    if not levels_present:
        levels_present = list(dict.fromkeys(df["level"]))

    facet = {}
    for lv in levels_present:
        sub = df[df["level"] == lv].copy()
        sub = sub.dropna(subset=["_rank"])
        sub = sub.sort_values("_rank", ascending=False).head(top_n)
        sub = sub.sort_values("_rank")  # ascending, so the strongest row lands at the top
        if not sub.empty:
            facet[lv] = sub
    levels_present = [lv for lv in levels_present if lv in facet]

    rows_per = [len(facet[lv]) for lv in levels_present]
    total_rows = sum(rows_per)
    if figsize is None:
        figsize = (9.0, max(3.0, 0.34 * total_rows + 0.9 * len(levels_present)))

    fig = plt.figure(figsize=figsize)
    gs = GridSpec(
        len(levels_present), 1, height_ratios=rows_per, hspace=0.45,
        figure=fig, left=0.22, right=0.80, top=0.90, bottom=0.14,
    )
    axes = []

    for i, lv in enumerate(levels_present):
        ax = fig.add_subplot(gs[i, 0])
        axes.append(ax)
        sub = facet[lv]
        yy = np.arange(len(sub))

        ax.axvline(0, color="black", linewidth=0.8, zorder=1)

        for row_i, (_, row) in zip(yy, sub.iterrows()):
            vals = [row[t] for t in tissues_present if pd.notna(row[t])]
            if len(vals) > 1:
                ax.hlines(row_i, min(vals), max(vals), color="0.75", linewidth=1.2, zorder=2)
            for t in tissues_present:
                v = row[t]
                if pd.notna(v):
                    ax.scatter(
                        v, row_i, s=45, color=tissue_colors.get(t, "0.5"),
                        edgecolor="white", linewidth=0.6, zorder=3,
                    )

        ax.set_yticks(yy)
        ax.set_yticklabels([_disp(c) for c in sub["cell_type"]], fontsize=8)
        ax.set_ylim(-0.6, len(sub) - 0.4)
        ax.set_title(lv, loc="left", fontsize=10, fontweight="bold")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)
        ax.tick_params(length=0)

    axes[-1].set_xlabel(metric_label, fontsize=9)

    handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=tissue_colors.get(t, "0.5"),
               markeredgecolor="white", markersize=8, label=t)
        for t in tissues_present
    ]
    fig.legend(
        handles=handles, loc="upper left", bbox_to_anchor=(0.82, 0.88),
        bbox_transform=fig.transFigure, frameon=False, fontsize=8,
        title="tissue", title_fontsize=8,
    )

    if title:
        fig.suptitle(title, fontsize=12, x=0.05, ha="left")
    return fig, axes

