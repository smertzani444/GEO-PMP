"""
Feature blocks:
- Residue-level physicochemical context
- Neighborhood composition (1 nm radius)
- No label-dependent features
"""

import pandas as pd
import numpy as np
import ast

def parse_neighbors(x):
    if pd.isna(x):
        return []
    try:
        return ast.literal_eval(x)
    except Exception:
        return []

HYDROPHOBIC = {"ALA","VAL","ILE","LEU","MET","PHE","TRP","TYR"}
CHARGED = {"ARG","LYS","ASP","GLU"}
POLAR = {"SER","THR","ASN","GLN","HIS","CYS"}
AROMATIC = {"PHE","TYR","TRP"}
SMALL = {"GLY","ALA","SER"}

def compute_neighbor_features(row, df_ref):
    idxs = row["neighbors"]

    if len(idxs) == 0:
        return {
            "n_neighbors": 0,
            "neighbor_frac_exposed": 0.0,
            "neighbor_frac_hydrophobic": 0.0,
            "neighbor_frac_charged": 0.0,
            "neighbor_frac_polar": 0.0,
            "neighbor_frac_aromatic": 0.0,
            "neighbor_frac_small": 0.0,
            "neighbor_mean_RSA": 0.0,
        }


    neigh = df_ref.loc[idxs]

    return {
        "n_neighbors": len(neigh),
        "neighbor_frac_exposed": neigh["exposed"].mean(),
        "neighbor_frac_hydrophobic": neigh["residue_name"].isin(HYDROPHOBIC).mean(),
        "neighbor_frac_charged": neigh["residue_name"].isin(CHARGED).mean(),
        "neighbor_mean_RSA": neigh["RSA_total_freesasa_tien"].mean(),
        "neighbor_frac_polar": neigh["residue_name"].isin(POLAR).mean(),
        "neighbor_frac_aromatic": neigh["residue_name"].isin(AROMATIC).mean(),
        "neighbor_frac_small": neigh["residue_name"].isin(SMALL).mean()
    }

def build_feature_table(df):
    """
    Build context-aware feature table from the full Tubiana dataset.
    Feature computation uses all residues.
    Filtering to exposed residues must be done downstream.
    """

    df = df.copy().reset_index(drop=True)

    # parse neighbors
    df["neighbors"] = df["neighboursList"].apply(parse_neighbors)

    # compute neighbor features
    neighbor_feats = df.apply(
        lambda row: compute_neighbor_features(row, df),
        axis=1,
        result_type="expand"
    )

    df = pd.concat([df, neighbor_feats], axis=1)

    return df
