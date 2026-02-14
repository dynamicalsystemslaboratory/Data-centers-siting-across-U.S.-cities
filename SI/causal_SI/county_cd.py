import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

import networkx as nx
from castle.algorithms import PC
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None  # default='warn'


#%% ------------------------------------------------------------------
# Two-file version (COUNTY)
# --------------------------------------------------------------------
COUNTS_CSV = "county_datacenter_counts_new.csv"
REGS_CSV   = "county_regressors_totals_nh_area_jan.csv"

# 1. Load CSVs
# --------------------------------------------------------------------
counts = pd.read_csv(COUNTS_CSV)
regs   = pd.read_csv(REGS_CSV)

print("Counts columns:", counts.columns.tolist())
print("Regs columns:", regs.columns.tolist())

# ---- Rename county outcome columns to standard names ----
counts = counts.rename(columns={
    "county Code":  "GEOID",      # county identifier
    "county Title": "CountyTitle" # county name
})

# ---- Standardize IDs to 5-digit strings ----
# County GEOIDs are 5-digit FIPS: STATE(2) + COUNTY(3)
counts["GEOID"] = pd.to_numeric(counts["GEOID"], errors="coerce").astype("Int64")
counts = counts.dropna(subset=["GEOID"]).copy()
counts["GEOID"] = counts["GEOID"].astype(int).astype(str).str.zfill(5)

possible_id_cols = ["GEOID", "geoid", "county_geoid", "fips", "FIPS", "county_fips"]
id_col = next((c for c in possible_id_cols if c in regs.columns), None)
if id_col is None:
    raise ValueError(f"Could not find a county ID column in REGs. Looked for {possible_id_cols}")

regs = regs.rename(columns={id_col: "GEOID"})
regs["GEOID"] = pd.to_numeric(regs["GEOID"], errors="coerce").astype("Int64")
regs = regs.dropna(subset=["GEOID"]).copy()
regs["GEOID"] = regs["GEOID"].astype(int).astype(str).str.zfill(5)

# ---- Keep outcome columns to merge ----
cols_keep = ["GEOID", "CountyTitle", "datacenter_count", "has_datacenter"]

# ---- Merge regressors + outcomes ----
data = regs.merge(
    counts[cols_keep],
    on="GEOID",
    how="left"
)

# Fill missing outcomes with zeros (counties with no DCs)
data["datacenter_count"] = data["datacenter_count"].fillna(0).astype(int)
data["has_datacenter"]   = data["has_datacenter"].fillna(0).astype(int)

# ---- Merge sanity checks ----
print("\nMerged data head:")
print(data.head())

matched = data["CountyTitle"].notna().mean()
print(f"\nMatch rate (regs rows that found a county name): {matched:.3%}")
if matched < 0.90:
    print("WARNING: Low match rate. Your REGs county IDs may not be county GEOIDs (5-digit FIPS).")

df = data


#%% ------------------------------------------------------------------
# PC algorithm + edge colouring by sign  (COUNTY)
# --------------------------------------------------------------------
cols = [
    "total_cap",
    "SAMI_energy_emp_it_total",
    "bqs",
    "nri_events_per_area",
    "water_cf",
    "prox_total",
    "retired_coal",
    "benefit_flag",
    "SAMI_energy_exp",
    "datacenter_count"
]

missing_cols = [c for c in cols if c not in df.columns]
if missing_cols:
    raise ValueError(f"These columns are missing in merged county df: {missing_cols}")

pc_dataset = np.array(df[cols])

pc_MI = PC(ci_test="Spearman", alpha=0.001)
pc_MI.learn(pc_dataset)

G_idx = nx.DiGraph(pc_MI.causal_matrix)

corr_mat = df[cols].corr(method="spearman").to_numpy()

edge_colors = []
for u, v in G_idx.edges():
    r = corr_mat[u, v]
    if r > 0:
        edge_colors.append("green")
    elif r < 0:
        edge_colors.append("red")
    else:
        edge_colors.append("gray")

names = [str(i) for i in range(1, len(cols))] + ["dc"]  # last is datacenter_count
MAPPING = {k: v for k, v in enumerate(names)}
learned_graph = nx.relabel_nodes(G_idx, MAPPING, copy=True)

pos = nx.circular_layout(learned_graph)

nx.draw(
    learned_graph,
    pos,
    with_labels=True,
    node_size=1800,
    font_size=18,
    font_color="white",
    edge_color=edge_colors
)
plt.axis("off")
plt.tight_layout()
plt.show()
