import warnings
warnings.filterwarnings("ignore")

import os
os.environ['CASTLE_BACKEND'] = 'pytorch'

import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from castle.algorithms import PC

pd.options.mode.chained_assignment = None  # default='warn'

#%% ------------------------------------------------------------------
# Two-file version
# --------------------------------------------------------------------
COUNTS_CSV = "msa_datacenter_counts_new2.csv"         # outcomes
REGS_CSV   = "msa_regressors_totals_nh_area_eia_noprox_ret.csv"      # regressors

# 1. Load and merge CSVs
# --------------------------------------------------------------------
counts = pd.read_csv(COUNTS_CSV)
regs   = pd.read_csv(REGS_CSV)

print("Counts columns:", counts.columns.tolist())
print("Regs columns:", regs.columns.tolist())

# Make the ID + name columns consistent
counts = counts.rename(columns={
    "MSA Code":  "MSA",       # ID
    "MSA Title": "MSATitle"   # name
})

# Standardize IDs to 5-digit strings
counts["MSA"] = counts["MSA"].astype(str).str.zfill(5)
regs["MSA"]   = regs["MSA"].astype(str).str.zfill(5)

# Select which outcome variables to attach from counts
cols_keep = ["MSA", "MSATitle", "datacenter_count", "has_datacenter"]

# Merge regressors + outcomes
data = regs.merge(
    counts[cols_keep],
    on="MSA",
    how="left"
)

# Fill missing outcomes with zeros
data["datacenter_count"] = data["datacenter_count"].fillna(0).astype(int)
data["has_datacenter"]   = data["has_datacenter"].fillna(0).astype(int)

print("\nMerged data head:")
print(data.head())

df = data

#%% ------------------------------------------------------------------
# PC algorithm + edge colouring by sign + LaTeX table output
# --------------------------------------------------------------------
cols = [
    "cap_total",
    "SAMI_energy_emp_it_total",
    "events_per_area",
    "bqs_mean",
    "msa_ret_coal",
    "benefit_flag",
    "SAMI_energy_exp",
    "datacenter_count",
]

# Dataset for PC
pc_dataset = np.array(df[cols])

#compute marginal Spearman rho + p-value matrices once
n = len(cols)
print("SAMI_energy_exp summary")
s = df["SAMI_energy_exp"]
print("n:", s.shape[0])
print("n_nan:", s.isna().sum())
print("n_unique (dropna):", s.dropna().nunique())
print("min/max (dropna):", s.dropna().min(), s.dropna().max())

rho_mat  = np.zeros((n, n), dtype=float)
pval_mat = np.zeros((n, n), dtype=float)

for i in range(n):
    for j in range(n):
        rho_mat[i, j], pval_mat[i, j] = spearmanr(df[cols[i]], df[cols[j]], nan_policy="omit")

# Learn PC graph
pc_MI = PC(ci_test="Spearman", alpha=0.001)
pc_MI.learn(pc_dataset)

# Graph in index space (0..n-1)
G_idx = nx.DiGraph(pc_MI.causal_matrix)

# Spearman correlations (for coloring) in the same order as `cols`
corr_mat = df[cols].corr(method='spearman').to_numpy()

# Edge colours by sign of correlation
edge_colors = []
for u, v in G_idx.edges():
    r = corr_mat[u, v]
    if r > 0:
        edge_colors.append('black')
    elif r < 0:
        edge_colors.append('black')
    else:
        edge_colors.append('black')

# Relabel nodes to short names
names = ['1', '2', '3', '4', '5', '6', '7', 'DC']
MAPPING_MI = {k: v for k, v in enumerate(names)}
learned_graph_MI = nx.relabel_nodes(G_idx, MAPPING_MI, copy=True)

# Plot the graph (improved styling)
pos = nx.circular_layout(learned_graph_MI)

plt.figure(figsize=(7.5, 7.5))

# --- node styling: white circles + one black circle for DC
node_list = list(learned_graph_MI.nodes())
node_facecolors = ['black' if n == 'DC' else 'white' for n in node_list]
node_edgecolors = ['black'] * len(node_list)

nx.draw_networkx_nodes(
    learned_graph_MI,
    pos,
    nodelist=node_list,
    node_size=2300,            # bigger circles
    node_color=node_facecolors,
    edgecolors=node_edgecolors,
    linewidths=2.6             # thicker node border
)

# --- labels: black text on white nodes, white text on black node
label_colors = {n: ('white' if n == 'DC' else 'black') for n in node_list}
for n, (x, y) in pos.items():
    plt.text(
        x, y, n,
        ha='center', va='center',
        fontsize=18, fontweight='bold',
        color=label_colors[n]
    )

# --- edges: thicker arrows, larger arrowheads
nx.draw_networkx_edges(
    learned_graph_MI,
    pos,
    edge_color='black',
    width=3.2,                 # thicker lines
    arrows=True,
    arrowstyle='-|>',          # clearer arrowhead
    arrowsize=28,              # bigger arrowhead
    min_source_margin=18,      # keep arrows off node fill
    min_target_margin=22,
    connectionstyle='arc3,rad=0.06'  # slight curve to reduce overlap
)

plt.axis('off')
plt.tight_layout()
plt.show()

