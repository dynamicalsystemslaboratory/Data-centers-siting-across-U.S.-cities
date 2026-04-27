from pathlib import Path
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
##################################################### STYLE #####################################################
def set_paper_style():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        "font.size": 13,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 12,

        "xtick.labelsize": 12,
        "ytick.labelsize": 12,

        "axes.linewidth": 1.0,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })

set_paper_style()

##################################################### PATHS #####################################################

data_path = "Cleaned Shapefile/DC_Shapefile.shp"
urban_path = "/Users/camilla/PycharmProjects/slides_nature_cities/Urban areas/tl_2020_us_uac20.shp"
msa_path = "tl_2024_us_cbsa/tl_2024_us_cbsa.shp"
state_path = "State/tl_2023_us_state.shp"
fips_path = r"/Users/camilla/PycharmProjects/slides_nature_cities/Contiguous US states FIPS.csv"

##################################################### LOAD DATA #####################################################
gdf_data_centers = gpd.read_file(data_path)
gdf_urban = gpd.read_file(urban_path)
gdf_msa = gpd.read_file(msa_path)
gdf_states = gpd.read_file(state_path)

df_fips = pd.read_csv(fips_path)
df_fips["FIPS Code"] = df_fips["FIPS Code"].astype(str).str.zfill(2)

##################################################### CLEAN GEOGRAPHY #####################################################
# Urban
gdf_urban.rename(columns={
    'GEOID20': 'GEOID',
    'NAME20': 'NAME',
    'NAMELSAD20': 'Urban area'
}, inplace=True)

gdf_urban['State_Abbr'] = gdf_urban['NAME'].str[-2:]
gdf_urban = gdf_urban[gdf_urban['State_Abbr'].isin(df_fips['Postal Abbr.'])]

# MSA
gdf_msa['State_Abbr'] = gdf_msa['NAME'].str[-2:]
gdf_msa = gdf_msa[gdf_msa['State_Abbr'].isin(df_fips['Postal Abbr.'])]

# States
gdf_states["STATEFP"] = gdf_states["STATEFP"].astype(str)
gdf_states = gdf_states[gdf_states["STATEFP"].isin(df_fips["FIPS Code"])]

##################################################### CRS #####################################################
TARGET_EPSG = 5070

gdf_data_centers = gdf_data_centers.to_crs(epsg=TARGET_EPSG)
gdf_urban = gdf_urban.to_crs(epsg=TARGET_EPSG)
gdf_msa = gdf_msa.to_crs(epsg=TARGET_EPSG)
gdf_states = gdf_states.to_crs(epsg=TARGET_EPSG)

##################################################### DISTANCE #####################################################
urban_boundary = gdf_urban.geometry.union_all()
gdf_data_centers['Dist_to_urban_km'] = gdf_data_centers.distance(urban_boundary) / 1000

##################################################### PLOT MAP #####################################################
##################################################### PLOT MAP #####################################################
fig, ax = plt.subplots(figsize=(10.5, 6.5))
fig.patch.set_alpha(0)
ax.set_facecolor("none")

# State boundaries
gdf_states.boundary.plot(
    ax=ax,
    linewidth=0.6,
    color="#d9d9d9",
    zorder=1
)

# MSA
gdf_msa.plot(
    ax=ax,
    color="#d3eced",
    edgecolor="white",
    linewidth=0.2,
    alpha=0.95,
    zorder=2
)

# Urban areas
gdf_urban.plot(
    ax=ax,
    color="#c51b8a",
    edgecolor="none",
    alpha=0.6,
    zorder=3
)

# Data centers
gdf_data_centers.plot(
    ax=ax,
    marker='o',
    facecolor='none',
    edgecolor="#e6c27a",
    linewidth=0.4,
    markersize=1,
    zorder=4
)

ax.set_aspect("equal")
ax.axis("off")

plt.tight_layout()
##################################################### LEGEND #####################################################

legend_elements = [

    # MSA
    Patch(
        facecolor="#d3eced",
        edgecolor="none",
        label="Cities"
    ),

    # Urban
    Patch(
        facecolor="#c51b8a",
        edgecolor="none",
        label="Urban areas"
    ),

    # Data centers
    Line2D(
        [0], [0],
        marker='o',
        color='none',
        markerfacecolor='none',
        markeredgecolor="#e6c27a",
        markeredgewidth=1,
        markersize=6,
        label="Data centers"
    ),
]

ax.legend(
    handles=legend_elements,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    frameon=True,
    framealpha=0.95,
    borderpad=0.6,
    handlelength=1.2,
    handletextpad=0.6,
    labelspacing=0.4
)
OUTPUT_PATH = "map_data_centers_urban.png"
fig.savefig(
    OUTPUT_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True
)
plt.show()

##################################################### HISTOGRAM #####################################################
distances = gdf_data_centers[gdf_data_centers['Dist_to_urban_km'] > 0]['Dist_to_urban_km']

mean_val = distances.mean()
median_val = distances.median()

fig, ax = plt.subplots(figsize=(7.5, 4.5))

ax.hist(
    distances,
    bins=40,
    color="grey",
    edgecolor="black",
    linewidth=0.3
)

ax.axvline(mean_val, color="#999999", linestyle="--", linewidth=0.8)
ax.axvline(median_val, color="#666666", linestyle=":", linewidth=0.8)

ax.set_xlabel("Distance to urban area (km)")
ax.set_ylabel("Number of data centers")

# Clean axes
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
OUTPUT_PATH ="hist_distances_urban.png"

fig.savefig(
    OUTPUT_PATH,
    dpi=600,
    bbox_inches="tight",
    pad_inches=0,
    transparent=True
)
plt.show()

##################################################### STATS #####################################################
thresholds = [0, 5, 10, 20]

for t in thresholds:
    share = (gdf_data_centers['Dist_to_urban_km'] <= t).mean() * 100
    print(f"<= {t} km: {share:.1f}%")

print(f"Mean: {mean_val:.2f} km")
print(f"Median: {median_val:.2f} km")