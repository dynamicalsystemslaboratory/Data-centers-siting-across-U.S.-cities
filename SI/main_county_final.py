# COUNTY SCRIPT
# GLM: Negative Binomial
# Moran's I
# clustering
# city/non city stats

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
import geopandas as gpd

from statsmodels.stats.outliers_influence import variance_inflation_factor
from esda import Moran
from libpysal.weights import KNN
import matplotlib as mpl

COUNTS_CSV = "nature_cities/Full_DC_Counts_County_REBUILT.csv"
REGS_CSV   = "county_regressors_totals_nh_area_eia_noprox_ret.csv"
SHAPEFILE  = "tl_2024_us_county/tl_2024_us_county.shp"

ID_COL_SHAPE = "GEOID"

label_map = {
    "water_cf":                "Water-stress index",
    "bqs":                     "Broadband quality score",
    "nri_events":              "Natural hazard events",
    "evi":                     "Energy grid stress",
    "benefit_flag":            "State-tax incentives",
    "retired_coal":            "Retired coal plants",
    "elec_var_ind_all":        "Electricity-price variability",
    "SAMI_energy_exp":         "Energy expenditure",
    "SAMI_energy_emp_it_total":"IT employment",
    "cap_total":               "Generation capacity",
}

# Load and merge CSVs

counts = pd.read_csv(COUNTS_CSV, dtype=str)

regs = pd.read_csv(REGS_CSV, dtype={"FIPS": str})

print("Counts columns:", counts.columns.tolist())
print("Regs columns:", regs.columns.tolist())

# build county key CID from counts GEOID
if "GEOID" not in counts.columns:
    raise ValueError("COUNTS_CSV missing 'GEOID' column.")
counts["CID"] = counts["GEOID"].astype(str).str.strip().str.zfill(5)

# outcomes
if "dc_count" not in counts.columns:
    raise ValueError("COUNTS_CSV missing 'dc_count' column.")
counts["datacenter_count"] = pd.to_numeric(counts["dc_count"], errors="coerce").fillna(0).astype(int)

if "has_dc" in counts.columns:
    counts["has_datacenter"] = pd.to_numeric(counts["has_dc"], errors="coerce").fillna(0).astype(int)
else:
    counts["has_datacenter"] = (counts["datacenter_count"] > 0).astype(int)

if "county_name_clean" in counts.columns:
    counts["CName"] = counts["county_name_clean"].astype(str)
else:
    counts["CName"] = ""

# build CID in regressors
if "CID" in regs.columns:
    regs["CID"] = regs["CID"].astype(str).str.strip().str.zfill(5)
elif "FIPS" in regs.columns:
    regs["CID"] = regs["FIPS"].astype(str).str.strip().str.zfill(5)
else:
    raise ValueError("REGS_CSV must contain a county id column named 'FIPS' or 'CID'.")

#  merge
data = regs.merge(
    counts[["CID", "CName", "datacenter_count", "has_datacenter"]],
    on="CID",
    how="left"
)

data["datacenter_count"] = pd.to_numeric(data["datacenter_count"], errors="coerce").fillna(0).astype(int)
data["has_datacenter"]   = pd.to_numeric(data["has_datacenter"],   errors="coerce").fillna(0).astype(int)

print("\nMerged data head:")
print(data.head())


#Define predictors and clean data (NaN / inf)

exclude = [
    "FIPS", "CID", "CName",
    "datacenter_count", "has_datacenter"
]

predictor_cols = [c for c in data.columns if c not in exclude]
print("\nPredictors used:", predictor_cols)

X_raw = data[predictor_cols].replace([np.inf, -np.inf], np.nan)
X_num = X_raw.apply(pd.to_numeric, errors="coerce")

y_count = pd.to_numeric(data["datacenter_count"], errors="coerce")
y_bin   = pd.to_numeric(data["has_datacenter"], errors="coerce")

mask = X_num.notnull().all(axis=1) & y_count.notnull() & y_bin.notnull()

data_clean = data.loc[mask].copy()
X_num_sub  = X_num.loc[mask].copy()
y_count_sub = y_count.loc[mask].copy()
y_bin_sub   = y_bin.loc[mask].copy()

# reset indices together
data_clean = data_clean.reset_index(drop=True)
X_num_sub  = X_num_sub.reset_index(drop=True)
y_count_sub = y_count_sub.reset_index(drop=True)
y_bin_sub   = y_bin_sub.reset_index(drop=True)

# CONUS filter
EXCLUDE_STATEFP = {"02","15","60","66","69","72","74","78"}  # AK, HI, territories
data_clean["STATEFP"] = data_clean["CID"].astype(str).str.zfill(5).str[:2]
keep = ~data_clean["STATEFP"].isin(EXCLUDE_STATEFP)

data_clean = data_clean.loc[keep].reset_index(drop=True)
X_num      = X_num_sub.loc[keep].reset_index(drop=True)
y_count    = y_count_sub.loc[keep].reset_index(drop=True).astype(float)
y_bin      = y_bin_sub.loc[keep].reset_index(drop=True).astype(int)

regs["STATEFP"] = regs["CID"].str[:2]
conus_regs = regs[~regs["STATEFP"].isin(EXCLUDE_STATEFP)]
print("rows in regs:", len(regs))
print("CONUS rows in regs:", len(conus_regs))
print("unique CONUS CIDs in regs:", conus_regs["CID"].nunique())

# z-score predictors
means = X_num.mean()
stds  = X_num.std(ddof=0)

zero_std = stds[stds == 0].index.tolist()
if zero_std:
    print("\n[WARN] Dropping zero-variance predictors:", zero_std)
    predictor_cols = [c for c in predictor_cols if c not in zero_std]
    X_num = X_num[predictor_cols]
    means = means[predictor_cols]
    stds  = stds[predictor_cols]

X_z = (X_num - means) / stds

for c in predictor_cols:
    data_clean[c] = X_z[c].astype(float)

# Build design matrices
X_pois  = sm.add_constant(X_z)
X_logit = X_pois.copy()

#  Negative Binomial
nb_res = sm.GLM(y_count, X_pois, family=NegativeBinomial()).fit()
print("\n=== Negative Binomial GLM results (county) ===")
print(nb_res.summary())

# Residuals for  diagnostics
data_clean["resid_nb"]    = nb_res.resid_response

# Attach geometry (county shapefile) and build GeoDataFrame

gdf_county = gpd.read_file(SHAPEFILE)
print("\nShapefile columns:", gdf_county.columns.tolist())
EXCLUDE_STATEFP = {"02","15","60","66","69","72","74","78"}  # AK, HI, territories

gdf_county = gpd.read_file(SHAPEFILE)

# keep only contiguous US counties
if "STATEFP" not in gdf_county.columns:
    raise ValueError("County shapefile missing STATEFP; cannot filter CONUS cleanly.")
gdf_county = gdf_county[~gdf_county["STATEFP"].astype(str).str.zfill(2).isin(EXCLUDE_STATEFP)].copy()

if ID_COL_SHAPE not in gdf_county.columns:
    raise ValueError(f"Shapefile missing '{ID_COL_SHAPE}' column.")

gdf_county = gdf_county.rename(columns={ID_COL_SHAPE: "CID"})
gdf_county["CID"] = gdf_county["CID"].astype(str).str.strip().str.zfill(5)

data_clean["CID"] = data_clean["CID"].astype(str).str.strip().str.zfill(5)

gdf = gdf_county[["CID", "geometry"]].merge(
    data_clean,
    on="CID",
    how="inner"
)

gdf = gdf.dropna(subset=["datacenter_count"])

print("\nGeoDataFrame shape after merge:", gdf.shape)
print(gdf[["CID"]].head())

# Geographic KNN (centroid-based)
K_NEIGHBORS = 8

w_geo = KNN.from_dataframe(gdf, k=K_NEIGHBORS)
w_geo.transform = "r"

print(f"\n[WEIGHTS] Geographic KNN (k={K_NEIGHBORS}):")
print("  Number of regions in w_geo:", len(w_geo.id_order))

#MORAN
resid_nb_sub    = gdf["resid_nb"].values

mi_nb    = Moran(resid_nb_sub, w_geo)


print("\n=== Moran's I for NB residuals (county, W=KNN) ===")
print("I:", mi_nb.I)
print("p-value (perm):", mi_nb.p_sim)

# Clustering analysis (MSA-level, k-means + silhouette)
# City vs non-city county statistics

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.stats import mannwhitneyu

# Facility-level file must have MSA id + lon/lat per datacenter
FACILITIES_CSV = "Full_DC_Counts_County_REBUILT.csv"
FAC_MSA_COL    = "MSA"
FAC_LON_COL    = "longitude"
FAC_LAT_COL    = "latitude"

# County-level counts
COUNTY_COUNTS_CSV = "/Users/camilla/PycharmProjects/datacenters/nature_cities/Full_DC_Counts_County_REBUILT.csv"

# Shapefiles
COUNTY_SHP = "tl_2024_us_county/tl_2024_us_county.shp"  # GEOID
CBSA_SHP   = "tl_2024_us_cbsa/tl_2024_us_cbsa.shp"      # CBSA polygons (metro+micro); has geometry + CBSAFP

# Output
FIG_DIR = "Figures"
OUT_FIG = os.path.join(FIG_DIR, "pan_dist_clustering.png")

# CLUSTERING PARAMETERS
K_MAX = 6
K_MIN = 2
N_INIT = 25
RANDOM_STATE = 0

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_CONUS = "EPSG:5070"

# Helpers
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def km(x_meters):
    return x_meters / 1000.0

def pairwise_centroid_distances(centroids):
    # centroids: (k,2)
    k = centroids.shape[0]
    if k < 2:
        return np.array([])
    dists = []
    for i in range(k):
        for j in range(i+1, k):
            dists.append(np.linalg.norm(centroids[i] - centroids[j]))
    return np.array(dists)

def choose_k_by_silhouette(X, k_min=2, k_max=6, n_init=25, random_state=0):
    # Returns (best_k, best_model, best_sil)
    best = (None, None, -np.inf)
    for k in range(k_min, min(k_max, len(X) - 1) + 1):
        model = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = model.fit_predict(X)
        # silhouette needs at least 2 clusters and no empty clusters; KMeans guarantees that if it converges
        sil = silhouette_score(X, labels, metric="euclidean")
        if sil > best[2]:
            best = (k, model, sil)
    return best


# CLUSTERING ANALYSIS
FACILITIES_SHP = "/Users/camilla/PycharmProjects/datacenters/Cleaned Shapefile/DC_Shapefile.shp"
CBSA_SHP       = "tl_2024_us_cbsa/tl_2024_us_cbsa.shp"  # CBSA polygons (metro+micro)

EXCLUDE_STATE_NAMES = {
    "Alaska", "Hawaii", "Puerto Rico", "Guam",
    "American Samoa", "Northern Mariana Islands",
    "U.S. Virgin Islands", "United States Virgin Islands",
    "Virgin Islands", "U.S. Minor Outlying Islands"
}

def run_clustering():
    if not os.path.exists(FACILITIES_SHP):
        raise FileNotFoundError(f"Facility shapefile not found: {FACILITIES_SHP}")
    if not os.path.exists(CBSA_SHP):
        raise FileNotFoundError(f"CBSA shapefile not found: {CBSA_SHP}")

    #  read facilities shapefile
    gfac = gpd.read_file(FACILITIES_SHP)

    # explicit contiguous-only filter (by full state name)
    if "state" in gfac.columns:
        gfac["state"] = gfac["state"].astype(str).str.strip()
        gfac = gfac[~gfac["state"].isin(EXCLUDE_STATE_NAMES)].copy()

    # ensure point geometry
    if not (gfac.geometry.geom_type == "Point").all():
        gfac = gfac.copy()
        gfac["geometry"] = gfac.geometry.centroid

    # ensure CRS
    if gfac.crs is None:
        gfac = gfac.set_crs(CRS_WGS84)

    #  read CBSA polygons
    cbsa = gpd.read_file(CBSA_SHP)
    if cbsa.crs is None:
        cbsa = cbsa.set_crs(CRS_WGS84)

    # project both to CONUS meters
    gfac = gfac.to_crs(CRS_CONUS)
    cbsa = cbsa.to_crs(CRS_CONUS)

    # CBSA id column
    cbsa_id = "GEOID" if "GEOID" in cbsa.columns else ("CBSAFP" if "CBSAFP" in cbsa.columns else None)
    if cbsa_id is None:
        raise ValueError("CBSA shapefile missing GEOID/CBSAFP column.")

    cbsa = cbsa[[cbsa_id, "geometry"]].rename(columns={cbsa_id: "CBSAFP"})

    # assign CBSA to each facility
    j = gpd.sjoin(gfac, cbsa, predicate="within", how="left").drop(columns="index_right")

    # keep only facilities that fall inside a CBSA (metro OR micro)
    j["CBSAFP"] = j["CBSAFP"].astype(str)
    j = j[j["CBSAFP"].str.fullmatch(r"\d{5}")].copy()

    # cluster at CBSA level (metro + micro together)
    j["CBSA"] = "CBSA_" + j["CBSAFP"]

    # only CBSAs with >= 3 facilities
    cbsa_counts = j.groupby("CBSA").size()
    keep_cbsas = cbsa_counts[cbsa_counts >= 3].index.tolist()
    j = j[j["CBSA"].isin(keep_cbsas)].copy()

    # per-MSA clustering stats
    rows = []
    for cbsa_id, sub in j.groupby("CBSA"):

        coords = np.vstack([sub.geometry.x.values, sub.geometry.y.values]).T
        n = coords.shape[0]
        if n < 3:
            continue

        best_k, best_model, best_sil = choose_k_by_silhouette(
            coords, k_min=K_MIN, k_max=K_MAX, n_init=N_INIT, random_state=RANDOM_STATE
        )

        labels = best_model.labels_
        centroids = best_model.cluster_centers_

        d = np.linalg.norm(coords - centroids[labels], axis=1)  # meters

        intra_mean = d.mean()
        intra_median = np.median(d)
        intra_max = d.max()
        intra_sum = d.sum()

        cd = pairwise_centroid_distances(centroids)  # meters
        cent_mean = cd.mean() if len(cd) else np.nan
        cent_median = np.median(cd) if len(cd) else np.nan

        rows.append({
            "CBSA": cbsa_id,
            "n_facilities": n,
            "k_hat": best_k,
            "silhouette": best_sil,
            "intra_mean_km": km(intra_mean),
            "intra_median_km": km(intra_median),
            "intra_max_km": km(intra_max),
            "intra_sum_km": km(intra_sum),
            "centroid_mean_km": km(cent_mean) if np.isfinite(cent_mean) else np.nan,
            "centroid_median_km": km(cent_median) if np.isfinite(cent_median) else np.nan,
        })

    df = pd.DataFrame(rows)

    intra_dist = df["intra_mean_km"].dropna().values
    cent_dist  = df["centroid_mean_km"].dropna().values

    intra_mean_all = float(np.mean(intra_dist)) if len(intra_dist) else np.nan
    intra_median_all = float(np.median(intra_dist)) if len(intra_dist) else np.nan
    cent_mean_all = float(np.mean(cent_dist)) if len(cent_dist) else np.nan
    cent_median_all = float(np.median(cent_dist)) if len(cent_dist) else np.nan

    ensure_dir(FIG_DIR)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(intra_dist, bins=40)
    ax.axvline(intra_mean_all, linestyle="--", linewidth=1)
    ax.axvline(intra_median_all, linestyle=":", linewidth=1)
    ax.set_xlabel("Average within-cluster distance (km)")
    ax.set_ylabel("MSA count")
    ax.set_title("Within-cluster distances across MSAs")
    plt.tight_layout()
    left_path = os.path.join(FIG_DIR, "_tmp_left.png")
    fig.savefig(left_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.hist(cent_dist, bins=40)
    ax.axvline(cent_mean_all, linestyle="--", linewidth=1)
    ax.axvline(cent_median_all, linestyle=":", linewidth=1)
    ax.set_xlabel("Average centroid-to-centroid distance (km)")
    ax.set_ylabel("MSA count")
    ax.set_title("Centroid distances across MSAs")
    plt.tight_layout()
    right_path = os.path.join(FIG_DIR, "_tmp_right.png")
    fig.savefig(right_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    import matplotlib.image as mpimg
    img1 = mpimg.imread(left_path)
    img2 = mpimg.imread(right_path)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    axes[0].imshow(img1); axes[0].axis("off")
    axes[1].imshow(img2); axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(OUT_FIG, dpi=300, bbox_inches="tight")
    plt.close(fig)

    try:
        os.remove(left_path)
        os.remove(right_path)
    except Exception:
        pass

    clustering_summary = {
        "n_msas_clustered": int(df.shape[0]),
        "k_max": K_MAX,
        "k_min": K_MIN,
        "intra_mean_of_means_km": intra_mean_all,
        "intra_median_of_means_km": intra_median_all,
        "centroid_mean_of_means_km": cent_mean_all,
        "centroid_median_of_means_km": cent_median_all,
        "k_hat_mean": float(df["k_hat"].mean()) if len(df) else np.nan,
        "k_hat_median": float(df["k_hat"].median()) if len(df) else np.nan,
    }

    return df, clustering_summary

#CITY VS NON-CITY COUNTY STATS
def run_city_stats():
    # county counts
    counts = pd.read_csv(COUNTY_COUNTS_CSV, dtype=str)
    if "GEOID" not in counts.columns or "dc_count" not in counts.columns:
        raise ValueError("COUNTY_COUNTS_CSV must have GEOID and dc_count.")

    counts["GEOID"] = counts["GEOID"].astype(str).str.strip().str.zfill(5)
    counts["dc_count"] = pd.to_numeric(counts["dc_count"], errors="coerce").fillna(0).astype(int)

    # county geometry
    EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "74", "78"}  # AK, HI, territories

    counties = gpd.read_file(COUNTY_SHP)[["GEOID", "STATEFP", "geometry"]].copy()
    counties["GEOID"] = counties["GEOID"].astype(str).str.strip().str.zfill(5)
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)

    # keep only contiguous US counties
    counties = counties[~counties["STATEFP"].isin(EXCLUDE_STATEFP)].copy()

    # join counts
    gcount = counties.merge(counts[["GEOID", "dc_count"]], on="GEOID", how="left")
    gcount["dc_count"] = gcount["dc_count"].fillna(0).astype(int)

    # CBSA polygons (metro + micro)
    cbsa = gpd.read_file(CBSA_SHP)[["geometry"]].copy()

    # centroid-based classification
    # compute centroids in a projected CRS to avoid geographic centroid weirdness
    gcount_proj = gcount.to_crs(CRS_CONUS)
    centroids = gcount_proj.geometry.centroid
    centroids_gdf = gpd.GeoDataFrame(gcount_proj[["GEOID", "dc_count"]].copy(),
                                     geometry=centroids, crs=CRS_CONUS)
    cbsa_proj = cbsa.to_crs(CRS_CONUS)

    # spatial join: centroid within CBSA => city county
    joined = gpd.sjoin(centroids_gdf, cbsa_proj, predicate="within", how="left")
    joined["is_city_county"] = joined["index_right"].notnull()

    city = joined.loc[joined["is_city_county"], "dc_count"].values
    non  = joined.loc[~joined["is_city_county"], "dc_count"].values
    pct_city_ge1 = 100 * (city >= 1).mean()
    pct_non_ge1 = 100 * (non >= 1).mean()
    print("Pct city counties with ≥1 DC:", pct_city_ge1)
    print("Pct non-city counties with ≥1 DC:", pct_non_ge1)

    # stats
    n_city = int(len(city))
    n_non  = int(len(non))

    mean_city = float(np.mean(city)) if n_city else np.nan
    mean_non  = float(np.mean(non)) if n_non else np.nan
    med_city  = float(np.median(city)) if n_city else np.nan
    med_non   = float(np.median(non)) if n_non else np.nan

    pct_city_ge1 = 100.0 * float(np.mean(city >= 1)) if n_city else np.nan
    pct_non_ge1  = 100.0 * float(np.mean(non >= 1)) if n_non else np.nan

    out = {
        "n_city": n_city,
        "n_non": n_non,
        "mean_city": mean_city,
        "mean_non": mean_non,
        "median_city": med_city,
        "median_non": med_non,
        "pct_city_ge1": pct_city_ge1,
        "pct_non_ge1": pct_non_ge1,
    }
    return out


# MAIN
def main():
    # clustering
    clustering_df, clustering_summary = run_clustering()

    # city stats
    city_stats = run_city_stats()

if __name__ == "__main__":
    main()

