import pandas as pd
import geopandas as gpd
from rasterstats import zonal_stats
import rasterio

# PATHS
SHAPEFILE = "tl_2024_us_county/tl_2024_us_county.shp"
SHAPEFILE2 = "tl_2024_us_cbsa/tl_2024_us_cbsa.shp"
LANDVAL_TIF = "/Users/camilla/Downloads/places_fmv_pnas_dryad/1 estimates/places_fmv_all.tif"
OUT_CSV = "county_land_prices.csv"

EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "74", "78"}  # drop AK, HI, territories

# Load counties
counties = gpd.read_file(SHAPEFILE)
# Clean FIPS
counties["CID"] = counties["GEOID"].astype(str).str.strip().str.zfill(5)
counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)

# Keep contiguous US only
counties = counties[~counties["STATEFP"].isin(EXCLUDE_STATEFP)].copy()

# Match CRS to raster
with rasterio.open(LANDVAL_TIF) as src:
    if counties.crs != src.crs:
        counties = counties.to_crs(src.crs)
    nodata = src.nodata

# Zonal stats (mean and median)
zs = zonal_stats(
    counties,
    LANDVAL_TIF,
    stats=["mean", "median"],
    nodata=nodata
)

# Build final table
df = pd.concat([counties[["CID"]].reset_index(drop=True),
                pd.DataFrame(zs)], axis=1)

df = df.rename(columns={
    "mean": "landval_mean_log",
    "median": "landval_med_log"
})

# Save to CSV
df.to_csv(OUT_CSV, index=False)

print("Saved:", OUT_CSV)


"""
# COUNTY SCRIPT — MODEL B
# Negative Binomial at county level
# Adds county-level land values from places_fmv_all.tif via zonal statistics
# Moran's I on NB residuals

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial
import geopandas as gpd
import rasterio

from rasterstats import zonal_stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from esda import Moran
from libpysal.weights import KNN

# =========================
# PATHS
# =========================
COUNTS_CSV = "nature_cities/Full_DC_Counts_County_REBUILT.csv"
REGS_CSV   = "county_regressors_totals_nh_area_eia_noprox_ret.csv"
SHAPEFILE  = "tl_2024_us_county/tl_2024_us_county.shp"
LANDVAL_TIF = "/Users/camilla/Downloads/places_fmv_pnas_dryad/1 estimates/places_fmv_all.tif"

ID_COL_SHAPE = "GEOID"

# =========================
# LABELS
# =========================
label_map = {
    "water_cf":                 "Water-stress index",
    "bqs":                      "Broadband quality score",
    "nri_events":               "Natural hazard events",
    "evi":                      "Energy grid stress",
    "benefit_flag":             "State-tax incentives",
    "retired_coal":             "Retired coal plants",
    "elec_var_ind_all":         "Electricity-price variability",
    "SAMI_energy_exp":          "Energy expenditure",
    "SAMI_energy_emp_it_total": "IT employment",
    "cap_total":                "Generation capacity",
    "landval_med_log":          "Land value (median log FMV)"
}

EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "74", "78"}  # AK, HI, territories

# =========================
# HELPERS
# =========================
def build_county_land_value(shapefile_path, raster_path, id_col="GEOID"):
  
    #Aggregate places_fmv_all.tif to county level using zonal statistics.

    #Output columns:
   #   CID
   #   landval_mean_log
   #   landval_med_log


    counties = gpd.read_file(shapefile_path)
    counties[id_col] = counties[id_col].astype(str).str.strip().str.zfill(5)
    counties["STATEFP"] = counties["STATEFP"].astype(str).str.zfill(2)

    # Keep contiguous US only
    counties = counties[~counties["STATEFP"].isin(EXCLUDE_STATEFP)].copy()

    with rasterio.open(raster_path) as src:
        raster_crs = src.crs
        nodata = src.nodata

    if counties.crs != raster_crs:
        counties = counties.to_crs(raster_crs)

    zs = zonal_stats(
        counties,
        raster_path,
        stats=["mean", "median"],
        nodata=nodata,
        all_touched=False
    )

    zs_df = pd.DataFrame(zs)
    out = pd.concat([counties[[id_col]].reset_index(drop=True), zs_df], axis=1)

    out = out.rename(columns={
        id_col: "CID",
        "mean": "landval_mean_log",
        "median": "landval_med_log",
    })

    out["CID"] = out["CID"].astype(str).str.strip()
    return out


def compute_vif(X_df):

    #Compute VIF on a numeric design matrix without constant.

    X_tmp = X_df.copy()
    X_tmp = X_tmp.replace([np.inf, -np.inf], np.nan).dropna()

    vif_rows = []
    for i, col in enumerate(X_tmp.columns):
        vif_rows.append({
            "variable": col,
            "VIF": variance_inflation_factor(X_tmp.values, i)
        })
    return pd.DataFrame(vif_rows).sort_values("VIF", ascending=False)


# =========================
# LOAD COUNTS + REGS
# =========================
counts = pd.read_csv(COUNTS_CSV, dtype=str)
regs = pd.read_csv(REGS_CSV, dtype={"FIPS": str})

print("Counts columns:", counts.columns.tolist())
print("Regs columns:", regs.columns.tolist())

# Build county key in counts
if "GEOID" not in counts.columns:
    raise ValueError("COUNTS_CSV missing 'GEOID' column.")
counts["CID"] = counts["GEOID"].astype(str).str.strip().str.zfill(5)

# Outcomes
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

# Build county key in regressors
if "CID" in regs.columns:
    regs["CID"] = regs["CID"].astype(str).str.strip().str.zfill(5)
elif "FIPS" in regs.columns:
    regs["CID"] = regs["FIPS"].astype(str).str.strip().str.zfill(5)
else:
    raise ValueError("REGS_CSV must contain a county id column named 'FIPS' or 'CID'.")

# =========================
# BUILD COUNTY LAND VALUE FROM RASTER
# =========================
landvals = build_county_land_value(
    shapefile_path=SHAPEFILE,
    raster_path=LANDVAL_TIF,
    id_col=ID_COL_SHAPE
)

print("\nLand-value table head:")
print(landvals.head())

print("\nMissing county land values:", landvals["landval_med_log"].isna().sum())

# Merge land values into regressors
regs = regs.merge(landvals, on="CID", how="left")

# =========================
# MERGE FULL ANALYSIS TABLE
# =========================
data = regs.merge(
    counts[["CID", "CName", "datacenter_count", "has_datacenter"]],
    on="CID",
    how="left"
)

data["datacenter_count"] = pd.to_numeric(data["datacenter_count"], errors="coerce").fillna(0).astype(int)
data["has_datacenter"] = pd.to_numeric(data["has_datacenter"], errors="coerce").fillna(0).astype(int)

print("\nMerged data head:")
print(data.head())

# =========================
# EXPLICIT MODEL B PREDICTORS
# =========================
predictor_cols = [
    "cap_total",
    "SAMI_energy_emp_it_total",
    "nri_events",
    "bqs",
    "retired_coal",
    "SAMI_energy_exp",
    "benefit_flag",
    "evi",
    "elec_var_ind_all",
    "water_cf",
    "landval_med_log",
]

missing_predictors = [c for c in predictor_cols if c not in data.columns]
if missing_predictors:
    raise ValueError(f"Missing predictors in merged data: {missing_predictors}")

print("\nPredictors used (Model B):", predictor_cols)

# =========================
# CLEAN DATA
# =========================
X_raw = data[predictor_cols].replace([np.inf, -np.inf], np.nan)
X_num = X_raw.apply(pd.to_numeric, errors="coerce")

y_count = pd.to_numeric(data["datacenter_count"], errors="coerce")
y_bin = pd.to_numeric(data["has_datacenter"], errors="coerce")

mask = X_num.notnull().all(axis=1) & y_count.notnull() & y_bin.notnull()

data_clean = data.loc[mask].copy().reset_index(drop=True)
X_num_sub = X_num.loc[mask].copy().reset_index(drop=True)
y_count_sub = y_count.loc[mask].copy().reset_index(drop=True).astype(float)
y_bin_sub = y_bin.loc[mask].copy().reset_index(drop=True).astype(int)

# CONUS filter again, just to be safe
data_clean["STATEFP"] = data_clean["CID"].astype(str).str.zfill(5).str[:2]
keep = ~data_clean["STATEFP"].isin(EXCLUDE_STATEFP)

data_clean = data_clean.loc[keep].reset_index(drop=True)
X_num = X_num_sub.loc[keep].reset_index(drop=True)
y_count = y_count_sub.loc[keep].reset_index(drop=True)
y_bin = y_bin_sub.loc[keep].reset_index(drop=True)

print("\nFinal analytic sample size:", len(data_clean))
print("Counties with >=1 data center:", int((y_count > 0).sum()))

# Drop zero-variance predictors
means = X_num.mean()
stds = X_num.std(ddof=0)

zero_std = stds[stds == 0].index.tolist()
if zero_std:
    print("\n[WARN] Dropping zero-variance predictors:", zero_std)
    predictor_cols = [c for c in predictor_cols if c not in zero_std]
    X_num = X_num[predictor_cols]
    means = means[predictor_cols]
    stds = stds[predictor_cols]

# z-score predictors
X_z = (X_num - means) / stds

for c in predictor_cols:
    data_clean[c] = X_z[c].astype(float)

# =========================
# VIF
# =========================
print("\n=== VIF (Model B predictors, z-scored) ===")
vif_df = compute_vif(X_z)
print(vif_df)

# =========================
# DESIGN MATRIX + NEGATIVE BINOMIAL
# =========================
X_nb = sm.add_constant(X_z, has_constant="add")

nb_res = sm.GLM(y_count, X_nb, family=NegativeBinomial()).fit()

print("\n=== Negative Binomial GLM results (county, Model B) ===")
print(nb_res.summary())

# Save residuals for spatial diagnostics
data_clean["resid_nb"] = nb_res.resid_response

# =========================
# ATTACH GEOMETRY
# =========================
gdf_county = gpd.read_file(SHAPEFILE)

if "STATEFP" not in gdf_county.columns:
    raise ValueError("County shapefile missing STATEFP; cannot filter CONUS cleanly.")
if ID_COL_SHAPE not in gdf_county.columns:
    raise ValueError(f"Shapefile missing '{ID_COL_SHAPE}' column.")

gdf_county["STATEFP"] = gdf_county["STATEFP"].astype(str).str.zfill(2)
gdf_county = gdf_county[~gdf_county["STATEFP"].isin(EXCLUDE_STATEFP)].copy()

gdf_county = gdf_county.rename(columns={ID_COL_SHAPE: "CID"})
gdf_county["CID"] = gdf_county["CID"].astype(str).str.strip().str.zfill(5)
data_clean["CID"] = data_clean["CID"].astype(str).str.strip().str.zfill(5)

gdf = gdf_county[["CID", "geometry"]].merge(
    data_clean,
    on="CID",
    how="inner"
)

gdf = gdf.dropna(subset=["datacenter_count"]).copy()

print("\nGeoDataFrame shape after merge:", gdf.shape)
print(gdf[["CID"]].head())

# =========================
# MORAN'S I ON NB RESIDUALS
# =========================
K_NEIGHBORS = 8

w_geo = KNN.from_dataframe(gdf, k=K_NEIGHBORS)
w_geo.transform = "r"

print(f"\n[WEIGHTS] Geographic KNN (k={K_NEIGHBORS})")
print("Number of regions in w_geo:", len(w_geo.id_order))

resid_nb_sub = gdf["resid_nb"].values
mi_nb = Moran(resid_nb_sub, w_geo)

print("\n=== Moran's I for NB residuals (county, Model B, W=KNN) ===")
print("I:", mi_nb.I)
print("p-value (perm):", mi_nb.p_sim)

# =========================
# OPTIONAL: NICE COEFFICIENT TABLE
# =========================
coef_table = pd.DataFrame({
    "variable": nb_res.params.index,
    "coef": nb_res.params.values,
    "se": nb_res.bse.values,
    "z": nb_res.tvalues.values,
    "p": nb_res.pvalues.values
})

coef_table["label"] = coef_table["variable"].map(label_map).fillna(coef_table["variable"])

print("\n=== Coefficient table (Model B) ===")
print(coef_table[["variable", "label", "coef", "se", "z", "p"]])

# Optional export
# coef_table.to_csv("county_modelB_nb_coefficients.csv", index=False)
# vif_df.to_csv("county_modelB_vif.csv", index=False)
# gdf[["CID", "resid_nb", "geometry"]].to_file("county_modelB_residuals.gpkg", driver="GPKG")
"""