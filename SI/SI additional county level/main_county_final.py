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
REGS_CSV   = 'county_regressors_totals_nh_area_eia_noprox_ret_review3.csv'#"county_regressors_totals_nh_area_eia_noprox_ret.csv"
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
    "land_prices":             "Land prices",
    "pov":                     "Poverty",
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
print("\n=== VIF ===")

X_vif = X_z.copy()

# drop rows with NaNs just in case
X_vif = X_vif.replace([np.inf, -np.inf], np.nan).dropna()

vif_list = []
for i, col in enumerate(X_vif.columns):
    vif_list.append({
        "variable": col,
        "VIF": variance_inflation_factor(X_vif.values, i)
    })

vif_df = pd.DataFrame(vif_list).sort_values("VIF", ascending=False)
print(vif_df)
for c in predictor_cols:
    data_clean[c] = X_z[c].astype(float)

# Build design matrices
X_pois  = sm.add_constant(X_z)
X_logit = X_pois.copy()

#  Negative Binomial
nb_res = sm.GLM(y_count, X_pois, family=NegativeBinomial()).fit()
print("\n=== Negative Binomial GLM results (county) ===")
print(nb_res.summary())



