# GLM(NB) + FOREST PLOT and DOMINANT DRIVER MAP (Figure 2) +
# POPULATION PROJECTIONS + SCENARIOS TRAJECTORIES and MAPS (Figure 4)

import matplotlib.font_manager as fm
import matplotlib as mpl
def set_big_font_style():
    mpl.rcParams.update({
        "font.size": 18,
        "axes.labelsize": 18,
        "axes.titlesize": 18,
        "legend.fontsize": 18,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.linewidth": 1.2,
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })

arial_path = "/System/Library/Fonts/Supplemental/Arial.ttf"
fm.fontManager.addfont(arial_path)

mpl.rcParams["font.family"] = "Arial"
mpl.rcParams.update({
    # font
    "font.size": 13,
    "axes.labelsize": 15,
    "axes.titlesize": 16,
    "legend.fontsize": 13,

    # ticks
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,

    # axes
    "axes.linewidth": 1.0,

    # output
    "figure.dpi": 300,
    "savefig.dpi": 600,
})

import numpy as np
import pandas as pd
import statsmodels.api as sm
import geopandas as gpd
from libpysal.weights import Queen
from esda import Moran
from matplotlib.patches import Patch
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
# INPUT FILES
COUNTS_CSV    = "nature_cities/DC_MSA_Micro_Final_REBUILT.csv"
REGS_CSV      = "msa_regressors_totals_nh_area_eia_noprox_ret.csv"
PROJ_CSV      = "msa_population_projections_SSP_wide_by_year.csv"
SHAPEFILE     = "tl_2024_us_cbsa/tl_2024_us_cbsa.shp"
STATE_SHP     = "State/tl_2023_us_state.shp"

BASELINE_YEAR = 2023

#labels
label_map = {
    "water_sum":                "Water-stress index",
    "bqs_mean":                 "Broadband quality",
    "events_sum_per_area":      "Natural hazard events",
    "events_per_area":          "Natural hazard events",
    "evi_mean":                 "Energy grid stress",
    "benefit_flag":             "State tax incentives",
    "msa_ret_coal":             "Retired coal plants",
    "elec_var_ind_all":         "Electricity-price variability",
    "SAMI_energy_exp":          "Energy expenditure",
    "SAMI_energy_emp_it_total": "IT employment",
    "cap_total":                "Nameplate capacity"
}


# Load states (contiguous US)
states = gpd.read_file(STATE_SHP)
states = states[~states["STUSPS"].isin(["AK", "HI", "PR", "GU", "VI", "MP", "AS"])].copy()


# Read inputs

counts = pd.read_csv(COUNTS_CSV, dtype=str)
regs   = pd.read_csv(REGS_CSV)

counts["MSA"] = counts["msa_code_c"].astype(str)
counts["MSATitle"] = counts["msa_name_clean"].astype(str)
counts["datacenter_count"] = pd.to_numeric(counts["dc_count"], errors="coerce").astype("Int64")
counts["has_datacenter"] = pd.to_numeric(counts["has_dc"], errors="coerce").astype("Int64")

# Merge regs + counts

regs["MSA"] = regs["MSA"].astype(str)
data = regs.merge(
    counts[["MSA", "MSATitle", "datacenter_count", "has_datacenter"]],
    on="MSA",
    how="left"
)
def show_drop(prev_df, next_df, label):
    prev = set(prev_df["MSA"].astype(str))
    nxt  = set(next_df["MSA"].astype(str))
    lost = sorted(list(prev - nxt))
    print(f"\n[TRACE] {label}")
    print("  prev rows:", len(prev_df), "prev unique MSA:", len(prev))
    print("  next rows:", len(next_df), "next unique MSA:", len(nxt))
    print("  LOST:", len(lost), "->", lost[:50])
    return lost

# after merge
print("\n[TRACE] after merge")
print("  rows:", len(data), "unique MSA:", data["MSA"].nunique())

# after dropping missing outcome vars
data1 = data.copy()
data1 = data1.dropna(subset=["datacenter_count", "has_datacenter"]).copy()
lost1 = show_drop(data, data1, "dropna(datacenter_count, has_datacenter)")

#after cleaning/standardizing predictors
exclude = ["MSA", "MSATitle", "datacenter_count", "has_datacenter"]
predictor_cols = [c for c in data1.columns if c not in exclude]

X_raw = data1[predictor_cols].replace([np.inf, -np.inf], np.nan)
mask = X_raw.notnull().all(axis=1)

data2 = data1.loc[mask].copy()
lost2 = show_drop(data1, data2, "mask predictors notnull(all)")

data = data.dropna(subset=["datacenter_count", "has_datacenter"]).copy()
data["datacenter_count"] = data["datacenter_count"].astype(int)
data["has_datacenter"]   = data["has_datacenter"].astype(int)

# Predictors

exclude = ["MSA", "MSATitle", "datacenter_count", "has_datacenter"]
predictor_cols = [c for c in data.columns if c not in exclude]
#predictor_cols = [
#    "water_sum", "bqs_mean", "events_sum_per_area", "evi_mean",
#    "benefit_flag", "msa_ret_coal", "elec_var_ind_all",
#    "SAMI_energy_exp", "SAMI_energy_emp_it_total", "cap_total"
#]

# Drop rows with NaN/inf in predictors
X_raw = data[predictor_cols].replace([np.inf, -np.inf], np.nan)
mask = X_raw.notnull().all(axis=1)
data_clean = data.loc[mask].reset_index(drop=True)
print(f"\nKept {mask.sum()} rows; dropped {len(mask) - mask.sum()} rows.")
# sample size actually used in nb_full
n_total = len(data_clean)  # all city in the model
n_with_dc = int(data_clean["has_datacenter"].sum())  # how many cities has DC>=1
n_zero_dc = int((data_clean["has_datacenter"] == 0).sum())

# Z-score predictors
X_raw_clean = data_clean[predictor_cols].astype(float)

means = X_raw_clean.mean()
stds  = X_raw_clean.std(ddof=0)

zero_std_cols = stds[stds == 0].index.tolist()

X_z = (X_raw_clean - means) / stds

for c in predictor_cols:
    data_clean[c] = X_z[c].astype(float)

X_pois  = sm.add_constant(X_z)
X_logit = X_pois.copy()

y_count = data_clean["datacenter_count"].astype(float)
y_bin   = data_clean["has_datacenter"].astype(int)

# Fit GLMs
nb_full = sm.GLM(y_count, X_pois, family=sm.families.NegativeBinomial(alpha=1.0)).fit()
print("\n=== Negative Binomial GLM ===")
print(nb_full.summary())


# attach residuals for Moran’s I
data_clean["resid_nb"]    = nb_full.resid_response

# Geometry merge
msa_suffix = data_clean["MSA"].astype(str).str.strip().str.replace(r"^C", "", regex=True)
msa_num = pd.to_numeric(msa_suffix, errors="coerce")
data_clean["MSA_geo"] = (msa_num * 10).astype("Int64")
mask_geo = data_clean["MSA_geo"].notna()

gdf_msa = gpd.read_file(SHAPEFILE)
gdf_msa["MSA_geo"] = gdf_msa["CBSAFP"].astype(int)

gdf = gdf_msa[["MSA_geo", "geometry"]].merge(
    data_clean[["MSA", "MSATitle", "MSA_geo", "datacenter_count", "has_datacenter",
                "resid_nb"] + predictor_cols],
    on="MSA_geo",
    how="inner"
)

TARGET_EPSG = 5070  # US Albers Equal Area

# States (CONUS) -> 5070
states = states.to_crs(epsg=TARGET_EPSG)

# CBSA -> 5070
gdf_msa = gpd.read_file(SHAPEFILE)
gdf_msa["MSA_geo"] = gdf_msa["CBSAFP"].astype(int)
gdf_msa = gdf_msa.to_crs(epsg=TARGET_EPSG)

# Dominant-driver map (significant NB regressors)

DOMINANT_PALETTE = [
    "#6c52a3",  # purple
    "#6db0ba",  # teal
    "#50b24e",  # green
    "#f2a7c6",  # pink
    "#a4c5dc",  # light blue
    "#7f7f7f",  # gray
    "#c7d8e7",  # very light blue
    "#2686bc",  # blue
    "#8d99ae",  # gray-blue
    "#f0c36d",  # sand
    "#e07a5f",  # coral

]

def build_dominant_driver_map(
    gdf: gpd.GeoDataFrame,
    model,
    predictor_cols: list[str],
    label_map: dict[str, str],
    alpha: float = 0.05,
    out_png: str = "dominant_driver_map.png",
    out_pdf: str = "dominant_driver_map.pdf",
    states_gdf: gpd.GeoDataFrame | None = None,
    palette: list[str] = DOMINANT_PALETTE,
    msa_edgecolor: str = "white",
    msa_linewidth: float = 0.25,
    state_color: str = "#d0d0d0",
    state_linewidth: float = 0.8,
    show_legend: bool = True,
):
    #  significant predictors
    pvals = model.pvalues
    sig_terms = [t for t in pvals.index if (t != "const") and (pvals[t] < alpha)]
    sig_terms = [t for t in sig_terms if t in predictor_cols]

    out = gdf.copy()

    # contribution = beta * z
    betas = model.params[sig_terms]
    contrib = pd.DataFrame(index=out.index, columns=sig_terms, dtype="float64")
    for t in sig_terms:
        z = pd.to_numeric(out[t], errors="coerce")
        contrib[t] = float(betas[t]) * z

    abs_contrib = contrib.abs()
    dom_term = abs_contrib.idxmax(axis=1)

    # if a row is all-NaN across sig_terms -> dom_term should be NaN
    all_nan = abs_contrib.isna().all(axis=1)
    dom_term = dom_term.astype("object")
    dom_term.loc[all_nan] = np.nan
    out["dominant_term"] = dom_term

    if out.crs is None or out.crs.to_epsg() != 5070:
        out = out.to_crs(epsg=5070)

    st = None
    if states_gdf is not None and len(states_gdf) > 0:
        st = states_gdf
        if st.crs is None or st.crs.to_epsg() != 5070:
            st = st.to_crs(epsg=5070)
        conus_union = st.geometry.union_all()
        out = gpd.clip(out, conus_union)

    driver_order = [
        "cap_total",
        "SAMI_energy_emp_it_total",
        "bqs_mean",
        "events_sum_per_area",
        "msa_ret_coal",
        "SAMI_energy_exp",
        "evi_mean",
        "elec_var_ind_all",
        "benefit_flag",
        "water_sum",
    ]

    present = pd.Series(out["dominant_term"]).dropna().unique().tolist()
    cats = [c for c in driver_order if c in present]

    colors = {}
    if cats:
        colors = {cat: palette[i % len(palette)] for i, cat in enumerate(cats)}
        out["plot_color"] = (
            out["dominant_term"]
              .map(colors)
              .astype("object")
              .where(lambda s: s.notna(), "white")
        )
    else:
        out["plot_color"] = "white"

    # plot
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    if st is not None:
        st.boundary.plot(ax=ax, linewidth=state_linewidth, color=state_color, zorder=10)
        minx, miny, maxx, maxy = st.total_bounds
        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)

    out.plot(
        ax=ax,
        color=out["plot_color"].tolist(),
        edgecolor=msa_edgecolor,
        linewidth=msa_linewidth,
        zorder=5
    )

    # LEGEND
    if show_legend and cats:
        legend_handles = [
            Patch(facecolor=colors[cat], edgecolor="none", label=label_map.get(cat, cat))
            for cat in cats
        ]
        legend_handles.append(
            Patch(facecolor="white", edgecolor="#bbbbbb", label="No data center / not dominant")
        )

        ax.legend(
            handles=legend_handles,
            title="Dominant predictor",
            loc="lower left",
            bbox_to_anchor=(0.02, 0.02),
            frameon=True,
            framealpha=0.95,
            borderpad=0.6,
            handlelength=1.2,
            handletextpad=0.6,
            labelspacing=0.4,
        )

    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(out_png, dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    print(f"Saved: {out_png}, {out_pdf}")
    return out


# Forest plot
def forest_plot_nb(
    model,
    label_map: dict[str, str],
    alpha: float = 0.05,
    out_png: str = "figure_nb_forestplot.png",
    out_pdf: str = "figure_nb_forestplot.pdf",
):
    params = model.params
    conf   = model.conf_int()
    pvals  = model.pvalues

    terms = [t for t in params.index if (t != "const") and (pvals[t] < alpha)]

    dfp = pd.DataFrame({
        "term": terms,
        "beta": params[terms].values,
        "lo": conf.loc[terms, 0].values,
        "hi": conf.loc[terms, 1].values,
        "p":  pvals[terms].values,
    })

    dfp["abs_beta"] = dfp["beta"].abs()
    dfp = dfp.sort_values("abs_beta", ascending=True).reset_index(drop=True)
    #  significance stars
    def p_to_stars(p):
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        else:
            return ""

    dfp["stars"] = [p_to_stars(p) for p in dfp["p"]]

    # append stars
    labels = [f"{label_map.get(t, t)} {s}".rstrip()
              for t, s in zip(dfp["term"], dfp["stars"])]


    y = np.arange(len(dfp))

    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.45 * len(dfp))))
    ax.axvline(0, color="#999999", linewidth=0.8, linestyle="--")

    ax.errorbar(
        dfp["beta"],
        y,
        xerr=[dfp["beta"] - dfp["lo"], dfp["hi"] - dfp["beta"]],
        fmt="o",
        capsize=3,
        linewidth=1.0,
        color="black"
    )

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Standardized coefficient (β)", fontsize=10)
    ax.set_title(f"Negative Binomial – significant predictors (p<{alpha})", fontsize=11)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=400, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=400, bbox_inches="tight")
    plt.close(fig)

    print(f"[OK] Saved forest plot: {out_png}, {out_pdf}")
    return dfp

# Run dominant map + forest plot

dominant_out = build_dominant_driver_map(
    gdf=gdf,
    model=nb_full,
    predictor_cols=predictor_cols,
    label_map=label_map,
    alpha=0.05,
    states_gdf=states,
    out_png="dominant_driver_map.png",
    out_pdf="dominant_driver_map.pdf",
    show_legend=False,
)


forest_df = forest_plot_nb(
    model=nb_full,
    label_map=label_map,
    alpha=0.05,
    out_png="figure_nb_forestplot.png",
    out_pdf="figure_nb_forestplot.pdf",
    show_irr=True,
)


from libpysal.weights import KNN
from esda import Moran

# Moran's I (residuals) with KNN

gdf_knn = gdf.to_crs(epsg=TARGET_EPSG).copy()
K = 8

# Build KNN weights using centroid coordinates
w = KNN.from_dataframe(gdf_knn, k=K, geom_col="geometry")
w.transform = "r"

mi_nb    = Moran(gdf_knn["resid_nb"].values,    w)


print("\nMoran's I (KNN) NB residuals")
print("I:", mi_nb.I, "p:", mi_nb.p_sim)


def compute_vif_df(X: pd.DataFrame) -> pd.DataFrame:
    """
    Compute VIF for each column in X.
    """
    Xv = X.astype(float)
    out = []
    for i, col in enumerate(Xv.columns):
        vif = variance_inflation_factor(Xv.values, i)
        out.append((col, float(vif)))
    df = pd.DataFrame(out, columns=["term", "vif"])
    return df


def plot_vif(
    vif_df: pd.DataFrame,
    label_map: dict[str, str],
    out_png: str = "figure_vif.png",
    out_pdf: str = "figure_vif.pdf",
    vline_5: float = 5,
    vline_10: float = 10,
):
    """
    Horizontal bar plot of VIF with colors mapped to a fixed range [1.0, 1.3],
    without colorbar extensions.
    """

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import pandas as pd

    df = vif_df.copy()
    df["label"] = df["term"].map(lambda t: label_map.get(t, t))

    # Order: lowest VIF at bottom
    df = df.sort_values("vif", ascending=True).reset_index(drop=True)
    y = np.arange(len(df))

    # color scale
    vmin_cb, vmax_cb = 1.0, 1.3
    cmap = plt.cm.RdBu_r
    norm = mpl.colors.Normalize(vmin=vmin_cb, vmax=vmax_cb, clip=True)

    colors = cmap(norm(df["vif"].values))

    fig, ax = plt.subplots(figsize=(8.2, 4.6))

    ax.barh(
        y,
        df["vif"].values,
        color=colors,
        edgecolor="black",
        linewidth=0.3,
    )

    ax.set_yticks(y)
    ax.set_yticklabels(df["label"].values, fontsize=9)
    ax.set_xlabel("Variance inflation factor (VIF)", fontsize=10)

    # Reference lines
    ax.axvline(vline_5,  color="gray", linestyle="--", linewidth=1, label=f"VIF = {int(vline_5)}")
    ax.axvline(vline_10, color="gray", linestyle=":",  linewidth=1, label=f"VIF = {int(vline_10)}")

    xmax = max(vline_10, df["vif"].max()) * 1.05
    ax.set_xlim(0, xmax)

    # Colorbar
    smap = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])

    cbar = fig.colorbar(
        smap,
        ax=ax,
        fraction=0.046,
        pad=0.03,
    )

    cbar.set_label("VIF value", rotation=90)
    ticks = [1.0, 1.1, 1.2, 1.3]
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{t:.2f}" for t in ticks])

    # Legend
    ax.legend(
        frameon=False,
        loc="upper right",
        bbox_to_anchor=(0.8, 1.0),
        fontsize=14,
    )

    # Style
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    fig.savefig(out_png, dpi=600, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=600, bbox_inches="tight")
    plt.close(fig)


vif_df = compute_vif_df(X_z)
print("\nVIF table:")
print(vif_df.sort_values("vif", ascending=False))

plot_vif(
    vif_df=vif_df,
    label_map=label_map,
    out_png="supp_figure_vif.png",
    out_pdf="supp_figure_vif.pdf",
    vline_5=5,
    vline_10=10,
)

#  FUTURE SSP PROJECTIONS + AGGREGATE & MAP FIGURES

# Style helpers
def set_paper_style():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.linewidth": 0.6,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "figure.dpi": 300,
        "savefig.dpi": 600,
        "figure.autolayout": False,
    })

def build_relative_change_table(proj_long: pd.DataFrame) -> pd.DataFrame:
    agg_r = (
        proj_long
        .groupby(["scenario", "year"])["ratio_nb"]
        .agg(
            median="median",
            q25=lambda x: x.quantile(0.25),
            q75=lambda x: x.quantile(0.75),
        )
        .reset_index()
    )
    return agg_r

def prepare_msa_geometries_for_proj(
    gdf_msa: gpd.GeoDataFrame,
    proj_long: pd.DataFrame,
    target_year: int
) -> dict[str, gpd.GeoDataFrame]:
    """
    Returns dict of scenario -> GeoDataFrame with 'ratio_nb' for the target year.
    Assumes MSA code in proj_long is like C#### and CBSAFP aligns with int(MSA_suffix)*10.
    """
    gdf_cbsa = gdf_msa.copy()

    # CBSAFP (e.g., 10180) -> MSA "C1018"
    gdf_cbsa["MSA"] = (
        "C" + (gdf_cbsa["CBSAFP"].astype(int) // 10).astype(str).str.zfill(4)
    )

    proj_yr = proj_long[proj_long["year"] == target_year].copy()
    if proj_yr.empty:
        raise ValueError(f"No projection rows in proj_long for year {target_year}")

    out = {}
    for sc in sorted(proj_yr["scenario"].unique()):
        sub = proj_yr[proj_yr["scenario"] == sc][["MSA", "ratio_nb"]].dropna()
        if sub.empty:
            continue
        gdf_sc = gdf_cbsa.merge(sub, on="MSA", how="inner")
        if not gdf_sc.empty:
            out[sc] = gdf_sc
    return out

# SIMPLE MODEL FOR PROJECTIONS:
#    Fit NB on observed DC counts using ACS 2023 population
#    (then apply to SSP projections starting in 2025)
import statsmodels.api as sm
import numpy as np
import pandas as pd

POP_CSV = "ACSDT5Y2023.B01003-Data.csv"
CROSSWALK_CSV = "/Users/camilla/PycharmProjects/datacenters/nature_cities/crosswalk_MSA_state.csv"

EXCLUDE_STATEFP = {"02","15","60","66","69","72","74","78"}  # AK, HI + territories

ABBR_TO_STATEFP = {
    "AL":"01","AK":"02","AZ":"04","AR":"05","CA":"06","CO":"08","CT":"09","DE":"10","DC":"11",
    "FL":"12","GA":"13","HI":"15","ID":"16","IL":"17","IN":"18","IA":"19","KS":"20","KY":"21",
    "LA":"22","ME":"23","MD":"24","MA":"25","MI":"26","MN":"27","MS":"28","MO":"29","MT":"30",
    "NE":"31","NV":"32","NH":"33","NJ":"34","NM":"35","NY":"36","NC":"37","ND":"38","OH":"39",
    "OK":"40","OR":"41","PA":"42","RI":"44","SC":"45","SD":"46","TN":"47","TX":"48","UT":"49",
    "VT":"50","VA":"51","WA":"53","WV":"54","WI":"55","WY":"56",
    "AS":"60","GU":"66","MP":"69","PR":"72","UM":"74","VI":"78"
}

cw = pd.read_csv(CROSSWALK_CSV, dtype=str)

cw["CBSAFP"] = pd.to_numeric(cw["GEOID"], errors="coerce").astype("Int64")
cw = cw.dropna(subset=["CBSAFP"]).copy()
cw["MSA"] = "C" + (cw["CBSAFP"].astype(int) // 10).astype(str).str.zfill(4)

abbr_col = "State_Abbr"
cw["STATEFP"] = cw[abbr_col].astype(str).str.strip().map(ABBR_TO_STATEFP)

cw_msa_state = (
    cw.dropna(subset=["MSA", "STATEFP"])[["MSA", "STATEFP"]]
      .drop_duplicates()
      .sort_values(["MSA", "STATEFP"])
      .drop_duplicates(subset=["MSA"], keep="first")
)

# ACS 2023 population
pop_data = pd.read_csv(POP_CSV, dtype=str)

pop_data = pop_data[["GEO_ID", "B01003_001E"]].copy()
pop_data = pop_data.iloc[1:].reset_index(drop=True)

pop_data["CBSAFP"] = pop_data["GEO_ID"].str.extract(r"US(\d{5})")[0]
pop_data = pop_data.dropna(subset=["CBSAFP"]).copy()

pop_data["pop_2023"] = pd.to_numeric(pop_data["B01003_001E"], errors="coerce")
pop_data = pop_data.dropna(subset=["pop_2023"]).copy()

# beta estimation

df_beta = counts[["CBSAFP", "msa_code_c", "datacenter_count", "has_datacenter"]].copy()
df_beta = df_beta.merge(pop_data[["CBSAFP", "pop_2023"]], on="CBSAFP", how="left")
df_beta = df_beta.dropna(subset=["pop_2023"]).copy()

# Keep only MSAs with >=1 DC
df_beta = df_beta[df_beta["has_datacenter"] == 1].copy()

df_beta["log_pop_2023"] = np.log(df_beta["pop_2023"].astype(float))
df_beta["log_dc"] = np.log(df_beta["datacenter_count"].astype(float))

# OLS
X = sm.add_constant(df_beta[["log_pop_2023"]].astype(float))
y = df_beta["log_dc"].astype(float)
ols_fit = sm.OLS(y, X).fit(cov_type="HC1")

print("\n=== OLS: log(datacenter_count) ~ log(pop_2023)  (robust HC1) ===")
print(ols_fit.summary())

# projections
proj = pd.read_csv(PROJ_CSV, dtype=str)
proj["MSA"] = proj["MSA"].astype(str)

# keep only MSAs used in model fit (>=1 DC + pop available)
msas_model = set(df_beta["msa_code_c"].unique().tolist())
proj = proj[proj["MSA"].isin(msas_model)].copy()

proj_long = proj.melt(id_vars="MSA", var_name="scenario_year", value_name="pop_proj")
proj_long[["scenario", "year"]] = proj_long["scenario_year"].str.extract(r"(SSP\d+)_(\d+)")
proj_long["year"] = pd.to_numeric(proj_long["year"], errors="coerce").astype("Int64")
proj_long["pop_proj"] = pd.to_numeric(proj_long["pop_proj"], errors="coerce")

proj_long = proj_long.dropna(subset=["scenario", "year", "pop_proj"]).copy()
proj_long["year"] = proj_long["year"].astype(int)

proj_long = proj_long[proj_long["year"] >= 2025].copy()

# log(pop) for predictions
proj_long["log_pop"] = np.where(proj_long["pop_proj"] > 0, np.log(proj_long["pop_proj"]), np.nan)
proj_long = proj_long.dropna(subset=["log_pop"]).copy()

X_future = sm.add_constant(proj_long[["log_pop"]].astype(float))
# predict log(mu) using OLS fitted on 2023
proj_long["log_mu"] = ols_fit.predict(X_future)
proj_long["mu_nb"] = np.exp(proj_long["log_mu"])
#  baseline for ratios must be 2025 because SSP starts there
BASE_RATIO_YEAR = 2025

available_years = sorted(proj_long["year"].unique().tolist())
print("[DIAG] SSP available years:", available_years)
if BASE_RATIO_YEAR not in available_years:
    raise ValueError(f"SSP projections do not include {BASE_RATIO_YEAR}. Available: {available_years}")

baseline_sc = (
    proj_long.loc[proj_long["year"] == BASE_RATIO_YEAR, ["MSA", "scenario", "mu_nb"]]
    .rename(columns={"mu_nb": "mu_nb_base"})
)

proj_long = proj_long.merge(baseline_sc, on=["MSA", "scenario"], how="left")
proj_long = proj_long.dropna(subset=["mu_nb_base"]).copy()

proj_long["ratio_nb"] = proj_long["mu_nb"] / proj_long["mu_nb_base"]
proj_long["log_ratio_nb"] = np.log(proj_long["ratio_nb"])

print("\n[DIAG] mu_nb summary:")
print(proj_long["mu_nb"].describe())
print("\n[DIAG] ratio_nb summary:")
print(proj_long["ratio_nb"].describe())

dfy = proj_long[(proj_long["year"]==2050) & (proj_long["scenario"]=="SSP5")]
print(dfy["ratio_nb"].min(), dfy["ratio_nb"].quantile([.01,.05,.1]), dfy["ratio_nb"].max())
print((dfy["ratio_nb"]<1).mean(), (dfy["ratio_nb"]<0.9).mean())


TARGET_EPSG = 5070  # CONUS Albers


def set_fig4_style_big():
    mpl.rcParams.update({
        "text.usetex": False,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],

        "font.size": 16,
        "axes.labelsize": 16,
        "axes.titlesize": 16,
        "legend.fontsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,

        # lines
        "axes.linewidth": 1.2,

        # output
        "figure.dpi": 300,
        "savefig.dpi": 600,
    })
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from matplotlib.cm import ScalarMappable

TARGET_EPSG = 5070  # CONUS Albers

def _safe_union(geom_series):
    if hasattr(geom_series, "union_all"):
        return geom_series.union_all()
    return geom_series.unary_union

def _cbsa_to_msa_code(gdf_msa: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    g = gdf_msa.copy()
    g["MSA"] = "C" + (g["CBSAFP"].astype(int) // 10).astype(str).str.zfill(4)
    return g

def _make_cmap():
    # blu -> grigio -> rosso
    return LinearSegmentedColormap.from_list(
        "BlueGreyRed",
        ["#2c7fb8", "#bdbdbd", "#d7301f"],
        N=256
    )

def _make_norm(rmin, rmax, vcenter=1.0, enforce_nonnegative=True):
    rmin = float(rmin)
    rmax = float(rmax)

    if enforce_nonnegative:
        rmin = max(0.0, rmin)
        rmax = max(0.0, rmax)

    if rmax <= rmin:
        rmax = rmin + 1e-9

    if (vcenter <= rmin) or (vcenter >= rmax):
        return mpl.colors.Normalize(vmin=rmin, vmax=rmax, clip=True)

    return TwoSlopeNorm(vmin=rmin, vcenter=vcenter, vmax=rmax)
def plot_one_scenario_map(
    dfy: pd.DataFrame,
    g: gpd.GeoDataFrame,
    st: gpd.GeoDataFrame,
    conus,
    bounds,
    sc: str,
    target_year: int,
    out_prefix: str,
    cmap,
    state_color="#d0d0d0",
    state_linewidth=0.8,
    msa_edgecolor="white",
    msa_linewidth=0.25,
    state_fill=None,
    missing_color="#d9d9d9",
    enforce_nonnegative=True,
):
    minx, miny, maxx, maxy = bounds

    sub = dfy[dfy["scenario"] == sc][["MSA", "ratio_nb"]].copy()
    if sub.empty:
        print(f"No data for {sc} {target_year}, skipping.")
        return

    g_sc = g.merge(sub, on="MSA", how="inner")
    if g_sc.empty:
        print(f" No geometry match for {sc} {target_year}, skipping.")
        return

    g_sc = gpd.clip(g_sc, conus)

    # TRUE min/max for this scenario (no symmetry)
    rmin = float(g_sc["ratio_nb"].min())
    rmax = float(g_sc["ratio_nb"].max())

    if enforce_nonnegative and (rmin < 0):
        raise ValueError(f"{sc} has negative ratio_nb (min={rmin}). Fix upstream or turn off enforce_nonnegative.")

    norm = _make_norm(rmin, rmax, vcenter=1.0, enforce_nonnegative=enforce_nonnegative)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    fig.patch.set_alpha(0)
    ax.set_facecolor("none")

    if state_fill is not None:
        st.plot(ax=ax, facecolor=state_fill, edgecolor="none", zorder=0)

    st.boundary.plot(ax=ax, linewidth=state_linewidth, color=state_color, zorder=10)

    g_sc.plot(
        ax=ax,
        column="ratio_nb",
        cmap=cmap,
        norm=norm,
        edgecolor=msa_edgecolor,
        linewidth=msa_linewidth,
        zorder=5,
        missing_kwds={"color": missing_color, "edgecolor": "none"},
    )

    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)
    ax.set_aspect("equal")
    ax.axis("off")

    smap = ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])

    cbar = fig.colorbar(smap, ax=ax, orientation="horizontal", fraction=0.05, pad=0.04)

    # ticks: endpoints + 1 (if inside range)
    ticks = [norm.vmin]
    if (1.0 > norm.vmin) and (1.0 < norm.vmax):
        ticks.append(1.0)
    ticks.append(norm.vmax)

    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r"$R_{i,s,%d}$" % target_year, fontsize=16)

    fname = f"{out_prefix}_{sc}_{target_year}"
    fig.savefig(f"{fname}.png", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)
    fig.savefig(f"{fname}.pdf", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    print(f" Saved: {fname}.png/.pdf  (rmin={rmin:.4f}, rmax={rmax:.4f})")
def plot_four_panel_maps(
    dfy: pd.DataFrame,
    g: gpd.GeoDataFrame,
    st: gpd.GeoDataFrame,
    conus,
    bounds,
    scenarios=("SSP1", "SSP2", "SSP3", "SSP4"),
    target_year: int = 2050,
    out_prefix: str = "fig_map_ratio_4panel",
    r_min: float | None = None,
    r_max: float | None = None,
    cmap=None,
    state_color="#d0d0d0",
    state_linewidth=0.8,
    msa_edgecolor="white",
    msa_linewidth=0.25,
    state_fill=None,
    missing_color="#d9d9d9",
    enforce_nonnegative=True,
):
    minx, miny, maxx, maxy = bounds

    # compute fixed extremes across SSP1..SSP4
    df_4 = dfy[dfy["scenario"].isin(scenarios)].copy()
    if df_4.empty:
        raise ValueError("No rows for SSP1-SSP4 in dfy.")

    if r_min is None:
        r_min = float(df_4["ratio_nb"].min())
    if r_max is None:
        r_max = float(df_4["ratio_nb"].max())

    if enforce_nonnegative and (r_min < 0):
        raise ValueError(f"SSP1-SSP4 have negative ratio_nb (min={r_min}). Fix upstream or turn off enforce_nonnegative.")

    norm = _make_norm(r_min, r_max, vcenter=1.0, enforce_nonnegative=enforce_nonnegative)

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    axes = axes.ravel()

    for ax, sc in zip(axes, scenarios):
        sub = dfy[dfy["scenario"] == sc][["MSA", "ratio_nb"]].copy()
        g_sc = g.merge(sub, on="MSA", how="inner")
        if g_sc.empty:
            ax.set_axis_off()
            ax.set_title(f"{sc} (no match)")
            continue

        g_sc = gpd.clip(g_sc, conus)

        ax.set_facecolor("none")
        if state_fill is not None:
            st.plot(ax=ax, facecolor=state_fill, edgecolor="none", zorder=0)

        st.boundary.plot(ax=ax, linewidth=state_linewidth, color=state_color, zorder=10)

        g_sc.plot(
            ax=ax,
            column="ratio_nb",
            cmap=cmap,
            norm=norm,
            edgecolor=msa_edgecolor,
            linewidth=msa_linewidth,
            zorder=5,
            missing_kwds={"color": missing_color, "edgecolor": "none"},
        )

        ax.set_xlim(minx, maxx)
        ax.set_ylim(miny, maxy)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(sc)

    # one shared colorbar
    smap = ScalarMappable(norm=norm, cmap=cmap)
    smap.set_array([])

    cbar = fig.colorbar(smap, ax=axes, orientation="horizontal", fraction=0.05, pad=0.06)

    ticks = [norm.vmin]
    if (1.0 > norm.vmin) and (1.0 < norm.vmax):
        ticks.append(1.0)
    ticks.append(norm.vmax)

    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:.2f}" for t in ticks])
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r"$R_{i,s,%d}$" % target_year, fontsize=16)

    fname = f"{out_prefix}_{target_year}"
    fig.savefig(f"{fname}.png", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)
    fig.savefig(f"{fname}.pdf", dpi=600, bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)

    print(f" Saved: {fname}.png/.pdf  (fixed r_min={r_min:.4f}, r_max={r_max:.4f})")
def plot_maps_split_scaling(
    proj_long: pd.DataFrame,
    gdf_msa: gpd.GeoDataFrame,
    states: gpd.GeoDataFrame,
    target_year: int = 2050,
    out_prefix_single: str = "fig_map_ratio",
    out_prefix_4panel: str = "fig_map_ratio_4panel",
    r_min: float | None = None,
    r_max: float | None = None,
    enforce_nonnegative: bool = True,
    # style
    state_color="#d0d0d0",
    state_linewidth=0.8,
    msa_edgecolor="white",
    msa_linewidth=0.25,
    state_fill=None,
    missing_color="#d9d9d9",
):
    mpl.rcParams.update({
        "font.family": "Arial",
        "font.size": 16,
        "axes.labelsize": 16,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "savefig.dpi": 600,
    })

    st = states.copy()
    if st.crs is None or st.crs.to_epsg() != TARGET_EPSG:
        st = st.to_crs(epsg=TARGET_EPSG)

    g = gdf_msa.copy()
    if g.crs is None or g.crs.to_epsg() != TARGET_EPSG:
        g = g.to_crs(epsg=TARGET_EPSG)
    g = _cbsa_to_msa_code(g)

    dfy = proj_long[(proj_long["year"] == target_year)].copy()
    dfy = dfy.dropna(subset=["MSA", "scenario", "ratio_nb"])
    if dfy.empty:
        raise ValueError(f"No rows in proj_long for target_year={target_year}")

    if enforce_nonnegative and (dfy["ratio_nb"].min() < 0):
        raise ValueError(f"Negative ratio_nb exists in year={target_year}. Min={dfy['ratio_nb'].min()}")

    conus = _safe_union(st.geometry)
    bounds = st.total_bounds  # minx, miny, maxx, maxy
    cmap = _make_cmap()

    # 4-panel SSP1-SSP4 with fixed extremes
    plot_four_panel_maps(
        dfy=dfy, g=g, st=st, conus=conus, bounds=bounds,
        scenarios=("SSP1", "SSP2", "SSP3", "SSP4"),
        target_year=target_year,
        out_prefix=out_prefix_4panel,
        r_min=r_min, r_max=r_max,   # if None, computed across SSP1-SSP4
        cmap=cmap,
        state_color=state_color, state_linewidth=state_linewidth,
        msa_edgecolor=msa_edgecolor, msa_linewidth=msa_linewidth,
        state_fill=state_fill, missing_color=missing_color,
        enforce_nonnegative=enforce_nonnegative,
    )

    # SSP5 single map with its own min/max (no symmetry)
    plot_one_scenario_map(
        dfy=dfy, g=g, st=st, conus=conus, bounds=bounds,
        sc="SSP5",
        target_year=target_year,
        out_prefix=out_prefix_single,
        cmap=cmap,
        state_color=state_color, state_linewidth=state_linewidth,
        msa_edgecolor=msa_edgecolor, msa_linewidth=msa_linewidth,
        state_fill=state_fill, missing_color=missing_color,
        enforce_nonnegative=enforce_nonnegative,
    )
plot_maps_split_scaling(
    proj_long=proj_long,
    gdf_msa=gdf_msa,
    states=states,
    target_year=2050,
    out_prefix_single="fig_map_ratio",
    out_prefix_4panel="fig_map_ratio_SSP1_4",
    r_min=None,
    r_max=None,
    enforce_nonnegative=False,
)
