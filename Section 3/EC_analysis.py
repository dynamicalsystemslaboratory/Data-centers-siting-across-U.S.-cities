"""
Spatially classifies U.S. data centers by Energy Community (EC) type and tests whether
under development vs. operational data centers are distributed differently across EC categories.

1. Load states, city, data center points, and two IRA EC layers.
2. Keep *CONUS only and reproject to EPSG:5070
3. Keep data centers inside cities.
4. Build mutually exclusive EC areas:
   - CC EC
   - ER EC
   - CC + ER
   - Non-EC
5. Label data centers:
   - current = Operational
   - future = Under Construction / Planned / Land Banked
6. Create a 2×4 contingency table and run a Chi-square test.
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.stats import chi2_contingency

STATE_SHP = r"/Users/camilla/PycharmProjects/datacenters/State/tl_2023_us_state.shp"
CBSA_SHP = r"/Users/camilla/PycharmProjects/datacenters/nature_cities/tl_2024_us_cbsa/tl_2024_us_cbsa.shp"
DATACENTERS_SHP = r"/Users/camilla/PycharmProjects/datacenters/Cleaned Shapefile/DC_Shapefile.shp"
fips_path = r"/Users/camilla/PycharmProjects/slides_nature_cities/Contiguous US states FIPS.csv"  # FIPS path
EXCLUDE_STATEFP = {"02", "15", "72", "60", "66", "69", "78"}  # AK, HI, PR, AS, GU, MP, VI

COAL_EC = {
    "path": "/Users/camilla/PycharmProjects/datacenters/ira_energy_communities_download/2024_Coal_Closure_Energy_Communities__00__2024_Coal_Closure_Energy_Communities.gpkg",
    "layer": None,
}

ER_EC = {
    "path": "/Users/camilla/PycharmProjects/datacenters/ira_energy_communities_download/2024_MSAs_NonMSAs_that_are_Energy_Communities__00__2024_MSAs_Non_MSAs_that_are_Energy_Communities.gpkg",
    "layer": None,
}

TARGET_EPSG = "EPSG:5070"  # meters

OUT_DIR = "ec_year_outputs_min"
os.makedirs(OUT_DIR, exist_ok=True)


def ensure_crs(gdf, epsg_if_missing="EPSG:4326"):
    if gdf.crs is None:
        gdf = gdf.set_crs(epsg_if_missing)
    return gdf


def conus_filter_states(states):
    if "STATEFP" in states.columns:
        return states[~states["STATEFP"].isin(EXCLUDE_STATEFP)].copy()

    name_col = "NAME" if "NAME" in states.columns else ("name" if "name" in states.columns else None)
    if name_col is None:
        return states
    return states[~states[name_col].isin(["Alaska", "Hawaii", "Puerto Rico"])].copy()


def read_layer(path, layer=None):
    if not os.path.exists(path):
        return None
    return gpd.read_file(path) if layer is None else gpd.read_file(path, layer=layer)


def repair_geoms(gdf):
    gdf = gdf.copy()
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    # buffer(0) is the standard quick fix for many topology issues
    gdf["geometry"] = gdf.geometry.buffer(0)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()
    return gdf


def presence_from_dissolved(points_gdf, dissolved_poly_gdf, predicate="within"):
    j = gpd.sjoin(
        points_gdf[["geometry"]].copy(),
        dissolved_poly_gdf[["geometry"]],
        predicate=predicate,
        how="left",
    )
    return j["index_right"].notna().astype(int).values


def main():
    states = gpd.read_file(STATE_SHP)
    states = ensure_crs(states, "EPSG:4326")
    states = conus_filter_states(states).to_crs(TARGET_EPSG)

    # CONUS mask for clipping EC layers
    conus_mask = states[["geometry"]].dissolve()

    # MSA mask for clipping EC layers within cities
    cbsa = gpd.read_file(CBSA_SHP)
    cbsa = ensure_crs(cbsa, "EPSG:4326").to_crs(TARGET_EPSG)
    df_fips = pd.read_csv(fips_path)  # Load FIPS
    df_fips["FIPS Code"] = df_fips["FIPS Code"].astype(str).str.zfill(2)  # Ensure string, add 0 to single digits
    cbsa['State_Abbr'] = cbsa['NAME'].str[-2:]  # Get state abbreviations
    contiguous_mask = cbsa['State_Abbr'].isin(df_fips['Postal Abbr.'])  # All contiguous states
    cbsa = cbsa[contiguous_mask].copy()  # Get contiguous states

    #  load datacenters and assign geographies
    dc = gpd.read_file(DATACENTERS_SHP)
    dc = dc.reset_index(drop=True)
    dc["dc_id"] = dc.index
    dc = ensure_crs(dc, "EPSG:4326").to_crs(TARGET_EPSG)

    dc_join = gpd.sjoin(
        dc[["dc_id", "geometry"]].copy(),
        cbsa[["geometry"]].copy(),
        how="left",
        predicate="within"
    )

    # Indicator: 1 if matched to a city polygon, else 0
    dc_in_city = dc_join.assign(
        dc_in_city=dc_join["index_right"].notna().astype(int))[["dc_id", "dc_in_city"]]

    dc = dc.merge(dc_in_city, on="dc_id", how="left")
    dc = dc[dc["dc_in_city"] == 1]

    # state
    g = gpd.sjoin(
        dc,
        states[["STATEFP", "geometry"]],
        predicate="within",
        how="inner",
    ).drop(columns="index_right")
    # DV
    g["Stage_future"] = g["stage"].isin(["Under Construction", "Planned", "Land Banked"])
    g["Stage_current"] = g["stage"].isin(["Operational"])

    #  EC layers -> indicators
    coal_ec = read_layer(COAL_EC["path"], COAL_EC.get("layer"))
    er_ec = read_layer(ER_EC["path"], ER_EC.get("layer"))

    coal_ec = ensure_crs(coal_ec, "EPSG:4326").to_crs(TARGET_EPSG)
    er_ec = ensure_crs(er_ec, "EPSG:4326").to_crs(TARGET_EPSG)

    coal_ec = coal_ec[coal_ec.geometry.notna() & ~coal_ec.geometry.is_empty].copy()
    er_ec = er_ec[er_ec.geometry.notna() & ~er_ec.geometry.is_empty].copy()

    #  clip to CONUS + repair geometries BEFORE dissolve/overlay
    coal_ec = repair_geoms(coal_ec) if not coal_ec.is_valid.all() else coal_ec
    er_ec = repair_geoms(er_ec) if not er_ec.is_valid.all() else er_ec

    coal_ec = gpd.clip(coal_ec, conus_mask)
    er_ec = gpd.clip(er_ec, conus_mask)

    coal_diss = coal_ec[["geometry"]].dissolve()
    er_diss = er_ec[["geometry"]].dissolve()

    #  build mutually exclusive EC areas
    coal_only = gpd.overlay(coal_diss, er_diss, how="difference", keep_geom_type=False)
    er_only = gpd.overlay(er_diss, coal_diss, how="difference", keep_geom_type=False)
    both_ec = gpd.overlay(coal_diss, er_diss, how="intersection", keep_geom_type=False)
    union_ec = gpd.overlay(coal_diss, er_diss, how="union", keep_geom_type=False)
    none_ec = gpd.overlay(conus_mask, union_ec, how="difference", keep_geom_type=False)

    # Repair overlay outputs
    coal_only = repair_geoms(coal_only) if len(coal_only) else coal_only
    er_only = repair_geoms(er_only) if len(er_only) else er_only
    both_ec = repair_geoms(both_ec) if len(both_ec) else both_ec
    none_ec = repair_geoms(none_ec) if len(none_ec) else none_ec

    coal_only = gpd.GeoDataFrame(geometry=coal_only.geometry, crs=TARGET_EPSG).dissolve()
    er_only = gpd.GeoDataFrame(geometry=er_only.geometry, crs=TARGET_EPSG).dissolve()
    both_ec = gpd.GeoDataFrame(geometry=both_ec.geometry, crs=TARGET_EPSG).dissolve()
    none_ec = gpd.GeoDataFrame(geometry=none_ec.geometry, crs=TARGET_EPSG).dissolve()

    g["cec"] = presence_from_dissolved(g, coal_only, predicate="within")
    g["erec"] = presence_from_dissolved(g, er_only, predicate="within")
    g["bothec"] = presence_from_dissolved(g, both_ec, predicate="within")
    g["noneec"] = presence_from_dissolved(g, none_ec, predicate="within")

    g["ec_cat"] = np.select(
        [
            (g["cec"] == 1),
            (g["erec"] == 1),
            (g["bothec"] == 1),
            (g["noneec"] == 1)
        ],
        ["coal_only", "er_only", "both_ec", "none_ec"],
        default="none"
    )

    print(pd.crosstab(index=g["ec_cat"], columns="count"))

    stage_grp = np.select(
        [g["Stage_future"], g["Stage_current"]],
        ["future", "current"],
        default="other"
    )

    # keep only future/current records
    g_test = g.loc[stage_grp != "other"].copy()
    g_test["stage_grp"] = stage_grp[stage_grp != "other"]

    col_order = ["coal_only", "er_only", "both_ec", "none_ec"]
    ct = pd.crosstab(g_test["stage_grp"], g_test["ec_cat"]).reindex(
        index=["future", "current"], columns=col_order, fill_value=0
    )

    print("\nContingency table (rows=stage, cols=ec_cat):")
    print(ct)

    ct_out_csv = os.path.join(OUT_DIR, "ct_stage_by_ec.csv")
    ct.to_csv(ct_out_csv)
    print(f"Saved contingency table to: {ct_out_csv}")

    chi2, p_chi2, dof, expected = chi2_contingency(ct.values, correction=False)

    print("\nChi-square test of independence (2x4):")
    print("chi2:", chi2)
    print("dof:", dof)
    print("p-value:", p_chi2)


if __name__ == "__main__":
    main()