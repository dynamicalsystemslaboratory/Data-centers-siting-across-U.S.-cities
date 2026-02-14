"""
1. Loads three shapefiles:
   - data center point locations
   - CBSA polygons
   - county polygons
2. Restricts the county universe to contiguous U.S. via the county `STATEFP` code.
3. Cleans data center coordinates and restricts points to a CONUS bounding box (sanity check).
4. Classifies each county:
   - city county if the county polygon intersects any CBSA polygon
   - non-city county otherwise
5. Counts the number of data centers per county .
6. Computes:
   - Welch two-sample t-test (one-sided: city > non-city)
7. Exports a summary CSV and a LaTeX macro file for direct inclusion in the Supplement.

---
"""
import geopandas as gpd
import pandas as pd
import numpy as np
from scipy import stats

CRS_WGS84 = "EPSG:4326"

DC_SHP   = "Cleaned Shapefile/DC_Shapefile.shp"
CBSA_SHP = "tl_2024_us_cbsa/tl_2024_us_cbsa.shp"
COUNTY_SHP = "tl_2024_us_county/tl_2024_us_county.shp"

# contiguous U.S. ( 48 + DC)
CONTIGUOUS_ABBR = {
    "AL","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND",
    "OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY","DC"
}

STATEFP_TO_ABBR = {
    "01":"AL","02":"AK","04":"AZ","05":"AR","06":"CA","08":"CO","09":"CT","10":"DE","11":"DC","12":"FL","13":"GA",
    "15":"HI","16":"ID","17":"IL","18":"IN","19":"IA","20":"KS","21":"KY","22":"LA","23":"ME","24":"MD","25":"MA",
    "26":"MI","27":"MN","28":"MS","29":"MO","30":"MT","31":"NE","32":"NV","33":"NH","34":"NJ","35":"NM","36":"NY",
    "37":"NC","38":"ND","39":"OH","40":"OK","41":"OR","42":"PA","44":"RI","45":"SC","46":"SD","47":"TN","48":"TX",
    "49":"UT","50":"VT","51":"VA","53":"WA","54":"WV","55":"WI","56":"WY",
}

def make_county_id(cty: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Create a stable county_id (prefer GEOID)."""
    if "GEOID" in cty.columns:
        cty["county_id"] = cty["GEOID"].astype(str)
    elif "STATEFP" in cty.columns and "COUNTYFP" in cty.columns:
        cty["county_id"] = cty["STATEFP"].astype(str).str.zfill(2) + cty["COUNTYFP"].astype(str).str.zfill(3)
    else:
        raise SystemExit(
            "County shapefile must contain GEOID or (STATEFP and COUNTYFP). "
            f"Found columns: {list(cty.columns)}"
        )
    return cty

def restrict_counties_to_contiguous_us(cty: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Filter county polygons to Lower 48 + DC using STATEFP."""
    if "STATEFP" not in cty.columns:
        raise SystemExit(
            "County shapefile missing STATEFP; cannot restrict to contiguous U.S. cleanly. "
            f"Found columns: {list(cty.columns)}"
        )
    statefp = cty["STATEFP"].astype(str).str.zfill(2)
    cty["state_abbr"] = statefp.map(STATEFP_TO_ABBR)
    cty = cty[cty["state_abbr"].isin(CONTIGUOUS_ABBR)].copy()
    return cty

def clean_dc_points(dc: gpd.GeoDataFrame, lon_col="longitude", lat_col="latitude") -> gpd.GeoDataFrame:
    """Coerce lon/lat numeric, drop invalid, restrict to CONUS bbox, rebuild point geometry in WGS84."""
    if lon_col not in dc.columns or lat_col not in dc.columns:
        raise SystemExit(
            f"DC shapefile must contain '{lon_col}' and '{lat_col}'. "
            f"Found columns: {list(dc.columns)}"
        )

    dc["lon"] = pd.to_numeric(dc[lon_col], errors="coerce")
    dc["lat"] = pd.to_numeric(dc[lat_col], errors="coerce")
    dc = dc.dropna(subset=["lon","lat"]).copy()

    # contiguous only
    dc = dc[
        dc["lon"].between(-125.0, -66.0) &
        dc["lat"].between(24.0, 50.0)
    ].copy()

    dc = gpd.GeoDataFrame(dc, geometry=gpd.points_from_xy(dc["lon"], dc["lat"]), crs=CRS_WGS84)
    return dc


def main():
    dc_raw = gpd.read_file(DC_SHP)
    cbsa = gpd.read_file(CBSA_SHP)
    cty = gpd.read_file(COUNTY_SHP)

    # Counties and CBSA are polygons with CRS in file; we convert to WGS84 for spatial joins with lon/lat points.
    if cbsa.crs is None:
        cbsa = cbsa.set_crs(CRS_WGS84)
    cbsa = cbsa.to_crs(CRS_WGS84)

    if cty.crs is None:
        cty = cty.set_crs(CRS_WGS84)
    cty = cty.to_crs(CRS_WGS84)

    #  restrict counties to contiguous U.S.
    cty = restrict_counties_to_contiguous_us(cty)

    #  county id
    cty = make_county_id(cty)

    #  clean DC coords and restrict to contiguous bbox
    dc = clean_dc_points(dc_raw, lon_col="longitude", lat_col="latitude")

    #  classify counties as "city county" if intersects any CBSA
    # county -> CBSA membership
    cty_city = gpd.sjoin(
        cty[["county_id","geometry"]],
        cbsa[["geometry"]],
        how="left",
        predicate="intersects"
    )
    cty_city["is_city_county"] = cty_city["index_right"].notna()
    cty_city = cty_city.drop_duplicates(subset=["county_id"])[["county_id","is_city_county"]]

    #  count data centers per county
    dc_in_county = gpd.sjoin(
        dc[["geometry"]],
        cty[["county_id","geometry"]],
        how="left",
        predicate="within"
    )

    dc_counts = (
        dc_in_county.dropna(subset=["county_id"])
        .groupby("county_id")
        .size()
        .rename("n_facilities")
        .reset_index()
    )

    #  build full county table (include zeros)
    tab = cty[["county_id"]].merge(cty_city, on="county_id", how="left")
    tab["is_city_county"] = tab["is_city_county"].fillna(False)
    tab = tab.merge(dc_counts, on="county_id", how="left")
    tab["n_facilities"] = tab["n_facilities"].fillna(0).astype(int)

    city = tab.loc[tab["is_city_county"], "n_facilities"].to_numpy()
    non  = tab.loc[~tab["is_city_county"], "n_facilities"].to_numpy()

    #  descriptive stats
    NCityCounties = int(tab["is_city_county"].sum())
    NNonCityCounties = int((~tab["is_city_county"]).sum())
    MeanCity = float(city.mean())
    MeanNon = float(non.mean())

    # Welch t-test (one-sided: city > non-city)
    res = stats.ttest_ind(city, non, equal_var=False)

    tstat = float(res.statistic)
    p_two = float(res.pvalue)
    df_welch = float(res.df)

    p_welch_one_sided = p_two / 2 if tstat > 0 else 1.0

    # ---------- export summary table ----------
    summary = pd.DataFrame([
        {
            "group": "city counties",
            "n_counties": NCityCounties,
            "mean": MeanCity,
            "median": float(np.median(city)),
            "share_zero": float((city == 0).mean()),
        },
        {
            "group": "non-city counties",
            "n_counties": NNonCityCounties,
            "mean": MeanNon,
            "median": float(np.median(non)),
            "share_zero": float((non == 0).mean()),
        },
    ])
    summary.to_csv("county_city_stats.csv", index=False)

    print("Welch one-sided p:", p_welch_one_sided)
    print(f"City counties: {NCityCounties} | Non-city counties: {NNonCityCounties}")
    print(f"Mean city: {MeanCity:.3f} | Mean non-city: {MeanNon:.3f}")
    print("Welch t:", tstat)
    print("Welch df:", df_welch)


if __name__ == "__main__":
    main()
