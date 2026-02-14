import pandas as pd
from pathlib import Path

OEWS_AREA_FILES = [
    Path("MSA_M2024_dl.xlsx"),  # metropolitan areas
    Path("BOS_M2024_dl.xlsx"),  # nonmetropolitan areas
]

AREA_DEFS_FILE = Path("area_definitions_m2024.xlsx")

LONG_OUTPUT_FILE = Path("data/it_maintenance_oews_area_tagged_to_counties.csv")
WIDE_OUTPUT_FILE = Path("data/it_maintenance_county_wide_equal_alloc.csv")

TARGET_SOC_CODES = [
    "15-1231",
    "15-1232",
    "15-1244",
    "15-1241",
    "11-3021",
]

STATE_ABBR_TO_FIPS = {
    "AL": "01", "AK": "02", "AZ": "04", "AR": "05", "CA": "06", "CO": "08", "CT": "09",
    "DE": "10", "FL": "12", "GA": "13", "HI": "15", "ID": "16", "IL": "17", "IN": "18",
    "IA": "19", "KS": "20", "KY": "21", "LA": "22", "ME": "23", "MD": "24", "MA": "25",
    "MI": "26", "MN": "27", "MS": "28", "MO": "29", "MT": "30", "NE": "31", "NV": "32",
    "NH": "33", "NJ": "34", "NM": "35", "NY": "36", "NC": "37", "ND": "38", "OH": "39",
    "OK": "40", "OR": "41", "PA": "42", "RI": "44", "SC": "45", "SD": "46", "TN": "47",
    "TX": "48", "UT": "49", "VT": "50", "VA": "51", "WA": "53", "WV": "54", "WI": "55",
    "WY": "56", "DC": "11", "PR": "72",
}

CONTIG48 = {
    "AL","AZ","AR","CA","CO","CT","DE","FL","GA",
    "ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH",
    "NJ","NM","NY","NC","ND","OH","OK","OR","PA",
    "RI","SC","SD","TN","TX","UT","VT","VA","WA",
    "WV","WI","WY","DC"
}

# HELPERS

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()
    return df


def load_oews_area_files() -> pd.DataFrame:
    frames = []
    for path in OEWS_AREA_FILES:
        print(f"Reading OEWS area file: {path}")
        df_raw = pd.read_excel(path)
        frames.append(normalize_columns(df_raw))
    return pd.concat(frames, ignore_index=True)


def detect_oews_columns(df: pd.DataFrame) -> dict:
    cols = df.columns

    if "area" in cols:
        area_col = "area"
    elif "area_code" in cols:
        area_col = "area_code"
    else:
        raise ValueError(f"Could not find area column. Columns: {list(cols)}")

    if "occ_code" in cols:
        occ_col = "occ_code"
    elif "soc_code" in cols:
        occ_col = "soc_code"
    else:
        raise ValueError(f"Could not find occupation code column. Columns: {list(cols)}")

    if "tot_emp" in cols:
        emp_col = "tot_emp"
    elif "employment" in cols:
        emp_col = "employment"
    else:
        raise ValueError(f"Could not find employment column. Columns: {list(cols)}")

    occ_title_col = "occ_title" if "occ_title" in cols else None

    return {
        "area_col": area_col,
        "occ_col": occ_col,
        "occ_title_col": occ_title_col,
        "emp_col": emp_col,
    }


def find_col(df: pd.DataFrame, substrings) -> str:
    for col in df.columns:
        for s in substrings:
            if s in col:
                return col
    raise ValueError(f"Could not find column with substrings {substrings}. Columns: {list(df.columns)}")


# MAIN

def main():
    # Load OEWS and filter SOCs
    oews = load_oews_area_files()
    ocols = detect_oews_columns(oews)
    area_col, occ_col, occ_title_col, emp_col = (
        ocols["area_col"], ocols["occ_col"], ocols["occ_title_col"], ocols["emp_col"]
    )

    oews[area_col] = oews[area_col].astype(str).str.strip()
    oews_filtered = oews[oews[occ_col].isin(TARGET_SOC_CODES)].copy()
    if oews_filtered.empty:
        raise RuntimeError("No rows after filtering by TARGET_SOC_CODES.")

    # Load area->county crosswalk and build county_fips
    print(f"Reading area definitions file: {AREA_DEFS_FILE}")
    area_defs = normalize_columns(pd.read_excel(AREA_DEFS_FILE))

    state_abbr_col   = find_col(area_defs, ["state abbreviation"])
    county_code_col  = find_col(area_defs, ["county code"])
    area_def_area_col = find_col(area_defs, ["msa code"])
    county_name_col  = find_col(area_defs, ["county name"])

    area_defs[county_code_col] = (
        area_defs[county_code_col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
        .astype("Int64")
    )

    area_defs["state_fips"] = (
        area_defs[state_abbr_col].astype(str).str.strip().map(STATE_ABBR_TO_FIPS)
    )

    area_defs["county_fips"] = (
        area_defs["state_fips"] + area_defs[county_code_col].astype(str).str.zfill(3)
    )

    area_defs[area_def_area_col] = area_defs[area_def_area_col].astype(str).str.strip()

    # Merge OEWS area x SOC to counties
    merged = oews_filtered.merge(
        area_defs,
        left_on=area_col,
        right_on=area_def_area_col,
        how="left",
        validate="m:m",
    )

    # Build long  output
    out_cols = {
        "area_code": area_col,
        "soc_code": occ_col,
        "tot_emp_area": emp_col,
        "county_fips": "county_fips",
        "county_name": county_name_col,
        "state_fips": "state_fips",
        "state_abbr": state_abbr_col,
    }
    if occ_title_col and occ_title_col in merged.columns:
        out_cols["occupation_title"] = occ_title_col

    tidy = merged[list(out_cols.values())].copy()
    tidy.columns = list(out_cols.keys())

    # Equal allocation: area employment split across counties in that area (per SOC)
    area_counts = (
        tidy.groupby(["area_code", "soc_code"])["county_fips"]
        .nunique()
        .reset_index(name="n_counties_in_area")
    )

    tidy = tidy.merge(area_counts, on=["area_code", "soc_code"], how="left", validate="m:1")
    tidy["tot_emp_area"] = pd.to_numeric(tidy["tot_emp_area"], errors="coerce")
    tidy["n_counties_in_area"] = pd.to_numeric(tidy["n_counties_in_area"], errors="coerce")
    tidy["tot_emp_county_alloc_equal"] = tidy["tot_emp_area"] / tidy["n_counties_in_area"]

    LONG_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(LONG_OUTPUT_FILE, index=False)
    print(f"Long file written: {LONG_OUTPUT_FILE} ({len(tidy):,} rows)")

    # Wide county x SOC (contiguous 48)
    df = tidy[tidy["state_abbr"].isin(CONTIG48)].copy()

    grouped = (
        df.groupby(["county_fips", "county_name", "state_abbr", "soc_code"], as_index=False)
          ["tot_emp_county_alloc_equal"]
          .sum()
    )

    wide = grouped.pivot(
        index=["county_fips", "county_name", "state_abbr"],
        columns="soc_code",
        values="tot_emp_county_alloc_equal",
    )

    wide = wide.rename(columns=lambda c: f"emp_{c.replace('-', '_')}")
    wide = wide.fillna(0).reset_index()

    WIDE_OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    wide.to_csv(WIDE_OUTPUT_FILE, index=False)
    print(f"Wide file written: {WIDE_OUTPUT_FILE} ({len(wide):,} counties)")


if __name__ == "__main__":
    main()