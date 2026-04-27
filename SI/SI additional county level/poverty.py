import pandas as pd
import numpy as np

INPUT_CSV = "/Users/camilla/Downloads/ACSST5Y2023.S1701_2026-04-11T134922/ACSST5Y2023.S1701-Data.csv"
OUTPUT_CSV = "EP_POV100_county.csv"

# Exclude Alaska, Hawaii, and territories
EXCLUDE_STATEFP = {"02", "15", "60", "66", "69", "72", "74", "78"}

# Read file
df = pd.read_csv(INPUT_CSV, skiprows=[1], dtype=str)

# Extract county FIPS from GEO_ID, e.g. 0500000US01001 -> 01001
df["FIPS"] = df["GEO_ID"].str.extract(r"US(\d{5})")

# Keep only rows with valid 5-digit county FIPS
df = df[df["FIPS"].notna()].copy()
df["FIPS"] = df["FIPS"].str.zfill(5)

# Derive state FIPS from county FIPS
df["STATEFP"] = df["FIPS"].str[:2]

# Keep contiguous US only
df = df[~df["STATEFP"].isin(EXCLUDE_STATEFP)].copy()

## Use estimate columns
variabile = pd.to_numeric(df["S1701_C03_001E"], errors="coerce")
df["EP_POV100"] = variabile
# Save output
out = df[["FIPS", "EP_POV100"]].copy()
out.to_csv(OUTPUT_CSV, index=False)

print(f"Saved {OUTPUT_CSV}")
print(out.head())

