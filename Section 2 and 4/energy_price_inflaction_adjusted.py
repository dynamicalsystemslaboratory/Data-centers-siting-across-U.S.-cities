"""
this script take the industrial electricity price at the state level
 for all the contiguous US, deflate them and compute their variability
"""


import pandas as pd

PRICES_CSV = "eia861_state_prices_total_and_industrial_2000_present.csv"
CPI_XLSX   = "historical-cpi-u-202508.xlsx"
BASE_YEAR  = 2020
OUT_CSV    = "eia861_state_prices_real_2020_with_volatility.csv"

CONTIG48 = {
    "AL","AZ","AR","CA","CO","CT","DE","FL","GA","ID","IL","IN","IA","KS","KY","LA","ME","MD",
    "MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA",
    "RI","SC","SD","TN","TX","UT","VT","VA","WA","WV","WI","WY"
}

prices = pd.read_csv(PRICES_CSV, dtype={"state": "string"})
prices["year"] = pd.to_numeric(prices["year"], errors="coerce").astype("Int64")
prices["price_total_cents_per_kwh"] = pd.to_numeric(prices["price_total_cents_per_kwh"], errors="coerce")
prices["price_industrial_cents_per_kwh"] = pd.to_numeric(prices["price_industrial_cents_per_kwh"], errors="coerce")

prices = prices.dropna(subset=["year"]).astype({"year": "int"})
prices = prices[(prices["year"].between(2000, 2025)) & (prices["state"].isin(CONTIG48))]

raw = pd.read_excel(CPI_XLSX, sheet_name="Index averages", header=None)
hdr = raw.index[raw.iloc[:, 1].astype(str).str.contains("Year", case=False, na=False)][0]
cpi = pd.read_excel(CPI_XLSX, sheet_name="Index averages", header=hdr)
cpi.columns = cpi.columns.astype(str).str.strip()

annual_col = next(c for c in cpi.columns if "annual" in c.lower() and "avg" in c.lower())
cpi = cpi[["Year", annual_col]].rename(columns={annual_col: "cpi"}).dropna()
cpi["Year"] = pd.to_numeric(cpi["Year"], errors="coerce").astype("Int64")
cpi["cpi"]  = pd.to_numeric(cpi["cpi"], errors="coerce")
cpi = cpi.dropna(subset=["Year","cpi"]).astype({"Year": "int"})
cpi = cpi[cpi["Year"].between(2000, 2025)]

cpi_base = float(cpi.loc[cpi["Year"].eq(BASE_YEAR), "cpi"].iloc[0])

#  merge + deflate
df = prices.merge(cpi, left_on="year", right_on="Year", how="inner")

df["price_total_real_2020"] = (
    df["price_total_cents_per_kwh"] * (cpi_base / df["cpi"])
)
df["price_industrial_real_2020"] = (
    df["price_industrial_cents_per_kwh"] * (cpi_base / df["cpi"])
)

#  variance
var_total_48 = df.groupby("state")["price_total_real_2020"].var()
var_ind_48   = df.groupby("state")["price_industrial_real_2020"].var()

df = df.merge(var_total_48.rename("var_total_48"), on="state", how="left")
df = df.merge(var_ind_48.rename("var_ind_48"), on="state", how="left")

df.to_csv(OUT_CSV, index=False)
print(f"Saved: {OUT_CSV}")
