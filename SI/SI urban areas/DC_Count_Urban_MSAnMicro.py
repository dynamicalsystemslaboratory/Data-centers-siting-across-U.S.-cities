import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


data_path = r"C:\Users\ol2234\Downloads\Project\Final data\Cleaned Shapefile\DC_Shapefile.shp"    # Data centers path
shapefile_path = r"C:\Users\ol2234\Downloads\Project\Data\Geography\2020\Urban areas\tl_2020_us_uac20.shp"  # Urban path
msa_path = r"C:\Users\ol2234\Downloads\Project\Data\Geography\2024\MSAs and Micro\tl_2024_us_cbsa.shp"  # MSA & Micro path
state_path = r"C:\Users\ol2234\Downloads\Project\Data\Geography\2023\State" # States path
fips_path = r"C:\Users\ol2234\Downloads\Project\Data\Geography\FIPS\Contiguous US states FIPS.csv" # FIPS path

##################################################### Data centers #####################################################
gdf_data_centers = gpd.read_file(data_path) # Data Shapefile

##################################################### Geography #####################################################
df_fips = pd.read_csv(fips_path)    # Load FIPS
df_fips["FIPS Code"] = df_fips["FIPS Code"].astype(str).str.zfill(2)    # Ensure string, add 0 to single digits

gdf_urban = gpd.read_file(shapefile_path)   # Load urban areas
gdf_urban.rename(columns={'GEOID20': 'GEOID', 'NAME20': 'NAME', 'NAMELSAD20': 'Urban area', 'LSAD20': 'LSAD'}, inplace=True) # Rename columns
gdf_urban['State_Abbr'] = gdf_urban['NAME'].str[-2:]    # Get state abbreviations
contiguous_mask = gdf_urban['State_Abbr'].isin(df_fips['Postal Abbr.']) # All contiguous states
gdf_urban_removed = gdf_urban[~contiguous_mask].copy()  # Non-contiguous territories
gdf_urban = gdf_urban[contiguous_mask].copy()   # Get contiguous states

##################################################### Urban Area Assignment #####################################################
gdf_urban = gdf_urban.to_crs(epsg=5070)   # Convert to the correct CRS
gdf_data_centers = gdf_data_centers.to_crs(epsg=5070) # Convert to the correct CRS

urban_boundary = gdf_urban.geometry.union_all() # Create a boundary for urban
gdf_data_centers['Dist_to_urban'] = gdf_data_centers.distance(urban_boundary)
gdf_data_centers['Dist_to_urban_km'] = gdf_data_centers['Dist_to_urban'] / 1000

joined_urban = gpd.sjoin(gdf_data_centers, gdf_urban, how="left", predicate="within")
gdf_data_centers["Urban"] = joined_urban.index_right.notnull().astype(int)

dc_counts = joined_urban.groupby(['GEOID']).size().reset_index(name='DC Count')  # Count data centers
gdf_urban = gdf_urban.merge(dc_counts[['GEOID', 'DC Count']], on='GEOID', how='left')   # Add to shapefile
gdf_urban['DC Count'] = gdf_urban['DC Count'].fillna(0).astype(int) # Fill nans

##################################################### Micro/MSA Assignment #####################################################
gdf_msa_micro = gpd.read_file(msa_path)  # Shapefile
gdf_msa_micro = gdf_msa_micro.to_crs(epsg=5070) # Convert to the correct CRS

gdf_msa_micro['State_Abbr'] = gdf_msa_micro['NAME'].str[-2:]
gdf_msa_micro = gdf_msa_micro[gdf_msa_micro['State_Abbr'].isin(df_fips['Postal Abbr.'])].copy()

joined_msa_micro = gpd.sjoin(gdf_data_centers, gdf_msa_micro, how="left", predicate="within")
gdf_data_centers["MicronMSA"] = joined_msa_micro.index_right.notnull().astype(int)

##################################################### States Assignment #####################################################
gdf_states = gpd.read_file(state_path)  # Shapefile
gdf_states = gdf_states.to_crs(epsg=5070) # Convert to the correct CRS

gdf_states["STATEFP"] = gdf_states["STATEFP"].astype(str)   # Ensure string
gdf_states = gdf_states[gdf_states["STATEFP"].isin(df_fips["FIPS Code"])].copy().reset_index(drop=True)  # Clean list

##################################################### Plotting #####################################################
fig, ax = plt.subplots(figsize=(10, 10))

# States
gdf_states.plot(
    ax=ax,
    color="white",
    edgecolor="black",
    linewidth=0.75)

# MSA/Micro
gdf_msa_micro.plot(
    ax=ax,
    color="lightblue",
    edgecolor="blue",
    alpha=0.3,
    linewidth=0.5)

# Urban areas
gdf_urban.plot(
    ax=ax,
    color="red",
    edgecolor="darkred",
    alpha=0.5,
    linewidth=0.5)

# Data centers
gdf_data_centers.plot(
    ax=ax,
    facecolor="none",
    edgecolor="black",
    alpha=0.6,
    markersize=10,
    marker="o",
    linewidth=0.5
)

plt.axis("off")
plt.show()

##################################################### Histogram #####################################################
distances = gdf_data_centers.loc[gdf_data_centers['Dist_to_urban_km'] > 0, 'Dist_to_urban_km']

# Histogram
plt.figure(figsize=(8, 5))
plt.hist(distances, bins=50)
plt.xlabel("Distance to Urban (km)")
plt.ylabel("Number of Data Centers")
plt.grid(True)

plt.show()

##################################################### Statistics #####################################################
thresholds = [0, 5, 10, 20]
total = len(gdf_data_centers)

for t in thresholds:
    share = (gdf_data_centers['Dist_to_urban_km'] <= t).mean() * 100
    print(f"<= {t} km: {share:.1f}%")

mean_val = distances.mean()
median_val = distances.median()

print(f"Mean (outside urban areas): {mean_val:.2f} km")
print(f"Median (outside urban areas): {median_val:.2f} km")