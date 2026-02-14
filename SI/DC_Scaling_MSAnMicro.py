import pandas as pd
import geopandas as gpd

##################################################### Paths #####################################################
data_path = r"C:\Users\ol2234\Downloads\Project\Data\Data centers\Shapefile\Data_centers_full.shp"    # Data centers path
original_data_path = r"C:\Users\ol2234\Downloads\Project\Data\Data centers\datacentermap.csv"    # Data centers path
pop_path = r"C:\Users\ol2234\Downloads\Project\Data\Population\MSA population data\ACSDT5Y2023.B01003-Data.csv" # Population
shapefile_path = r"C:\Users\ol2234\Downloads\Project\Data\Geography\2024\MSAs and Micro\tl_2024_us_cbsa.shp"

##################################################### Data centers #####################################################
gdf_data_centers = gpd.read_file(data_path) # Data Shapefile
df_original_data_centers = pd.read_csv(original_data_path) # Original data centers data

# Fix data
mask_zero_id = gdf_data_centers['id'] == 0  # Rows with id = 0

gdf_data_centers.loc[mask_zero_id, 'name'] = gdf_data_centers.loc[mask_zero_id, 'USER_name']
gdf_data_centers.loc[mask_zero_id, 'id'] = gdf_data_centers.loc[mask_zero_id, 'USER_id']
gdf_data_centers.loc[mask_zero_id, 'company_na'] = gdf_data_centers.loc[mask_zero_id, 'USER_compa']
gdf_data_centers.loc[mask_zero_id, 'company_id'] = gdf_data_centers.loc[mask_zero_id, 'USER_com_1']
gdf_data_centers.loc[mask_zero_id, 'stage'] = gdf_data_centers.loc[mask_zero_id, 'USER_stage']
gdf_data_centers.loc[mask_zero_id, 'type'] = gdf_data_centers.loc[mask_zero_id, 'USER_type']
gdf_data_centers.loc[mask_zero_id, 'latitude'] = gdf_data_centers.loc[mask_zero_id, 'USER_latit']
gdf_data_centers.loc[mask_zero_id, 'longitude'] = gdf_data_centers.loc[mask_zero_id, 'USER_longi']
gdf_data_centers.loc[mask_zero_id, 'city'] = gdf_data_centers.loc[mask_zero_id, 'USER_city']
gdf_data_centers.loc[mask_zero_id, 'state'] = gdf_data_centers.loc[mask_zero_id, 'USER_state']
gdf_data_centers.loc[mask_zero_id, 'whitespa_1'] = gdf_data_centers.loc[mask_zero_id, 'USER_whi_1']
gdf_data_centers.loc[mask_zero_id, 'building_1'] = gdf_data_centers.loc[mask_zero_id, 'USER_bui_1']
gdf_data_centers.loc[mask_zero_id, 'power_mw'] = gdf_data_centers.loc[mask_zero_id, 'USER_power']
gdf_data_centers.loc[mask_zero_id, 'pue'] = gdf_data_centers.loc[mask_zero_id, 'USER_pue']
gdf_data_centers.loc[mask_zero_id, 'year_opera'] = gdf_data_centers.loc[mask_zero_id, 'USER_year_']

gdf_data_centers = gdf_data_centers.rename(columns={'company_na': 'company_name', 'whitespa_1': 'whitespace_sqm', 'building_1': 'building_sqm', 'year_opera': 'year_operational'}) # Rename columns
gdf_data_centers = gdf_data_centers[['name', 'id', 'company_name', 'company_id', 'stage', 'type', 'latitude', 'longitude', 'city', 'state', 'whitespace_sqm', 'building_sqm', 'power_mw', 'pue', 'year_operational', 'geometry']]    # Relevant columns
gdf_data_centers = gdf_data_centers.fillna(0)   # Replace nans by 0s

df_original_data_centers = df_original_data_centers[['name', 'id', 'company_name', 'company_id', 'stage', 'type', 'latitude', 'longitude', 'city', 'state', 'whitespace_sqm', 'building_sqm', 'power_mw', 'pue', 'year_operational']]    # Relevant columns
df_original_data_centers = df_original_data_centers.fillna(0)   # Replace nans by 0s

# Sanity check
compare_cols = ['whitespace_sqm', 'building_sqm', 'power_mw', 'pue'] # Columns to compare
for col in compare_cols:    # Convert to numerical
    gdf_data_centers[col] = pd.to_numeric(gdf_data_centers[col], errors='coerce')
    df_original_data_centers[col] = pd.to_numeric(df_original_data_centers[col], errors='coerce')

gdf_compare = gdf_data_centers[compare_cols + ['id']].copy() # Shapefile to compare
df_compare = df_original_data_centers[compare_cols + ['id']].copy()  # Original data to compare

gdf_compare = gdf_compare.sort_values('id').reset_index(drop=True)  # Sort by id
df_compare = df_compare.sort_values('id').reset_index(drop=True)    # Sort by id

compare_cols_sorted = compare_cols  # Columns to compare

for col in compare_cols_sorted:    # Ensure same data types
    df_compare[col] = df_compare[col].astype(gdf_compare[col].dtype)

are_identical = gdf_compare[compare_cols_sorted].equals(df_compare[compare_cols_sorted])  # Compare
if not are_identical:   # If not identical, find differences
    diff = gdf_compare[compare_cols_sorted].compare(df_compare[compare_cols_sorted])

##################################################### Population #####################################################
df_pop = pd.read_csv(pop_path)   # Load data
df_pop = df_pop[['GEO_ID', 'NAME', 'B01003_001E']].copy()  # Take relevant columns
df_pop.rename(columns={'GEO_ID': 'GEOID', 'NAME': 'MSA', 'B01003_001E': 'Population'}, inplace=True) # Rename columns
df_pop = df_pop.iloc[1:].reset_index(drop=True) # Remove first row
df_pop['GEOID'] = df_pop['GEOID'].str[-5:]    # Get GEOID
df_pop["GEOID"] = df_pop["GEOID"].astype(str) # GEOID as string

##################################################### Geography #####################################################
msa_shapefile_path = r"C:\Users\ol2234\Downloads\Project\Data\Geography\2024\MSAs and Micro\tl_2024_us_cbsa.shp"    # MSA & Micro
fips_path = r"C:\Users\ol2234\Downloads\Project\Data\Geography\FIPS\Contiguous US states FIPS.csv"  # FIPS

df_fips = pd.read_csv(fips_path)    # Load FIPS
df_fips["FIPS Code"] = df_fips["FIPS Code"].astype(str).str.zfill(2)    # Ensure string, add 0 to single digits

gdf_msa_micro = gpd.read_file(msa_shapefile_path)   # Load MSA & Micro
gdf_msa_micro['State_Abbr'] = gdf_msa_micro['NAME'].str[-2:]    # Get state abbreviations
contiguous_mask = gdf_msa_micro['State_Abbr'].isin(df_fips['Postal Abbr.']) # All contiguous states
gdf_msa_micro_removed = gdf_msa_micro[~contiguous_mask].copy()  # Non-contiguous territories
gdf_msa_micro = gdf_msa_micro[contiguous_mask].copy()   # Get contiguous states
valid_geoid = gdf_msa_micro["GEOID"]    # Contiguous GEOIDs

# Guarantee same CRS
if gdf_msa_micro.crs != gdf_data_centers.crs:
    gdf_data_centers = gdf_data_centers.to_crs(gdf_msa_micro.crs)

# Assign data centers to MSA and Micro
gdf_us_data = gpd.sjoin(gdf_data_centers, gdf_msa_micro[['geometry', 'GEOID', 'NAMELSAD', 'LSAD']], how="left", predicate="within")
gdf_us_data = gdf_us_data.drop(columns=['index_right']) # Remove index column
gdf_us_data = gdf_us_data.dropna(subset=['GEOID'])  # Remove non MSA/Micro data centers

# Data centers counts
dc_counts = gdf_us_data.groupby(['GEOID']).size().reset_index(name='DC Count')  # Count data centers
gdf_msa_micro_count = gdf_msa_micro.merge(dc_counts[['GEOID', 'DC Count']], on='GEOID', how='left')   # Add to shapefile
gdf_msa_micro_count['DC Count'] = gdf_msa_micro_count['DC Count'].fillna(0).astype(int) # Fill nans
dc_count_scaling_data = dc_counts.merge(df_pop[['GEOID', 'Population']], on='GEOID', how='left')   # Create clean dataset
dc_count_scaling_data = dc_count_scaling_data[dc_count_scaling_data["GEOID"].isin(valid_geoid)]   # Data for contiguous states

# Data centers MW
mw_stats = gdf_us_data.loc[gdf_us_data['power_mw'].notna() & (gdf_us_data['power_mw'] > 0)].groupby('GEOID').agg(Mean_Power_MW=('power_mw', 'mean')).reset_index()  # Average MW in each city
dc_mw_estimated = dc_counts.merge(mw_stats, on='GEOID', how='left') # Add DC counts
dc_mw_estimated['Total_Power_MW'] = dc_mw_estimated['DC Count'] * dc_mw_estimated['Mean_Power_MW']  # Estimated total power
mw_coverage = gdf_us_data.assign(has_mw=lambda x: x['power_mw'].notna() & (x['power_mw'] > 0)).groupby('GEOID').agg(DC_With_MW=('has_mw', 'sum')).reset_index()    # Track power coverage
dc_mw_estimated = dc_mw_estimated.merge(mw_coverage, on='GEOID', how='left')
dc_mw_estimated = dc_mw_estimated.dropna(subset=['Mean_Power_MW'])

gdf_msa_micro_mw = gdf_msa_micro.merge(dc_mw_estimated[['GEOID', 'Total_Power_MW']], on='GEOID', how='left')   # Add to shapefile
dc_scaling_mw_data = dc_mw_estimated.merge(df_pop[['GEOID', 'Population']], on='GEOID', how='left')   # Create clean dataset
dc_scaling_mw_data = dc_scaling_mw_data[dc_scaling_mw_data["GEOID"].isin(valid_geoid)]   # Data for contiguous states

##################################################### Scaling #####################################################
dc_count_scaling_data['Population'] = pd.to_numeric(dc_count_scaling_data['Population'], errors='coerce') # Data to numeric
dc_count_scaling_data['DC Count'] = pd.to_numeric(dc_count_scaling_data['DC Count'], errors='coerce') # Data to numeric

mask_count = (dc_count_scaling_data['Population'] > 0) & (dc_count_scaling_data['DC Count'] > 0)  # Find 0s
dc_count_scaling_data = dc_count_scaling_data[mask_count] # Remove 0s

dc_scaling_mw_data['Population'] = pd.to_numeric(dc_scaling_mw_data['Population'], errors='coerce') # Data to numeric
dc_scaling_mw_data['Total_Power_MW'] = pd.to_numeric(dc_scaling_mw_data['Total_Power_MW'], errors='coerce') # Data to numeric

mask_mw = (dc_scaling_mw_data['Population'] > 0) & (dc_scaling_mw_data['Total_Power_MW'] > 0)  # Find 0s
dc_scaling_mw_data = dc_scaling_mw_data[mask_mw] # Remove 0s

##################################################### Export #####################################################
output_count_path = r"C:\Users\ol2234\Downloads\Project\Figures\New\Scaling\DC_count.csv"    # output path
dc_count_scaling_data.to_csv(output_count_path, index=False)  # Save

# output_power_path = r"C:\Users\ol2234\Downloads\Project\Figures\New\Scaling\DC_power.csv"    # output path
# dc_scaling_mw_data.to_csv(output_power_path, index=False)  # Save