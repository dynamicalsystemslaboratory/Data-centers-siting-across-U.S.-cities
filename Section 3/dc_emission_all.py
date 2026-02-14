"""
Estimates state-level CO_2 emissions allocated to U.S. data centers using their IT power capacity data,
state nameplate capacity and eGRID state annual CO_2 emissions , then visualizes the results.

1. Load data center shapefile, eGRID spreadsheet, state FIPS table, and U.S. state polygons.
2. Keep CONUS only and clean state identifiers.
3. Compute a state emissions factor:
   - annual CO₂ emissions / state nameplate capacity (tons CO₂ per MW).
4. Aggregate data centers by state:
   - count, mean, median, min, max capacity (MW).
5. Estimate data-center emissions per state by multiplying capacity stats by the emissions factor.
6. Merge estimates with state geometries.
7. Produce plots:
   - scatter plot: capacity vs emissions factor.
   - choropleth maps (log scale): mean and median emission estimates.
"""
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

######################################### Load data #########################################
data_center_path = r"/Users/camilla/PycharmProjects/datacenters/Cleaned Shapefile/DC_Shapefile.shp"   # Data centers path
emissions_path = r"/Users/camilla/PycharmProjects/datacenters/egrid2023_data_rev2.xlsx"   # eGRID path
fips_path = r"/Users/camilla/PycharmProjects/datacenters/Contiguous US states FIPS.csv"   # State FIPS path
states_path = r"/Users/camilla/PycharmProjects/datacenters/State/tl_2023_us_state.shp"   # States shapefile path

gpd_data_center = gpd.read_file(data_center_path)   # Load data centers data
egrid = pd.ExcelFile(emissions_path)    # eGRID data
df_emissions = egrid.parse("ST23")   # Load eGRID data
df_fips = pd.read_csv(fips_path)  # Load FIPS
gpd_state = gpd.read_file(states_path)   # Load states shapefile

######################################### Process data #########################################
df_data_center = gpd_data_center[['State', 'power_mw']]    # Take relevant columns
df_data_center = df_data_center[df_data_center['power_mw'] > 0]  # Take data centers with power data

df_fips['FIPS Code'] = df_fips['FIPS Code'].astype(str)  # FIPS to string
df_fips['FIPS Code'] = df_fips['FIPS Code'].str.zfill(2)    # Pad FIPS

gpd_state = gpd_state.merge(df_fips[['FIPS Code', 'State']], left_on='STATEFP', right_on='FIPS Code', how='left')  # Add full state name

df_emissions = df_emissions[['State abbreviation', 'State nameplate capacity (MW)', 'State annual CO2 emissions (tons)']]  # Columns to take
df_emissions = df_emissions.iloc[1:].reset_index(drop=True)  # Remove secondary column acronym
df_emissions.rename(columns={
    'State abbreviation': 'Postal Abbr.',
    'State nameplate capacity (MW)': 'Nameplate',
    'State annual CO2 emissions (tons)': 'State emissions'}, inplace=True)    # Rename columns
exclude_abbr = ['HI', 'AK', 'PR']  # States to exclude
df_emissions = df_emissions[~df_emissions['Postal Abbr.'].isin(exclude_abbr)]  # Take only contiguous states
df_emissions['Emissions factor'] = df_emissions['State emissions']/df_emissions['Nameplate']    # Tons of CO2 per MW
df_emissions = df_emissions.merge(df_fips[['Postal Abbr.', 'State']], on='Postal Abbr.', how='left')    # Add full state name

df_emissions_estimation = (df_data_center.groupby('State').agg(
        **{'Number of DCs': ('power_mw', 'count'),
            'Mean DC capacity': ('power_mw', 'mean'),
            'Median DC capacity': ('power_mw', 'median'),
            'Maximal DC capacity': ('power_mw', 'max'),
            'Minimal DC capacity': ('power_mw', 'min')}).reset_index())    # Aggregate data centers by state

df_emissions_estimation = df_emissions[['State', 'Nameplate', 'State emissions', 'Emissions factor']].merge(df_emissions_estimation, on='State', how='left')    # Join all data
num_cols = df_emissions_estimation.select_dtypes(include=np.number).columns  # Columns to complete
df_emissions_estimation[num_cols] = df_emissions_estimation[num_cols].fillna(0)  # Complete missing data
df_emissions_estimation["Mean DC emission estimate"] = df_emissions_estimation['Mean DC capacity']*df_emissions_estimation['Emissions factor']  # Estimation
df_emissions_estimation["Median DC emission estimate"] = df_emissions_estimation['Median DC capacity']*df_emissions_estimation['Emissions factor']  # Estimation
df_emissions_estimation["Maximal DC emission estimate"] = df_emissions_estimation['Maximal DC capacity']*df_emissions_estimation['Emissions factor']  # Estimation
df_emissions_estimation["Minimal DC emission estimate"] = df_emissions_estimation['Minimal DC capacity']*df_emissions_estimation['Emissions factor']  # Estimation
estimate_cols = ['Mean DC emission estimate', 'Median DC emission estimate', 'Maximal DC emission estimate', 'Minimal DC emission estimate']    # Estimates
df_emissions_estimation[estimate_cols] = df_emissions_estimation[estimate_cols].round(2)    # Round

gdf_emissions_estimation = df_emissions_estimation.merge(gpd_state[['State', 'geometry']], on='State', how='left')  # Add geometry
gdf_emissions_estimation = gpd.GeoDataFrame(gdf_emissions_estimation, geometry='geometry')  # Convert to gdf

######################################### Plotting #########################################
# Sanity check - estimation components comparison
plt.figure(figsize=(10, 6))  # Initialize plot
plt.grid(True)  # Grid
plt.scatter(df_emissions_estimation['Mean DC capacity'], df_emissions_estimation['Emissions factor'],
            color='orange', label='Mean capacity', alpha=0.7, s=120)  # Plot capacity vs factor (mean)
plt.scatter(df_emissions_estimation['Median DC capacity'], df_emissions_estimation['Emissions factor'],
            color='grey', label='Median capacity', alpha=0.7, s=120)  # Plot capacity vs factor (mean)
plt.xlabel('Data center capacity (MW)', fontsize=22)  # X label
plt.ylabel('Tons CO$_2$ per MW', fontsize=22)  # Y label
plt.legend(fontsize=20)    # Legend
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)

plt.savefig("plot_crappy", dpi=600, bbox_inches="tight", transparent=True)

plt.show()  # Actually plot

# Maps
gdf_mean = gdf_emissions_estimation.copy()  # Mean DC emissions
gdf_median = gdf_emissions_estimation.copy()    # Median DC emissions

gdf_mean['log_mean_emission'] = gdf_mean['Mean DC emission estimate'].apply(lambda x: np.log10(x) if x > 0 else np.nan)  # Mean emissions masked for 0s
gdf_median['log_median_emission'] = gdf_median['Median DC emission estimate'].apply(lambda x: np.log10(x) if x > 0 else np.nan)  # Median emissions masked for 0s

vmin = 1    # Min range
vmax = 6    # Max range

fig, ax = plt.subplots(1, 1, figsize=(15, 10))  # Initialize plot

gdf_mean.plot(column='log_mean_emission', cmap='RdYlGn_r', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
              vmin=vmin, vmax=vmax, missing_kwds={'color': 'white', 'label': '0 emission'})   # Plot logged mean emissions
ax.set_title('Mean DC Emission Estimates')    # Title
ax.axis('off')  # No axes
plt.show()  # Actually plot

fig, ax = plt.subplots(1, 1, figsize=(15, 10))  # Initialize plot

gdf_median.plot(column='log_median_emission', cmap='RdYlGn_r', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True,
                vmin=vmin, vmax=vmax, missing_kwds={'color': 'white', 'label': '0 emission'})  # Plot logged median emissions
ax.set_title('Median DC Emission Estimates')   # Title
ax.axis('off')  # No axes
plt.show()  # Actually plot
