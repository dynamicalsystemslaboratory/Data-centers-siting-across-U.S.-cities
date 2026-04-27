County Additional Regressors (SI)

This section builds extra county-level regressors used for robustness
checks in the county Negative Binomial model.

It adds: 
1) Poverty 
2) Land prices 

WORKFLOW 
1) Aggregate raw county features (MATLAB) 
2) Select final regressors and merge additional variables 
3) Build poverty and land-price county datasets 
4) Run updated county NB model 

county_agg_add_feat.m - Build base county regressor matrix from raw datasets

county_sel_add_feat.m - Select final county regressors analysis

poverty.py 
- Extract county poverty variable from ACS 2023 
- Uses S1701_C03_001E 
- Keeps contiguous U.S. only
- Output:EP_POV100_county.csv

land_prices.py 
- Compute county land values from places_fmv_all.tif 
-Uses zonal statistics (mean + median) 
- Keeps contiguous U.S. only 
- Output: county_land_prices.csv

main_county_final.py 
- County GLM Negative Binomial model 
- Adds poverty + land prices

OUTPUTS 
- Updated county regressors 
- Poverty and land-value datasets 
- NB model coefficients 

