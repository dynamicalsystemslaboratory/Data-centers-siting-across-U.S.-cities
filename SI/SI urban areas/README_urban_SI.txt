Urban Areas vs Cities (SI)

This section compares U.S. data center locations with: 1) Census Urban
Areas 2) MSA / MicroSA boundaries (Cities)

WORKFLOW 
1) Assign data centers to Urban Areas 
2) Assign data centers to MSA / MicroSA boundaries 
3) Measure distance to nearest urban boundary
4) Produce maps, histograms, and summary statistics

DC_Count_Urban_MSAnMicro.py - Spatial joins for Urban Areas and cities boundaries 
- Distance to urban calculations 
- Urban counts and proximity statistics 
- National map + histogram

urban_plot.py 
- Publication-quality SI figures 
- Final map of cities + urban areas + data centers 
- Histogram of distance to urban areas 
- Export-ready PNG figures

OUTPUTS 
- map_data_centers_urban.png 
- hist_distances_urban.png 
-Distance statistics 
- Shares within 0, 5, 10, and 20 km
