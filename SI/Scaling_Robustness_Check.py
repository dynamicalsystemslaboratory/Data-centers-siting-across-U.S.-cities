import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

data_path = r"C:\Users\ol2234\Downloads\Project\Final data\MSA_Total_Nameplate_Capacity.csv"    # Total nameplate
pop_path = r"C:\Users\ol2234\Downloads\Project\Data\Population\MSA population data\ACSDT5Y2023.B01003-Data.csv" # Population

##################################################### Population #####################################################
df_pop = pd.read_csv(pop_path)   # Load data
df_pop = df_pop[['GEO_ID', 'NAME', 'B01003_001E']].copy()  # Take relevant columns
df_pop.rename(columns={'GEO_ID': 'GEOID', 'NAME': 'MSA', 'B01003_001E': 'Population'}, inplace=True) # Rename columns
df_pop = df_pop.iloc[1:].reset_index(drop=True) # Remove first row
df_pop['GEOID'] = df_pop['GEOID'].str[-5:]    # Get GEOID
df_pop["GEOID"] = df_pop["GEOID"].astype(str) # GEOID as string

##################################################### Nameplate #####################################################
df_data = pd.read_csv(data_path) # Data nameplate
df_data["GEOID"] = df_data["GEOID"].astype(str) # GEOID as string

df_data = df_data.merge(df_pop[['GEOID', 'Population']], on='GEOID', how='left')

################################################### Plotting ###################################################
plot_df = df_data.dropna().copy()  # Drop nan
plot_df['Population'] = pd.to_numeric(plot_df['Population'])
plot_df['Nameplate Capacity (MW)'] = pd.to_numeric(plot_df['Nameplate Capacity (MW)'])

plot_df = plot_df[(plot_df['Population'] > 0) & (plot_df['Nameplate Capacity (MW)'] > 0)].copy()

plot_df['log_power'] = np.log(plot_df['Nameplate Capacity (MW)'])
plot_df['log_population'] = np.log(plot_df['Population'])

derrible = 0
if derrible:
    p_low, p_high = plot_df['Population'].quantile([0.05, 0.95])
    plot_df = plot_df[(plot_df['Population'] >= p_low) &  (plot_df['Population'] <= p_high)].copy()

# Fit OLS regression
X = sm.add_constant(plot_df['log_population'])
y = plot_df['log_power']
model = sm.OLS(y, X).fit(cov_type='HC1')
r2 = model.rsquared

# Extract slope and 95% confidence interval
slope = model.params['log_population']
ci_lower, ci_upper = model.conf_int().loc['log_population']

print(f"Slope: {slope:.3f}")
print(f"95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]")

# Plot scatter
plt.figure(figsize=(7,5))
plt.scatter(plot_df['log_population'], plot_df['log_power'], alpha=0.6, edgecolor='k', s=50)

# Regression line
x_range = np.linspace(plot_df['log_population'].min(), plot_df['log_population'].max(), 100)
y_pred = model.predict(sm.add_constant(x_range))
plt.plot(x_range, y_pred, color='red', linewidth=2, label='OLS regression')

# Annotate slope and confidence interval on the plot
plt.text(
    0.05, 0.95,
    f"Slope = {slope:.2f}\n95% CI = [{ci_lower:.2f}, {ci_upper:.2f}]\nR² = {r2:.2f}",
    transform=plt.gca().transAxes,
    fontsize=12,
    verticalalignment='top',
    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

plt.xlabel('Population (log scale)')
plt.ylabel('Power (log scale)')

ax = plt.gca()

xticks = ax.get_xticks()
yticks = ax.get_yticks()

ax.set_xticks(xticks)
ax.set_yticks(yticks)

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda xt, _: f"{np.exp(xt):.0f}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda yt, _: f"{np.exp(yt):.1f}"))

plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()
plt.tight_layout()
plt.show()