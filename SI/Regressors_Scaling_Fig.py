import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
pd.options.mode.chained_assignment = None  # default='warn'
import statsmodels.api as sm
from matplotlib.ticker import LogLocator, NullFormatter

##################################################### Data #####################################################
nameplate_path = r"C:\Users\ol2234\Downloads\Project\Data\Regressors scaling\MSA_Total_Nameplate_Capacity.csv"    # Data centers count path
df_nameplate = pd.read_csv(nameplate_path) # Data centers count data

it_path = r"C:\Users\ol2234\Downloads\Project\Data\Regressors scaling\msa_it_emp_GEOID.csv"    # Data centers count path
df_it = pd.read_csv(it_path) # Data centers count data

energy_exp_path = r"C:\Users\ol2234\Downloads\Project\Data\Regressors scaling\msa_energy_exp_GEOID.csv"    # Data centers count path
df_energy_exp = pd.read_csv(energy_exp_path) # Data centers count data

pop_path = r"C:\Users\ol2234\Downloads\Project\Data\Population\MSA population data\ACSDT5Y2023.B01003-Data.csv" # Population

for df_2fix in [df_nameplate, df_it, df_energy_exp]:
    df_2fix['GEOID'] = df_2fix['GEOID'].astype(str)

##################################################### Population #####################################################
df_pop = pd.read_csv(pop_path)   # Load data
df_pop = df_pop[['GEO_ID', 'NAME', 'B01003_001E']].copy()  # Take relevant columns
df_pop.rename(columns={'GEO_ID': 'GEOID', 'NAME': 'MSA', 'B01003_001E': 'Population'}, inplace=True) # Rename columns
df_pop = df_pop.iloc[1:].reset_index(drop=True) # Remove first row
df_pop['GEOID'] = df_pop['GEOID'].str[-5:]    # Get GEOID
df_pop["GEOID"] = df_pop["GEOID"].astype(str) # GEOID as string

df_nameplate = df_nameplate.merge(df_pop[['GEOID', 'Population']], on='GEOID', how='left')   # Create clean dataset
df_it = df_it.merge(df_pop[['GEOID', 'Population']], on='GEOID', how='left')   # Create clean dataset
df_energy_exp = df_energy_exp.merge(df_pop[['GEOID', 'Population']], on='GEOID', how='left')   # Create clean dataset

for df_2fix, col in [(df_nameplate, "Nameplate Capacity (MW)"), (df_it, "IT"), (df_energy_exp, "Energy_Exp")]:
    df_2fix["Population"] = pd.to_numeric(df_2fix["Population"], errors='coerce')
    df_2fix[col] = pd.to_numeric(df_2fix[col], errors='coerce')

##################################################### SAMIs #####################################################
def get_samis(x0,y0):
    x = np.log10(x0)    # Logged x data
    y = np.log10(y0)    # Logged y data
    res = linregress(x,y)   # Linear regression
    y_hat = res.intercept + res.slope*x # Fit
    samis_fun = y - y_hat   # SAMIs
    return samis_fun

##################################################### Scaling #####################################################
def scaling(df, y_name, name, fig_name, x_ticks, y_ticks):
    df = df[(df[y_name] > 0) & (~df[y_name].isna())]    # Ensure no 0s

    derrible = 0
    if derrible:
        p_low, p_high = df['Population'].quantile([0.05, 0.95])
        df = df[(df['Population'] >= p_low) & (df['Population'] <= p_high)].copy()

    x0 = df["Population"].values    # Get population values
    y0 = df[y_name].values  # Get scaling variable

    samis = get_samis(x0, y0)   # Get SAMIs

    print("Var = ", np.round(np.var(samis), 2)) # Print variance of SAMIs
    print(len(x0))  # Get number of data points

    x = np.log10(x0)    # Logged x data
    y = np.log10(y0)    # Logged y data

    x_var = sm.add_constant(x)  # Constant in regression
    model = sm.OLS(y, x_var)    # Ordinary Least Squares
    fit = model.fit(cov_type='HC1') # Heteroscedasticity-consistent estimator

    intercept, slope = fit.params   # Get fitting parameters

    x_0 = np.sort(x0)[0]    # Minimum x0 value
    y_0 = 10**(slope*np.log10(x_0)+intercept)    # Minimum y0 value from fit

    x_f = np.sort(x0)[-1]   # Maximum x0 value
    y_f = 10**(slope*np.log10(x_f)+intercept)   # Maximum y0 value from fit

    betta = f"{slope:.3f}"  # Slope with 3 decimals
    r2 = f"{fit.rsquared:.3f}"  # R² with 3 decimals
    beta_lowerbound, beta_upper = fit.conf_int()[1]
    beta_lowerbound = f"{beta_lowerbound:.3f}"  # Lower CI with 3 decimals
    beta_upper = f"{beta_upper:.3f}"  # Upper CI with 3 decimals

    plt.rcParams.update({
        'font.size': 20,
        "lines.linewidth": 2,
        "font.family": "arial",
        # "font.serif": ["Computer Modern Roman"],
        "mathtext.fontset": "cm",
        "mathtext.default": "rm",
        "mathtext.rm": "arial",
    })  # Plot setup

    fig, (ax) = plt.subplots(1, 1, sharey='row', figsize=(6, 6))    # Initialize plot
    ax.scatter(x0, y0, facecolors='none', edgecolors='k')   # Scatter plot
    ax.plot([x_0, x_f], [y_0, y_f], lw=2, color='k')    # Plot range

    ax.text(0.02, 0.95, r'$\beta = {}$'.format(betta) + r'$ \, \in \,({}$'.format(beta_lowerbound) + r'$,{})$'.format(beta_upper),
            ha='left', va='top', transform=ax.transAxes)    # Add slope CI
    ax.text(0.02, 0.85, r'$\mathit{R}^2 = $' + r'${}$'.format(r2), ha='left', va='top', transform=ax.transAxes) # Add R2

    ax.set_yscale("log")    # X log scale
    ax.set_xscale("log")    # Y log scale
    ax.set_xlabel("Population") # X axis label
    ax.set_ylabel(name) # Y axis label

    ax.set_xlim(x_ticks[0], x_ticks[-1])  # Use the provided x_ticks as range
    ax.set_ylim(y_ticks[0], y_ticks[-1])  # Use the provided y_ticks as range

    ax.set_xticks(x_ticks)  #   Set X axis ticks
    ax.set_yticks(y_ticks)  #   Set Y axis ticks

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[5.0, 10.0], numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())

    if derrible == 0:
        plt.savefig(rf"C:\Users\ol2234\Downloads\Project\Figures\New\Scaling\{fig_name}.pdf", bbox_inches="tight")
    else:
        plt.savefig(rf"C:\Users\ol2234\Downloads\Project\Figures\New\Scaling\{fig_name}minmax.pdf", bbox_inches="tight")

    plt.show()
    return


##################################################### Plotting #####################################################
scaling(df_nameplate, "Nameplate Capacity (MW)", "Nameplate Capacity (MW)", "Nameplate Scaling", [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8], [10 ** (-1), 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
scaling(df_it, "IT", "IT employment", "IT employment Scaling", [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8], [10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6])
scaling(df_energy_exp, "Energy_Exp", "Energy expenditure (USD)", "Energy Expenditure Scaling", [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8], [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8, 10 ** 9, 10 ** 10, 10 ** 11])
