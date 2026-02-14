import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
pd.options.mode.chained_assignment = None  # default='warn'
import statsmodels.api as sm
from matplotlib.ticker import LogLocator, NullFormatter

##################################################### Data #####################################################
data_center_count_path = r"C:\Users\ol2234\Downloads\Project\Figures\New\Scaling\DC_count.csv"    # Data centers count path
df_data_centers_count = pd.read_csv(data_center_count_path) # Data centers count data

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
    df = df[df[y_name] != 0]    # Ensure no 0s
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

    if y_ticks[0] == 10**0:
        y_ticks0 = 0.9
    else:
        y_ticks0 = y_ticks[0]

    ax.set_xlim(min(float(x_0), x_ticks[0]), max(float(x_f), x_ticks[-1])) #   Set X axis range
    ax.set_ylim(min(float(y_0), y_ticks0), max(float(y_f), y_ticks[-1])) #   Set Y axis range

    ax.set_xticks(x_ticks)  #   Set X axis ticks
    ax.set_yticks(y_ticks)  #   Set Y axis ticks

    ax.xaxis.set_major_locator(LogLocator(base=10.0, subs=[5.0, 10.0], numticks=10))
    ax.xaxis.set_minor_formatter(NullFormatter())

    plt.savefig(rf"C:\Users\ol2234\Downloads\Project\Figures\New\Scaling\{fig_name}.pdf", bbox_inches="tight")

    plt.show()
    return


##################################################### Plotting #####################################################
scaling(df_data_centers_count, "DC Count", "Data center count", "Data Center Count Scaling", [10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7, 10 ** 8], [10 ** (-1), 10 ** 0, 10 ** 1, 10 ** 2, 10 ** 3])
