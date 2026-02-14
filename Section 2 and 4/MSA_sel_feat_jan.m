clear
clc

%% Load dataset from aggregate_feature_MSA_one and take the SAMIs 
% for the variables that scales with population

% county-level
county_all = readtable("county_features_all_nh_area_eia_noprox.csv", ...
    'VariableNamingRule','preserve');

% city-level
msa_all = readtable("msa_features_all_def_eia_noprox_ret.csv", ...
    'VariableNamingRule','preserve');

county_all.FIPS = string(county_all.FIPS);
msa_all.MSA     = string(msa_all.MSA);


%%  variables (energy expenditure, IT jobs) SAMI

%% county-level
T= readtable("pop_county.csv");
% SAMIs 
county_all.SAMI_energy_exp        = SAMIs_fun(T.Pop, county_all.energy_exp);
county_all.SAMI_energy_emp_it_total = SAMIs_fun(T.Pop, county_all.emp_it_total);

%% city-level
T2= readtable("pop_msa.csv");

msa_all.SAMI_energy_exp        = SAMIs_fun(T2.Value, msa_all.energy_exp_msa);
msa_all.SAMI_energy_emp_it_total = SAMIs_fun(T2.Value, msa_all.emp_it_total);

%%selected regressors 

%% county-level
county_base_vars_full = { ...
    'water_cf', ...                     % water stress
    'bqs', ...                          % broadband quality
    'nri_events_per_area', ...          % total NRI events normalized by area
    'evi', ...                          % energy grid stress index
    'retired_coal', ...                 % retired coal plants
    'benefit_flag', ...                 % state tax incentives
    'elec_var_ind_all', ...             % electric-price volatility
    'SAMI_energy_exp', ...              % energy expenditure
    'SAMI_energy_emp_it_total',...      % IT employement
    'tot_cap', ...                      % nameplate capacity
    };                     
county_all = county_all(:,[1:7 9 11:end] );
county_regressors_full = county_all(:, ['FIPS', county_base_vars_full]);


%% city-level
msa_base_vars_full = { ...
    'water_sum', ...                   % water stress
    'bqs_mean', ...                     % broadband quality
    'events_per_area', ...              % total NRI events normalized by area
    'evi_mean', ...                     % energy grid stress index
    'benefit_flag', ...                 % state tax incentives
    'msa_ret_coal', ...                % retired coal plants
    'elec_var_ind_all', ...             % electric-price volatility
    'SAMI_energy_exp', ...            % energy expenditure
    'SAMI_energy_emp_it_total',...  % IT employement
    'cap_total'};                    % nameplate capacity
    
msa_all = msa_all(:,[1:8 10 12:end] );
msa_all.msa_ret_coal = fillmissing(msa_all.msa_ret_coal, 'constant', 0);
msa_all.cap_total = fillmissing(msa_all.cap_total, 'constant', 0);
msa_regressors_full = msa_all(:, ['MSA', msa_base_vars_full]);

%% SAVE REGRESSOR MATRIX

%writetable(county_regressors_full, "county_regressors_totals_nh_area_eia_noprox.csv");
writetable(msa_regressors_full,    "msa_regressors_totals_nh_area_eia_noprox_ret.csv");

function sami = SAMIs_fun(popVec, featVec)

    popVec = double(popVec);
    featVec = double(featVec);

    valid = isfinite(popVec) & isfinite(featVec) & popVec > 0 & featVec > 0;

    sami = NaN(size(popVec)); 

    xlog = log10(popVec(valid));
    ylog = log10(featVec(valid));

    % fit log-log: ylog = slope * xlog + intercept
    p = polyfit(xlog, ylog, 1);
    yhat = polyval(p, xlog);

    % SAMI = residuals in log10  
    sami(valid) = ylog - yhat;
end