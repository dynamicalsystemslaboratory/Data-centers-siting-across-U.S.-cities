clear
clc

%% Load dataset from aggregate_feature_MSA_one and take the SAMIs 
% for the variables that scales with population

% county-level
county_all = readtable("county_features_all_nh_area_eia_noprox_ret_review3.csv", ...
    'VariableNamingRule','preserve');

% city-level
msa_all = readtable("msa_features_all_def_eia_noprox_ret_review4.csv", ...
    'VariableNamingRule','preserve');

county_all.FIPS = string(county_all.FIPS);
msa_all.MSA     = string(msa_all.MSA);


%%  variables (energy expenditure, IT jobs) SAMI

%% county-level
T= readtable("pop_county.csv");
%% SCALING CHECK (county-level)
Tpop = readtable("pop_county.csv");

Tpop.FIPS = string(Tpop.FIPS);
county_all.FIPS = string(county_all.FIPS);

J = innerjoin(county_all, Tpop, 'Keys', 'FIPS');

pop = double(J.Pop);

% LAND
land = (double(J.landval_mean_log));

valid = isfinite(pop) & isfinite(land) & pop > 0;
beta_land = polyfit(log(pop(valid)), land(valid), 1);
fprintf('beta_land = %.3f\n', beta_land(1));
%% SCALING CHECK (MSA-level)
Tpop_msa = readtable("pop_msa.csv");
Tpop_msa.MSA = string(Tpop_msa.MSA);
Tpop_msa.MSA = regexprep(Tpop_msa.MSA, '0$', '');
Tpop_msa.MSA = "C" + Tpop_msa.MSA;
msa_all.MSA = string(msa_all.MSA);
J = innerjoin(msa_all, Tpop_msa, 'Keys', 'MSA');

pop = double(J.Value);

% LAND
land = (double(J.landval_mean_log));

valid = isfinite(pop) & isfinite(land) & pop > 0;
beta_land = polyfit(log(pop(valid)), land(valid), 1);
fprintf('beta_land = %.3f\n', beta_land(1));
scatter(log(pop(valid)), land(valid), '.');
xlabel('log(pop)');
ylabel('log(land)');
title('Scaling check: land prices');
figure;
scatter(log10(pop(valid)), log10(pov(valid)), '.');
xlabel('log10(pop)');
ylabel('log10(pov)');
title('Scaling check: poverty');
%  POVERTY 
pov = double(J.pov100);

valid = isfinite(pop) & isfinite(pov) & pop > 0 & pov > 0;
beta_pov = polyfit(log10(pop(valid)), log10(pov(valid)), 1);
fprintf('beta_pov = %.3f\n', beta_pov(1));

fprintf('Scaling exponent (poverty): %.3f\n', beta_pov(1)); figure;
scatter(log(pop(valid)), land(valid), '.');
xlabel('log(pop)');
ylabel('log(land)');
title('Scaling check: land prices');
figure;
scatter(log10(pop(valid)), log10(pov(valid)), '.');
xlabel('log10(pop)');
ylabel('log10(pov)');
title('Scaling check: poverty');

% SAMIs 
county_all.SAMI_energy_exp        = SAMIs_fun(T.Pop, county_all.energy_exp);
county_all.SAMI_energy_emp_it_total = SAMIs_fun(T.Pop, county_all.emp_it_total);
county_all.SAMI_land_prices = SAMIs_fun(T.Pop, exp(county_all.landval_mean_log));
%% city-level
T2= readtable("pop_msa.csv");

msa_all.SAMI_energy_exp        = SAMIs_fun(T2.Value, msa_all.energy_exp_msa);
msa_all.SAMI_energy_emp_it_total = SAMIs_fun(T2.Value, msa_all.emp_it_total);
msa_all.SAMI_land_prices = SAMIs_fun(T2.Value, exp(msa_all.landval_mean_log));
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
    'SAMI_land_prices',...              % land prices 
    'pov100',...                        % poverty
    };                     
county_all = county_all(:,[1:7 9 11:end] );
county_regressors_full = county_all(:, ['FIPS', county_base_vars_full]);


%% city-level
% msa_base_vars_full = { ...
%     'water_sum', ...                   % water stress
%     'bqs_mean', ...                     % broadband quality
%     'events_per_area', ...              % total NRI events normalized by area
%     'evi_mean', ...                     % energy grid stress index
%     'benefit_flag', ...                 % state tax incentives
%     'msa_ret_coal', ...                % retired coal plants
%     'elec_var_ind_all', ...             % electric-price volatility
%     'SAMI_energy_exp', ...            % energy expenditure
%     'SAMI_energy_emp_it_total',...  % IT employement
%     'cap_total'};                    % nameplate capacity
msa_base_vars_full = { ...
'water_sum', ...
'bqs_mean', ...
'events_per_area', ...
'evi_mean', ...
'benefit_flag', ...
'msa_ret_coal', ...
'elec_var_ind_all', ...
'SAMI_energy_exp', ...
'SAMI_energy_emp_it_total',...
'cap_total', ...
'SAMI_land_prices', ...
'pov100'};
msa_all = msa_all(:,[1:8 10 12:end] );
msa_all.msa_ret_coal = fillmissing(msa_all.msa_ret_coal, 'constant', 0);
msa_all.cap_total = fillmissing(msa_all.cap_total, 'constant', 0);
msa_regressors_full = msa_all(:, ['MSA', msa_base_vars_full]);

%% SAVE REGRESSOR MATRIX

writetable(county_regressors_full, "county_regressors_totals_nh_area_eia_noprox_ret_review3.csv");
writetable(msa_regressors_full,    "msa_regressors_totals_nh_area_eia_noprox_ret_review4.csv");

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