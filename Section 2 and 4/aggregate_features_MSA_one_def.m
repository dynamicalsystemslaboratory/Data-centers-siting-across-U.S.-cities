%% this script take all the data ad build a matrix used 
% for the regression matrix at county and city level
clear
clc
close all


%% path and crosswalk
addpath /Users/camilla/Dropbox/feature_file_MSA_crosswalk

crosswalk = readtable("crosswalk_county_msa.xlsx", ...
    'VariableNamingRule','preserve');

% normalize the county and MSA code
crosswalk.CountyCode = pad(string(crosswalk.("County Code")), 5, 'left','0');
crosswalk.MSACode    = string(crosswalk.("MSA Code"));

% remove Alaska, Hawaii and Puerto Rico 
crosswalk = crosswalk(~startsWith(crosswalk.CountyCode,"72"), :);
crosswalk = crosswalk(~startsWith(crosswalk.CountyCode,"02"), :);
crosswalk = crosswalk(~startsWith(crosswalk.CountyCode,"15"), :);
% all cities (metro + micro)
all_msa = unique(crosswalk.MSACode);
%% county fips and state mapping
crosswalk_county_state = readtable("Full_County_List.csv", ...
    'VariableNamingRule','preserve');
cw_cs       = crosswalk_county_state; 
cw_cs.FIPS5 = pad(string(cw_cs.("FIPS")), 5, 'left','0');  % 01001 etc


dc_county = readtable("Full_DC_Counts_County.csv", ...
    'VariableNamingRule','preserve');
dc_county.FIPS5 = pad(string(dc_county.FIPS), 5, 'left','0');

% county all is a table with county FIPS as rows
county_all = table(cw_cs.FIPS5, 'VariableNames', {'FIPS'});
county_all.FIPS = string(county_all.FIPS);

% add county state fip
[tf_sf, loc_sf] = ismember(county_all.FIPS, cw_cs.FIPS5);
county_all.state_fips_2 = NaN(height(county_all),1);
county_all.state_fips_2(tf_sf) = cw_cs.state_fips_2(loc_sf(tf_sf));
C_base = innerjoin( ...
    cw_cs(:,{'FIPS5','state_fips_2'}), ...
    crosswalk(:,{'CountyCode','MSACode'}), ...
    'LeftKeys','FIPS5', ...
    'RightKeys','CountyCode' );

%%%%% adding regressors %%%%%%%%%
%% WATER 
T  = readtable("AWARE_US_CF.csv", 'VariableNamingRule','preserve');
water_county = T(:,[1,4]);  

% county-level
FIPS_aw   = pad(string(T{:,1}), 5, 'left','0');  % column 1 = FIPS
water_val = force_numeric(T{:,4});               % column 4 = AWARE value
water_val(~isfinite(water_val) | water_val < 0) = NaN;
county_all = add_var_county(county_all, FIPS_aw, water_val, 'water_cf');

% city-level
msa_water = county_to_msa_by_fips( ...
    "AWARE_US_CF.csv", 4, 1, crosswalk, 'mean');


%% BENEFIT (state-level -> MSA) 
T3 = readtable("benefits_state_DC_fips.xlsx");
T3 = T3(1:end-1, [15,2]);   
state_fips = T3.state_fips;  
binary     = T3.binary;      

C = innerjoin(C_base, T3(:,{'state_fips','binary'}), ...
              'LeftKeys','state_fips_2', ...
              'RightKeys','state_fips');

% aggregate to cities: 1 if any county in the city is in a state with tax incentives 
[G, M] = findgroups(C.MSACode);
val    = splitapply(@nanmax, C.binary, G);
msa_benefit = table(M, val, 'VariableNames', {'MSA','Value'});

% all the cities should be there, add zeros if they are missing
msa_benefit = outerjoin( ...
    table(all_msa,'VariableNames',{'MSA'}), ...
    msa_benefit, ...
    'Keys','MSA', 'MergeKeys',true, 'Type','left');

%msa_benefit.Value(isnan(msa_benefit.Value)) = 0;

%% BENEFIT (state-level -> COUNTY) 
% dummy county-level: 1 if the state of the county has an incentive, 0 otherwise 

state_benefit = table( double(T3.state_fips), double(T3.binary), ...
    'VariableNames', {'state_fips_2','benefit_flag'} );

% join with county_all via state_fips_2
county_all = outerjoin(county_all, state_benefit, ...
    'LeftKeys','state_fips_2', ...
    'RightKeys','state_fips_2', ...
    'MergeKeys', true, 'Type','left');

% make it numeric
county_all.benefit_flag = double(county_all.benefit_flag > 0);

state_abbrev = ["AL","AZ","AR","CA","CO","CT","DE","DC","FL","GA",...
    "ID","IL","IN","IA","KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT",...
    "NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD",...
    "TN","TX","UT","VT","VA","WA","WV","WI","WY"]';
state_fips_vec = [...
     1,  4,  5,  6,  8,  9, 10, 11, 12, 13, 16, 17, 18, 19, ...
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, ...
    36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50, 51, 53, 54, 55, 56]';


%% REAL ELECTRICITY PRICE & VOLATILITY (STATE -> COUNTY / MSA) 
Tvr = readtable("eia861_state_prices_real_2020_with_volatility.csv", ...
    'VariableNamingRule','preserve'); 

% take only the 2020
Year_vr = force_numeric(Tvr.year);
mask2020 = (Year_vr == 2020);
Tvr = Tvr(mask2020, :);

STATE_r = string(Tvr.state);
[tf_r, loc_r] = ismember(STATE_r, state_abbrev);

state_fips_r = state_fips_vec(loc_r(tf_r));

var_ind_all        = force_numeric(Tvr.var_ind_48(tf_r));

state_price_real = table(state_fips_r, ...
                         var_ind_all, ...                        
    'VariableNames', {'state_fips_2', ...
                      'elec_var_ind_all'});
% %% county-level 
 county_all = outerjoin(county_all, state_price_real, ...
                        'LeftKeys','state_fips_2', ...
                        'RightKeys','state_fips_2', ...
                        'MergeKeys',true, 'Type','left');

% %% city-level
C_price_real = innerjoin(C_base, state_price_real, ...
                          'LeftKeys','state_fips_2', ...
                          'RightKeys','state_fips_2');

[Gr_msa, M_msa] = findgroups(C_price_real.MSACode);

var_ind_all_msa     = splitapply(@nanmean, C_price_real.elec_var_ind_all,       Gr_msa);
msa_elec_var_ind_all  = table(M_msa, var_ind_all_msa, ...
    'VariableNames', {'MSA','Value'});


%% population
acsFile = "ACSDT5Y2023.B01003-Data_county.csv"; 
T = readtable(acsFile, "TextType","string");

% Drop the metadata row
T = T(T.GEO_ID ~= "Geography", :);

% Extract FIPS code 
T.GEO_ID = extractAfter(T.GEO_ID, strlength(T.GEO_ID) - 5);
T.STATEFP = extractBetween(T.GEO_ID, strlength(T.GEO_ID) - 4, strlength(T.GEO_ID) - 3);
T = T(~startsWith(T.STATEFP,"02"), :);
T = T(~startsWith(T.STATEFP,"15"), :);
T = T(~startsWith(T.STATEFP,"72"), :);
T = T(~startsWith(T.STATEFP,"60"), :);
T = T(~startsWith(T.STATEFP,"66"), :);
T = T(~startsWith(T.STATEFP,"69"), :);
T = T(~startsWith(T.STATEFP,"78"), :);
T.Properties.VariableNames{'GEO_ID'} = 'FIPS';
T.Properties.VariableNames{'B01003_001E'} = 'Pop';
T_pop = T(:, {'FIPS', 'Pop'});
writetable(T_pop,"pop_county.csv")

%% MSA-level population (from ACSDT5Y2023.B01003-Data.csv) ------------------

acsFile = "ACSDT5Y2023.B01003-Data.csv"; 
T = readtable(acsFile, "TextType","string");

% Drop the metadata row
T = T(T.GEO_ID ~= "Geography", :);

% Extract CBSA code (last 5 digits from GEO_ID like 310M700US10180)
T.msa_codes = extractAfter(T.GEO_ID, strlength(T.GEO_ID) - 5);
msa_list = readtable ("crosswalk_MSA_state.csv", "VariableNamingRule","preserve");
rows = ismember(T.msa_codes, string(msa_list.GEOID));
T = T(rows, :);

% Build output 
msa_pop = table(T.msa_codes, T.B01003_001E, 'VariableNames', {'MSA','Value'});
writetable(msa_pop,"pop_msa.csv")
%%  natural hazards events 
T6 = readtable("NRI_Table_counties.csv", 'VariableNamingRule','preserve');

filename = {'HRCN_EVNTS', 'ISTM_EVNTS', 'LNDS_EVNTS', 'LTNG_EVNTS', 'RFLD_EVNTS', ...
    'SWND_EVNTS', 'TRND_EVNTS', 'TSUN_EVNTS', 'VLCN_EVNTS',  'WFIR_EVNTS', 'WNTW_EVNTS',...
    'AVLN_EVNTS', 'CFLD_EVNTS', 'CWAV_EVNTS' , 'DRGT_EVNTS', 'ERQK_EVNTS', 'HAIL_EVNTS','HWAV_EVNTS'};

T6.TOT_NRI = nansum(T6{:, filename}, 2);
T_nri = T6(:,{'STCOFIPS','AREA','TOT_NRI'});
FIPS_ev = pad(string(T_nri{:,1}), 5, 'left','0');
val_ev  = force_numeric(T_nri{:,3});
AREA_ev = force_numeric(T_nri{:,2});
county_all = add_var_county(county_all, FIPS_ev, val_ev, 'nri_events');
county_all = add_var_county(county_all, FIPS_ev, AREA_ev, 'area');

county_all.nri_events_per_area = county_all.nri_events ./ county_all.area;
county_all.nri_events_per_area(county_all.area<= 0) = NaN;  

% MSA-level: (sum events) / (sum area)
J_evA = outerjoin( ...
    table(county_all.FIPS, county_all.nri_events, county_all.area, ...
          'VariableNames', {'FIPS','Events','Area'}), ...
    crosswalk(:,{'CountyCode','MSACode'}), ...
    'LeftKeys','FIPS', ...
    'RightKeys','CountyCode');

mask_msa = ~ismissing(J_evA.MSACode);
J_evA = J_evA(mask_msa, :);

[G_evA, msa_evA] = findgroups(J_evA.MSACode);

events_sum_msa = splitapply(@nansum, J_evA.Events, G_evA);
area_sum_msa   = splitapply(@nansum, J_evA.Area,   G_evA);

events_per_area_msa = events_sum_msa ./ area_sum_msa;
events_per_area_msa(area_sum_msa < 0) = NaN;

msa_events_per_area = table(string(msa_evA), events_per_area_msa, ...
    'VariableNames', {'MSA','Value'});
%% EVI_county_level.csv 
T8 = readtable("EVI_county_level.csv", 'VariableNamingRule','preserve');
EVI_county = T8(:,[1,4]);  

FIPS_evi = pad(string(T8{:,1}), 5, 'left','0');
val_evi  = force_numeric(T8{:,4});
county_all = add_var_county(county_all, FIPS_evi, val_evi, 'evi');

msa_EVI = county_to_msa_by_fips( ...
    "EVI_county_level.csv", 4, 1, crosswalk, 'mean');

%%  LATENCY / BROADBAND QUALITY 
%% read me I renamed this file "data-iwybr" from Purdue link
T10 = readtable("latency_county.csv", 'VariableNamingRule','preserve');
BQS_county = T10(:,[1,8]);  

FIPS_bqs = pad(string(T10{:,1}), 5, 'left','0');
val_bqs  = force_numeric(T10{:,8});
county_all = add_var_county(county_all, FIPS_bqs, val_bqs, 'bqs');

msa_BQS = county_to_msa_by_fips( ...
    "latency_county_with_msacode.csv", 8, 1, crosswalk, 'mean');

%% ENERGY EXPENDITURE
T14 = readtable("energy_consumption_expenditure_business_as_usual_county.csv", 'VariableNamingRule','preserve');
T14 = T14(T14.Year == 2024, :);
T14 = T14(T14.Sector == "industrial", :);

valueVar = "Expenditure US Dollars";

% sum elec+ng
Gname = groupsummary(T14, {'County Name','State Name'}, "sum", valueVar);

Gname.Properties.VariableNames{'sum_' + valueVar} = 'energy_exp';
Gname.CountyKey = lower(strtrim(string(Gname.("County Name"))));
Gname.StateKey  = lower(strtrim(string(Gname.("State Name"))));

cw_cs.CountyKey = lower(strtrim(string(cw_cs.Name_Clean)));
cw_cs.StateKey  = lower(strtrim(string(cw_cs.state_name)));

if ~ismember("FIPS5", string(cw_cs.Properties.VariableNames))
    cw_cs.FIPS5 = pad(string(cw_cs.FIPS), 5, 'left', '0');
end

alias = table( ...
    ["iowa"; "maryland"; "maryland"; "maryland"], ...
    ["o brien"; "prince george s"; "queen anne s"; "st. mary s"], ...
    ["o'brien"; "prince george's"; "queen anne's"; "st. mary's"], ...
    'VariableNames', {'StateKey','From','To'} );

for i = 1:height(alias)
    hit = (Gname.StateKey == alias.StateKey(i)) & (Gname.CountyKey == alias.From(i));
    Gname.CountyKey(hit) = alias.To(i);
end

dropStates = ["Alaska","Hawaii"];
Gname = Gname(~ismember(string(Gname.("State Name")), dropStates), :);

J = innerjoin(Gname, cw_cs(:, {'CountyKey','StateKey','FIPS5'}), ...
    'LeftKeys',  {'CountyKey','StateKey'}, ...
    'RightKeys', {'CountyKey','StateKey'}); FIPS_eexp = J.FIPS5;
val_eexp  = force_numeric(J.energy_exp);

county_all = add_var_county(county_all, FIPS_eexp, val_eexp, 'energy_exp');
Jleft = outerjoin(Gname, cw_cs(:, {'CountyKey','StateKey','FIPS5'}), ...
    'LeftKeys',  {'CountyKey','StateKey'}, ...
    'RightKeys', {'CountyKey','StateKey'}, ...
    'Type','left', 'MergeKeys', true);

cw = crosswalk;
JM = innerjoin( ...
    J(:, {'FIPS5','energy_exp'}), ...
    cw(:, {'CountyCode','MSA Code','MSA Title'}), ...
    'LeftKeys','FIPS5', ...
    'RightKeys','CountyCode' ...
);
msa_energy_exp = groupsummary( ...
    JM, ...
    {'MSA Code','MSA Title'}, ...
    'sum', ...
    'energy_exp' ...
);

msa_energy_exp.Properties.VariableNames("sum_energy_exp") = "energy_exp_msa";
msa_energy_exp = msa_energy_exp(:,[1,4]);


%% RETIRED COAL
T_rc = readtable("county_retired_coal_indicator.csv", 'VariableNamingRule','preserve');
FIPS_rc = pad(string(T_rc{:,1}), 5, 'left','0');
val_rc  = force_numeric(T_rc{:,2});
county_all = add_var_county(county_all, FIPS_rc, val_rc, 'retired_coal');

% msa_ret_coal = county_to_msa_by_fips( ...
%     "county_retired_coal_indicator.csv", 2, 1, crosswalk, 'sum');
msa_ret_coal = readtable("MSA_Total_Retired_Generators.csv", ...
    'VariableNamingRule','preserve');

% GEOID -> string 
g = msa_ret_coal.GEOID;
g = string(g);
g = strtrim(g);

% if GEOID ends with a letter, remove 
hasLetterSuffix = ~ismissing(g) & ~cellfun(@isempty, regexp(cellstr(g), '[A-Za-z]$', 'once'));
g(hasLetterSuffix) = extractBefore(g(hasLetterSuffix), strlength(g(hasLetterSuffix)));

% remove the 0 at the end
hasTrailingZero = endsWith(g, "0");
g(hasTrailingZero) = extractBefore(g(hasTrailingZero), strlength(g(hasTrailingZero)));
g_num = regexp(g, '\d+', 'match', 'once');
g_num = string(g_num);

% Pad 4 digit + C
g_num = pad(g_num, 4, 'left', '0');
msa_ret_coal.MSA_Code = "C" + g_num;

%adjust 2 cities manually
msa_ret_coal.MSA_Code(string(msa_ret_coal.GEOID) == "25775") = "C257A";
msa_ret_coal.MSA_Code(string(msa_ret_coal.GEOID) == "36837") = "C368A";

%rename
msa_ret_coal = renamevars(msa_ret_coal, "Number_Ret_Gen", "msa_ret_coal");
msa_ret_coal = renamevars(msa_ret_coal, "MSA_Code", "MSA");
%add_var MSA + 1 value
msa_ret_coal.MSA = strtrim(string(msa_ret_coal.MSA));
msa_ret_coal = msa_ret_coal(:, {'MSA','msa_ret_coal'});


%% IT MAINTENANCE
T_it = readtable("it_maintenance_county_wide_equal_alloc.csv", ...
    'VariableNamingRule','preserve');

FIPS_it = pad(string(T_it.county_fips), 5, 'left', '0');

emp_11_3021 = force_numeric(T_it.emp_11_3021);
emp_15_1231 = force_numeric(T_it.emp_15_1231);
emp_15_1232 = force_numeric(T_it.emp_15_1232);
emp_15_1241 = force_numeric(T_it.emp_15_1241);
emp_15_1244 = force_numeric(T_it.emp_15_1244);

emp_it_total = emp_11_3021 + emp_15_1231 + emp_15_1232 + emp_15_1241 + emp_15_1244;

county_all = add_var_county(county_all, FIPS_it, emp_it_total, 'emp_it_total');

J_it_msa = outerjoin( ...
    table(FIPS_it, emp_it_total, ...
          'VariableNames', {'FIPS','emp_it_total'}), ...
    crosswalk(:,{'CountyCode','MSACode'}), ...
    'LeftKeys','FIPS', ...
    'RightKeys','CountyCode');

J_it_msa.MSA = string(J_it_msa.MSACode);

mask_msa = ~ismissing(J_it_msa.MSA);
J_it_msa = J_it_msa(mask_msa, :);

[G_it, msa_it] = findgroups(J_it_msa.MSA);

sum_it_total = splitapply(@nansum, J_it_msa.emp_it_total, G_it);

msa_it_total   = table(msa_it, sum_it_total, 'VariableNames', {'MSA','Value'});
%% new capacity 
county_name_cap  = readtable("County_Total_Nameplate_Capacity.csv", ...
    'VariableNamingRule','preserve'); 

% county-level
FIPS_cap   = pad(string(county_name_cap{:,1}), 5, 'left','0');  % column 1 = FIPS
cap_val = force_numeric(county_name_cap{:,2});               
cap_val(~isfinite(cap_val) | cap_val < 0) = NaN;
county_all = add_var_county(county_all, FIPS_cap, cap_val, 'tot_cap');



msa_name_cap = readtable("MSA_Total_Nameplate_Capacity.csv", ...
    'VariableNamingRule','preserve');

% GEOID -> string 
g = msa_name_cap.GEOID;
g = string(g);
g = strtrim(g);

% if GEOID ends with a letter, remove 
hasLetterSuffix = ~ismissing(g) & ~cellfun(@isempty, regexp(cellstr(g), '[A-Za-z]$', 'once'));
g(hasLetterSuffix) = extractBefore(g(hasLetterSuffix), strlength(g(hasLetterSuffix)));

% remove the 0 at the end
hasTrailingZero = endsWith(g, "0");
g(hasTrailingZero) = extractBefore(g(hasTrailingZero), strlength(g(hasTrailingZero)));
g_num = regexp(g, '\d+', 'match', 'once');
g_num = string(g_num);

% Pad 4 digit + C
g_num = pad(g_num, 4, 'left', '0');
msa_name_cap.MSA_Code = "C" + g_num;

%adjust 2 cities manually
msa_name_cap.MSA_Code(string(msa_name_cap.GEOID) == "25775") = "C257A";
msa_name_cap.MSA_Code(string(msa_name_cap.GEOID) == "36837") = "C368A";

% match with crosswalk
cw_msa = string(crosswalk.MSACode);
missingMatch = ~ismember(msa_name_cap.MSA_Code, cw_msa);

if any(missingMatch)
    fprintf("[WARN] %d righe in msa_name_cap non matchano crosswalk.MSACode\n", sum(missingMatch));
    disp(msa_name_cap(missingMatch, {'GEOID','MSA_Code'}));
end
%rename
msa_name_cap = renamevars(msa_name_cap, "Nameplate Capacity (MW)", "cap_total");
msa_name_cap = renamevars(msa_name_cap, "MSA_Code", "MSA");
%add_var MSA + 1 value
msa_name_cap.MSA = strtrim(string(msa_name_cap.MSA));
msa_name_cap = msa_name_cap(:, {'MSA','cap_total'});
%% building all table
msa_all = table(unique(string(crosswalk.MSACode)), ...
                'VariableNames', {'MSA'});

msa_all = add_var(msa_all, msa_name_cap,         'cap_total');
msa_all = add_var(msa_all, msa_water,            'water_sum');
msa_all = add_var(msa_all, msa_benefit,          'benefit_flag');
msa_all = add_var(msa_all, msa_events_per_area, 'events_per_area');
msa_all = add_var(msa_all, msa_EVI,              'evi_mean');
msa_all = add_var(msa_all, msa_BQS,              'bqs_mean');
msa_all = add_var(msa_all, msa_ret_coal,         'msa_ret_coal');
msa_all = add_var(msa_all, msa_it_total,         'emp_it_total');
msa_all = add_var(msa_all, msa_elec_var_ind_all, 'elec_var_ind_all');
msa_all = add_var(msa_all, msa_energy_exp,  'energy_exp_msa');


sum_vars = ["water_sum","events_per_area"];
for v = sum_vars
    if ismember(v, msa_all.Properties.VariableNames)
        msa_all.(v) = fillmissing(msa_all.(v), 'constant', 0);
    end
end

writetable(msa_all, 'msa_features_all_def_eia_noprox_ret.csv');

%%  COUNTY-LEVEL 
county_all.FIPS = string(county_all.FIPS);
county_all = sortrows(county_all, 'FIPS');
% remove county_all.nri_events, county_all.area
county_all = county_all(:,[1 3:5 8:end]);
%writetable(county_all, 'county_features_all_nh_area_eia_noprox.csv');

%%%%%%%%%%%%% FUNCTIONS %%%%%%%%%%%%%%%%%%%%%
function msa_tbl = county_to_msa_by_fips(file, valCol, fipsCol, crosswalk, method)
    T = readtable(file, 'VariableNamingRule','preserve');

    FIPS = string(T{:, fipsCol});
    FIPS = pad(FIPS, 5, 'left', '0');

    Var = T{:, valCol};
    if ~isnumeric(Var)
        Var = string(Var);
        Var = regexprep(Var, '[,\s]', '');
        Var = str2double(Var);
    end

    n_tot  = numel(Var);
    n_nan  = sum(isnan(Var));
    n_zero = sum(Var == 0 & ~isnan(Var));
    fprintf('\n[DIAG %s] raw county-level: n=%d, NaN=%d, zeros=%d\n', ...
            file, n_tot, n_nan, n_zero);

    J = outerjoin( ...
        table(FIPS, Var), ...
        crosswalk(:,{'CountyCode','MSACode'}), ...
        'LeftKeys','FIPS', ...
        'RightKeys','CountyCode');

    unmatched   = ismissing(J.MSACode);
    n_unmatched = sum(unmatched);
    n_nan_after = sum(isnan(J.Var));

    nan_rows = isnan(J.Var) & ~unmatched;
    if any(nan_rows)
        ncols = width(J);
        disp(J(nan_rows, 1:min(5, ncols)));
    end

    [G, msa] = findgroups(J.MSACode);

    switch lower(method)
        case 'sum'
            aggfun = @nansum;
        case 'mean'
            aggfun = @nanmean;
        otherwise
            error('error', method);
    end

    val = splitapply(aggfun, J.Var, G);
    msa_tbl = table(msa, val, 'VariableNames', {'MSA','Value'});

    rows_bad = unmatched & ~isnan(J.Var);
 
    if any(rows_bad)
        ncols = width(J);
        disp(J(rows_bad, 1:min(5, ncols)));
    end
end

function x = force_numeric(x)
    if isnumeric(x), return; end

    x = string(x);
    x = strtrim(x);

    % Common missing tokens
    miss = (x=="" | lower(x)=="na" | lower(x)=="n/a" | lower(x)=="nan" | x=="-");
    x(miss) = "NaN";

    % Percent flag (we'll convert % to fraction)
    isPct = contains(x, "%");
    x = erase(x, "%");

    % Parentheses for negatives: (123) -> -123
    isPar = startsWith(x,"(") & endsWith(x,")");
    x(isPar) = "-" + extractBetween(x(isPar), 2, strlength(x(isPar))-1);

    % Remove currency symbols and spaces
    x = regexprep(x, "[\s\$\€\£]", "");

    % Normalize decimal separator:
    % - If only comma present => comma is decimal
    % - If both comma and dot present => the LAST one is decimal, the other is thousands
    for k = 1:numel(x)
        s = x(k);
        if s == "NaN", continue; end

        hasC = contains(s, ",");
        hasD = contains(s, ".");

        if hasC && ~hasD
           
            s = replace(s, ",", ".");
        elseif hasC && hasD
            lastC = find(char(s)==',', 1, 'last');
            lastD = find(char(s)=='.', 1, 'last');

            if lastC > lastD
                s = erase(s, ".");      
                s = replace(s, ",", "."); 
            else
                s = erase(s, ",");      
            end
        end

        s = regexprep(s, "[^0-9eE\+\-\.]", "");
        x(k) = s;
    end

    x = str2double(x);

    % Convert percent to fraction (12.3% -> 0.123)
    x(isPct) = x(isPct) ./ 100;
end

function B = add_var(B, T, newname)
    vn = T.Properties.VariableNames;
    if ismember('MSA Code', vn)
        T = renamevars(T, 'MSA Code', 'MSA');
    elseif ismember('MSA_Code', vn)
        T = renamevars(T, 'MSA_Code', 'MSA');
    end

    if ~ismember('MSA', T.Properties.VariableNames)
        error('add_var: table T does not have "MSA" or "MSA Code".');
    end

    vn = T.Properties.VariableNames;
    vn_val = setdiff(vn, {'MSA'});
    if numel(vn_val) ~= 1
        error('dd_var: table T does not have "MSA" or "MSA Code');
    end

    T = renamevars(T, vn_val{1}, newname);

    B.MSA = string(B.MSA);
    T.MSA = string(T.MSA);
    B = outerjoin(B, T, 'Keys','MSA', 'MergeKeys',true, 'Type','left');
end

function B = add_var_county(B, FIPS, Var, newname)
    F = string(FIPS);
    F = pad(F, 5, 'left', '0');

    if ~isnumeric(Var)
        Var = force_numeric(Var);
    end

    vn = string({'FIPS', newname});

    T = table(F, Var, 'VariableNames', vn);

    B.FIPS = string(B.FIPS);
    T.FIPS = string(T.FIPS);

    B = outerjoin(B, T, 'Keys','FIPS', 'MergeKeys',true, 'Type','left');
end
