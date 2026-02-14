%% this script preprocess the data for the population projections filtering from 2025 to 2050
% and organizing them by cities and scenarios

clear
clc
close all

crosswalk = readtable("C:/Users/Owner/Downloads/crosswalk_county_msa.xlsx", 'VariableNamingRule','preserve');

% normalize county/MSA codes as in the main script
crosswalk.CountyCode = pad(string(crosswalk.("County Code")), 5, 'left','0');
crosswalk.MSACode    = string(crosswalk.("MSA Code"));

% % drop Puerto Rico, Alaska and Hawaii
crosswalk = crosswalk(~startsWith(crosswalk.CountyCode,"72"), :);
crosswalk = crosswalk(~startsWith(crosswalk.CountyCode,"02"), :);
crosswalk = crosswalk(~startsWith(crosswalk.CountyCode,"15"), :);

%% Population projections 
% County-level SSP projections -> MSA-level sums, years <= 2050
Tssp = readtable("C:/Users/Owner/Downloads/SSP_asrc.csv", 'VariableNamingRule','preserve');

% keep only years up to 2050
Tssp = Tssp(Tssp.YEAR <= 2050, :);

% Build 5-digit county GEOID (same padding logic as the rest)
Tssp.GEOID = pad(string(Tssp.GEOID), 5, 'left', '0');
FIPS_ssp = Tssp.GEOID;

% Drop Puerto Rico (state FIPS 72)
mask_pr = startsWith(FIPS_ssp, "72");
Tssp    = Tssp(~mask_pr, :);
FIPS_ssp = FIPS_ssp(~mask_pr);

% Drop Alaska (state FIPS 02)
mask_pr = startsWith(FIPS_ssp, "02");
Tssp    = Tssp(~mask_pr, :);
FIPS_ssp = FIPS_ssp(~mask_pr);

% Drop Hawaii (state FIPS 15)
mask_pr = startsWith(FIPS_ssp, "15");
Tssp    = Tssp(~mask_pr, :);
FIPS_ssp = FIPS_ssp(~mask_pr);

% identify SSP scenario columns 
vn = Tssp.Properties.VariableNames;
ssp_vars = vn(contains(vn, "SSP"));

% force them to numeric 
for k = 1:numel(ssp_vars)
    Tssp.(ssp_vars{k}) = force_numeric(Tssp.(ssp_vars{k}));
end

% join with county–MSA crosswalk on 5-digit county FIPS
J_left = table(FIPS_ssp, Tssp.YEAR, 'VariableNames', {'FIPS','YEAR'});
for k = 1:numel(ssp_vars)
    vname = ssp_vars{k};
    J_left.(vname) = Tssp.(vname);
end

%% checks
% Data completness - No missing data across years/SSPs
hasMissing = any(any(ismissing(Tssp(:, ssp_vars))));
if hasMissing
    disp('Missing SSP data');
end

% No missing combinations
unique_FIPS_ssp = unique(FIPS_ssp);
Year_ssp = unique(Tssp.YEAR);
Sex_ssp = unique(Tssp.SEX);
Race_ssp = unique(Tssp.RACE);
Age_ssp = unique(Tssp.AGE);

expectedPerCounty = numel(Sex_ssp) * numel(Race_ssp) * numel(Age_ssp) * numel(Year_ssp);

[G_comb, GEOID_comb, ~, ~, ~, ~] = findgroups(Tssp.GEOID, Tssp.SEX, Tssp.RACE, Tssp.AGE, Tssp.YEAR);
counts_per_comb = splitapply(@numel, Tssp.GEOID, G_comb);

[G_county, county_u] = findgroups(Tssp.GEOID);
actualCombinations = splitapply(@sum, counts_per_comb, G_county);

missingCounties = county_u(actualCombinations < expectedPerCounty);

if ~isempty(missingCounties)    % Remove missing counties 
    mask_keep = ~ismember(Tssp.GEOID, missingCounties);
    Tssp = Tssp(mask_keep, :);
    FIPS_ssp = FIPS_ssp(mask_keep);
end

if height(Tssp) == expectedPerCounty*numel(unique(Tssp.GEOID))
    disp('Data is complete');
end

%% filter counties not in metro/micro sas
uniqueCountiesCrosswalk = unique(crosswalk.CountyCode); % Unique city counties
missingInSSP = ~ismember(uniqueCountiesCrosswalk, unique_FIPS_ssp);  % Missing counties from SSP
FIPS_present = uniqueCountiesCrosswalk(~missingInSSP); % SSP counties in crosswalk
J_left = J_left(ismember(J_left.FIPS, FIPS_present), :);    % Remove missing counties

idx = ismember(crosswalk.CountyCode, FIPS_present);     % Corresponding rows in crosswalk
msa_for_SSP = unique(crosswalk.MSACode(idx));   % Corresponding cities 

%join to crosswalk 
J_ssp = outerjoin( ...
    J_left, ...
    crosswalk(:, {'CountyCode','MSACode'}), ...
    'LeftKeys','FIPS', ...
    'RightKeys','CountyCode' );

% group by (MSA, YEAR) and sum each SSP scenario
[G_ssp, msa_u, year_u] = findgroups(J_ssp.MSACode, J_ssp.YEAR);
msa_pop_proj = table(string(msa_u), year_u, ...
    'VariableNames', {'MSA','YEAR'});
for k = 1:numel(ssp_vars)
    v = ssp_vars{k};
    val = splitapply(@sum, J_ssp.(v), G_ssp);
    msa_pop_proj.(v) = val;
end

validMSA = ismember(msa_pop_proj.MSA, msa_for_SSP); % Valid MSAs
msa_pop_proj = msa_pop_proj(validMSA, :);   % Keep only valid MSAs

% save
writetable(msa_pop_proj, "msa_population_projections_SSP_to2050.csv");

%% WIDE: one row for each MSA, columns for SSPs 

% start from a table with all the MSAs as rows
msa_pop_wide = table(unique(msa_pop_proj.MSA), ...
                     'VariableNames', {'MSA'});

for i = 1:numel(ssp_vars)
    v = ssp_vars{i};

    % take only MSA, YEAR and the corresponding SSP column
    Ttmp = msa_pop_proj(:, {'MSA','YEAR', v});

    % YEAR becomes a column 
    Ttmp = unstack(Ttmp, v, 'YEAR');  

    % rename the columns (SSP1_2020, SSP1_2025, ...)
    years    = sort(unique(msa_pop_proj.YEAR));      % [2020 2025 ... 2050]
    oldNames = Ttmp.Properties.VariableNames(2:end); % avoid title 'MSA'
    newNames = compose("%s_%d", v, years);           % "SSP1_2020", ...
    Ttmp = renamevars(Ttmp, oldNames, newNames);
    
    % merge 
    msa_pop_wide = outerjoin(msa_pop_wide, Ttmp, ...
                             'Keys','MSA', 'MergeKeys',true, 'Type','left');
end

% remove the 2020 columns
cols = msa_pop_wide.Properties.VariableNames;
mask2020 = endsWith(cols, "_2020");
msa_pop_wide = removevars(msa_pop_wide, cols(mask2020));

% save
writetable(msa_pop_wide, "msa_population_projections_SSP_wide_by_year.csv");

function x = force_numeric(x)
    if isnumeric(x), return; end
    x = string(x);
    x = regexprep(x,'[,\s]','');
    x = regexprep(x,'%$','');
    x = str2double(x);
end