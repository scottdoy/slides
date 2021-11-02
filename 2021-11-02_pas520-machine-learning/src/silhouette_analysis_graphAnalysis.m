%% Silhouette Analysis: graphAnalysis
% Script to analyze the graph features returned by `_createFeatures.m`
%
% 2017-01-12 SD

close all; clear;

%% Set up folders
if ispc
    base_dir = fullfile('E:', 'projects', 'base_matlab');
    proj_dir = fullfile('E:', 'projects', 'occ_quant_risk_score');
    data_dir = fullfile(proj_dir, 'data', 'Silhouettes');
elseif isunix
    base_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'base_matlab');
    proj_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'occ_quant_risk_score');
    data_dir = fullfile(proj_dir, 'data', 'Silhouettes');
else
    fprintf(1, 'Unknown filesystem, please edit folder setup!\n');
    return;
end

% Set up paths
pathCell = regexp(path, pathsep, 'split');
if ispc
  base_dir_onPath = any(strcmpi(base_dir, pathCell));
  proj_dir_onPath = any(strcmpi(proj_dir, pathCell));
else
  base_dir_onPath = any(strcmp(base_dir, pathCell));
  proj_dir_onPath = any(strcmp(proj_dir, pathCell));
end
if ~base_dir_onPath
    fprintf(1, 'Adding base_dir to path\n');
    addpath(genpath(base_dir));
end
if ~proj_dir_onPath
    fprintf(1, 'Adding base_dir to path\n');
    addpath(genpath(fullfile(proj_dir, 'scripts')));
    addpath(genpath(fullfile(proj_dir, 'module')));
end

% Features
orig_dir = fullfile(data_dir, 'Original');
feat_dir = fullfile(data_dir, 'Features');
% feat_list = dir(fullfile(feat_dir, '*.mat'));
feat_path = fullfile(feat_dir, 'all_features.csv');
% label_path = fullfile(data_dir, 'Silhouettes.csv');
names_path = fullfile(feat_dir, 'feature_names.xlsx');
output_dir = fullfile(proj_dir, 'figs', 'silhouette_feature_histograms');

% Create a list of image names to use
% base_names = cellfun(@(x) strrep(x, '_geometry.mat', ''), {feat_list.name}, 'UniformOutput', false)';

% Init for debugging
iimg = 1;

%% Data

% Get feature data
if(~exist(feat_path, 'file'))
    fprintf(1, 'Feature data does not exist! Run `_createFeatures` first!\n');
    return;
end
feature_data = readtable(feat_path);
feature_prog = feature_data.prog;
case_names = feature_data.names;
feature_poi = feature_data.poi;
feature_data = feature_data{:,4:end};

% Get feature names
if(~exist(names_path, 'file'))
    fprintf(1, 'Feature names file does not exist!\n');
    return;
end
feature_names = readtable(names_path);
feature_names = feature_names.FeatureTitle;

nimgs = size(feature_data,1);
nfeats = size(feature_data,2);

%% Merge Features from Same Patient

% Merge the data from the same patient
merged_names = cellfun(@(x) x(1:3), case_names', 'UniformOutput', false)';
unique_names = unique(merged_names);

merged_data = [];
merged_prog = [];
merged_poi = [];

for iname = 1:length(unique_names)
    name_idx = ismember(merged_names, unique_names{iname});
    
    % Take the average
%     merged_data = cat(1, merged_data, mean(feature_data(name_idx, :),1));
    % Take the first value
    merged_data = cat(1, merged_data, feature_data(find(name_idx, 1), :));
    
    merged_prog = cat(1, merged_prog, feature_prog(find(name_idx, 1), :));
    merged_poi = cat(1, merged_poi, feature_poi(find(name_idx, 1), :));
end

%% Progression Label
% First cycle through the data and generate histograms of each feature
% based on progression

for ifeat = [1,3, 40, 41]%1:nfeats
    save_hist = fullfile(output_dir, ['prog_feature_' num2str(ifeat) '_' feature_names{ifeat} '.png']);
    
    if(~exist(save_hist, 'file'))
        feat = merged_data(:,ifeat);
        pos = feat(merged_prog>0);
        neg = feat(merged_prog==0);
        
        dist_prog = makedist('Normal', 'mu', mean(pos), 'sigma', std(pos));
        dist_nonprog = makedist('Normal', 'mu', mean(neg), 'sigma', std(neg));
        
        x = linspace(min(feat), max(feat), 10000);
        
        pdf_prog = pdf(dist_prog, x);
        pdf_neg = pdf(dist_nonprog, x);
        
        figure;
        plot(x, pdf_prog, 'LineWidth', 3); hold on;
        plot(x, pdf_neg, 'LineWidth', 3);
        xlim([min(feat), max(feat)]);
        title(['Feature ' num2str(ifeat) ': ' feature_names{ifeat}], 'FontSize', 18);
        xlabel(['$x_{' num2str(ifeat) '}$'], 'Interpreter', 'latex', 'FontSize', 18); %Feature ' num2str(ifeat) ': ' feature_names{ifeat}], 'FontSize', 18);
        ylabel(['$p(\omega_{i} | x_{' num2str(ifeat) '})$'], 'Interpreter', 'latex', 'FontSize', 18);
        ax = gca;
        legend({'Progressor', 'Non-progressor'}, 'Location', 'best', 'FontSize', 18);
        print(save_hist, '-dpng');
        close all;
%         figure;
%         plot(x, pdf_prog); hold on;
%         plot(x, pdf_neg);
%         xlim([min(feat), max(feat)]);
%         title([num2str(ifeat) ': ' feature_names{ifeat}]);
%         export_fig(save_hist, '-m2', gcf);
%         close all;
    end
end

% Scatterplots
fselect = [20,34,40];
feats = merged_data(:,fselect);
pos = feats(merged_prog>0, :);
neg = feats(merged_prog==0, :);

figure;
scatter3(pos(:,1), pos(:,2), pos(:,3), 'r'); hold on;
scatter3(neg(:,1), neg(:,2), neg(:,3), 'k');
xlabel(feature_names{fselect(1)});
ylabel(feature_names{fselect(2)});
zlabel(feature_names{fselect(3)});

%% POI Label
% First cycle through the data and generate histograms of each feature
% based on progression

for ifeat = [1,2,3,4]%1:nfeats
    save_hist = fullfile(output_dir, ['poi_feature_' num2str(ifeat) '_' feature_names{ifeat} '.png']);
    
    if(~exist(save_hist, 'file'))
        feat = merged_data(:,ifeat);
        pos = feat(merged_poi==4);
        neg = feat(merged_poi==5);
        
        dist_prog = makedist('Normal', 'mu', mean(pos), 'sigma', std(pos));
        dist_nonprog = makedist('Normal', 'mu', mean(neg), 'sigma', std(neg));
        
        x = linspace(min(feat), max(feat), 10000);
        
        pdf_prog = pdf(dist_prog, x);
        pdf_neg = pdf(dist_nonprog, x);
        
        figure;
        plot(x, pdf_prog, 'LineWidth', 3); hold on;
        plot(x, pdf_neg, 'LineWidth', 3);
        xlim([min(feat), max(feat)]);
        title(['Feature ' num2str(ifeat) ': ' feature_names{ifeat}], 'FontSize', 18);
        xlabel(['$x_{' num2str(ifeat) '}$'], 'Interpreter', 'latex', 'FontSize', 18); %Feature ' num2str(ifeat) ': ' feature_names{ifeat}], 'FontSize', 18);
        ylabel(['$p(\omega_{i} | x_{' num2str(ifeat) '})$'], 'Interpreter', 'latex', 'FontSize', 18);
        ax = gca;
        legend({'WPOI 4', 'WPOI 5'}, 'Location', 'best', 'FontSize', 18);
        print(save_hist, '-dpng');
        close all;
    end
end

%% POI Scatterplots
fselect = [3,18,16];
feats = merged_data(:,fselect);
pos = feats(merged_poi==4, :);
neg = feats(merged_poi==5, :);

figure;
scatter3(pos(:,1), pos(:,2), pos(:,3), 50, 'filled'); hold on;
scatter3(neg(:,1), neg(:,2), neg(:,3), 50, 'filled');
xlabel(feature_names{fselect(1)});
ylabel(feature_names{fselect(2)});
zlabel(feature_names{fselect(3)});
legend('WPOI 4', 'WPOI 5', 'Location', 'best');
title('POI Labels');

save_scatter = fullfile(output_dir, ['poi_scatterplot_features_' num2str(fselect(1)) '-' num2str(fselect(2)) '-' num2str(fselect(3)) '.png']);
%export_fig(save_scatter, '-native', gcf);
export_fig(save_scatter, '-m2', gcf);

%% Print class means
pos = merged_data(merged_poi == 4, :);
neg = merged_data(merged_poi == 5, :);

fprintf(1, 'Feature\tWPOI 4 Mean\tWPOI 4 STD\tWPOI 5 Mean\tWPOI 5 STD\tMean Difference\n');
for ifeat = 1:nfeats
    fprintf(1, '%s\t%3.3f\t%3.3f\t%3.3f\t%3.3f\t%3.3f\n', feature_names{ifeat}, mean(pos(:,ifeat)), std(pos(:,ifeat)), mean(neg(:,ifeat)), std(neg(:,ifeat)), mean(pos(:,ifeat)) - mean(neg(:,ifeat)));
end

