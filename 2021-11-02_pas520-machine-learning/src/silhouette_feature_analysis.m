%% Silhouette Examples

close all; clear;
addpath(genpath('/media/scottdoy/Vault/projects/base_matlab'));

%% Load feature sets
% img_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'occ_quant_risk', 'figs', 'silhouettes');
img_dir = fullfile('/', 'home', 'scottdoy', 'projects', 'occ_quant_risk', 'figs', 'silhouettes');
feat_dir = fullfile(img_dir, 'features');

feat_list = dir(fullfile(feat_dir, '*.mat'));

img_nums = [1:9, 11:26]';
img_scales = [500, 200, 500, 200, 1000, 500, 1000, 500, 500, 500, 200, 200, 200, 1000,...
    200, 200, 500, 500, 500, 200, 500, 500, 500, 500, 500]';
img_labels = [0	1   1	1	0	0	0	0	0	0	1	1	1	0	1	0	0	1	0	0	0	1	1	0	0]';

% Identify images to exclude, if any
remove_me = [20];
for ii = 1:length(remove_me)
    img_nums(remove_me(ii)) = [];
    img_scales(remove_me(ii)) = [];
    img_labels(remove_me(ii)) = [];
end

total_features = [];
for ifeat = 1:length(feat_list)
    load(fullfile(feat_dir, feat_list(ifeat).name));
    total_features = [total_features; img_features, cnfs];
    
    % Display
    figure;
    imshow(img_bin); hold on;
    scatter(sat_Y, sat_X, '.r');
    scatter(tum_Y, tum_X, '.g');
    
    % Calculate the distance between all satellites
    neighbor_dist = squareform(pdist([sat_X, sat_Y]));
    
    % For each sattelite, calculate the minimum distance to the tumor
    % boundaries
    sat_dist = [];
    for isat = 1:length(sat_X)
        this_sat_dist = sqrt((tum_X - sat_X(isat)).^2 + (tum_Y - sat_Y(isat)).^2);
        [this_sat_dist, itum] = min(this_sat_dist);
        sat_dist = [sat_dist; this_sat_dist];
        
        % Display
        line([sat_Y(isat); tum_Y(itum)], [sat_X(isat); tum_X(itum)], 'Color', 'b');
    end
    
    [~,isat] = max(sat_dist);
    
    % Display
    max_sat_dist = sqrt((tum_X - sat_X(isat)).^2 + (tum_Y - sat_Y(isat)).^2);
    [this_sat_dist, itum] = min(max_sat_dist);
    
    line([sat_Y(isat); tum_Y(itum)], [sat_X(isat); tum_X(itum)], 'Color', 'r');
    
    print(gcf, [img_dir '/graphs/' num2str(img_nums(ifeat)) '_map.png'], '-dpng');
    close all;
    clear cnfs img_bin sat_bin tum_bin
    clear sat_X sat_Y tum_X tum_Y
    clear img_features
end
img_features = total_features;
clear total_features

%%


%% Feature Analysis

% Clear out the NaNs
nimgs = size(img_features, 1);
clear_row = [];
for ii = 1:nimgs
    if(any(isnan(img_features(ii,:))))
        clear_row = [clear_row; ii];
    end
end
img_features(clear_row, :) = [];
img_nums(clear_row) = [];
img_scales(clear_row) = [];
img_labels(clear_row) = [];

% Whitening
X = (img_features - repmat(mean(img_features), size(img_features,1), 1)) ./ repmat(var(img_features), size(img_features,1), 1);



%% MRMR Feature Selection
% selfeats = mrmr_mid_d(X, img_labels, 10);
sf = mrmr_miq_d(X, img_labels, 10);

% %% Use sat_max_dist to discriminate clusters
% sat_label = sat_max_dist > 2500;
% 
% tp = sum(sat_label == 1 & img_labels == 1);
% fp = sum(sat_label == 1 & img_labels == 0);
% tn = sum(sat_label == 0 & img_labels == 0);
% fn = sum(sat_label == 0 & img_labels == 1);
% 
% acc = (tp + tn) / (tp + tn + fp + fn);
% ppv = (tp) / (tp + fp);
% npv = (tn) / (fn + tn);
% 
% 
% % Perfcurve
% [~,~,T,AUC,OPTROCPT] = perfcurve(img_labels, 1- sat_max_dist./max(sat_max_dist), 1);

%% Create Classifier

pos_idx = find(img_labels == 1);
neg_idx = find(img_labels == 0);

pos_idx = pos_idx(randperm(length(pos_idx)));
neg_idx = neg_idx(randperm(length(neg_idx)));



bayes_model = fitcnb(X([pos_idx(1:5);neg_idx(1:10)],sf), img_labels([pos_idx(1:5);neg_idx(1:10)]));

results = predict(bayes_model, X([pos_idx(6:end); neg_idx(11:end)],sf));



%% Visualization of Progression

% selfeats = [13, 18, 52];
% selfeats = [13, 37, 36];

for ii = 1:length(sf)
    for jj = ii:length(sf)
%         for kk = 1:52
%             selfeats = [ii, jj, kk];
            selfeats = [sf(ii), sf(jj)];
            
            %%
            figure;
            pos_idx = find(img_labels == 1);
            for idx = 1:length(pos_idx)
%                 scatter(img_features(pos_idx(idx), selfeats(1)), img_features(pos_idx(idx), selfeats(2)), img_features(pos_idx(idx), selfeats(3)), 'r', 'filled'); hold on;
%                 text(img_features(pos_idx(idx), selfeats(1)), img_features(pos_idx(idx), selfeats(2)), img_features(pos_idx(idx), selfeats(3)), num2str(pos_idx(idx))); hold on;
                scatter(img_features(pos_idx(idx), selfeats(1)), img_features(pos_idx(idx), selfeats(2)), 'r', 'filled'); hold on;
                text(img_features(pos_idx(idx), selfeats(1)), img_features(pos_idx(idx), selfeats(2)), num2str(pos_idx(idx))); hold on;

            end
            
            neg_idx = find(img_labels == 0);
            for idx = 1:length(neg_idx)
%                 scatter3(img_features(neg_idx(idx), selfeats(1)), img_features(neg_idx(idx), selfeats(2)), img_features(neg_idx(idx), selfeats(3)), 'b','filled'); hold on;
%                 text(img_features(neg_idx(idx), selfeats(1)), img_features(neg_idx(idx), selfeats(2)), img_features(neg_idx(idx), selfeats(3)), num2str(neg_idx(idx))); hold on;
                scatter(img_features(neg_idx(idx), selfeats(1)), img_features(neg_idx(idx), selfeats(2)), 'b','filled'); hold on;
                text(img_features(neg_idx(idx), selfeats(1)), img_features(neg_idx(idx), selfeats(2)), num2str(neg_idx(idx))); hold on;

            end
            
            % scatter3(img_features(img_labels==0, selfeats(1)), img_features(img_labels==0, selfeats(2)), img_features(img_labels==0, selfeats(3)), 'ob');
            xlabel(num2str(selfeats(1))); ylabel(num2str(selfeats(2))); %zlabel(num2str(selfeats(3)));
            %%
            
            % % Look at just the max_sat_dist feature
            % figure;
            % scatter(1:length(pos_idx), sat_max_dist(pos_idx), 'r'); hold on;
            % scatter(1:length(neg_idx), sat_max_dist(neg_idx), 'b');
            
            % Save 
%             print(gcf, fullfile(img_dir, 'plots', [num2str(ii) '-' num2str(jj) '-' num2str(kk) '_plot.png']), '-dpng');
            print(gcf, fullfile(img_dir, 'plots', [num2str(ii) '-' num2str(jj) '_sf_plot.png']), '-dpng');
            close all;
%         end
    end
end


%% Specific Features

selfeats = mrmr_mid_d(X, img_labels, 10);
% selfeats = [13, 11, 52];
figure;
pos_idx = find(img_labels == 1);
for idx = 1:length(pos_idx)
    scatter3(X(pos_idx(idx), selfeats(1)), X(pos_idx(idx), selfeats(2)), X(pos_idx(idx), selfeats(3)), 'r', 'filled'); hold on;
    text(X(pos_idx(idx), selfeats(1)), X(pos_idx(idx), selfeats(2)), X(pos_idx(idx), selfeats(3)), num2str(pos_idx(idx))); hold on;
end

neg_idx = find(img_labels == 0);
for idx = 1:length(neg_idx)
    scatter3(X(neg_idx(idx), selfeats(1)), X(neg_idx(idx), selfeats(2)), X(neg_idx(idx), selfeats(3)), 'b','filled'); hold on;
    text(X(neg_idx(idx), selfeats(1)), X(neg_idx(idx), selfeats(2)), X(neg_idx(idx), selfeats(3)), num2str(neg_idx(idx))); hold on;
end

xlabel(num2str(selfeats(1))); ylabel(num2str(selfeats(2))); zlabel(num2str(selfeats(3)));
