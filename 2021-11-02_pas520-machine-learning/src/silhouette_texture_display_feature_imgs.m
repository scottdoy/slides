%% Silhouette Texture Feature Display
% Script to save texture features from the images
% 2017-02-26 SD

close all; clear; clc;
format compact; warning off;

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
    fprintf(1, 'Adding proj_dir to path\n');
    addpath(genpath(fullfile(proj_dir, 'scripts')));
    addpath(genpath(fullfile(proj_dir, 'module')));
end

%% Organize paths
feat_dir = fullfile(data_dir, 'Features');

% Get the list of extracted features
feat_list = dir(fullfile(feat_dir, '*_rgb_features.mat'));
nfeats = length(feat_list);

% Init for debugging
iimg = 1;

% Resize image by this amount
RES_MPP = 2;

% Use these features for display
% use_feats = [1,2,3,4,16,17,18,19,31,32,33,34,46,47,48,49,...
% 126,127,129,141,142,144,156,157,159,171,172,174,184,...
% 251, 252, 254,266,267];
use_feats = [47, 49, 184, 159, 171, 167];

%% Split features off the large file

for ifeat = [17]%29%1:16
    
    %% Set up filename and paths
    feat_path = fullfile(feat_dir, feat_list(ifeat).name);
    [~, base_name, ~] = fileparts(feat_path);
    base_name = strsplit(base_name, '_');
    base_name = base_name{1};
    fprintf(1, 'Processing %s\n', base_name);

    geom_path = fullfile(feat_dir, [base_name '_geometry.mat']);
    
    %% Check to make sure images exist and are compatible
    
    %% Load features
    
    % Set up file handler
    feats = matfile(feat_path);
    feats_rgb_names = feats.feats_rgb_names;
    
    for this_feat = use_feats
        
        featsave = fullfile(feat_dir, 'texture_imgs', strrep(feat_list(ifeat).name, 'rgb_features.mat', ['Feature-' num2str(this_feat) '-' strrep(feats_rgb_names{this_feat}, ':', '') '.png']));
        
        if ~exist(featsave, 'file')
            
            % Load the slice of features we want
            feats_loaded = feats.feats_rgb(:,:,this_feat);
            
            imwrite(uint8(feats_loaded./max(max(feats_loaded)).*255), featsave);
            
            clear feats_loaded
        end
    end
end