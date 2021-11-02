%% Silhouette Checking Script
% Script to check files being used in the silhouette project.
% This will cycle through the list of files it finds in the 'Marked',
% 'Unmarked', and 'Original' folders and report whether any of them are
% missing, and also whether any of them are mismatched. We are using the
% 'Unmarked' folder as the index.
% 
% 2017-01-20 SD

close all; clear; clc;
warning('off');

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
    addpath(genpath(proj_dir));
end

%% Set up folders
sil_dir = fullfile(data_dir, 'Unmarked');
mark_dir = fullfile(data_dir, 'Marked');
orig_dir = fullfile(data_dir, 'Original');
feat_dir = fullfile(data_dir, 'Features');

% List of silhouettes, marked images, and originals
sil_list = dir(fullfile(sil_dir, '*.jpg'));
mark_list = dir(fullfile(mark_dir, '*.jpg'));
orig_list = dir(fullfile(orig_dir, '*.jpg'));

%% Check Sils
% Start cycling through silhouette image list
for iimg = 1:length(sil_list)

    fprintf(1,'%s\n', sil_list(iimg).name);
    fprintf(1,'%s\n', repmat('=', 1, length(sil_list(iimg).name)));
    
    % Pathnames to sil and mark
    sil_path = fullfile(sil_dir, sil_list(iimg).name);
%     sil_img = imread(sil_path);
%     [sil_height, sil_width, sil_depth] = size(sil_img);
    sil_info = imfinfo(sil_path);
    sil_height = sil_info.Height;
    sil_width = sil_info.Width;
    
    % Get filename for saving
    [~, img_name, ~] = fileparts(sil_path);
    
    % Check marked image
    mark_path = fullfile(mark_dir, [img_name, 'MARKED.jpg']);
    if(~exist(mark_path, 'file'))
        fprintf(1,'\tThe Marked image is not found at:\t%s\n', mark_path);
    else
        mark_info = imfinfo(mark_path);
        mark_height = mark_info.Height;
        mark_width = mark_info.Width;
        
        if(mark_height ~= sil_height)
            fprintf(1,'\tThe marked image height is %d, silhouette height is %d\n', mark_height, sil_height);
        end
        
        if(mark_width ~= sil_width)
            fprintf(1,'\tThe marked image width is %d, silhouette width is %d\n', mark_width, sil_width);
        end
    end
    
    % Check for original image
    orig_path = fullfile(orig_dir, [img_name, '.jpg']);
    if(~exist(orig_path, 'file'))
        fprintf(1,'\tThe Original image is not found at:\t%s\n', orig_path);
    else
        orig_info = imfinfo(orig_path);
        orig_height = orig_info.Height;
        orig_width = orig_info.Width;
        
        if(orig_height ~= sil_height)
            fprintf(1,'\tThe Original image height is %d, silhouette height is %d\n', orig_height, sil_height);
        end
        
        if(orig_width ~= sil_width)
            fprintf(1,'\tThe Original image width is %d, silhouette width is %d\n', orig_width, sil_width);
        end
    end
    
end
