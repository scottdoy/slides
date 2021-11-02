% Script to pull patches from the image uniformly
%
% 2017-03-20 SD 
% 2017-03-15 SD - Modified to take in multiple classes
% 2017-03-13 SD (Modified due to low performance on Keras)
% 2017-01-29 SD


% Set up workspace
format compact;
close all;
clear;
clc;
warning('off');

%% Set up folders

% Project head
if ispc
    base_dir = fullfile('E:', 'projects', 'base_matlab');
    proj_dir = fullfile('E:', 'projects', 'occ_quant_risk_score');
elseif isunix
    base_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'base_matlab');
    proj_dir = fullfile('/', 'media', 'scottdoy', 'Vault', 'projects', 'occ_quant_risk_score');
else
    fprintf(1, 'Unknown filesystem, please edit folder setup!\n');
    return;
end

% Add stuff to the path
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

% Data Sources
data_dir = fullfile(proj_dir, 'data', 'Silhouettes');
orig_dir = fullfile(data_dir, 'Original');
label_dir = fullfile(data_dir, 'Labels_Multiclass');


%% Set up initialization parameters

PATCH_SIZE  = 32; % AlexNet Size

% Whether to display figures
debugFlag = false;

% Label Codes for each class
class_code = struct('ClassName', {'Satellite_Tumor', 'Main_Tumor', 'Smooth_Keratin', 'Rough_Stroma', 'Lymphocytes', 'Smooth_Stroma', 'Mucosa', 'Whorls', 'Blood', 'Background', 'Avoid'}, ...
    'RGB', {[0,0,0], [0,0,255], [255,0,0], [0,255,0], [255,255,0], [255,0,255], [0,255,255], [128,0,0], [0,128,0], [100,100,100],[255,255,255]});

% Ratio to resize images (if desired)
% Conversion: Images are 1/RESIZE_RATIO times smaller
% Images should start at 0.25 microns per pixel, so:
%
% | RESIZE_RATIO | mpp  |
% |--------------|------|
% | 1.0          | 0.25 | (Disabled)
% | 0.5          | 0.5  |
% | 0.25         | 1.0  |
% | 0.125        | 2.0  |
% | 0.0625       | 4.0  |

% Set to 1 to disable
RESIZE_RATIO = 1;

% List of label images
label_list = dir(fullfile(label_dir, '*.png'));

% Init for debugging
iimg = 1;

% Save Directories
save_dir = fullfile(data_dir, 'patches', ['unsupervised_mpp-' num2str(1 / (RESIZE_RATIO / .25)) '_px-' num2str(PATCH_SIZE)]);

% Folder Creation / Destruction
if(~exist(save_dir, 'dir'))
    mkdir(save_dir);
else
    fprintf(1, 'WARNING: unsupervised directory exists!\n');
end

%% Begin processing images

for iimg = 2:4%1:length(label_list)
    % Load the geometry for the current image
    base_name = strrep(label_list(iimg).name, '.png', '');
    
    fprintf(1, 'Creating patches from %s\n', base_name);
    
    % Before we do anything, check that the sils and the original are the
    % same resolution
    label_path = fullfile(label_dir, [base_name '.png']);
    orig_path = fullfile(orig_dir, [base_name '.jpg']);
    
    % Check existence
    if(~exist(orig_path, 'file') || ~exist(label_path, 'file'))
        fprintf(1, '\tSkipping: Original or Silhouette file does not exist!\n');
        continue;
    end
    
    label_info = imfinfo(label_path);
    orig_info = imfinfo(orig_path);
    
    if(label_info.Width ~= orig_info.Width || label_info.Height ~= orig_info.Height)
        fprintf(1, '\tSkipping: Original and silhouette are not the same size.\n');
        continue;
    end
    
    % Check existence of geometry (should be redundant)
    label_path = fullfile(label_dir, label_list(iimg).name);
    
    if(~exist(label_path, 'file'))
        fprintf(1, '\tSkipping: File %s does not exist, check for the labeled image.\n', label_list(iimg).name);
        continue;
    end
    
    % Load the label
    fprintf(1, '\tLoading Label image...\n');
    
    % Load original image
    imgOrig = imread(orig_path);
    
    % Resize images if desired
    if(RESIZE_RATIO ~= 1)
        fprintf(1, '\tResizing again...\n');
        imgOrig = imresize(imgOrig, RESIZE_RATIO);
    end
    
    [h, w, d] = size(imgOrig);
    
    % Get a list of all the possible indices that we can try
    % (pad the arrays to remove possible edge points)
    padding = padarray(true(h-PATCH_SIZE*2,w-PATCH_SIZE*2), [PATCH_SIZE, PATCH_SIZE], 0);
    
    fprintf(1, '\tStarting Class Extraction\n');
    
    % Pull apart the image into equally-spaced, half-overlapping samples
    cols = ceil(PATCH_SIZE/2)+1:floor(PATCH_SIZE/4):w-ceil(PATCH_SIZE/2);
    rows = ceil(PATCH_SIZE/2)+1:floor(PATCH_SIZE/4):h-ceil(PATCH_SIZE/2);
    [R,C] = meshgrid(rows, cols);
    
    foundPatchesIdx = sub2ind([h,w], R(:), C(:));
    
    imgPatches = pull_patches(foundPatchesIdx, imgOrig, PATCH_SIZE);
    
    % Cycle through imgPatches
    for iPatch = 1:size(imgPatches,4)
        patch_path = fullfile(save_dir, sprintf('%s-unsup_patch-%d.png', base_name, iPatch));
        imwrite(squeeze(imgPatches(:,:,:,iPatch)), patch_path);
    end
    
    
end
    

