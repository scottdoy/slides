% Script to generate patches of tumor / not-tumor from original images.
% Refer to the Silhouette data spreadsheet to identify samples that should
% be left out (those where the original image is missing, misaligned, or
% not the same size / resolution as the marked and unmarked images).
% We also do some checking here to make sure the images are the same sizes.
% This script should be run after `silhouette_analysis_getGeometry.m`.
%
% 2017-03-15 SD - Modified to take in multiple classes
% 2017-03-13 SD (Modified due to low performance on Keras)
% 2017-01-29 SD


% Set up workspace
format compact;
close all;
clear;
clc;
% warning('off');

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

PATCH_SIZE  = 128; % AlexNet Size
NUM_PATCHES = 2000; % Max number of patches per image, per class

% Parameters for train and test / val split
% (Suggested by Andrew Ng, not set in stone)
train_pct = 0.7;
test_pct = 0.3;

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
RESIZE_RATIO = 0.25;

% List of label images
label_list = dir(fullfile(label_dir, '*.png'));

% Init for debugging
iimg = 1;

% Train/Test/Val Directories
train_dir = fullfile(data_dir, 'patches', ['training_multiclass_mpp-' num2str(1 / (RESIZE_RATIO / .25)) '_px-' num2str(PATCH_SIZE)]);
val_dir = fullfile(data_dir, 'patches', ['validation_multiclass_mpp-' num2str(1 / (RESIZE_RATIO / .25)) '_px-' num2str(PATCH_SIZE)]);
test_dir = fullfile(data_dir, 'patches', ['testing_multiclass_mpp-' num2str(1 / (RESIZE_RATIO / .25)) '_px-' num2str(PATCH_SIZE)]);

% Folder Creation / Destruction
if(~exist(train_dir, 'dir'))
    mkdir(train_dir);
else
    fprintf(1, 'WARNING: training directory exists!\n');
    %     rmdir(train_dir);
    %     mkdir(train_dir);
end

if(~exist(test_dir, 'dir'))
    mkdir(test_dir);
else
    fprintf(1, 'WARNING: testing directory exists!\n');
    %     rmdir(test_dir);
    %     mkdir(test_dir);
end

if(~exist(val_dir, 'dir'))
    mkdir(val_dir);
else
    fprintf(1, 'WARNING: validation directory exists!\n');
    %     rmdir(val_dir);
    %     mkdir(val_dir);
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
    imgLabel = imread(label_path);
    
    % Load original image
    imgOrig = imread(orig_path);
    
    % Resize images if desired
    if(RESIZE_RATIO ~= 1)
        fprintf(1, '\tResizing again...\n');
        imgOrig = imresize(imgOrig, RESIZE_RATIO);
        imgLabel = imresize(imgLabel, RESIZE_RATIO, 'nearest');
    end
    
    [h, w, d] = size(imgLabel);
    
    % Get a list of all the possible indices that we can try
    % (pad the arrays to remove possible edge points)
    padding = padarray(true(h-PATCH_SIZE*2,w-PATCH_SIZE*2), [PATCH_SIZE, PATCH_SIZE], 0);
    
    fprintf(1, '\tStarting Class Extraction\n');
    
    % Cycle through the labeled classes
    for iclass = 1:length(class_code)
        fprintf(1, '\t\tClass %s\n', class_code(iclass).ClassName);
        
        if(~strcmp(class_code(iclass).ClassName, 'Avoid'))
            % Create a binary mask from the label image
            imgMask = imgLabel(:,:,1) == class_code(iclass).RGB(1) & ...
                imgLabel(:,:,2) == class_code(iclass).RGB(2) & ...
                imgLabel(:,:,3) == class_code(iclass).RGB(3);
            if(sum(imgMask(:)) ~= 0)
                % Erode image to avoid edge cases
                %imgMask = imerode(imgMask, strel('disk', PATCH_SIZE/4));
                
                % Find the set of eligible patches by randomly sampling the
                % imgMask, and deleting patches as you go
                foundPatchesIdx = [];
                
                while numel(foundPatchesIdx) < NUM_PATCHES
                    posIdx = find(imgMask .* padding);
                    
                    
                    % Assume the image mask is nonzero
                    if(numel(posIdx) <= 0)
                        break;
                    end
                    
                    % Grab a random index index (is datasample best thing to use?)
                    posIdx = datasample(posIdx, 1, 'Replace', false);
                    
                    foundPatchesIdx = [foundPatchesIdx, posIdx];
                    [iRow, iCol] = ind2sub([h,w], posIdx);
                    imgMask(iRow-floor(PATCH_SIZE/10):iRow+ceil(PATCH_SIZE/10)-1, iCol-floor(PATCH_SIZE/10):iCol+ceil(PATCH_SIZE/10)-1) = 0;
                end
                
                nPatches = numel(foundPatchesIdx);
                
                % Shuffle found patches and pull out training / testing
                foundPatchesIdx = foundPatchesIdx(randperm(nPatches));
                
                nTraining = ceil(nPatches .* train_pct);
                nTesting = floor(nPatches .* test_pct);
                
                for iPatch = 1:nPatches
                    
                    imgPatch = pull_patches(foundPatchesIdx(iPatch), imgOrig, PATCH_SIZE);
                    [iRow, iCol] = ind2sub([h,w], foundPatchesIdx(iPatch));
                    
                    if(iPatch <= nTraining)
                        save_dir = train_dir;
                    else
                        save_dir = test_dir;
                    end
                    
                    if(~exist(fullfile(save_dir, class_code(iclass).ClassName), 'dir'))
                        mkdir(fullfile(save_dir, class_code(iclass).ClassName));
                    end
                    
                    patch_path = fullfile(save_dir, class_code(iclass).ClassName, sprintf('%s-R%d-C%d-%s.png', base_name, iRow, iCol, class_code(iclass).ClassName));
                    imwrite(squeeze(imgPatch), patch_path);
                end
            end
        end
    end
end % Continue cycling images

