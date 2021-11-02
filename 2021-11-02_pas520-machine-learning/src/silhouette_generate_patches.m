% Script to generate patches of tumor / not-tumor from original images.
% Refer to the Silhouette data spreadsheet to identify samples that should
% be left out (those where the original image is missing, misaligned, or
% not the same size / resolution as the marked and unmarked images).
% We also do some checking here to make sure the images are the same sizes.
% This script should be run after `silhouette_analysis_getGeometry.m`.
% 
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
sil_dir = fullfile(data_dir, 'Unmarked');
mark_dir = fullfile(data_dir, 'Marked');
orig_dir = fullfile(data_dir, 'Original');
geometry_dir = fullfile(data_dir, 'Features');


%% Set up initialization parameters

PATCH_SIZE  = 224; % AlexNet Size
NUM_PATCHES = 5000; % Max number of patches per image, per class

% Parameters for train and test / val split
% (Suggested by Andrew Ng, not set in stone)
train_pct = 0.7;
test_pct = 0.3;

% Whether to display figures
debugFlag = false;

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

% List of features
geometry_list = dir(fullfile(geometry_dir, '*geometry.mat'));

% Init for debugging
iimg = 1;

% Train/Test/Val Directories
train_dir = fullfile(data_dir, 'patches', ['training_mpp-' num2str(1 / (RESIZE_RATIO / .25)) '_px-' num2str(PATCH_SIZE)]);
val_dir = fullfile(data_dir, 'patches', ['validation_mpp-' num2str(1 / (RESIZE_RATIO / .25)) '_px-' num2str(PATCH_SIZE)]);
test_dir = fullfile(data_dir, 'patches', ['testing_mpp-' num2str(1 / (RESIZE_RATIO / .25)) '_px-' num2str(PATCH_SIZE)]);

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

for iimg = 6:length(geometry_list)
    % Load the geometry for the current image
    base_name = strrep(geometry_list(iimg).name, '_geometry.mat', '');
    
    fprintf(1, 'Creating patches from %s\n', base_name);
    
    % Before we do anything, check that the sils and the original are the
    % same resolution
    sil_path = fullfile(sil_dir, [base_name '.jpg']);
    orig_path = fullfile(orig_dir, [base_name '.jpg']);
    
    % Check existence
    if(~exist(orig_path, 'file') || ~exist(sil_path, 'file'))
        fprintf(1, '\tSkipping: Original or Silhouette file does not exist!\n');
        continue;
    end
    
    sil_info = imfinfo(sil_path);
    orig_info = imfinfo(orig_path);
    
    if(sil_info.Width ~= orig_info.Width || sil_info.Height ~= orig_info.Height)
        fprintf(1, '\tSkipping: Original and silhouette are not the same size.\n');
        continue;
    end
    
    % Check existence of geometry (should be redundant)
    geometry_path = fullfile(geometry_dir, geometry_list(iimg).name);
    
    if(~exist(geometry_path, 'file'))
        fprintf(1, '\tSkipping: File %s does not exist, check "silhouette_analysis_getGeometry.m" again.\n', geometry_list(iimg).name);
        continue;
    end
    
    % Load the geometry
    fprintf(1, '\tLoading Geometry...\n');
    load(geometry_path, 'sat_mask', 'tum_mask');
    
    % Resize mask back to the original resolution, create fused mask
    fprintf(1, '\tResizing...\n');
    imgMask = logical(imresize(sat_mask, [sil_info.Height, sil_info.Width]) + ...
        imresize(tum_mask, [sil_info.Height, sil_info.Width]));
    clear sat_mask tum_mask
    
    % Erode image to avoid edge cases
    imgMask = imerode(imgMask, strel('disk', PATCH_SIZE/8));
    
    % Load original image
    imgOrig = imread(orig_path);
    
    % Resize images if desired
    if(RESIZE_RATIO ~= 1)
        fprintf(1, '\tResizing again...\n');
        imgOrig = imresize(imgOrig, RESIZE_RATIO);
        imgMask = logical(imresize(imgMask, RESIZE_RATIO));
    end
    
    [h, w] = size(imgMask);
    
    % Get a list of all the possible indices that we can try
    % (pad the arrays to remove possible edge points)
    padding = padarray(true(h-PATCH_SIZE*2,w-PATCH_SIZE*2), [PATCH_SIZE, PATCH_SIZE], 0);
    posIdx = find(imgMask .* padding);
    negIdx = find(~imgMask .* padding);
    
    % Shuffle the indices, select the target number from each
    % This does NOT explicitly cover overlaps; we assume that the number of
    % overlaps should (statistically) be minimal.
    posIdx = datasample(posIdx, NUM_PATCHES, 'Replace', false);
    negIdx = datasample(negIdx, NUM_PATCHES, 'Replace', false);
    
    % Divide into train / test sets according to percentages above
    posIdxTrain = posIdx(1:floor(numel(posIdx)*train_pct));
    posIdxTest = posIdx(floor(numel(posIdx)*train_pct)+1:end);
    negIdxTrain = negIdx(1:floor(numel(negIdx)*train_pct));
    negIdxTest = negIdx(floor(numel(negIdx)*train_pct)+1:end);
    
    %% Training
    % Positive
    className = 'tumor';
    posTrainImgs = pull_patches(posIdxTrain, imgOrig, PATCH_SIZE);
    for ipatch = 1:length(posIdxTrain)
        % Grab the current position
        % FOR DISPLAY ONLY, change the iRow & iCol positions
        [iRow, iCol] = ind2sub([h,w], posIdxTrain(ipatch));
        
        % Save File
        if(~exist(fullfile(train_dir, className), 'dir'))
            mkdir(fullfile(train_dir, className));
        end
        
        patch_path = fullfile(train_dir, className, sprintf('%s-R%d-C%d-%s.png', base_name, iRow, iCol, className));
        imwrite(posTrainImgs(:,:,:,ipatch), patch_path);
        
        if(debugFlag)
            imgBB = [iRow-floor(PATCH_SIZE/2), iRow+ceil(PATCH_SIZE/2), iCol-floor(PATCH_SIZE/2), iCol+floor(PATCH_SIZE/2)];
            figure;
            subplot(2,1,1); imshow(imgOrig); hold on;
            scatter(iCol, iRow, '*g');
            line([imgBB(3), imgBB(3)], [imgBB(1), imgBB(2)], 'Color', 'g', 'LineWidth', 5);
            line([imgBB(4), imgBB(4)], [imgBB(1), imgBB(2)], 'Color', 'g', 'LineWidth', 5);
            line([imgBB(3), imgBB(4)], [imgBB(1), imgBB(1)], 'Color', 'g', 'LineWidth', 5);
            line([imgBB(3), imgBB(4)], [imgBB(2), imgBB(2)], 'Color', 'g', 'LineWidth', 5);
            hold off;
            subplot(2,2,3); imshow(imgMask(iRow-floor(PATCH_SIZE/2):iRow+ceil(PATCH_SIZE/2)-1, iCol-floor(PATCH_SIZE/2):iCol+ceil(PATCH_SIZE/2)-1));
            subplot(2,2,4); imshow(posTrainImgs(:,:,:,ipatch));
            pause;
            close all;
            clear imgBB
        end
    end
    
    clear posTrainImgs
    
    % Negative
    className = 'nontumor';
    negTrainImgs = pull_patches(negIdxTrain, imgOrig, PATCH_SIZE);
    for ipatch = 1:length(negIdxTrain)
        % Grab the current position
        % FOR DISPLAY ONLY, change the iRow & iCol positions
        [iRow, iCol] = ind2sub([h,w], negIdxTrain(ipatch));
        
        % Save File
        if(~exist(fullfile(train_dir, className), 'dir'))
            mkdir(fullfile(train_dir, className));
        end
        patch_path = fullfile(train_dir, className, sprintf('%s-R%d-C%d-%s.png', base_name, iRow, iCol, className));
        imwrite(negTrainImgs(:,:,:,ipatch), patch_path);
    end
    clear negTrainImgs
    
    %% Testing
    
    % Positive
    className = 'tumor';
    posTestImgs = pull_patches(posIdxTest, imgOrig, PATCH_SIZE);
    for ipatch = 1:length(posIdxTest)
        % Grab the current position
        % FOR DISPLAY ONLY, change the iRow & iCol positions
        [iRow, iCol] = ind2sub([h,w], posIdxTest(ipatch));
        
        % Save File
        if(~exist(fullfile(test_dir, className), 'dir'))
            mkdir(fullfile(test_dir, className));
        end
        patch_path = fullfile(test_dir, className, sprintf('%s-R%d-C%d-%s.png', base_name, iRow, iCol, className));
        imwrite(posTestImgs(:,:,:,ipatch), patch_path);
    end
    clear posTestImgs
    
    % Negative
    className = 'nontumor';
    negTestImgs = pull_patches(negIdxTest, imgOrig, PATCH_SIZE);
    for ipatch = 1:length(negIdxTest)
        % Grab the current position
        % FOR DISPLAY ONLY, change the iRow & iCol positions
        [iRow, iCol] = ind2sub([h,w], negIdxTest(ipatch));
        
        % Save File
        if(~exist(fullfile(test_dir, className), 'dir'))
            mkdir(fullfile(test_dir, className));
        end
        patch_path = fullfile(test_dir, className, sprintf('%s-R%d-C%d-%s.png', base_name, iRow, iCol, className));
        imwrite(negTestImgs(:,:,:,ipatch), patch_path);
    end
    clear negTestImgs
    
    
end % Continue cycling images

exit
