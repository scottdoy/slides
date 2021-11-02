%% Silhouette Binarize Ground Truth
% Script to create a new set of ground truth masks that have specific 
% colors associated with each class. This script intended to run only on 
% the "Marked" and "Unmarked" images, creating a new "Labeled" category.
% 
% 2017-02-15 SD

close all; clear;

if ispc
    base_dir = fullfile('E:', 'projects', 'base_matlab');
    proj_dir = fullfile('E:', 'projects', 'occ_quant_risk_score');
    data_dir = fullfile(proj_dir, 'data', 'Silhouettes', 'merge');
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
label_dir = fullfile(data_dir, 'Labels');

% List of silhouettes (Use these to pull the others)
sil_list = dir(fullfile(sil_dir, '*.jpg'));

%% Cycle through silhouettes

for iimg = 115:length(sil_list)
    
    fprintf(1,'Merging masks for %s\n', sil_list(iimg).name);
    
    % Pathnames to sil and mark
    sil_path = fullfile(sil_dir, sil_list(iimg).name);
    mark_path = fullfile(mark_dir, strrep(sil_list(iimg).name, '.jpg', 'MARKED.jpg'));
    
    % Check to make sure all the images are there (sil and mark) and the
    % filenames correspond
    if(~exist(sil_path, 'file'))
        fprintf(1, '\tThe Silhouette file cannot be found. Skipping...\n');
        continue;
    elseif(~exist(mark_path, 'file'))
        fprintf(1, '\tThe Marked file cannot be found. Skipping...\n');
        continue;
    end

    % Load up the silhouette and the marked image
    fprintf(1, '\tLoading up %s...\n', sil_path);
    sil = imread(sil_path);
    
    % Check that it is a binary image
    if(size(sil,3) > 1)
        fprintf(1, '\tSilhouette is 3D, converting with im2bw...\n');
        img_mask = ~im2bw(sil);
    elseif(islogical(sil))
        fprintf(1, '\tSilhouette is logical, so inverting...\n');
        img_mask = ~sil;
    else
        fprintf(1, '\tSilhouette is not logical, so thresholding at 250...\n');
        img_mask = sil < 250;
    end
    
    img_mask = imopen(img_mask, strel('disk', 10));
    
    fprintf(1, '\tLoading up %s...\n', mark_path);
    img_marked = imread(mark_path);
    
    % Check that the marked and masked images are the same size
    [hmarked, wmarked, dmarked] = size(img_marked);
    [hmasked, wmasked, dmasked] = size(img_mask);
    
    if(hmarked ~= hmasked || wmarked ~= wmasked)
        fprintf(1, '\tERROR: the img_marked and img_mask files are not the same size!\n');
        continue;
    end
    
    % Pull out main tumor region from marked image
    marked_color = [3 175 241];
    marked_adjust = [50 50 50];
    
    mt_mask = img_marked(:,:,1) >= marked_color(1)-marked_adjust(1) & img_marked(:,:,1) < marked_color(1)+marked_adjust(1) & ...
        img_marked(:,:,2) >= marked_color(2)-marked_adjust(2) & img_marked(:,:,2) < marked_color(2)+marked_adjust(2) &...
        img_marked(:,:,3) >= marked_color(3)-marked_adjust(3);
    fprintf(1,'\tNumber of detected MT pixels: %d\n', sum(mt_mask(:)));
    clear img_marked
    
    img_props = regionprops(img_mask, 'PixelIdxList');
    mt_pixelidxlist = find(mt_mask);
    
    % Create fused map
    label_mask = zeros(size(img_mask,1) * size(img_mask,2), 3, 'uint8');

    fprintf(1, '\tSeparating tumor from satellite...\n');
    for iblob = 1:length(img_props)
        these_pixels = img_props(iblob).PixelIdxList;
        if(any(ismember(these_pixels, mt_pixelidxlist)));
            label_mask(these_pixels,:) = repmat([0, 0, 255], length(these_pixels), 1);
        else
            label_mask(these_pixels,:) = repmat([0, 255, 0], length(these_pixels), 1);
        end
    end
    
    % Get filename for saving
    [~, img_name, ~] = fileparts(sil_path);
    
    imwrite(reshape(label_mask, size(img_mask,1), size(img_mask,2), 3), fullfile(label_dir, [img_name '.png']));
end
