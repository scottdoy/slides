%% Silhouette Comparison Script
% Script to compare two silhouettes

close all; clear;

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
sil_before = fullfile(data_dir, 'Before_After', 'Before');
sil_after = fullfile(data_dir, 'Before_After', 'After');

% List of silhouettes, marked images, and originals
sil_before = dir(fullfile(sil_before, '*.jpg'));
sil_after = dir(fullfile(sil_after, '*.jpg'));


%% Create features
% Start cycling through images to pull out tumor boundary and sat centroid

for iimg = 1:length(sil_before)
    
    if(exist(fullfile(feat_dir, strrep(sil_list(iimg).name, '.jpg', '_geometry.mat')), 'file'))
        fprintf(1,'Features exist for %s\n', sil_list(iimg).name);
        continue;
    end
    
    fprintf(1,'Analyzing %s\n', sil_list(iimg).name);
    
    % Pathnames to sil and mark
    sil_before_path = fullfile(sil_dir, sil_list(iimg).name);
    mark_path = fullfile(mark_dir, mark_list(iimg).name);
    
    % Check to make sure all the images are there (sil and mark) and the
    % filenames correspond
    if(~exist(sil_path, 'file'))
        fprintf(1, '\tThe Silhouette file cannot be found. Skipping...\n');
        continue;
    elseif(~exist(mark_path, 'file'))
        fprintf(1, '\tThe Marked file cannot be found. Skipping...\n');
        continue;
    elseif(~strcmp(sil_list(iimg).name, strrep(lower(mark_list(iimg).name), 'marked', '')))
        fprintf(1, '\tThe Silhouette and Marked filenames do not match. Please check. Skipping...\n');
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
   
    % Resize the images
    img_marked = imresize(img_marked, 0.25);
    img_mask = imresize(img_mask, 0.25);
    
    % Pull out main tumor region from marked image
    mt_mask = img_marked(:,:,1) >= 0 & img_marked(:,:,1) < 100 & ...
        img_marked(:,:,2) >= 100 & img_marked(:,:,2) < 200 &...
        img_marked(:,:,3) >= 200;
    fprintf(1,'\tNumber of detected MT pixels: %d\n', sum(mt_mask(:)));
    clear img_marked
    
    img_props = regionprops(img_mask, 'PixelIdxList');
    mt_pixelidxlist = find(mt_mask);
    
    sat_mask = false(size(img_mask));
    tum_mask = false(size(img_mask));
    
    fprintf(1, '\tSeparating tumor from satellite...\n');
    for iblob = 1:length(img_props)
        these_pixels = img_props(iblob).PixelIdxList;
        if(any(ismember(these_pixels, mt_pixelidxlist)));
            tum_mask(these_pixels) = 1;
        else
            sat_mask(these_pixels) = 1;
        end
    end
    
    % Get filename for saving
    [~, img_name, ~] = fileparts(sil_path);
    
    imwrite(imresize(sat_mask, 0.25),fullfile(sil_dir, 'tmp', [img_name '_sat_mask.jpg']));
    imwrite(imresize(tum_mask, 0.25),fullfile(sil_dir, 'tmp', [img_name '_tum_mask.jpg']));
    
    % Get the boundary points of the "tumor"
    fprintf(1, '\tGetting boundary points of the tumor...\n');
    tum_boundary = bwboundaries(tum_mask);
    X_tum = []; Y_tum = [];
    for ibound = 1:length(tum_boundary)
        X_tum = [X_tum; tum_boundary{ibound}(:,1)];
        Y_tum = [Y_tum; tum_boundary{ibound}(:,2)];
    end
    
    sat_props = regionprops(sat_mask>0, 'Area', 'Centroid');
    
    sat_boundary = bwboundaries(imfill(sat_mask, 'holes'));
    X_sat = []; Y_sat = [];
    for ibound = 1:length(sat_boundary)
        X_sat = [X_sat; sat_boundary{ibound}(:,1)];
        Y_sat = [Y_sat; sat_boundary{ibound}(:,2)];
    end
    
    save(fullfile(feat_dir, strrep(sil_list(iimg).name, '.jpg', '_geometry.mat')), 'X_tum', 'Y_tum', 'X_sat', 'Y_sat', 'sat_props', 'sat_mask', 'tum_mask', '-v7.3');    
    
end
