%% Silhouette Classification Construction
% Script to train a classifier on the silhouette data
% Requires "silhouette_get_geometry" to have been run

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
    fprintf(1, 'Adding proj_dir to path\n');
    addpath(genpath(proj_dir));
    addpath(genpath(proj_dir));
end

%% Organize paths
orig_dir = fullfile(data_dir, 'Original');
feat_dir = fullfile(data_dir, 'Features');
sil_dir = fullfile(data_dir, 'Unmarked');
marked_dir = fullfile(data_dir, 'Marked');

% Get the list of silhouettes (use this to extract features)
sil_list = dir(fullfile(sil_dir, '*.jpg'));
nimgs = length(sil_list);

% Init for debugging
iimg = 1;

% Resize image by this amount
IMG_RESIZE = 0.05;

%% Process Dataset

for iimg = 1:nimgs
    %% Set up filename and paths
    sil_path = fullfile(sil_dir, sil_list(iimg).name);
    [~, base_name, ~] = fileparts(sil_path);
    
    fprintf(1, 'Processing %s\n', base_name);
    
    marked_path = fullfile(marked_dir, [base_name 'MARKED.jpg']);
    orig_path = fullfile(orig_dir, [base_name '.jpg']);
    feats_rgb_path = fullfile(feat_dir, [base_name '_' num2str(IMG_RESIZE) '_rgb_features.mat']);
    feats_hsv_path = fullfile(feat_dir, [base_name '_' num2str(IMG_RESIZE) '_rgb_features.mat']);
    feats_lab_path = fullfile(feat_dir, [base_name '_' num2str(IMG_RESIZE) '_lab_features.mat']);
    
    %% Check to make sure images exist and are compatible
    
    % File Existence
    if(~exist(marked_path, 'file')), fprintf(1, '\tMarked file does not exist! Skipping...\n'); continue; end    
    if(~exist(orig_path, 'file')), fprintf(1, '\tOriginal file does not exist! Skipping...\n'); continue; end
    
    % File Info (size)   
    sil_info = imfinfo(sil_path);
    marked_info = imfinfo(marked_path);
    orig_info = imfinfo(orig_path);
    
    if(sil_info.Width ~= marked_info.Width), fprintf(1, '\tMarked and Sil have mismatched width, skipping...\n'); continue; end;
    if(sil_info.Height ~= marked_info.Height), fprintf(1, '\tMarked and Sil have mismatched height, skipping...\n'); continue; end;
    if(sil_info.Width ~= orig_info.Width), fprintf(1, '\tOrig and Sil have mismatched width, skipping...\n'); continue; end;
    if(sil_info.Height ~= orig_info.Height), fprintf(1, '\tOrig and Sil have mismatched height, skipping...\n'); continue; end;
    
    % Existence of features
    if(exist(feats_rgb_path, 'file') && exist(feats_hsv_path, 'file') && exist(feats_lab_path, 'file')), fprintf(1, '\tPixel features exist, skipping...\n'); continue; end
    
    %% Still here? Good, begin processing
    
    % Load image
    img = imread(orig_path);
    
    % Resize image by this amount
    img = imresize(img, IMG_RESIZE);

    %% Begin extraction
    fprintf(1, '\tStarting feature extraction...\n');
    
    % RGB Features
    fprintf(1, '\t\tRGB Features...\n');
    if(~exist(feats_rgb_path, 'file'))
        
        [feats_rgb, feats_rgb_names] = silhouette_texture_features(img);
        feats_rgb = single(feats_rgb);
        save(feats_rgb_path,'feats_rgb', 'feats_rgb_names', '-v7.3');
        clear feats_rgb
    else
        fprintf(1, '\t\tRGB feats already processed\n');
    end
    
    % HSV Features
    fprintf(1, '\t\tHSV Features...\n');
    img_hsv = rgb2hsv(img);
    if(~exist(feats_hsv_path, 'file'))
        [feats_hsv, feats_hsv_names] = get_texture_features(img_hsv);
        feats_hsv = single(feats_hsv);
        save(feats_hsv_path, 'feats_hsv', 'feats_hsv_names', '-v7.3');
        clear feats_hsv
    else
        fprintf(1, '\t\tHSV feats already processed\n');
    end
    
    
    fprintf(1, '\t\tLAB Features...\n');
    img_lab = rgb2lab(img);
    if(~exist(feats_lab_path, 'file'))
        [feats_lab, feats_lab_names] = get_texture_features(img_lab);
        feats_lab = single(feats_lab);
        save(feats_lab_path, 'feats_lab', 'feats_lab_names', '-v7.3');
        clear feats_lab
    else
        fprintf(1, '\t\tHSV feats already processed\n', img_name);
    end

end

% Clean up
clear img_list num_imgs img_idx img_dir img_name img_ext img_path
clear process_path img_orig img_process


%% --[ Generate Masks from Annotations ]-- %%

for case_idx = 4:4
    % Get the list of annotations contained in the XML directory
    case_dir = fullfile(data_dir, case_list(case_idx).name);
    img_list = dir(fullfile(case_dir, 'tif', '*.tiff'));
    xml_list = dir(fullfile(case_dir, 'xml', '*.xml'));
    num_imgs = length(img_list); num_xmls = length(xml_list);
    
    % Get list of names from img_list and xml_list (no extensions)
    for img_idx = 1:num_imgs
        [~, tmpname, tmpext] = fileparts(img_list(img_idx).name);
        img_names{img_idx} = tmpname;
    end
    
    for xml_idx = 1:num_xmls
        [~, tmpname, tmpext] = fileparts(xml_list(xml_idx).name);
        xml_names{xml_idx} = tmpname;
    end
    
    % Cycle through the xmls
    for xml_idx = 1:num_xmls
        % Quick check that each xml file has a corresponding tif
        if(any(strcmp(xml_names{xml_idx}, img_names)))

            % Figure out which of the images is a match with this xml
            this_img_idx = find(ismember(img_names, xml_names{xml_idx}));
            
            % Get the info for the tiff file
            img_info = imfinfo(fullfile(case_dir, 'tif', [img_names{this_img_idx} '.tiff']));
            h = img_info(1).Height;
            w = img_info(1).Width;
            
            % Load the annotation file
            A = getAnnotationNew(fullfile(case_dir, 'xml', [xml_names{xml_idx} '.xml']));
            
            % Create a new annotation mask for each layer found in here
            for anno_idx = 1:length(A)
                img_mask = false(h,w);
                annotation = A{anno_idx}.regions;
                for region_idx = 1:length(annotation)
                    mask = poly2mask(annotation(region_idx).X.*0.0625,...
                        annotation(region_idx).Y.*0.0625,...
                        h, w);
                    img_mask(mask) = true;
                    clear mask
                end
                
                % Save in the masks directory
                imwrite(img_mask, fullfile(case_dir, 'masks', [xml_names{xml_idx}, '_' A{anno_idx}.desc '.png']));
                clear img_mask annotation
            end
        end
    end
end

%% --[ Training Data Collection ]-- %%

X = [];
Y = [];

for case_idx = 4:4
    
    % Set up the list of images
    case_dir = fullfile(data_dir, case_list(case_idx).name);
    
    % Get the list of images (used to peruse the multiple masks / feature
    % files)
    img_list = dir(fullfile(case_dir, 'tif', '*.tiff'));
    num_imgs = length(img_list);
    
    % Get list of names from img_list and xml_list (no extensions)
    for img_idx = 1:num_imgs
        [~, tmpname, tmpext] = fileparts(img_list(img_idx).name);
        img_names{img_idx} = tmpname;
    end
    
    % Cycle through the image names to pull out features
    for img_idx = 1:1
        
        % Get the list of feature files (this is our data now)
        feat_list = dir(fullfile(case_dir, 'feats', [img_names{img_idx} '*.mat']));
        
        % Also get the list of corresponding masks
        mask_list = dir(fullfile(case_dir, 'masks', [img_names{img_idx} '*.png']));

        % Cycle through the feature list
        for feat_idx = 1:length(feat_list)
            feat_path = fullfile(case_dir, 'feats', feat_list(feat_idx).name);
            feat_data = load(feat_path);
            
            % Need to grab the feature data
            name = fieldnames(feat_data);
            feat_data = getfield(feat_data, name{1});
            
            [h,w,f] = size(feat_data);
            
            feat_data = reshape(feat_data, h*w,f);
            tmp_data = []; tmp_label = [];
            
            % Cycle through the masks and pull out the relevant training data
            for mask_idx = 1:length(mask_list)
                mask_path = fullfile(case_dir, 'masks', mask_list(mask_idx).name);
                
                % Load up this mask
                this_mask = imread(mask_path);
                this_mask = reshape(this_mask, h*w, 1);
                
                % Pull out the features of this mask
                this_data = feat_data(this_mask, :);
                
                % Switch the label depending on the string matching with the
                % mask name
                % Label Codes:
                %   Background  = 0
                %   Tumor       = 1
                %   Stroma      = 2
                %   Blood       = 3
                %   Ink         = 4
                %   Lymphocytes = 5
                %   Muscle      = 6
                if ~isempty(strfind(mask_list(mask_idx).name, 'Background'))
                    this_label = 0 .* ones(size(this_data,1),1);
                elseif ~isempty(strfind(mask_list(mask_idx).name, 'Tumor'))
                    this_label = 1 .* ones(size(this_data,1),1);
                elseif ~isempty(strfind(mask_list(mask_idx).name, 'Stroma'))
                    this_label = 2 .* ones(size(this_data,1),1);
                elseif ~isempty(strfind(mask_list(mask_idx).name, 'Blood'))
                    this_label = 3 .* ones(size(this_data,1),1);
                elseif ~isempty(strfind(mask_list(mask_idx).name, 'Ink'))
                    this_label = 4 .* ones(size(this_data,1),1);
                elseif ~isempty(strfind(mask_list(mask_idx).name, 'Lymphocytes'))
                    this_label = 5 .* ones(size(this_data,1),1);
                elseif ~isempty(strfind(mask_list(mask_idx).name, 'Muscle'))
                    this_label = 6 .* ones(size(this_data,1),1);
                else
                    this_data = [];
                    this_label = [];
                    fprintf(1, 'No acceptable mask name found; skipping!\n');
                end
                tmp_data = [tmp_data; this_data];
                tmp_label = [tmp_label; this_label];
            end
            selfeats(feat_idx,:) = mrmr_miq_d(double(tmp_data), tmp_label, 20);
            tmp_data = tmp_data(:,selfeats(feat_idx,:));
            X = [X, tmp_data];
            Y = [Y, tmp_label];
        end
    end
end

%% --[ Create Bayes Classifier ]-- %%

sf = reshape(selfeats, 1, 9*20);
usefeats = 1:180;

bayes_model = fitcnb(X(:,sf(usefeats)), Y(:,1));

%%
test_data = [];
for case_idx = 4:4
    
    % Set up the list of images
    case_dir = fullfile(data_dir, case_list(case_idx).name);
    
    % Get the list of images (used to peruse the multiple masks / feature
    % files)
    img_list = dir(fullfile(case_dir, 'tif', '*.tiff'));
    num_imgs = length(img_list);
    
    % Get list of names from img_list and xml_list (no extensions)
    for img_idx = 1:num_imgs
        [~, tmpname, tmpext] = fileparts(img_list(img_idx).name);
        img_names{img_idx} = tmpname;
    end
    
    % Cycle through the image names to pull out features
    for img_idx = 1:1
        
        % Get the list of feature files (this is our data now)
        feat_list = dir(fullfile(case_dir, 'feats', [img_names{img_idx} '*.mat']));
        
        % Also get the list of corresponding masks
        mask_list = dir(fullfile(case_dir, 'masks', [img_names{img_idx} '*.png']));

        % Cycle through the feature list
        for feat_idx = 1:length(feat_list)
            feat_path = fullfile(case_dir, 'feats', feat_list(feat_idx).name);
            feat_data = load(feat_path);
            
            % Need to grab the feature data
            name = fieldnames(feat_data);
            feat_data = getfield(feat_data, name{1});
            
            [h,w,f] = size(feat_data);
            
            feat_data = reshape(feat_data, h*w,f);
            test_data = [test_data, feat_data(:,selfeats(feat_idx,:))];
        end
    end
end

%%
usefeats = 1:180;

sf = reshape(selfeats, size(selfeats,1)*size(selfeats,2), 1);
[test_bin, test_predict] = predict(bayes_model, test_data(:, sf(usefeats)));

% Class names
class_names = {'Background', 'Tumor', 'Stroma', 'Blood', 'Ink', 'Lymphocytes', 'Muscle'};

for ii = 1:7
    figure; 
    imshow(reshape(test_predict(:,ii), h, w),[]); 
    title(class_names{ii});
    drawnow;
end