%% Silhouette Texture Feature Classification
% Script to load up the silhouette features and create a classifier from
% them
% 2017-02-26 SD

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
    addpath(genpath(fullfile(proj_dir, 'scripts')));
    addpath(genpath(fullfile(proj_dir, 'module')));
end

%% Organize paths
feat_dir = fullfile(data_dir, 'Features');

% Get the list of features
feat_list = dir(fullfile(feat_dir, '*_rgb_features.mat'));
nfeats = length(feat_list);

% Init for debugging
ifeat = 1;

% Resize image by this amount
RES_MPP = 2;

%% Process Dataset

Xtot = [];
Ytot = [];

for ifeat = 1:nfeats
    %% Set up the base name for this image
    feat_path = fullfile(feat_dir, feat_list(ifeat).name);
    [~, base_name, ~] = fileparts(feat_path);
    base_name = strsplit(base_name, '_');
    base_name = base_name{1};
    
    % Check existence of each of the feature files
    if(exist(fullfile(feat_dir, [base_name, '_0.125_rgb_features_1-100.mat']), 'file') && ...
            exist(fullfile(feat_dir, [base_name, '_0.125_rgb_features_101-200.mat']), 'file') && ...
            exist(fullfile(feat_dir, [base_name, '_0.125_rgb_features_201-300.mat']), 'file'))
        fprintf(1, 'Loading %s\n', base_name);
        
        %% Load the three sets of features
        load(fullfile(feat_dir, [base_name, '_0.125_rgb_features_1-100.mat']));
        X1 = X; Y1 = Y;
        load(fullfile(feat_dir, [base_name, '_0.125_rgb_features_101-200.mat']));
        X2 = X; Y2 = Y;
        load(fullfile(feat_dir, [base_name, '_0.125_rgb_features_201-300.mat']));
        X3 = X; Y3 = Y;
        load(fullfile(feat_dir, [base_name, '_0.125_rgb_features_300-375.mat']));
        X4 = X; Y4 = Y;
        
        % Fuse them together
        Xtot = [Xtot; X1, X2, X3, X4];
        Ytot = logical([Ytot; Y]);
        clear X1 X2 X3 X4
        clear Y1 Y2 Y3 Y4
        clear X Y
    end
end

% %% Clean features
% remove_feats = false(1,size(Xtot,2));
% for ifeat = 1:size(Xtot,2)
%     idx_nan = any(isnan(Xtot(:,ifeat)));
%     idx_degen = length(unique(Xtot(:,ifeat))) < 10;
%     
%     % Remove nan features
%     if(idx_nan==1)
%         remove_feats(ifeat) = true;
%     end
%     if(idx_degen==1)
%         remove_feats(ifeat) = true;
%     end
% end
% 
% Xtot(:,remove_feats) = [];

%% Select features

% % Based on histograms
% selfeats = [34, 48, 49, 126, 127, 141, 142, 144, 156, 157, 158, 159, 171, 172, 173, 174];

% % Based on sequential FS
% c = cvpartition(Ytot, 'k', 10);
% opts = statset('display', 'iter');
% fun = @(XT,yT, Xt, yt)...
%     (sum(~strcmp(yt, classify(Xt, XT, yT, 'quadratic'))));
% [fs, history] = sequentialfs(fun, Xtot, Ytot, 'cv', c, 'options', opts);

% Based on Visual Inspection
% selfeats = [47, 49, 159, 167, 171, 184];
% selfeats = [47, 167, 171];

% Based on mrmr
selfeats = mrmr_mid_d(Xtot, uint8(Ytot), 10);

%% Create a classifier

MdlNB = fitcnb(Xtot(:,selfeats), Ytot);


% Compute ROC curve
scores = MdlGLM.Fitted.Probability;
[X,Y,T,AUC] = perfcurve(Ytot, scores, 1);
figure; plot(X,Y);

save(fullfile(proj_dir, 'data', 'classifiers', 'silhouette_texture_classifier_3feats.mat'), 'MdlNB', 'selfeats');

% % Save the file
% pixel_path = fullfile(feat_dir, 'texture_data_1-100.mat');
% save(pixel_path, 'X', 'Y', 'RES_MPP', '-v7.3');

% % Clean up
% clear img_list num_imgs img_idx img_dir img_name img_ext img_path
% clear process_path img_orig img_process


% %% --[ Generate Masks from Annotations ]-- %%
% 
% for case_idx = 4:4
%     % Get the list of annotations contained in the XML directory
%     case_dir = fullfile(data_dir, case_list(case_idx).name);
%     img_list = dir(fullfile(case_dir, 'tif', '*.tiff'));
%     xml_list = dir(fullfile(case_dir, 'xml', '*.xml'));
%     num_imgs = length(img_list); num_xmls = length(xml_list);
%     
%     % Get list of names from img_list and xml_list (no extensions)
%     for img_idx = 1:num_imgs
%         [~, tmpname, tmpext] = fileparts(img_list(img_idx).name);
%         img_names{img_idx} = tmpname;
%     end
%     
%     for xml_idx = 1:num_xmls
%         [~, tmpname, tmpext] = fileparts(xml_list(xml_idx).name);
%         xml_names{xml_idx} = tmpname;
%     end
%     
%     % Cycle through the xmls
%     for xml_idx = 1:num_xmls
%         % Quick check that each xml file has a corresponding tif
%         if(any(strcmp(xml_names{xml_idx}, img_names)))
% 
%             % Figure out which of the images is a match with this xml
%             this_img_idx = find(ismember(img_names, xml_names{xml_idx}));
%             
%             % Get the info for the tiff file
%             img_info = imfinfo(fullfile(case_dir, 'tif', [img_names{this_img_idx} '.tiff']));
%             h = img_info(1).Height;
%             w = img_info(1).Width;
%             
%             % Load the annotation file
%             A = getAnnotationNew(fullfile(case_dir, 'xml', [xml_names{xml_idx} '.xml']));
%             
%             % Create a new annotation mask for each layer found in here
%             for anno_idx = 1:length(A)
%                 img_mask = false(h,w);
%                 annotation = A{anno_idx}.regions;
%                 for region_idx = 1:length(annotation)
%                     mask = poly2mask(annotation(region_idx).X.*0.0625,...
%                         annotation(region_idx).Y.*0.0625,...
%                         h, w);
%                     img_mask(mask) = true;
%                     clear mask
%                 end
%                 
%                 % Save in the masks directory
%                 imwrite(img_mask, fullfile(case_dir, 'masks', [xml_names{xml_idx}, '_' A{anno_idx}.desc '.png']));
%                 clear img_mask annotation
%             end
%         end
%     end
% end
% 
% %% --[ Training Data Collection ]-- %%
% 
% X = [];
% Y = [];
% 
% for case_idx = 4:4
%     
%     % Set up the list of images
%     case_dir = fullfile(data_dir, case_list(case_idx).name);
%     
%     % Get the list of images (used to peruse the multiple masks / feature
%     % files)
%     img_list = dir(fullfile(case_dir, 'tif', '*.tiff'));
%     num_imgs = length(img_list);
%     
%     % Get list of names from img_list and xml_list (no extensions)
%     for img_idx = 1:num_imgs
%         [~, tmpname, tmpext] = fileparts(img_list(img_idx).name);
%         img_names{img_idx} = tmpname;
%     end
%     
%     % Cycle through the image names to pull out features
%     for img_idx = 1:1
%         
%         % Get the list of feature files (this is our data now)
%         feat_list = dir(fullfile(case_dir, 'feats', [img_names{img_idx} '*.mat']));
%         
%         % Also get the list of corresponding masks
%         mask_list = dir(fullfile(case_dir, 'masks', [img_names{img_idx} '*.png']));
% 
%         % Cycle through the feature list
%         for feat_idx = 1:length(feat_list)
%             feat_path = fullfile(case_dir, 'feats', feat_list(feat_idx).name);
%             feat_data = load(feat_path);
%             
%             % Need to grab the feature data
%             name = fieldnames(feat_data);
%             feat_data = getfield(feat_data, name{1});
%             
%             [h,w,f] = size(feat_data);
%             
%             feat_data = reshape(feat_data, h*w,f);
%             tmp_data = []; tmp_label = [];
%             
%             % Cycle through the masks and pull out the relevant training data
%             for mask_idx = 1:length(mask_list)
%                 mask_path = fullfile(case_dir, 'masks', mask_list(mask_idx).name);
%                 
%                 % Load up this mask
%                 this_mask = imread(mask_path);
%                 this_mask = reshape(this_mask, h*w, 1);
%                 
%                 % Pull out the features of this mask
%                 this_data = feat_data(this_mask, :);
%                 
%                 % Switch the label depending on the string matching with the
%                 % mask name
%                 % Label Codes:
%                 %   Background  = 0
%                 %   Tumor       = 1
%                 %   Stroma      = 2
%                 %   Blood       = 3
%                 %   Ink         = 4
%                 %   Lymphocytes = 5
%                 %   Muscle      = 6
%                 if ~isempty(strfind(mask_list(mask_idx).name, 'Background'))
%                     this_label = 0 .* ones(size(this_data,1),1);
%                 elseif ~isempty(strfind(mask_list(mask_idx).name, 'Tumor'))
%                     this_label = 1 .* ones(size(this_data,1),1);
%                 elseif ~isempty(strfind(mask_list(mask_idx).name, 'Stroma'))
%                     this_label = 2 .* ones(size(this_data,1),1);
%                 elseif ~isempty(strfind(mask_list(mask_idx).name, 'Blood'))
%                     this_label = 3 .* ones(size(this_data,1),1);
%                 elseif ~isempty(strfind(mask_list(mask_idx).name, 'Ink'))
%                     this_label = 4 .* ones(size(this_data,1),1);
%                 elseif ~isempty(strfind(mask_list(mask_idx).name, 'Lymphocytes'))
%                     this_label = 5 .* ones(size(this_data,1),1);
%                 elseif ~isempty(strfind(mask_list(mask_idx).name, 'Muscle'))
%                     this_label = 6 .* ones(size(this_data,1),1);
%                 else
%                     this_data = [];
%                     this_label = [];
%                     fprintf(1, 'No acceptable mask name found; skipping!\n');
%                 end
%                 tmp_data = [tmp_data; this_data];
%                 tmp_label = [tmp_label; this_label];
%             end
%             selfeats(feat_idx,:) = mrmr_miq_d(double(tmp_data), tmp_label, 20);
%             tmp_data = tmp_data(:,selfeats(feat_idx,:));
%             X = [X, tmp_data];
%             Y = [Y, tmp_label];
%         end
%     end
% end
% 
% %% --[ Create Bayes Classifier ]-- %%
% 
% sf = reshape(selfeats, 1, 9*20);
% usefeats = 1:180;
% 
% bayes_model = fitcnb(X(:,sf(usefeats)), Y(:,1));
% 
% %%
% test_data = [];
% for case_idx = 4:4
%     
%     % Set up the list of images
%     case_dir = fullfile(data_dir, case_list(case_idx).name);
%     
%     % Get the list of images (used to peruse the multiple masks / feature
%     % files)
%     img_list = dir(fullfile(case_dir, 'tif', '*.tiff'));
%     num_imgs = length(img_list);
%     
%     % Get list of names from img_list and xml_list (no extensions)
%     for img_idx = 1:num_imgs
%         [~, tmpname, tmpext] = fileparts(img_list(img_idx).name);
%         img_names{img_idx} = tmpname;
%     end
%     
%     % Cycle through the image names to pull out features
%     for img_idx = 1:1
%         
%         % Get the list of feature files (this is our data now)
%         feat_list = dir(fullfile(case_dir, 'feats', [img_names{img_idx} '*.mat']));
%         
%         % Also get the list of corresponding masks
%         mask_list = dir(fullfile(case_dir, 'masks', [img_names{img_idx} '*.png']));
% 
%         % Cycle through the feature list
%         for feat_idx = 1:length(feat_list)
%             feat_path = fullfile(case_dir, 'feats', feat_list(feat_idx).name);
%             feat_data = load(feat_path);
%             
%             % Need to grab the feature data
%             name = fieldnames(feat_data);
%             feat_data = getfield(feat_data, name{1});
%             
%             [h,w,f] = size(feat_data);
%             
%             feat_data = reshape(feat_data, h*w,f);
%             test_data = [test_data, feat_data(:,selfeats(feat_idx,:))];
%         end
%     end
% end
% 
% %%
% usefeats = 1:180;
% 
% sf = reshape(selfeats, size(selfeats,1)*size(selfeats,2), 1);
% [test_bin, test_predict] = predict(bayes_model, test_data(:, sf(usefeats)));
% 
% % Class names
% class_names = {'Background', 'Tumor', 'Stroma', 'Blood', 'Ink', 'Lymphocytes', 'Muscle'};
% 
% for ii = 1:7
%     figure; 
%     imshow(reshape(test_predict(:,ii), h, w),[]); 
%     title(class_names{ii});
%     drawnow;
% end