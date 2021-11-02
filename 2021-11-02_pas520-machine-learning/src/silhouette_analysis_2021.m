%% Silhouette Analysis Script
% Script to run analysis of the OCC interface using graph-based feature
% extraction.

close all; clear;

%% Set up Directories

% % Main project directory (change if you move to another computer)
% projDir = fullfile('..', '', 'occ_quant_risk_score');
% 
% % Source code modules
% moduleDir = fullfile(projDir, 'module');
% 
% % Data Parent
% dataDir = fullfile(projDir, 'data', 'Silhouettes');
% 
% % Label Subfolder contains manually labeled images
% labelDir = fullfile(dataDir, 'Labels');
% 
% % Orig Subfolder contains H&E slides
% origDir = fullfile(dataDir, 'Original_jpgs');
% 
% % Geometry Files - Contains geometry calculated from label images
% geomDir = fullfile(dataDir, 'geometry');
% 
% % Feature Files - Output from graph-based calculations
% featsDir = fullfile(dataDir, 'features');
featsDir = fullfile('..', 'data', 'features');

% % Output Figures - Saved figures (histograms, etc.)
% figDir = fullfile(projDir, 'figs', 'Silhouette');
% 
% % Output Directory Checking
% if(~isdir(geomDir)), mkdir(geomDir); end
% if(~isdir(featsDir)), mkdir(featsDir); end
% if(~isdir(figDir)), mkdir(figDir); end
% 
% % Add modules to path
% addpath(genpath(moduleDir));

%% Set up parameters

% % Display figures;
% disp_figures = false;
% 
% % Define label parameters
% labelParams(1).rgb = [0, 0, 255];
% labelParams(1).name = 'Main Tumor';
% labelParams(2).rgb = [0, 255, 0];
% labelParams(2).name = 'Satellite';
% 
% % Millimeters per pixel
% % Image geometry will be scaled to 689.0100 nm / pixel
% % So to convert to mm: 689.01 nm / pixel = 0.00068 mm / pixel
% mmpp = 0.00068;

%% Load preliminary data

% Pull out dataset summary sheet
labelFile = fullfile('..', 'data', 'Silhouettes_Dataset.xlsx');
labelTable = readtable(labelFile, 'ReadRowNames', true);

% %% Calculate Geometry from Labeled Images
% 
% % Get list of labeled images (basis for calculating geometry)
% labelList = dir(fullfile(labelDir, '*.png'));
% nLabels = length(labelList);
% 
% % Cycle through each label image
% for iImg = 1:nLabels
%     
%     % Pull out the base name of this silhouette
%     imgName = strrep(labelList(iImg).name, '.png', '');
%     
%     fprintf(1,'Analyzing %s\n', imgName);
%     
%     % Get path to image & Geometry
%     labelPath = fullfile(labelDir, labelList(iImg).name);
%     
%     % Set path of geometry
%     geomPath = fullfile(geomDir, [imgName '_geometry.mat']);
%     
%     % Check to ensure that geometry doesn't exist
%     if(exist(geomPath, 'file'))
%         fprintf(1, '\tFile %s already exists! Skipping...\n', geomPath);
%         continue;
%     end
%     
%     % Load label image
%     imgLabel = imread(labelPath);
%     
%     % Resize image to 10x (lowest-common-denominator) 
%     % 10x == 689.01 nm/pixel == 0.68901 microns / pixel
%     switch labelTable.SourceNm_pixel(imgName)
%         case 172.6300 % 40x, resize by 0.125
%             imgLabel = imresize(imgLabel, 0.125, 'Method', 'nearest');
%         case 345.5300 % 20x, resize by 0.25
%             imgLabel = imresize(imgLabel, 0.25, 'Method', 'nearest');
%         case 689.0100 % 10x, don't resize
%             imgLabel = imgLabel;
%         otherwise
%             fprintf(1, '\tUndetermined resolution, skipping...\n');
%             continue;
%     end
%     
%     % Extract geometry from the image
%     imgGeom = extract_geometry(imgLabel, labelParams);
%     
%     % Save geometry
%     save(geomPath, 'imgGeom', '-v7.3');
%     
%     clear imgLabel imgGeom labelPath geomPath
% end
% 
% clear iImg
    
%% Use Geometry to Get Graphs and Features

% % Get list of extracted geometry
% geomList = dir(fullfile(geomDir, '*_geometry.mat'));
% nGeometry = length(geomList);
% 
% % Cycle through extracted geometry
% for iGeom = 1:nGeometry
%     
%     fprintf(1,'Analyzing %s\n', geomList(iGeom).name);
%     
%     % Grab paths to geometry, label, and extracted features
%     imgName = strrep(geomList(iGeom).name, '_geometry.mat', '');
%     geomPath = fullfile(geomDir, geomList(iGeom).name);
%     labelPath = fullfile(labelDir, labelList(iGeom).name);
%     featsPath = fullfile(featsDir, strrep(geomList(iGeom).name, '_geometry.mat', '_graph_feats.mat'));
%     
%     % Skip if: 
%     % 1. We do not have geometry
%     % 2. We have already extracted features
%     if(~exist(geomPath, 'file'))
%         fprintf(1, '\tGeometry file %s does not exist, please try re-extracting geometry.\n', geomPath);
%         continue;
%     end
%     
%     if(exist(featsPath, 'file'))
%         fprintf(1, '\tFeatures %s exist, please delete if you want to re-extract.\n', featsPath);
%         continue;
%     end
%     
%     % Load the geometry
%     load(geomPath);
%     
%     %% REFACTOR THIS CODE =================================================
%     % Process Geometry
%     % Get indices for target geometry names (e.g., satellites vs. main tumor)
%     satIdx = find(strcmpi({imgGeom.name}, 'Satellite'));
%     tumIdx = find(strcmpi({imgGeom.name}, 'Main Tumor'));
%     
%     % Pull out satellite and tumor information
%     satMask = imgGeom(satIdx).binary;
%     satProps = imgGeom(satIdx).properties;
%     satBoundary = imgGeom(satIdx).boundary;
%     satCentroid = [satProps.Centroid];
%     
%     tumMask = imgGeom(tumIdx).binary;
%     
%     Y = satCentroid(1:2:end)';
%     X = satCentroid(2:2:end)';
%     
%     % Get boundary points for main tumor
%     tumBoundary = imgGeom(tumIdx).boundary;
%     
%     
%     % Extract Delaunay Triangulation Features
%     
%     % Calculate triangulation
%     DT = delaunayTriangulation(Y,X);
%     
% %     if(disp_figures)
% %         figure;
% %         subplot(1,2,1); imshow(satMask + tumMask); hold on; triplot(DT); title('Original Triangulation'); hold off;
% %     end
%     
%     % Break down into a regular triangulation, crop out edges crossing MT
%     faces       = DT.ConnectivityList;
%     vertcoords  = DT.Points;
%     edgelist    = edges(DT);
%     newfaces    = faces;
%     
%     % Cycle through the edge list
%     for iedge = 1:size(edgelist,1)
%         
%         % Pull coordinates for this edge
%         X_edge = [vertcoords(edgelist(iedge,1),2), vertcoords(edgelist(iedge,2),2)];
%         Y_edge = [vertcoords(edgelist(iedge,1),1), vertcoords(edgelist(iedge,2),1)];
%         
%         % Cycle through boundary objects in MT
%         for iBound = 1:length(tumBoundary)
%             
%             % Extract this boundary set
%             thisBoundary = tumBoundary{iBound};
%             X_tum = thisBoundary(:,1);
%             Y_tum = thisBoundary(:,2);
%             
%             % Check for intersections
%             [xi,yi] = intersections(X_tum,Y_tum,X_edge,Y_edge);
%             
%             if(~isempty(xi))
%                 newfaces(sum(ismember(newfaces, edgelist(iedge,:)),2) == 2, :) = [];
%             end
%         end
%         
%     end
%     
%     % If the deconstruction process removed too many edges, then we need to
%     % fill in the rows with NaNs
%     if(isempty(newfaces))
%         edge_feats = descriptive_statistics(NaN);
%         tri_feats = descriptive_statistics(NaN);
%     else
%         % Re-create triangulation (no longer Delaunay)
%         DT = triangulation(newfaces, vertcoords);
%         
% %         if(disp_figures)
% %             subplot(1,2,2); imshow(satMask + tumMask); hold on; triplot(DT); title('Post-Processed Graph');
% %         end
%         
%         % Get deconstructed triangulation points
%         faces = DT.ConnectivityList;
%         vcoords = DT.Points;
%         edgelist = edges(DT);
%         nedges = size(edgelist,1);
%         nvert = max(faces(:));
%         nfaces = size(faces,1);
%         A = zeros(nvert);
%         
%         % Edge weights
%         elengths = zeros(1, nedges);
%         for iedge = 1:nedges
%             elengths(iedge) = sqrt((DT.Points(edgelist(iedge,1),1) - DT.Points(edgelist(iedge,2),1))^2 + (DT.Points(edgelist(iedge,1),2) - DT.Points(edgelist(iedge,2),2))^2);
%         end
%         
%         edge_feats = descriptive_statistics(elengths);
%         
%         % Triangle areas
%         tareas = zeros(1, nfaces);
%         for iface = 1:nfaces
%             tareas(iface) = polyarea([DT.Points(faces(iface,1),1), DT.Points(faces(iface,2),1), DT.Points(faces(iface,3),1)], ...
%                 [DT.Points(faces(iface,1),2), DT.Points(faces(iface,2),2), DT.Points(faces(iface,3),2)]);
%         end
%         tri_feats = descriptive_statistics(tareas);
%         
%     end
%     
%     % Rename variables by replacing the VariableNames property of the
%     % table with a cell of strings created by 'cellfun', where 'cellfun'
%     % appends 'edge_' or 'tri_' to each of the cell contents.
%     edge_feats.Properties.VariableNames = cellfun(@(x) ['edge_' x], edge_feats.Properties.VariableNames, 'UniformOutput', false);
%     tri_feats.Properties.VariableNames = cellfun(@(x) ['tri_' x], tri_feats.Properties.VariableNames, 'UniformOutput', false);
%     
%     % Extract Satellite Statistic / Distance Features
%     
%     % Satellite counts
%     sat_counts = numel(X);
%     
%     % Satellite area / boundary area ratio
%     k = boundary(X,Y);
%     a = polyarea(X(k), Y(k));
%     sat_area_ratio = sum(satMask(:)) / a;
%     
%     % Straight-line minimum distance between each satellite and MT boundary
%     sat_dist = pdist2([X,Y], [X_tum, Y_tum]);
%     sat_dist = min(sat_dist, [], 2);
%     
%     % Statistics on minimum distances
%     sat_dist_feats = descriptive_statistics(sat_dist);
%     
%     % Rename sat distance features
%     sat_dist_feats.Properties.VariableNames = cellfun(@(x) ['sat_dist_' x], sat_dist_feats.Properties.VariableNames, 'UniformOutput', false);
% 
%     
%     % Extract Wave Features
%     
%     % Run satellite wave function
%     sat_waves = silhouette_analysis_satelitewaves(imgGeom(tumIdx).binary,imgGeom(satIdx).binary,imgGeom(satIdx).properties,strel('disk',10),0);
%     
%     % Calculate wave graph
%     [sat_waves,G,G2,wave_stats,figs] = silhouette_wave_graphs(imgGeom(tumIdx).binary, imgGeom(satIdx).binary, sat_waves, imgGeom(tumIdx).X, imgGeom(tumIdx).Y);
%     
%     % Edge features
%     wave_weights = G.Edges.Weight;
%     wave_weights = descriptive_statistics(wave_weights);
%     wave_weights.Properties.VariableNames = cellfun(@(x) ['wave_weights_' x], wave_weights.Properties.VariableNames, 'UniformOutput', false);
%     
%     % Branch features
%     branch_stats = [...
%         wave_stats.max_distance,...
%         wave_stats.average_distance_to_parent,...
%         wave_stats.average_distance_to_mt,...
%         wave_stats.linearity,...
%         wave_stats.num_conncomp];
%     stat_names = {'max_distance', 'avg_dist_to_parent', 'avg_dist_to_mt', 'linearity', 'num_conncomp'};
%     branch_feats = array2table(branch_stats, 'VariableNames', cellfun(@(x) ['wave_branches_' x], stat_names, 'UniformOutput', false));
%     
% %     % Graph clustering
% %     undirG = graph(adjacency(G), 'upper');
% %     graphLaplacian = laplacian(undirG);
% %     [graphV, graphD] = eigs(graphLaplacian, 2, 'sm');
% %     w = graphV(:,1);
%     
%     % Concat
%     wave_feats = [wave_weights branch_feats];
%     
%     % Combination Features
%     
%     % Max straight-line distance divided by max branch distance
%     
%     
%     % Max straight-line distance divided by that sat's nearest neighbor
%     
%     
%     % Combine Feature Sets Together
%     
%     graph_feats = [edge_feats tri_feats sat_dist_feats wave_feats];
%     
%     % Add a name to the row indicating what patient this came from
%     graph_feats.Properties.RowNames = {imgName};
%     
%     % Save this image's features
%     save(featsPath, 'graph_feats', '-v7.3');
%     
%     % Clean up
%     close all;
%     
%     clear DT edge_feats edgelist elengths
%     clear faces figs G G2 graph_feats iBound iedge iface
%     clear sat_area_ratio sat_counts sat_dist sat_dist_feats sat_props satBoundary
%     clear satCentroid satIdx satMask satProps
%     clear nedges newfaces nfaces imgName 
%     clear featsPath labelPath geomPath
%     % END REFACTOR ========================================================
% end
% 
% clear iGeom
    
    
%% Load Features

% Get list of features
featsList = dir(fullfile(featsDir, '*_graph_feats.mat'));
nFeats = length(featsList);

% Placeholder for the full feature table
allFeats = table;

% Cycle through extracted geometry
for iFeat = 1:nFeats
    
    fprintf(1,'Loading %s\n', featsList(iFeat).name);
    
    imgName = strrep(featsList(iFeat).name, '_graph_feats.mat', '');
    featPath = fullfile(featsDir, featsList(iFeat).name);
    
    % Load the features for this image
    load(featPath);
    
    % Edit the row names (previously had '_1' appended)
    thisName = graph_feats.Properties.RowNames;
    thisName = cellfun(@(x) strrep(x, '_1', ''), thisName, 'UniformOutput', false);
    graph_feats.Properties.RowNames = thisName;
    
    allFeats = [allFeats; graph_feats];
    
    clear imgName featPath graph_feats
end

% Attach sample info to the loaded features

% Join tables together, get rid of the label tables
allFeats = join(allFeats, labelTable, 'Keys', 'RowNames');

% Set up categorical label variable
allFeats.proggroup = categorical(allFeats.Progression>0, [true false], {'progression', 'no progression'});
allFeats.poigroup = categorical(allFeats.POI, [4 5 45 NaN], {'POI 4', 'POI 5', 'POI 4-5', 'NaN'});

% Drop non-data, non-label columns
allFeats(:,90:104) = [];

%% Modify data in table

% Remove rows that have missing values in the features data
TF = ismissing(allFeats(:,1:89));
completeFeats = allFeats(~any(TF,2),:);
% clear allFeats

% Remove rows by hand for any other reason
% Removing images EXCEPT those targeted by Shirley in WPOI 4, 5 (below)
% Everything else gets removed
wpoi4_keep = {'006a', '007a', '011a', '038b', '060a', '060c', '107a', '111a', '114a', '117a', '117b', '120c', '123a', '149a', '149b'};
wpoi5_keep = {'002b', '013a', '018a', '037a', '037b', '065a', '070a', '083a', '108a', '115e', '118b', '118e', '121b', '121d', '127c', '133a', '159a'};

% Round 2
wpoi4_keep = {wpoi4_keep{:}, '004a', '004b', '008a', '030a', '107c', '120b'};
wpoi5_keep = {wpoi5_keep{:}, '032a', '032b', '099a', '115c', '118a', '121e'};

% Round 3
wpoi4_keep = {wpoi4_keep{:}, '040c', '044a', '055c', '060b', '071a', '076a', '098a', '105d', '113a', '120a', '128a', '134b', '136a', '141a', '148a'};
wpoi5_keep = {wpoi5_keep{:}, '002a', '009b', '012a','013b', '014a', '056a', '059a', '066a', '075a', '083b', '099c', '110b', '115a', '116b', '118c', '121a', '122b', '127a', '138a', '145a', '159b', '233b'};
% 028a -- '45' label
% 107b -- '45' label
% 147b -- '45' label

nameList = completeFeats.Properties.RowNames;
nameList = cellfun(@(x) strrep(x, '_1', ''), nameList, 'UniformOutput', false);
remImages = ~(ismember(nameList, wpoi4_keep) | ismember(nameList, wpoi5_keep));
completeFeats(remImages, :) = [];
clear wpoi4_keep wpoi5_keep nameList remImages

% Identify groups
progIdx = (completeFeats.proggroup == 'progression');
noprogIdx = (completeFeats.proggroup == 'no progression');

poi4Idx = (completeFeats.poigroup == 'POI 4');
poi5Idx = (completeFeats.poigroup == 'POI 5');
poi45Idx = (completeFeats.poigroup == 'POI 4-5');

variableNames = completeFeats.Properties.VariableNames;
variableNames = cellfun(@(x) strrep(x, '_', ' '), variableNames, 'UniformOutput', false);

sampleNames = completeFeats.Properties.RowNames;
sampleNames = cellfun(@(x) strrep(x, '_1', ''), sampleNames, 'UniformOutput', false);

%% Calculate transforms of the feature data
% Scale the data: subtract mean, divide by standard deviation
scaledFeats = completeFeats;
scaledFeats{:,1:89} = zscore(scaledFeats{:,1:89});

% Binarize the data: perform density estimation with 2 means (POI)
discFeats = completeFeats;
for iFeature = 1:89
    [~, binedges] = histcounts(completeFeats{:, iFeature});
    disc = discretize(completeFeats{:, iFeature}, binedges);
    discFeats{:, iFeature} = disc;
end

%% Display / Save Histograms for Each Feature
useData = completeFeats;

for iFeature = 1:89

    if(exist(fullfile(figDir, 'feature_histograms_poi', ['Feature Values ' num2str(iFeature) '.png']), 'file'))
        continue;
    end
    
    muPos = mean(useData{poi4Idx, iFeature});
    muNeg = mean(useData{poi5Idx, iFeature});
    
    stdPos = std(useData{poi4Idx, iFeature});
    stdNeg = std(useData{poi5Idx, iFeature});
    
    x = linspace(min([useData{poi4Idx, iFeature}; useData{poi5Idx, iFeature}]), ...
        max([useData{poi4Idx, iFeature}; useData{poi5Idx, iFeature}]), 1000);
    p1 = pdf('Normal', x, muPos, stdPos); 
    p2 = pdf('Normal', x, muNeg, stdNeg);
    
    figure;
    h1 = histogram( useData{poi4Idx, iFeature}, 'Normalization', 'probability');
    hold on
    h2 = histogram( useData{poi5Idx, iFeature}, 'Normalization', 'probability');
    clist = get(gca, 'ColorOrder');
    plot(x, p1, 'Color', clist(1,:));
    plot(x, p2, 'Color', clist(2,:));
    
%     set(gca, 'XLim', [min([h1.BinLimits h2.BinLimits])-1, max([h1.BinLimits h2.BinLimits])+1]);
    
    xlabel(strrep(useData.Properties.VariableNames{iFeature}, '_', ' '));
    ylabel('Number of Images');
    legend('WPOI 4','WPOI 5');
    title(sprintf('Feature %d: %s', iFeature, variableNames{iFeature}));
    hold off
    saveas(gcf, fullfile(figDir, 'feature_histograms_poi', ['Feature Values ' num2str(iFeature)]), 'png');
    close all;
end

%% Selecting Data to Analyze
% Select the type of data to use: 
%   'scaledFeats'
%   'completeFeats', or
%   'discFeats'

useData = scaledFeats(poi4Idx | poi5Idx, :);
useSampleNames = sampleNames(poi4Idx | poi5Idx);

% Extract the feature data from the data table
useFeats = 1:89;
X = useData{:, useFeats};

% Binarize the category groups into indices
% For progression:
%   'progression' is 1
%   'no progression' is 0
y = useData.proggroup == 'progression';

% For POI:
%   'POI 5' is 1
%   'POI 4' is 0
% y = useData.poigroup == 'POI 5';


% Grab the feature names (for plotting)
featNames = useData.Properties.VariableNames;
featNames = featNames(useFeats);
featNames = cellfun(@(x) strrep(x, '_', ' '), featNames, 'UniformOutput', false);

% Identify samples we want to remove (for whatever reason)
rSamps = [];
X(rSamps,:) = [];
y(rSamps) = [];

%% Feature Selection
% Using MRMR methods
addpath(genpath(fullfile('..', '..', 'doyle_lab','code', 'matlab','feature_selection', 'mRMR_0.9_compiled')));
addpath(genpath(fullfile('..', '..', 'doyle_lab','code', 'matlab','mutual_information', 'mi')));
selfeats = mrmr_mid_d(X, uint8(y), 10);

% Print out feature names
fprintf(1, 'Feature num\tName\n');
for ifeat = 1:length(selfeats)
    fprintf(1, '%d\t%s\n', ifeat, featNames{selfeats(ifeat)});
end

% Create a logical index for the selected features
fs = zeros(1, size(X,2), 'logical');
fs(selfeats) = 1;

%% Create scatter plots
nSelected = sum(fs);

% jitter = rand(length(y),2);
jitter = 0;

switch nSelected
    case 0
        fprintf(1,'No features selected; why?\n');
    case 1
        fprintf(1,'Only one feature selected, skipping\n');
    otherwise
        if(nSelected >= 2)
            % Plot 2d
            figure; 
            scatter(X(y,selfeats(1)), X(y,selfeats(2)), 150, 'g', 'filled'); 
            hold on;
            scatter(X(~y,selfeats(1)), X(~y,selfeats(2)), 150, 'r', 'filled');
            xlabel(sprintf('%d: %s', selfeats(1), featNames{selfeats(1)})); ylabel(sprintf('%d: %s', selfeats(2), featNames{selfeats(2)}));
            title('Feature Values (Top 2 Selected)');
            colormap 'prism'
            for iSamp = 1:length(useSampleNames)
                txtmove = 0.05;
                text(X(iSamp,selfeats(1))+txtmove, X(iSamp,selfeats(2))+txtmove, strrep(useSampleNames{iSamp}, '_1', ''));
            end
%             legend('POI 5', 'POI 4');
            legend('Progression', 'No Progression');
            
        end
        if(nSelected >= 3)
            % Plot 3d
            figure; 
            scatter3(X(y,selfeats(1)), X(y,selfeats(2)), X(y,selfeats(3)), 150, 'g', 'filled'); 
            hold on;
            scatter3(X(~y,selfeats(1)), X(~y,selfeats(2)), X(~y,selfeats(3)), 150, 'r', 'filled');
            xlabel(sprintf('%d: %s', selfeats(1), featNames{selfeats(1)})); ylabel(sprintf('%d: %s', selfeats(2), featNames{selfeats(2)})); zlabel(sprintf('%d: %s', selfeats(3), featNames{selfeats(3)}));
            title('Feature Values (Top 3 Selected)');
            colormap 'prism'
            for iSamp = 1:length(useSampleNames)
                txtmove = 0.05;
                text(X(iSamp,selfeats(1))+txtmove, X(iSamp,selfeats(2))+txtmove, X(iSamp,selfeats(3))+txtmove, strrep(useSampleNames{iSamp}, '_1', ''));
            end
%             legend('POI 5', 'POI 4');
            legend('Progression', 'No Progression');
        end
end


%% T-SNE Plots
% Note 2017-10-10: Not really enough points to justify using this; results
% are not as easily discriminatory as using the discritized feature values
% (above)

if(nSelected > 3)
    Y = tsne(X(:,selfeats), 'NumDimensions', 3);
    figure; 
    scatter3(Y(y,1), Y(y,2), Y(y,3), 150, 'g', 'filled'); hold on;
    scatter3(Y(~y,1), Y(~y,2), Y(~y,3), 150, 'r', 'filled');
    title('t-SNE Embedding (3D)');
    colormap 'prism'
    for iSamp = 1:length(useSampleNames)
        txtmove = 0.1;
%         text(Y(iSamp,1)+txtmove, Y(iSamp,2)+txtmove, Y(iSamp, 3)+txtmove, strrep(useSampleNames{iSamp}, '_1', ''));
    end
    legend('Progression', 'No Progression');
    
end

%% Save rotating 3D plot?

% Set up recording parameters (optional), and record
OptionZ.FrameRate=15; OptionZ.Duration=5.5; OptionZ.Periodic=true;
CaptureFigVid([-20,10;-110,10;-190,80;-290,10;-380,10], 'WellMadeVid', OptionZ)


%% SVM Classification
X = X(:,selfeats);

% Change the range of the parameters that will be optimized
% SVMParams = hyperparameters('fitcsvm', X, y);
% SVMParams(1).Range = [1e-5, 1e5];
% SVMParams(2).Range = [1e-5, 1e5];

SVMModel = fitcsvm(X,y,'OptimizeHyperparameters','auto',...
    'HyperparameterOptimizationOptions',struct('AcquisitionFunctionName',...
    'expected-improvement-plus'));
CVSSVMModel = crossval(SVMModel, 'KFold', 3);
% CVSSVMModel = crossval(SVMModel, 'Leaveout', 'on');
classLoss = kfoldLoss(CVSSVMModel);
fprintf(1, 'SVM kfoldLoss:\t%f\n', classLoss);

sv = SVMModel.SupportVectors;
%%
d = 0.1;
% [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
%     min(X(:,2)):d:max(X(:,2)));
% xGrid = [x1Grid(:),x2Grid(:)];

[x1Grid,x2Grid,x3Grid] = meshgrid(min(X(:,1))-1:d:max(X(:,1))+1,...
    min(X(:,2))-1:d:max(X(:,2))+1,...
    min(X(:,3))-1:d:max(X(:,3))+1);
xGrid = [x1Grid(:), x2Grid(:), x3Grid(:)];

[YGrid,score] = predict(SVMModel, xGrid);
% YGrid = YGrid == 'POI 5';
% figure;
% scatter(xGrid(YGrid,1), xGrid(YGrid,2), 1, 'g', 'filled');
% hold on;
% scatter(xGrid(~YGrid,1), xGrid(~YGrid,2), 1, 'r', 'filled');
% scatter(X(y,1), X(y,2), 150, 'g', 'filled');
% scatter(X(~y,1), X(~y,2), 150, 'r', 'filled');
% plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 20);
% legend('POI 5 Region', 'POI 4 Region', 'POI 5', 'POI 4', 'Support Vector');
% hold off;

% 3D -- Create a 3D contour of the separation
fv = isosurface(x1Grid, x2Grid, x3Grid, reshape(YGrid, size(x1Grid)), 0.5);
figure;
% scatter3(xGrid(YGrid,1), xGrid(YGrid,2), xGrid(YGrid,3), 1, 'g', 'filled');
% hold on;
% scatter3(xGrid(~YGrid,1), xGrid(~YGrid,2), xGrid(~YGrid,3), 1, 'r', 'filled');
scatter3(X(y,1), X(y,2), X(y,3), 150, 'g', 'filled'); hold on;
scatter3(X(~y,1), X(~y,2), X(~y,3), 150, 'r', 'filled');
plot3(sv(:,1), sv(:,2), sv(:,3), 'ko', 'MarkerSize', 20);
p = patch(fv);
p.FaceColor = 'red';
p.FaceAlpha = 0.2;
legend('POI 5 Region', 'POI 4 Region', 'POI 5', 'POI 4', 'Support Vector');
hold off;


% %% Linear Classifier
% LinModel = fitclinear(X,y);
% 
% d = 0.02;
% [x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
%     min(X(:,2)):d:max(X(:,2)));
% xGrid = [x1Grid(:),x2Grid(:)];
% [YGrid,score] = predict(LinModel, xGrid);
% 
% figure;
% scatter(xGrid(YGrid,1), xGrid(YGrid,2), 1, 'g', 'filled');
% hold on;
% scatter(xGrid(~YGrid,1), xGrid(~YGrid,2), 1, 'r', 'filled');
% scatter(X(y,1), X(y,2), 150, 'g', 'filled');
% scatter(X(~y,1), X(~y,2), 150, 'r', 'filled');
% % plot(sv(:,1), sv(:,2), 'ko', 'MarkerSize', 20);
% legend('POI 5 Region', 'POI 4 Region', 'POI 5', 'POI 4');
% hold off;
% 
% % 
% % CVLinModel = crossval(LinModel, 'Leaveout', 'on');
% % classLoss = kfoldLoss(CVLinModel);
% % fprintf(1, 'Linear kfoldLoss:\t%f\n', classLoss);
% 
