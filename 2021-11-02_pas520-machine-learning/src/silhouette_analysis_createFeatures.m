%% Silhouette Analysis: createFeatures
% Script to create the feature file for the silhouette project.
%
% 2017-01-12 SD

% Set up workspace
format compact;
close all; 
clear;
clc;
warning('off');

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
    fprintf(1, 'Adding base_dir to path\n');
    addpath(genpath(proj_dir));
end

% Paths for existing data
orig_dir = fullfile(data_dir, 'Original');
feat_dir = fullfile(data_dir, 'Features');
geometry_list = dir(fullfile(feat_dir, '*.mat'));

% Path for saved features
feat_path = fullfile(feat_dir, 'all_features.csv');

% If the output exists, move it to a backup first!
if(exist(feat_path, 'file'))
    fprintf(1, 'Output exists! Backing up to "all_features_backup.csv"\n');
    [status] = system(['mv ' feat_path ' ' strrep(feat_path, 'all_features.csv', 'all_features_backup.csv')]);
end

% Create a list of image names to use
base_names = cellfun(@(x) strrep(x, '_geometry.mat', ''), {geometry_list.name}, 'UniformOutput', false)';

% Load up the main .odt file and pull progression figures from it
prog_path = fullfile(data_dir, 'patient_progression.csv');
patient_data = readtable(prog_path);

% Init for debugging
iimg = 1;

% Holder for output variables (to adjust for skipped images)
all_feats = [];
kept_idx = 1;
kept_names = {};
kept_prog = [];
kept_poi = [];


%% Select an image from the list
for iimg = 1:length(geometry_list)
    fprintf(1, 'Analyzing graphs on: %s\n', base_names{iimg});
    this_feats = [];
    
    %% Load up the data

    base_name = base_names{iimg};
    geometry_path = fullfile(feat_dir, geometry_list(iimg).name);
    
    % Load geometry data
    load(geometry_path, 'sat_props', 'X_sat', 'Y_sat', 'X_tum', 'Y_tum', 'sat_mask');

    % Grab sat centroids
    centroids = [sat_props(:).Centroid];
    X = centroids(2:2:end);
    Y = centroids(1:2:end);

    % Resize
    X = X .* 0.5;
    Y = Y .* 0.5;
    X_tum = X_tum .* 0.5;
    Y_tum = Y_tum .* 0.5;
    X_sat = X_sat .* 0.5;
    Y_sat = Y_sat .* 0.5;

    sat_mask = imresize(sat_mask, 0.5);
    % Check if there are enough sats for feature extraction
    if(length(X) <= 3)
        fprintf(1, '\tImage has 3 or fewer satellites, not enough for a Delaunay. Skipping.\n');
        continue;
    end
    
    % If we're still here, add this name to the kept_names cell
    kept_names(kept_idx) = {base_name};
    kept_idx = kept_idx + 1;
    
    % Grab the prognosis using the table of patient data
    ipatient = ismember(patient_data.xCase, base_name);
    kept_prog = [kept_prog; patient_data.prog(ipatient)];
    kept_poi = [kept_poi; patient_data.poi(ipatient)];
    
    
    %% Satellite Features
    
    % Satellite counts
    this_feats = cat(2, this_feats,...
        numel(X));
    
    % Satellite area / boundary area ratio
    k = boundary(X_sat,Y_sat);
    a = polyarea(X_sat(k), Y_sat(k));
    area_ratio = sum(sat_mask(:)) / a;
    
    this_feats = cat(2, this_feats,...
        area_ratio);
    
    % Distances between satellite and MT
    sat_dist = pdist2([X',Y'], [X_tum, Y_tum]);
    sat_dist = min(sat_dist, [], 2);
    
    % Descriptive statistics
    this_feats = cat(2, this_feats, ...
        descriptive_statistics(sat_dist));

    %% Triangulation
    DT = delaunayTriangulation(Y',X');
    
    % Break down into a regular triangulation, crop out edges crossing MT
    faces       = DT.ConnectivityList;
    vertcoords  = DT.Points;
    edgelist    = edges(DT);
    newfaces    = faces;
    
    % Cycle through the edge list
    for iedge = 1:size(edgelist,1)
        
        % Pull coordinates for this edge
        X_edge = [vertcoords(edgelist(iedge,1),2), vertcoords(edgelist(iedge,2),2)];
        Y_edge = [vertcoords(edgelist(iedge,1),1), vertcoords(edgelist(iedge,2),1)];
        
        % Check for intersections
        [xi,yi] = intersections(X_tum,Y_tum,X_edge,Y_edge);
        
        if(~isempty(xi))
            newfaces(sum(ismember(newfaces, edgelist(iedge,:)),2) == 2, :) = [];
        end
        
    end
    
    % Re-create triangulation (no longer Delaunay)
    DT = triangulation(newfaces, vertcoords);
    
    %% Delaunay Features
    faces = DT.ConnectivityList;
    vcoords = DT.Points;
    edgelist    = edges(DT);
    nedges = size(edgelist,1);
    nvert = max(faces(:));
    nfaces = size(faces,1);
    A = zeros(nvert);
    
    % Edge weights
    elengths = zeros(1, nedges);
    for iedge = 1:nedges
        elengths(iedge) = sqrt((DT.Points(edgelist(iedge,1),1) - DT.Points(edgelist(iedge,2),1))^2 + (DT.Points(edgelist(iedge,1),2) - DT.Points(edgelist(iedge,2),2))^2);
    end
    
    % Triangle areas
    tareas = zeros(1, nfaces);
    for iface = 1:nfaces
        tareas(iface) = polyarea([DT.Points(faces(iface,1),1), DT.Points(faces(iface,2),1), DT.Points(faces(iface,3),1)], ...
            [DT.Points(faces(iface,1),2), DT.Points(faces(iface,2),2), DT.Points(faces(iface,3),2)]);
    end
    
    % DT Features
    dtfeatures = [descriptive_statistics(elengths), descriptive_statistics(tareas)];
    
    this_feats = cat(2, this_feats, dtfeatures);
    
    % MST
    
    
    all_feats = cat(1, all_feats, this_feats);
%     % Write out this feature vector
%     dlmwrite(save_path, this_feats, '-append');
%     A = zeros(nvert);
% 
%     for i = 1:nface
%         % Put the distance between points into the graph adjacency matrix
%         A(faces(i,1),faces(i,2)) = sqrt((DT.Points(faces(i,1),1) - DT.Points(faces(i,2),1))^2 + (DT.Points(faces(i,1),2) - DT.Points(faces(i,2),2))^2);
%         A(faces(i,2),faces(i,3)) = sqrt((DT.Points(faces(i,2),1) - DT.Points(faces(i,3),1))^2 + (DT.Points(faces(i,2),2) - DT.Points(faces(i,3),2))^2);
%         A(faces(i,3),faces(i,1)) = sqrt((DT.Points(faces(i,3),1) - DT.Points(faces(i,1),1))^2 + (DT.Points(faces(i,3),2) - DT.Points(faces(i,1),2))^2);
%         %  make sure that all edges are symmetric
%         A(faces(i,2),faces(i,1)) = sqrt((DT.Points(faces(i,2),1) - DT.Points(faces(i,1),1))^2 + (DT.Points(faces(i,2),2) - DT.Points(faces(i,1),2))^2);
%         A(faces(i,3),faces(i,2)) = sqrt((DT.Points(faces(i,3),1) - DT.Points(faces(i,2),1))^2 + (DT.Points(faces(i,3),2) - DT.Points(faces(i,2),2))^2);
%         A(faces(i,1),faces(i,3)) = sqrt((DT.Points(faces(i,1),1) - DT.Points(faces(i,3),1))^2 + (DT.Points(faces(i,1),2) - DT.Points(faces(i,3),2))^2);
%     end
%     
%     
%     %% Convert to graph
%     % Create adjacency graph
%     faces = DT.ConnectivityList;
%     vcoords = DT.Points;
%     nvert = max(faces(:));
%     nface = size(faces,1);
%     A = zeros(nvert);
% 
%     for i = 1:nface
%         % Put the distance between points into the graph adjacency matrix
%         A(faces(i,1),faces(i,2)) = sqrt((DT.Points(faces(i,1),1) - DT.Points(faces(i,2),1))^2 + (DT.Points(faces(i,1),2) - DT.Points(faces(i,2),2))^2);
%         A(faces(i,2),faces(i,3)) = sqrt((DT.Points(faces(i,2),1) - DT.Points(faces(i,3),1))^2 + (DT.Points(faces(i,2),2) - DT.Points(faces(i,3),2))^2);
%         A(faces(i,3),faces(i,1)) = sqrt((DT.Points(faces(i,3),1) - DT.Points(faces(i,1),1))^2 + (DT.Points(faces(i,3),2) - DT.Points(faces(i,1),2))^2);
%         %  make sure that all edges are symmetric
%         A(faces(i,2),faces(i,1)) = sqrt((DT.Points(faces(i,2),1) - DT.Points(faces(i,1),1))^2 + (DT.Points(faces(i,2),2) - DT.Points(faces(i,1),2))^2);
%         A(faces(i,3),faces(i,2)) = sqrt((DT.Points(faces(i,3),1) - DT.Points(faces(i,2),1))^2 + (DT.Points(faces(i,3),2) - DT.Points(faces(i,2),2))^2);
%         A(faces(i,1),faces(i,3)) = sqrt((DT.Points(faces(i,1),1) - DT.Points(faces(i,3),1))^2 + (DT.Points(faces(i,1),2) - DT.Points(faces(i,3),2))^2);
%     end
%     G = graph(A, 'OmitSelfLoops');
% 
%     % Destroy edges crossing MT  
%     rmedges = []; irmedges = [];
%     for iedge = 1:height(G.Edges)
%         edgepoly = table2array(G.Edges(iedge,1));
%         
%     
%         % Set up the polygon coordinates
%         nodecoords = [vcoords(edgepoly(:),:)];
%         
%         % Check for intersections
%         [xi,yi] = intersections(X_tum,Y_tum,nodecoords(:,2),nodecoords(:,1));
%         
%         if(~isempty(xi))
%             rmedges = [rmedges; edgepoly];
%             irmedges = [irmedges; iedge];
%         end
%     end
%     
%     for iedge = 1:size(rmedges,1)
%         G = rmedge(G,rmedges(iedge,1), rmedges(iedge,2));
%     end
%     
%     % It's not a Delaunay anymore, but...
%     faces(irmedges,:) = [];
%     DT = triangulation(faces, vcoords);
%     
% %     %% Check that it worked
% % 	f_original = plot_overlay(img, sat_mask,...
% %         'LColor', 'k', 'PatchColor', 'k');
% %     f_original = plot_overlay(img, tum_mask, 'f', f_original,...
% %         'LColor', 'g', 'PatchColor', 'g');
% %     figure(f_original);
% %     hold on;
% %     LWidths = 5*G.Edges.Weight/max(G.Edges.Weight);
% %     plot(G, '-k', 'XData', vcoords(:,1), 'YData', vcoords(:,2),...
% %         'NodeLabel', '', 'Marker', '.', 'MarkerSize', 10,...
% %         'LineWidth', LWidths, 'EdgeAlpha', 1.0, 'EdgeLabel', G.Edges.Weight);
% %     % There should be no lines that cross the main tumor!
%     
%     %% Calculate features from G
%     
%     % First identify connected components
%     gbins = conncomp(G);
%     nbins = max(gbins);
%     gfeats = [];
%     
%     % If nbins > 1, then pick the largest bin and analyze on that
%     if nbins > 1
%         vertex_idx = find(gbins == mode(gbins));
%         H = subgraph(G,vertex_idx);
%     else
%         H = G;
%     end
%     
%     nnodes = numnodes(H);
%     
%     % Distance features
%     gdist = distances(H);
%     
%     gfeats = cat(1, gfeats,...
%         max(gdist(:)));
%     
%     % Get an unordered "bag of distances"
%     gd = tril(gdist);
%     gd = gd(gd~=0);
%     
%     % Calculate stats from this list
%     gfeats = cat(2, gfeats,...
%         descriptive_statistics(gd));
%     
%     % Degree Features
%     gdegrees = degree(H);
%     gfeats = cat(2, gfeats,...
%         descriptive_statistics(gdegrees));
%     
%     % Flow Features
%     gflow = zeros(nnodes);
%     for ni = 1:nnodes
%         for nj = ni+1:nnodes
%             gflow(ni,nj) = maxflow(H, ni, nj);
%         end
%     end
%     gflow = gflow(gflow~=0);
%     
%     gfeats = cat(2, gfeats,...
%         descriptive_statistics(gflow));
%     
%     % Previous: Calculate features for each subgraph
% %     for ibin = 1:nbins
% %         gfeats = [];
% %         
% %         % Create a subgraph of this bin
% %         H = subgraph(G, find(gbins==ibin));
% %         nnodes = numnodes(H);
% %         
% %         % Skip this subgraph if it has fewer than 3 points
% %         if(nnodes < 3)
% %             continue;
% %         end
% %         
% %         % Distance features
% %         gdist = distances(H);
% %         
% %         gfeats = cat(1, gfeats,...
% %             max(gdist(:)));
% %         
% %         % Get an unordered "bag of distances"
% %         gd = tril(gdist);
% %         gd = gd(gd~=0);
% %         
% %         % Calculate stats from this list
% %         gfeats = cat(2, gfeats,...
% %             descriptive_statistics(gd));
% %         
% %         % Degree Features
% %         gdegrees = degree(H);
% %         gfeats = cat(2, gfeats,...
% %             descriptive_statistics(gdegrees));
% %         
% %         % Flow Features
% %         gflow = zeros(nnodes);
% %         for ni = 1:nnodes
% %             for nj = ni+1:nnodes
% %                 gflow(ni,nj) = maxflow(H, ni, nj);
% %             end
% %         end
% %         gflow = gflow(gflow~=0);
% %         
% %         gfeats = cat(2, gfeats,...
% %             descriptive_statistics(gflow));
% %         
% %         gfeatures = cat(1, gfeatures, gfeats);
% %     end
    
end

%% Create table output of features
feat_table = array2table(all_feats);
feat_table.names = kept_names';
feat_table.prog = kept_prog;
feat_table.poi = kept_poi;

% Rearrange
feat_table = feat_table(:, [end, end-1, end-2, 1:end-3]);

% Save!
writetable(feat_table, feat_path);
