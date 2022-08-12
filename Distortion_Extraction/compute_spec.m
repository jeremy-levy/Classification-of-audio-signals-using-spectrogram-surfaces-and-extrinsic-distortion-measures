function compute_spec(baseFileName, base_path_save, data_dir)

%% Compute the spectrogram

filename = strcat(data_dir, baseFileName);
[signal_1D,fs] = audioread(filename);

signal_1D = wdenoise(signal_1D,9,'Wavelet','sym4');

[s, w, t] = melSpectrogram(signal_1D,fs, ...
                               'WindowLength',2048,...
                               'OverlapLength',1024, ...
                               'FFTLength',8192, ...
                               'NumBands',150, ...
                               'FrequencyRange', [0, 2e3]);
% figure;
% surf(t,w,s,'EdgeColor','none');
% xlabel('Time (s)')
% ylabel('Frequency (Hz)')
% view([0,90])
% axis([t(1) t(end) w(1) w(end)])

s = 10*log10(s+eps);
w = w(ones(1,size(t, 1)), :);
t = t(:, ones(1,size(w, 2)));

t = t(:);
w = w(:);
s = abs(s(:));

t(s < quantile(s, 0.05)) = [];
w(s < quantile(s, 0.05)) = [];
s(s < quantile(s, 0.05)) = [];

% Run ABCD algorithm
mesh_data = delaunayTriangulation(t,w);

T = mesh_data.ConnectivityList;
V = [mesh_data.Points s];
uv = disk_harmonic_map_cured(T,V,-1);
% fV = [mesh_data.Points s_vector]; %initialization
fV = [uv(:,1),uv(:,2),zeros(size(uv,1),1)];%Tutte Embedding inot a disc

load(strcat(base_path, 'Distortion_Extraction/save_for_ABCD/general_options.mat'));
load(strcat(base_path, 'Distortion_Extraction/save_for_ABCD/fixer_optimizer_spec.mat'));
optimizer_spec.use_pardiso =0;

options.free_bnd       = 1; %fixed boundary 
options.fixed_vertices = [];%no additional anchors 
options.is_uv_mesh     = 1; %planar mesh
options.draw_mesh = false;

%% loading source/ in mesh

tmesh.T = T;
tmesh.V = V;
d = size(tmesh.T,2);
if d==3 && isfield(options,'is_uv_mesh') && ~options.is_uv_mesh
     tmesh.source_dim = 2; %remove zero column from planar mesh
else
     tmesh.source_dim = 3;
end
% disp(['Loading source mesh ' num2str(toc) ' sec.']);

%% loading target mesh

tmesh.fV = fV;
if size(tmesh.fV,2) > d-1
    tmesh.fV = tmesh.fV(:,1:d-1);
end

%%
tmesh.vn = size(tmesh.V,1);
tmesh.tn = size(tmesh.T,1);
is_1_based_index =0; %0-based index for C++
[tmesh.is_bnd_v, tmesh.vv_mesh_edges, tmesh.vert_neighbors, tmesh.vert_simplices, energy0, elemnent_dist0, sing_val0]  = ...
             GetMeshData_mex(tmesh.T,tmesh.V(:,1:tmesh.source_dim),tmesh.tn, tmesh.vn, 0,tmesh.fV,optimizer_spec);

num_of_flipped   =  sum(prod(sign(sing_val0),2)<0);
num_of_collapsed =  sum(abs(sing_val0(:,end))<10^-8);

%% mesh positional constraints
tmesh.is_fixed_vert = zeros(tmesh.vn,1);
if isfield(options,'fixed_vertices')  
    tmesh.is_fixed_vert(options.fixed_vertices) =1;
end

if ~options.free_bnd %check for fixed boundary
    tmesh.is_fixed_vert   =   tmesh.is_fixed_vert | tmesh.is_bnd_v;  
end
disp(['vertices / anchors # :'   num2str(tmesh.vn) ' / ' num2str(sum(tmesh.is_fixed_vert))  ...
      '; simplices / invalids # :' num2str(tmesh.tn) ' / ' num2str(num_of_flipped+num_of_collapsed) ]);
%% optimization
fV = ABCD_MexWrapperFunction(tmesh,fixer_spec,optimizer_spec,options);
%% visualization
if options.draw_mesh
    if d==3
        figure; trimesh(tmesh.T,tmesh.V(:,1),tmesh.V(:,2),tmesh.V(:,3),'facecolor',[0.7 0.7 0.7], 'edgecolor','black'); view([0 0 1]); axis equal
        title('source');

        log_element_dist0 = log(elemnent_dist0+1);
        [~, ~, ~,~, ~, elemnent_dist, sing_val]  = GetMeshData_mex(tmesh.T,tmesh.V(:,1:tmesh.source_dim), ...
                                                                   tmesh.tn, tmesh.vn, 0,fV,optimizer_spec);
        log_element_dist = log(elemnent_dist+1);

        drawOpt.color_range = [min([log_element_dist; log_element_dist0]) ...
            max([log_element_dist; log_element_dist0]) ];

        figure;
        plot_triangle_mesh(tmesh.T,tmesh.fV, log(elemnent_dist0+0.1), sing_val0, drawOpt, tmesh.is_fixed_vert);
        title('initial target');
            

    figure;
    plot_triangle_mesh(tmesh.T,fV, log(elemnent_dist+0.1), sing_val, drawOpt, tmesh.is_fixed_vert);
    title('final target');
    else
        disp('Drawing tetrahedral meshes ...');
        VV = (1:tmesh.vn)';
        VV(logical(tmesh.is_bnd_v)) =0;
        is_bnd_tet=any(VV(T),2);
        is_fixed=logical(tmesh.is_fixed_vert);
        
        is_flliped  = prod(sign(sing_val0),2)<0;
        figure; tetramesh(T(is_bnd_tet,:),tmesh.fV, ... 
                          log(elemnent_dist0(is_bnd_tet)+0.1),'FaceAlpha',0.3);
        hold on; tetramesh(T(is_flliped,:),tmesh.fV,'FaceAlpha',0.7,'FaceColor','y');
        scatter3(tmesh.fV(is_fixed,1),tmesh.fV(is_fixed,2),tmesh.fV(is_fixed,3), ...
                 100,'blue','filled','MarkerEdgeColor','black');
        title('initial target');
        if isfield(drawOpt,'col_map')
            colormap(drawOpt.col_map);
        else
            colormap cool
        end
        
        figure; tetramesh(tmesh.T(is_bnd_tet,:),fV,'FaceAlpha',0.3,'FaceColor',[.7 .7 .7]);
        hold on;

        scatter3(fV(is_fixed,1),fV(is_fixed,2),fV(is_fixed,3), ...
                 100,'blue','filled','MarkerEdgeColor','black');
        title('final target');
    end
end
%% 
if  ~optimizer_spec.is_parallel  || ~fixer_spec.is_parallel
    disp( ['-W- Slow non-parallel mode!' newline ...
           '    Set optimizer_spac.is_parallel=1 and fixer_spec.is_parallel=1 to optimize  blocks in parallel' newline ...
           '   (also check parallel_energy flag)']);
end
if ~optimizer_spec.use_pardiso
    disp( ['-W- Slow Eigen linear solver!' newline ...
           '    Set optimizer_spec.use_pardiso=1 to use faster Pardiso solver']);
end


[tmesh.is_bnd_v, tmesh.vv_mesh_edges, tmesh.vert_neighbors, tmesh.vert_simplices, energy0, elemnent_dist0, sing_val_final]  = ... 
GetMeshData_mex(tmesh.T,tmesh.V(:,1:tmesh.source_dim),tmesh.tn, tmesh.vn, 0,fV,optimizer_spec);

area_triangles = calc_area_triangles(tmesh.T, tmesh.V);
frequencies_triangles = calc_frequencies(tmesh.T, tmesh.V);

csvwrite(strcat(base_path_save, 'energy/', baseFileName(1: end-4), '.csv'),energy0)
csvwrite(strcat(base_path_save, 'dist/', baseFileName(1: end-4), '.csv'),elemnent_dist0)
csvwrite(strcat(base_path_save, 'sing_val/', baseFileName(1: end-4), '.csv'),sing_val_final)
csvwrite(strcat(base_path_save, 'area/', baseFileName(1: end-4), '.csv'),area_triangles)
csvwrite(strcat(base_path_save, 'freq/', baseFileName(1: end-4), '.csv'),frequencies_triangles)
end