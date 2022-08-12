%Run the demo from 'lung_respiratory_master\Distortion_Extraction\adaptive_sampling'
clear all
close all

addpath('../geometry_processing_tutte');
ABCD_path = '../../../ABCD_Algorithm/';
data_path = './';


%% Adaptive sampling of a surface according to its  mean curvature 
load([data_path '/peak_surface.mat'])%loading T,V matrices of a source mesh
%subdivision params (should be tunned for real data)
opt.max_subdivision_iter =3;  %maximal number of iterations 
opt.curve_thresholds =  0.8*( 2.^((1:opt.max_subdivision_iter) -1)); %high mean curvature
opt.draw=1; %visualize subdivisions
[T,V] = curvature_based_sampling(T,V,opt); %Resampling 
save([data_path '/peak_surface_resampled.mat'],'T','V');%saving resampled mesh

%% parametrization of original  VS  resampled surfaces
addpath([ABCD_path './visualization']);
addpath([ABCD_path './wrapper']);
addpath(genpath([ABCD_path './mexed_solver']));

load([ABCD_path './data/general_options.mat']);
load([ABCD_path './data/fixer_optimizer_spec.mat']);
optimizer_spec.use_pardiso =0;%<--Set Eigen or Pardiso

%%% uniformly sampled surface
load([data_path '/peak_surface.mat']);
uv = disk_harmonic_map_cured(T,V,-1);
fV = [uv(:,1),uv(:,2),zeros(size(uv,1),1)];%Tutte Embedding into a disc
options.free_bnd       = 1;
options.is_uv_mesh     = 1;

in_mesh     =  [data_path 'peak_surface_init.mat'];
target_mesh =  [data_path 'peak_surface.mat'];
save(in_mesh,'fV');
run ABCD_MexWrapperScript.m

%%% adaptively  sampled surface
load([data_path '/peak_surface_resampled.mat']);
uv = disk_harmonic_map_cured(T,V,-1);
fV = [uv(:,1),uv(:,2),zeros(size(uv,1),1)];%Tutte Embedding into a disc
options.free_bnd       = 1;
options.is_uv_mesh     = 1;

in_mesh     =  [data_path 'peak_surface_resampled_init.mat'];
target_mesh =  [data_path 'peak_surface_resampled.mat'];
save(in_mesh,'fV');
run ABCD_MexWrapperScript.m
