clear all 
close all
Set_ABCD_Paths
addpath('./geometry_processing_tutte')
load('./data/general_options.mat');
load('./data/fixer_optimizer_spec.mat');
addpath('./geometry_processing_tutte') 
optimizer_spec.use_pardiso =0;%<--Set Eigen or Pardiso


x = rand([1000,1]);
y = rand([1000,1]);
z = rand([1000,1]); %height
mesh_data = delaunayTriangulation(x,y);%


T = mesh_data.ConnectivityList;
V = [mesh_data.Points z];
uv = disk_harmonic_map_cured(T,V,-1);
fV = [uv(:,1),uv(:,2),zeros(size(uv,1),1)];%Tutte Embedding inot a disc
% if source orientation is negative, then try to reflect
% the initialization fV = [-uv(:,1),uv(:,2),zeros(size(uv,1),1)]

save('./data/random_signal_intit.mat','fV');
save('./data/random_signal_source.mat','T','V');


in_mesh     =  './data/random_signal_intit.mat';
target_mesh =  './data/random_signal_source.mat';
options.free_bnd       = 1;
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m