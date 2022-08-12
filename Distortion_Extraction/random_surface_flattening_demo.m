clear all 
close all
Set_ABCD_Paths
load('./data/general_options.mat');
load('./data/fixer_optimizer_spec.mat');
optimizer_spec.use_pardiso =0;%<--Set Eigen or Pardiso


x = rand([20,1]);
y = rand([20,1]);
z = rand([20,1]); %height
mesh_data = delaunayTriangulation(x,y);%


T = mesh_data.ConnectivityList;
V = [mesh_data.Points z];
fV = [mesh_data.Points zeros(size(V,1),1)]; %initialization

save('./data/random_signal_intit.mat','fV');
save('./data/random_signal_source.mat','T','V');


in_mesh     =  './data/random_signal_intit.mat';
target_mesh =  './data/random_signal_source.mat';
options.free_bnd       = 1;
options.is_uv_mesh     = 1;
run ABCD_MexWrapperScript.m

[tmesh.is_bnd_v, tmesh.vv_mesh_edges, tmesh.vert_neighbors, tmesh.vert_simplices, energy0, elemnent_dist0, sing_val_final]  = ... 
GetMeshData_mex(tmesh.T,tmesh.V(:,1:tmesh.source_dim),tmesh.tn, tmesh.vn, 0,fV,optimizer_spec);
