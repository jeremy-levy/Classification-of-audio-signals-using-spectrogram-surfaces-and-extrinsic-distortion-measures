%% Adaptive sampling based on mean curvature
% input:  T,V - source mesh
% output: T,V- mesh after adaptive sampling/ subdivision
% opt - options with fields:
%       max_subdivision_iter
%       curve_thresholds
%       draw  = 0,1
function [T,V] = curvature_based_sampling(T_orig,V_orig,opt)
if ~isfield(opt,'draw')
    opt.draw =0;
end
iter=1;
T=T_orig;
V=V_orig;

while iter <= opt.max_subdivision_iter
    [gauss_curv_vertex mean_curv_vertex]= curvatures(V(:,1),V(:,2),V(:,3),T);
    
    if opt.draw
        figure;
        trimesh(T,V(:,1),V(:,2),V(:,3),'FaceVertexCData',mean_curv_vertex,'FaceColor','interp','EdgeColor','black');
        axis equal
        title(['iteration' num2str(iter)]);
        colorbar
    end
    triangle_mean_curve= mean(abs(mean_curv_vertex(T)),2);
    
    
    I_large_curve = find(triangle_mean_curve > opt.curve_thresholds(iter));
    if isempty(I_large_curve)
        break;
    end
    % subdivide triangles with high curvature
    centroid_bary = repmat(1/3, length(I_large_curve),3);
    TR = triangulation(T,V);
    centroid_cart = barycentricToCartesian(TR, I_large_curve,centroid_bary);
    if opt.draw
        hold on;
        scatter3(centroid_cart(:,1),centroid_cart(:,2),centroid_cart(:,3),'r','fill');
    end
    
    %re-triangulate
    V = [V; centroid_cart];
    mesh_data_new = delaunayTriangulation(V(:,1),V(:,2));
    T = mesh_data_new.ConnectivityList;
    
    iter =iter+1;
end
