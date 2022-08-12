%% disk harmonic map
% Disk harmonic map of a 3D simply-connected surface.
%
%% Syntax
%   uv = disk_harmonic_map(face,vertex)
%
%% Description
%  face  : double array, nf x 3, connectivity of mesh
%  vertex: double array, nv x 3, vertex of mesh
%
%  uv: double array, nv x 2, uv coordinates of vertex on 2D circle domain
%
%% Contribution
%  Author : Wen Cheng Feng
%  Created: 2014/03/18
%  Revised: 2014/03/24 by Wen, add doc
%
%  Copyright 2014 Computational Geometry Group
%  Department of Mathematics, CUHK
%  http://www.math.cuhk.edu.hk/~lmlui
% 
%  Alex Naitsat, added check  

function uv = disk_harmonic_map_cured(face,vertex,orientation)
if ~exist('orientation','var')
    orientation =1;
end
nv = size(vertex,1);
nf  = size(face,1);
bd = compute_bd(face);
% bl is boundary edge length
db = vertex(bd,:) - vertex(bd([2:end,1]),:);
bl = sqrt(dot(db,db,2));
t = cumsum(bl)/sum(bl)*2*pi;
t = t([end,1:end-1]);
% use edge length to parameterize boundary
uvbd = [cos(t),sin(t)];
uv = zeros(nv,2);
uv(bd,:) = uvbd;
in = true(nv,1);
in(bd) = false;
%A = laplace_beltrami(face,vertex); %due to nuemricla issues cotangences can be NaN
A = laplace_beltrami_my(face,vertex,'CotFinite'); %my version where I tranculated weights on bad vertices

Ain = A(in,in);
rhs = -A(in,bd)*uvbd;
uvin = Ain\rhs;
uv(in,:) = uvin;


%%check if there are flips in uv map
Vec1 = [uv(face(:,2),:) - uv(face(:,1),:),zeros(nf,1)];
Vec2= [uv(face(:,1),:) - uv(face(:,3),:), zeros(nf,1)];
Normal = cross(Vec1,Vec2);
%detT  =  norms(cross(Vec1,Vec2),[],2);
is_inverted = any(Normal(:,3)<0) && any(Normal(:,3)>0);
%EPS =10^-9;
EPS =0;
is_collapsed = any(abs(Normal(:,3))<EPS);
if is_inverted || is_collapsed  %check if orientations are consitent and no collapses 
    if is_inverted
        disp('-W- Found foldovers in uv map, remapping using uniform weights');
    else
        disp('-W- Found collapses in uv map, remapping using uniform weights');
    end
    %fix flips using uniform Laplacian (same hack as in libigl based code) 
    A = laplace_beltrami_my(face,vertex,'Uniform'); %To save runtime return in prev. run non zero indices, and then reset these indices to 1 minus row sums 
    Ain = A(in,in);
    rhs = -A(in,bd)*uvbd;
    uvin = Ain\rhs;
    uv(in,:) = uvin;
    
    %check if uv attians negative orientation and fix it to positive
    Vec1 = [uv(face(1,2),:) - uv(face(1,1),:),0];
    Vec2= [uv(face(1,1),:) - uv(face(1,3),:), 0];
    Normal = cross(Vec1,Vec2);
    if orientation*Normal(3) < 0
        uv(:,2) = -uv(:,2); %reflect to invert orientation 
    end
    
end

