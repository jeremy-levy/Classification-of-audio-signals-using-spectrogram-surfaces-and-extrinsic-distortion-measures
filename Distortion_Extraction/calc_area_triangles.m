function [area_triangles] = calc_area_triangles(T, V)

area_triangles = zeros(size(T, 1), 1);

for k = 1:size(T, 1)
    P1 = V(T(k, 1),:);
    P2 = V(T(k, 2),:);
    P3 = V(T(k, 3),:);
    
    area_triangles(k) = 1/2*norm(cross(P2-P1,P3-P1));
end

end