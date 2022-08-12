function [frequencies_triangles] = calc_frequencies(T, V)

frequencies_triangles = zeros(size(T, 1), 1);

for k = 1:size(T, 1)
    P1 = V(T(k, 1),:);
    P2 = V(T(k, 2),:);
    P3 = V(T(k, 3),:);
    
    frequencies_triangles(k) = (P1(1) + P2(1) + P3(1)) / 3;
end

end