function dVI = compute_boundary(mesh)
dVI = compute_bd(mesh.T);
dVI= [dVI(2:end); dVI(1)];
end