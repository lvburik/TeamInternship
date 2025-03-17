
pdeplot3D(ThermalModel,"ColorMapData",thermalresults.Temperature(:,end))
colormap("gray")
clim([22 55])
view(90, -90)
zoom(1)


nodes = find(ThermalModel.Mesh.Nodes(3, :) == 0)';
temps = thermalresults.Temperature(nodes, :);
locs = ThermalModel.Mesh.Nodes(1:2, nodes)';


results = [nodes locs temps];

csvwrite('results.csv', results)