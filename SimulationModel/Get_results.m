%create plot and show plot (not needed)
pdeplot3D(ThermalModel,"ColorMapData",thermalresults.Temperature(:,end))
colormap("gray")
clim([22 55])
view(90, -90)
zoom(1)

%find all nodes in the XY plane
nodes = find(ThermalModel.Mesh.Nodes(3, :) == 0)';

%extract all temperature values for these nodes
temps = thermalresults.Temperature(nodes, :);

%find location of nodes
locs = ThermalModel.Mesh.Nodes(1:2, nodes)';

%combine node IDs, location and temperature data into one matrix
results = [nodes locs temps];

%save CSV (first column = node_ID, second column = X, third column = Y, remaining columns belong to each timestamp)
csvwrite('results.csv', results)
