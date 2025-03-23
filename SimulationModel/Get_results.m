%%process results into labeled csv file

function Get_results(ThermalModel, tlist, ThermalResults, filename)

    %find all nodes in the XY plane
    node_IDs = find(ThermalModel.Mesh.Nodes(3, :) == 0.02)';

    %extract all temperature values for these nodes
    TemperatureData = ThermalResults.Temperature(node_IDs, :);

    %find location of nodes
    locs = ThermalModel.Mesh.Nodes(1:2, node_IDs)';
    X = locs(:, 1); Y = locs(:, 2);

    TimestampStrings = "T_" + string(tlist);

    %create a table
    DataTable = array2table([node_IDs, X, Y, TemperatureData], ...
        'VariableNames',['Node_ID', 'X', 'Y', TimestampStrings]);

    %save table
    writetable(DataTable, filename)

    figure;
    scatter(X, Y, 10, 'filled'); % Small dots, filled markers
    title('Node Locations in XY Plane');
    axis equal; % Keep aspect ratio
    grid on;

end


