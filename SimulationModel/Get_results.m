%% process results into labeled csv file


function Get_results(ThermalModel, tlist, ThermalResults, filename, labelface_id, g)

    %find all nodes in the XY plane

    node_IDs = find(ThermalModel.Mesh.Nodes(3, :) == 0)';

    %extract all temperature values for these nodes
    TemperatureData = ThermalResults.Temperature(node_IDs, :);

    %find location of nodes
    locs = ThermalModel.Mesh.Nodes(1:2, node_IDs)';
    X = locs(:, 1); Y = locs(:, 2);
    
    %find labels for damaged nodes
    face_IDs = nearestFace(g, locs);
    labels = 1 - double(face_IDs == labelface_id)';

    TimestampStrings = "T_" + string(tlist);

    %create a table
    DataTable = array2table([node_IDs, X, Y, labels, TemperatureData], ...
        'VariableNames',['Node_ID', 'X', 'Y', 'label', TimestampStrings]);

    %save table
    writetable(DataTable, filename)

    figure;
    scatter(X, Y, 10, 'filled'); % Small dots, filled markers
    title('Node Locations in XY Plane');
    axis equal; % Keep aspect ratio
    grid on;

end


