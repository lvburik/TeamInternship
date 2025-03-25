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
    labels = (face_IDs == labelface_id)';

    TimestampStrings = "T_" + string(tlist);

    %create a table
    DataTable = array2table([node_IDs, X, Y, labels, TemperatureData], ...
        'VariableNames',['Node_ID', 'X', 'Y', 'label', TimestampStrings]);

    %save table
    writetable(DataTable, filename)

end


