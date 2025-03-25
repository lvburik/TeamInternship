function [g, labelface_ID] = Multiple_defects(model, number_of_defects)
    % Sample dimensions
    L = 0.3;  % Length
    th = 0.05; % Thickness (not used in 2D)
    d = 0.05*th+0.9*th*rand();  % Defect depth

    % Define the sample as a rectangle
    S = [3 4 -L/2 L/2 L/2 -L/2 -L/2 -L/2 L/2 L/2];  

    % Initialize defect storage
    defects = cell(1, number_of_defects);

    shapelist = ["ci", "sq", "tr"];
    for ii = 1:number_of_defects
        shape = shapelist(randi(3));
    
        size = 5*L/11 * rand() + L/32;
        offset_X = (L/2 - size) * (2*rand() - 1);  % Ensure shape stays inside the rectangle
        offset_Y = (L/2 - size) * (2*rand() - 1); 

        switch shape
            case "ci"           
                % Define circle geometry [1 x_center y_center radius]
                defect = [1 offset_X offset_Y size/2];  
                defect = [defect, zeros(1, 6)];  % Pad to match the expected format
            case "sq"       
                defect = [3 4 -size/2+offset_X size/2+offset_X size/2+offset_X -size/2+offset_X size/2+offset_Y size/2+offset_Y -size/2+offset_Y -size/2+offset_Y];
            case "tr"
                % Define a triangle with three points
                defect = [2 3 offset_X offset_X-size/2 offset_X+size/2 offset_Y+size*sqrt(3)/4  offset_Y-size*sqrt(3)/4  offset_Y-size*sqrt(3)/4 0 0];
            case "poly"
                %to be implemented?
            case "el"
                %to be implemented?
        end
        % Store defect in cell array
        defects{ii} = defect;
    end

    % Convert defect cells to matrix form
    defectsMat = cell2mat(defects');  

    % Combine sample and defects
    gd = [S; defectsMat]';

    % Create decomposed geometry
    dl = decsg(gd);

    % Assign the geometry to the PDE model
    g = geometryFromEdges(model, dl);

    %extrude undamaged plate upto defect depth
    f = extrude(g, d);

    model.Geometry = f;

    %extrude damaged plate until final thickness
    extrude(f, nearestFace(f, [-0.999*L/2 0.999*L/2 d]), th-d);
    
    %merge cells into one
    while f.NumCells > 1
        cellfaces = cellFaces(f, 1, "internal");
        for jj = 2:f.NumCells
            for face = cellfaces
                if ismember(face, cellFaces(f,jj, "internal"))
                    neighbor_ID = jj;
                end
            end
        end

        mergeCells(f, [1, neighbor_ID]);
    end
    
    %assign final geometry to model
    model.Geometry = f;

    %find the face number for undamaged profile
    labelface_ID = nearestFace(g, [-0.999*L/2 0.999*L/2]);
end



