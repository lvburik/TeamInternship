function g = Load_Geometry(ThermalModel, shape)
    %sample dimension
    L = 0.3;
    th = 0.05;
    d = 0.02;
    
    % Define the square vertices
    sample_x = [0 1 1 0]*L-L/2;
    sample_y = [0 0 1 1]*L-L/2;
    
    % Create the sample as polyshape object
    sample = polyshape(sample_x, sample_y);
    
    % load the damage as a polyshape object
    switch shape
        case 'circle'
            [damage, inner_edges] = Circular_damage;
        case 'square'
            [damage, inner_edges] = Square_damage;
        case 'triangle'
            [damage, inner_edges] = Triangle_damage;
    end
    
    % Subtract the damage from the sample to create the shape with the hole
    shape = subtract(sample, damage);
    shape = simplify(shape);
    figure
    plot(shape)
    
    
    TR = triangulation(shape);
    
    stlwrite(TR, 'test.stl')
    
    model = ThermalModel;
    g = importGeometry(model,'test.stl');
    
    % Check geometry
    pdegplot(g, 'EdgeLabels', 'on', 'FaceAlpha', 0.5);
    
    %add a face in the defect region
    addFace(g, inner_edges);
    %extrude undamaged plate upto defect depth
    extrude(g, d);
    %extrude damaged plate until final thickness
    extrude(g, 1, th-d);
    
    mergeCells(g,[1 2]);
    mergeCells(g,[1 2]);
end




