%sample dimension
L = 0.3;
th = 0.05;

% Define the square vertices
sample_x = [0 1 1 0]*L-L/2;
sample_y = [0 0 1 1]*L-L/2;

% Create the sample as polyshape object
sample = polyshape(sample_x, sample_y);

% load the damage as a polyshape object
Circular_damage;

% Subtract the damage from the sample to create the shape with the hole
shape = subtract(sample, damage);
plot(shape)
TR = triangulation(shape);

stlwrite(TR, 'test.stl')

model = ThermalModel;
g = importGeometry(model,'test.stl');


%add a face in the defect region
addFace(g, 5:g.NumEdges);
%extrude undamaged plate upto defect depth
extrude(g, d);
%extrude damaged plate until final thickness
extrude(g, 3, th-d);

mergeCells(g,[1 2]);
mergeCells(g,[1 2]);

