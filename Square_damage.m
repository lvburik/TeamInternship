

%sample dimension
L = 0.3;
th = 0.05;

%damage dimension
radius = 0.02;
d = 0.02;
offset_x = .0;
offset_y = .0;

% Define the square vertices
sample_x = [0 1 1 0]*L-L/2;
sample_y = [0 0 1 1]*L-L/2;

% Define the damage (using parametric equations)

%%%%TO DO
damage_x = ;
damage_y = ;


% Create the square and the circle as polyshape objects
sample = polyshape(sample_x, sample_y);
damage = polyshape(damage_x, damage_y);

% Subtract the circle from the square to create the shape with the hole
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

