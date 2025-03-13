%damage dimension
radius = 0.02;
d = 0.02;
offset_x = .0;
offset_y = .0;

% load damage polyshape
theta = linspace(0, 2*pi, 100);  % Angle for circle

damage_x = offset_x + radius * cos(theta);  % X coordinates of the circle
damage_y = offset_y + radius * sin(theta);  % Y coordinates of the circle

damage = polyshape(damage_x, damage_y);
