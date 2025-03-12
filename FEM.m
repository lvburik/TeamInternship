clear

%%define the model
ThermalModel = createpde('thermal', 'transient');

%run the geometry script to create the sample
Circular_damage

%plot the geometry
pdegplot(ThermalModel,'FaceLabels','on','FaceAlpha',0.5)

%material properties
thermalProperties(ThermalModel,'ThermalConductivity',1,...
                               'MassDensity',1,...
                               'SpecificHeat',1);

%boundary conditions:
thermalBC(ThermalModel,'Face',1:ThermalModel.Geometry.NumFaces,'ConvectionCoefficient',5, 'AmbientTemperature',22);


heatFlux = @(x, y) x + y;
thermalBC(ThermalModel,'Face',[1, 2],'HeatFlux',@(location, state) heatFlux(location.x, location.y));

%initial conditions
thermalIC(ThermalModel,0);

%generate and display mesh
generateMesh(ThermalModel, Hmax = 0.01);
figure
pdemesh(ThermalModel)

%%solve
tlist = [0 0.1];

thermalresults = solve(ThermalModel,tlist);




