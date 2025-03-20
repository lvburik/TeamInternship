clear
%% simulation parameters
tlist = 0:2:10;
filename = 'results.csv';

%environment properties
Ambient_T = 22; %[°C]

%material properties
conductivity = 0.38;% [W/m*°K]
density = 980; %[kg/m^3]
specific_heat = 1200; %[J/°K]

convex_coeff = 40; %[W/m*°K]
emis_coeff = 0.9;

%% set up the simulation

% define the model
ThermalModel = createpde('thermal', 'transient');

%run the geometry script to create the sample
Load_Geometry(ThermalModel, 'triangle');

%plot the geometry
figure
pdegplot(ThermalModel,'FaceLabels','on','FaceAlpha',0.5);


%material properties
thermalProperties(ThermalModel,'ThermalConductivity',conductivity,...
                               'MassDensity',density,...
                               'SpecificHeat',specific_heat);

%boundary conditions:
thermalBC(ThermalModel,'Face',1:ThermalModel.Geometry.NumFaces,'ConvectionCoefficient',convex_coeff, 'AmbientTemperature',Ambient_T);
ThermalModel.StefanBoltzmannConstant = 5.670373E-8;
thermalBC(ThermalModel, 'Face', 1:ThermalModel.Geometry.NumFaces, 'Emissivity',emis_coeff, 'AmbientTemperature', Ambient_T);


thermalBC(ThermalModel,'Face',[1, 2],'HeatFlux',@heatFluxFunction);

%initial conditions
thermalIC(ThermalModel,Ambient_T);

%generate and display mesh

generateMesh(ThermalModel, Hmax = 0.007);

figure
pdemesh(ThermalModel)

%% solve the model

thermalresults = solve(ThermalModel,tlist);

function q = heatFluxFunction(region, state)
    x = region.x; % X-coordinates of the surface
    y = region.y; % Y-coordinates of the surface
    t = state.time; % Current simulation time

    
    q = 1100;

    if t > 300
        q = 0;
    end
end

Get_results(ThermalModel, tlist, thermalresults, filename)


