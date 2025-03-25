clear

%run number of simulations with random defects at a random depth and save
%results

%% simulation parameters
tlist = 0:2:2;
number_of_simulations = 1;
number_of_defects = 3; %still bit buggy with more
sample_thickness = 0.05; %[m]
file_name = 'results'; %simulation number and .csv will be added




for ii = 1:number_of_simulations
    [Model, results, labelface_ID, g] = FEM_simulation(tlist, number_of_defects);
    
    filename = append(file_name, "_" ,string(ii),".csv");

    Get_results(Model, tlist, results, filename, labelface_ID, g)
    
end

function [ThermalModel, thermalresults, labelface_ID, g] = FEM_simulation(tlist, number_of_defects)
    %% environment & sample properties
    Ambient_T = 22; %[°C]

    %material properties
    conductivity = 0.38;% [W/m*°K]
    density = 980; %[kg/m^3]
    specific_heat = 1200; %[J/°K]
    convex_coeff = 5; %[W/m*°K]
    emis_coeff = 0.9;

    %% set up the simulation

    % define the model
    ThermalModel = createpde('thermal', 'transient');

    %run the geometry script to create the sample
    [g, labelface_ID] = Multiple_defects(ThermalModel, number_of_defects);
    
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
    
    %heat flux through top face
    thermalBC(ThermalModel,'Face',1:ThermalModel.Geometry.NumFaces,'HeatFlux',@heatFluxFunction);

    %initial conditions
    thermalIC(ThermalModel,Ambient_T);

    %generate and display mesh
    generateMesh(ThermalModel, Hmax = 0.007);

    %% solve the model
    thermalresults = solve(ThermalModel,tlist);
end

function q = heatFluxFunction(region, state)
    x = region.x; % X-coordinates of the surface
    y = region.y; % Y-coordinates of the surface
    z = region.z; % Z-coordinate
    t = state.time; % Current simulation time
    q = 0;
    if z == 0
        if t < 300
            q = 350; %could be a x and y dependent function
        end
    end    
end


