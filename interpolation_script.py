import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

def extract_simulation_data(sim_data):
    # extract coordinates and temperatures
    x = sim_data['X'].values  # x-coordinates of nodes
    y = sim_data['Y'].values  # t-coordinates of nodes
    temperatures = sim_data.iloc[:, 3:].values  # temperature of nodes
    
    return x, y, temperatures
def interpolate_temperatures(x, y, temperatures):
    # create meshgrid for pixel data
    grid_x, grid_y = np.meshgrid(
        np.linspace(x.min(), x.max(), 640),
        np.linspace(y.min(), y.max(), 480)
    )

    # initialize pixel grid for temperatures
    grid_temperatures = np.zeros((grid_x.shape[0], grid_x.shape[1], temperatures.shape[1]))

    # interpolate temperatures for each time step
    for i in range(temperatures.shape[1]):
        grid_temperatures [:,:,i] = griddata(
        points=(x, y), # input coordinates
        values=temperatures[:,i], # temperatures at input coordinates
        xi=(grid_x, grid_y), # grid coordinates for interpolation
        method='nearest'  
    )
        
    return grid_temperatures

def main(path):

    # load simulation data
    simulation_data = pd.read_csv(os.path.join(path, "SimulationModel/results.csv"))
    print("simulation data shape: ", simulation_data.shape)

    # extract simulation data
    x, y, temperatures = extract_simulation_data(simulation_data)
    print(f"loaded data: {x.shape}, {y.shape}, {temperatures.shape} ")
    print("num time steps: ", temperatures.shape[1])

    # interpolate temperatures
    grid_temperatures = interpolate_temperatures(x, y, temperatures)
    print("interpolated temperatures: ", grid_temperatures.shape)

    # transpose to make time axis the first dimension
    grid_temperatures_transposed = np.transpose(grid_temperatures, (2, 0, 1))

    # flatten the grid to have shape (time steps, 307200)
    flattened_temperatures = grid_temperatures_transposed.reshape(temperatures.shape[1], -1)  
    print("final Shape: ", flattened_temperatures.shape) # should be (num time steps, 307200)

    # dataframe to save the interpolated data
    flattened_temperatures_df = pd.DataFrame(flattened_temperatures)

    # save the interpolated temperatures in a CSV file
    output_csv_path = os.path.join(path, "interpolated_temperatures.csv")
    flattened_temperatures_df.to_csv(output_csv_path, index=False)
    print(f"interpolated temperatures saved to {output_csv_path}")

if __name__ == "__main__":
    path = "./"  # set path to working directory
    main(path)



