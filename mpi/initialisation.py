# initialisation.py

import os
import numpy as np


class InitialiseSimulation:
    """
    Class to initialise the simulation with different choices of obstacles.

    Attributes:
        sim (Parameters): Object with all the input parameters for the fluid and lattice
        obstacle_options (dict): A dictionary of obstacle filepaths, fluid y-randomness and obstacle placement on the lattice

    """

    def __init__(self, parameters):
        self.sim = parameters  # Store the Parameters instance
        self.obstacle_options = { # Information about the obstacle filepath, placement and y direction of the fluid
            'a': {"file": "plane wing mask with spaces.txt", "random": (0, 2), "placement": (50, 150, 138, 63)}, # `random` values here give an incline to plane wing
            'b': {"file": "parachute mask with spaces.txt", "random": (-1, 1), "placement": (50, 150, 138, 63)},
            'c': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "placement": (50, 250, 75, 0)},
            'd': {"file": "circle mask with spaces.txt", "random": (-1, 1), "placement": (84, 116, 116, 84)},
            'e': {"file": "circle mask with spaces.txt", "random": (-1, 1), "placement": (84, 116, 116, 84)},
            **{ch: {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "placement": (50, 250, 75, 0), # Front car setup for slipstreaming cars
                    "offset": offset} for ch, offset in zip('fghijklm', [1, 300, 257, 214, 171, 128, 85, 42])},
        }


    def load_mask(self, choice):
        """
        Load the mask data from text file for a given choice of obstacle.

        Arguments:
            choice (str): Indicates which obstacle to select

        Returns:
            np.ndarray: 2D NumPy array of binary containing obstacle data
        """

        mask_file = self.obstacle_options[choice]["file"]
        mask_dir = os.path.join(os.path.dirname(__file__), "..", "masks")
        mask_path = os.path.join(mask_dir, mask_file) # Find mask file from parent directory

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file for choice {choice} not found: {mask_path}")

        # Load the binary mask data (1s correspond to obstacle, 0s correspond to no obstacle)
        mask_data = np.genfromtxt(mask_path, dtype=np.int32)
        return np.flip(np.transpose(mask_data), axis=1)


    def special_conditions(self, choice, mask_data, placement):
        """
        Apply modifications to mask for specific obstacles.

        Arguments:
            choice (str): Indicates which obstacle to select
            mask_data (np.ndarray): 2D NumPy array of binary containing obstacle data
            placement (tuple): Placement of the mask on the lattice
        """

        sim = self.sim
        xleft, xright, ytop, ybottom = placement

        if choice =='c': # Creating a tarmac road and tunnel
            sim.mask[:, 190:200] = 1
            sim.mask[:, 0:10] = 1
            sim.mask2[xleft:xright, ybottom:ytop] = mask_data

        elif choice == 'e': # Add multiple circles
            sim.mask2[xleft:xright, ybottom:ytop] = mask_data
            offsets = [
                (148, 180, 116, 148),   #1
                (148, 180, 52, 84),     #2
                (212, 244, 148, 180),   #3
                (212, 244, 84, 116),    #4
                (212, 244, 20, 52),     #5
                (276, 308, 116, 148),   #6
                (276, 308, 52, 84),     #7
            ]
            for offset in offsets:
                xleft, xright, ytop, ybottom = offset
                sim.mask[xleft:xright, ybottom:ytop] = mask_data

            # Adjust scale for visualisation plots
            sim.scalemin = -0.04
            sim.scalemax = 0.04

        elif choice in ['f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']: # Slipstreaming car cases
            offset = { # Different car separations
                'f': 1, 'g': 300, 'h': 257, 'i': 214,
                'j': 171, 'k': 128, 'l': 85, 'm': 42
            }[choice]
            sim.mask[250 + offset:450 + offset, 0:75] = mask_data
            sim.mask2[250 + offset:450 + offset, 0:75] = mask_data # Force only calculated for the 2nd car, not whole mask (including the road etc)
            sim.mask[:, 190:200] = 1  # Tunnel ceiling
            sim.mask[:, 0:10] = 1 # Tarmac road

        else:
            sim.mask2[xleft:xright, ybottom:ytop] = mask_data # Portion of mask to calculate force on


    def initialise_turbulence(self, choice, start_x, end_x):
        """
        Sets up the mask, and initialises the fluid density and velocity for the simulation.

        Arguments:
            choice (str): Indicates which obstacle to select

        Returns:
            rho (np.ndarray): 2D array of the fluid density at each lattice point
            u (np.ndarray): 3D array of the fluid x & y velocity at each lattice point

        """

        if choice not in self.obstacle_options:
            raise ValueError(f"Invalid obstacle choice: '{choice}'.")
        sim = self.sim
        obstacle = self.obstacle_options[choice] # Select the obstacle
        random_range = obstacle["random"]

        # Load and apply the mask
        mask_data = self.load_mask(choice)
        placement = obstacle["placement"]
        xleft, xright, ytop, ybottom = placement
        sim.mask[xleft:xright, ybottom:ytop] = mask_data

        # Apply additional mask modifications for specific obstacles
        self.special_conditions(choice, mask_data, placement)

        # The below is slightly modified for MPI implementation

        local_num_x = end_x - start_x # Calculate the subdomain size for thread

        # Initialise velocity field for subdomain
        ux_initial = np.full((local_num_x, sim.num_y), sim.u0)
        ux = np.where(sim.mask[start_x:end_x], 0, ux_initial)

        r = np.random.uniform(random_range[0], random_range[1], (local_num_x, sim.num_y)) # These values will alter angle of incoming fluid, leading to faster turbulent flow
        uy = np.where(sim.mask[start_x:end_x], 0, np.full((local_num_x, sim.num_y), (1/10) * sim.u0 * r))
        u = np.stack((ux, uy), axis=2) # sets the velocity of the fluid as u0 in the x-direction

        # Initialise density field for subdomain
        rho_initial = np.full((local_num_x, sim.num_y), sim.rho0)
        rho = np.where(sim.mask[start_x:end_x], 0.0001, rho_initial)

        return rho, u