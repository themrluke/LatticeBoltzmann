# initialisation.py

import os
import numpy as np


def initial_turbulence(sim):
    # Create the initial flow. Also initialise the mask, indicating the
    # presence of the obstacle. 
    
    #Read the mask from the text file
    """
    Here are your obstacle options:
          a =  Airfoil used in plane wings
          b =  Parachute
          c =  Cybertruck
          d =  Circle
          e =  Multiple circles
          f =  Slipstreamed Cybertruck, Spacing: 1 px
          g =  Slipstreamed Cybertruck, Spacing: 300 px
          h =  Slipstreamed Cybertruck, Spacing: 257 px
          i =  Slipstreamed Cybertruck, Spacing: 214 px
          j =  Slipstreamed Cybertruck, Spacing: 171 px
          k =  Slipstreamed Cybertruck, Spacing: 128 px
          l =  Slipstreamed Cybertruck, Spacing: 85 px
          m =  Slipstreamed Cybertruck, Spacing: 42 px
    """

    obstacle_options = {
        'a': {"file": "plane wing mask with spaces.txt", "random": (0, 2), "dimensions": (50, 150, 138, 63)},
        'b': {"file": "parachute mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 150, 138, 63)},
        'c': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'd': {"file": "circle mask with spaces.txt", "random": (-1, 1), "dimensions": (84, 116, 116, 84)},
        'e': {"file": "circle mask with spaces.txt", "random": (-1, 1), "dimensions": (84, 116, 116, 84)},
        'f': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'g': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'h': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'i': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'j': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'k': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'l': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
        'm': {"file": "cybertruck mask with spaces.txt", "random": (-1, 1), "dimensions": (50, 250, 75, 0)},
    }

    choice = 'm'
    if choice not in obstacle_options:
        raise ValueError("Invalid obstacle choice!")
    
    # Get obstacle settings
    obstacle = obstacle_options[choice]
    mask_file = obstacle["file"]
    random_range = obstacle["random"]
    xleft, xright, ytop, ybottom = obstacle["dimensions"]
    
    # Construct relative path for mask file
    mask_dir = os.path.join(os.path.dirname(__file__), "..", "masks")
    mask_path = os.path.join(mask_dir, mask_file)
    if not os.path.exists(mask_path):
        raise FileNotFoundError(f"Mask file not found: {mask_path}")
    
    # Load the mask data
    numbers = np.genfromtxt(mask_path, dtype=np.int32)
    numbers = np.flip(np.transpose(numbers), axis=1)
    sim.mask[xleft:xright, ybottom:ytop] = numbers

    # Some extra conditions for special cases based on user obstacle choices
    if choice =='c':
        sim.mask[:,190:200] = 1 #creating a tarmac road and tunnel as an obstacle
        sim.mask[:,0:10] = 1
        sim.mask2[xleft:xright, ybottom:ytop] = numbers
        

    elif choice == 'e':
        # Add multiple circles
        sim.mask2[xleft:xright, ybottom:ytop] = numbers
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
            xleft, xright, ybottom, ytop = offset
            sim.mask[xleft:xright, ybottom:ytop] = numbers
        sim.scalemin = -0.04
        sim.scalemax = 0.04

    elif choice in ['f', 'g', 'h', 'i', 'j', 'k', 'l', 'm']:
        # Slipstream cases
        offset = {
            'f': 1, 'g': 300, 'h': 257, 'i': 214,
            'j': 171, 'k': 128, 'l': 85, 'm': 42
        }[choice]
        sim.mask[250 + offset:450 + offset, 0:75] = numbers
        sim.mask2[250 + offset:450 + offset, 0:75] = numbers
        sim.mask[:, 190:200] = 1  # Tarmac road
        sim.mask[:, 0:10] = 1
        
    else:
        sim.mask2[xleft:xright, ybottom:ytop] = numbers
        
    #initialise the flow
    ux_initial = np.full((sim.num_x, sim.num_y), sim.u0)
    ux = np.where(sim.mask, 0, ux_initial)
    
    r = np.random.uniform(random_range[0], random_range[1], (sim.num_x,sim.num_y)) # change numbers here to alter angle of incoming fluid. keep a range of 2 between them to lead to faster turbulant flow
    uy = np.where(sim.mask, 0, np.full((sim.num_x, sim.num_y), (1/10)*sim.u0*r))
    u = np.stack((ux, uy), axis=2) # sets the velocity of the fluid as u0 in the x-direction
    
    rho_initial = np.full((sim.num_x, sim.num_y), sim.rho0)
    rho = np.where(sim.mask,0.0001,rho_initial)
    
    
    # Returns numpy arrays of density and velocity data, of shape (sim.num_x,
    # sim.num_y) and (sim.num_x, sim.num_y, 2) respectively.    
    
    return rho, u