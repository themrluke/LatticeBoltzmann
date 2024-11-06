# initialisation.py
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
    choice = 'm'
    
    #Adding some randomness to initial flow y-direction
    random1=-1
    random2=1
    
    #dimensions for 100x75 grid template
    xleft=50
    xright=150
    ytop=138
    ybottom=63
    
    if choice == 'a':
        mask_data = "/home/themrluke/projects/Advanced_Computational/miniproject/masks/plane wing mask with spaces.txt"
        random1=0 #adds an incline to the wing by adjusting the oncoming fluid
        random2=2
    elif choice =='b':
        mask_data = "/home/themrluke/projects/Advanced_Computational/miniproject/masks/parachute mask with spaces.txt"
    elif choice =='c':
        mask_data = "/home/themrluke/projects/Advanced_Computational/miniproject/masks/cybertruck mask with spaces.txt"
        xleft=50
        xright=250
        ytop=75
        ybottom=0
    elif choice in ['d', 'e']:
        mask_data = "/home/themrluke/projects/Advanced_Computational/miniproject/masks/circle mask with spaces.txt"
        xleft=84
        xright=116
        ytop=116
        ybottom=84
    elif choice in ['f','g','h','i','j','k','l','m']:
        #for all slipstreaming cars, front car is setup here
        mask_data = "/home/themrluke/projects/Advanced_Computational/miniproject/masks/cybertruck mask with spaces.txt"
        xleft=50
        xright=250
        ytop=75
        ybottom=0
        
    else:
        print('That was not a valid choice!')
    
    numbers = np.genfromtxt(mask_data, dtype=np.int64)
    numbers = np.flip(np.transpose(numbers.astype(bool)), axis = 1)
    
    sim.mask[xleft:xright, ybottom:ytop] = numbers 
    
    #some extra conditions for special cases based on user obstacle choices
    if choice =='c':
        sim.mask[:,190:200]=True #creating a tarmac road and tunnel as an obstacle
        sim.mask[:,0:10]=True
        sim.mask2[xleft:xright, ybottom:ytop] = numbers
        
    elif choice =='e':          #positioning multiple circles
        sim.mask[148:180, 116:148] = numbers    #1
        sim.mask[148:180, 52:84] = numbers      #2
        sim.mask[212:244, 148:180] = numbers    #3
        sim.mask[212:244, 84:116] = numbers     #4
        sim.mask[212:244, 20:52] = numbers      #5
        sim.mask[276:308, 116:148] = numbers    #6
        sim.mask[276:308, 52:84] = numbers      #7
        
        sim.scalemin = -0.04
        sim.scalemax = 0.04
        
        sim.mask2[xleft:xright, ybottom:ytop] = numbers
    
    elif choice =='f':
        sim.mask[250:450, 0:75] = numbers      # adding 2nd car behind first one separation 1px
        sim.mask2[250:450, 0:75] = numbers     # ensuring that the force is only being calculated for the 2nd car and not on the whole mask (including the road etc)
        sim.mask[:,190:200]=True # creating a tarmac road and tunnel as an obstacle
        sim.mask[:,0:10]=True
        
    elif choice =='g':
        sim.mask[550:750, 0:75] = numbers      # 2nd car separation 300px
        sim.mask2[550:750, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
    
    elif choice =='h':
        sim.mask[507:707, 0:75] = numbers      # 2nd car separation 257px
        sim.mask2[507:707, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='i':
        sim.mask[464:664, 0:75] = numbers      # 2nd car separation 214px
        sim.mask2[464:664, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='j':
        sim.mask[421:621, 0:75] = numbers      # 2nd car separation 171px
        sim.mask2[421:621, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='k':
        sim.mask[378:578, 0:75] = numbers      # 2nd car separation 128px
        sim.mask2[378:578, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='l':
        sim.mask[335:535, 0:75] = numbers      # 2nd car separation 85px
        sim.mask2[335:535, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
    
    elif choice =='m':
        sim.mask[292:492, 0:75] = numbers      # 2nd car separation 42px
        sim.mask2[292:492, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    else:
        sim.mask2[xleft:xright, ybottom:ytop] = numbers
        
    #initialise the flow
    ux_initial = np.full((sim.num_x, sim.num_y), sim.u0)
    ux = np.where(sim.mask,0,ux_initial)
    
    r = np.random.uniform(random1,random2,(sim.num_x,sim.num_y)) # change numbers here to alter angle of incoming fluid. keep a range of 2 between them to lead to faster turbulant flow
    uy = np.where(sim.mask, 0, np.full((sim.num_x, sim.num_y), (1/10)*sim.u0*r))
    u = np.stack((ux, uy), axis=2) # sets the velocity of the fluid as u0 in the x-direction
    
    rho_initial = np.full((sim.num_x, sim.num_y), sim.rho0)
    rho = np.where(sim.mask,0.0001,rho_initial)
    
    
    # Returns numpy arrays of density and velocity data, of shape (sim.num_x,
    # sim.num_y) and (sim.num_x, sim.num_y, 2) respectively.    
    
    return rho, u