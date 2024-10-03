import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

class Parameters():
    def __init__(self, num_x, num_y, tau, u0):
        self.num_x = num_x
        self.num_y = num_y
        self.tau = tau  # Decay timescale
        self.u0 = u0  # Initial speed
        self.nu = (2.0 * tau - 1) / 6.0  # Kinematic viscosity
        self.Re = num_x * u0 / self.nu  # Reynolds number
        self.inv_tau = 1 / tau
        self.cs = 1 / np.sqrt(3)  # Speed of sound
        self.rho0 = 1.0  # Fluid density
        self.num_vel = 9
        self.c = np.array([[1, 0], [0, 1], [-1, 0], [0, -1], [1, 1], [-1, 1],
                           [-1, -1], [1, -1], [0, 0]])
        self.w = np.array([1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36, 4/9])
        self.reflection = np.array([2, 3, 0, 1, 6, 7, 4, 5, 8])
        self.mask = np.full((num_x, num_y), False)
        self.mask2 = np.full((num_x, num_y), False) #the part of the total mask which we are measuring the force on

        scalemax = float(input("""What would you like the maximum Vorticity scale value to be? Enter:\n
                                    0.06  For Airfoil
                                    0.06  For Parachute
                                    0.015 For Cybertruck
                                    0.08  For one circle
                                    0.04  For muliple circles\n
                                    Please make your choice here: """))
        self.scalemax = scalemax
        if scalemax == 0.015:
            self.scalemin = -0.03 #-scalemax for all shapes other than cybertruck set to: -0.03
        else:
            self.scalemin = -scalemax

def plot_solution(sim, t, u, vor, t_plot):
    # We have naturally used x as the first index in our multidimensional
    # arrays and y as the second index. However, matplotlib essentially plots
    # *matrices*, with first index increasing *down* the matrix and the second
    # index increasing *across*. This explains the use of 'transpose' and
    # 'origin = 'lower'' in the code below.
    
    x = np.arange(sim.num_x) + 0.5
    y = np.arange(sim.num_y) + 0.5
    
    [fig, ax] = plt.subplots(3, 1, figsize=(16,9))
    
    #needed to make the length of system longer to avoid interference from vortices passing out the back and appearing in front again
    #cutoff value allows the graph to remain focused on the interesting part of the vortex near the cars
    #cutff limits how far the graphs are plotted along x-direction. 
    #cutoff = sim.num_x for true scale of simulated fluid
    cutoff = 1300
    
    # Density field
    
    c = ax[0].imshow(rho[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y], vmax=1.2) 
    bar = fig.colorbar(c,ax=ax[0])
    bar.set_label(r'$\rho$')
    ax[0].set_title(r'Density $\rho$')
    ax[0].set_xlabel('$X$')
    ax[0].set_ylabel('$Y$')

    # Velocity field
    speed = np.sqrt(np.einsum('xyv->xy', u*u) ) / sim.u0
    c = ax[1].imshow(speed[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y],
                       vmin=0, vmax=1)
    bar = fig.colorbar(c, ax=ax[1])
    bar.set_label('$|\mathbf{u}|/u_0$')
    ax[1].streamplot(x[:cutoff], y, u[:cutoff,:,0].transpose(),
                       u[:cutoff,:,1].transpose(), color=[1,1,1], density = 1, linewidth = 0.7, arrowsize = 0.7)
    ax[1].set_title(r'Velocity $\mathbf{u}$')
    ax[1].set_xlabel('$X$')
    ax[1].set_ylabel('$Y$')
    
    # Vorticity field
    c = ax[2].imshow(vor[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y])
    bar = fig.colorbar(c,ax=ax[2])
    bar.set_label('$v$')
    ax[2].set_title(r'Vorticity $v$')
    ax[2].set_xlabel('$X$')
    ax[2].set_ylabel('$Y$')
    
    plt.subplots_adjust(hspace=0.85)
    #plt.savefig('DVV1car_{}.png'.format(int((t+t_plot)/t_plot)), dpi=300)
    plt.show()
    
    
    # Vorticity field with streamlines for u
    #colours for colormap below:
    #twilight
    #terrain
    #nipy_spectral
    #gist_ncar
    c = plt.imshow(vor[:cutoff, :].transpose(), origin='lower', extent=[0,cutoff,0,sim.num_y], cmap='gist_ncar', vmin=sim.scalemin, vmax=sim.scalemax)
    bar2 = plt.colorbar(c)
    bar2.set_label('$v$')
    plt.title(r'Vorticity with Streamlines $v$', fontsize = '8')
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.streamplot(x[:cutoff], y, u[:cutoff, :, 0].transpose(), u[:cutoff,:,1].transpose(), color=[1,1,1], density = 0.87, linewidth = 0.4, arrowsize = 0.4)
    
    plt.savefig('streamlines1car_{}.png'.format(int((t+t_plot)/t_plot)), dpi=300)
    plt.show()
    
    #TEST voricity plot to check if vortices are interfering with front of car due to going off graph at back
    c = plt.imshow(vor.transpose(), origin='lower', extent=[0,sim.num_x,0,sim.num_y], cmap='gist_ncar', vmin=sim.scalemin, vmax=sim.scalemax)
    bar2 = plt.colorbar(c)
    bar2.set_label('$v$')
    plt.title(r'Vorticity with Streamlines $v$', fontsize = '8')
    plt.xlabel('$X$')
    plt.ylabel('$Y$')
    plt.streamplot(x, y, u[:, :, 0].transpose(), u[:,:,1].transpose(), color=[1,1,1], density = 0.87, linewidth = 0.4, arrowsize = 0.4)
    
    plt.show()
    
    
    #TESTING THE SIM.MASK2 TO CHECK IT IS BEING INITIALISED CORRECTLY
    colors = np.array([[1, 0, 0], [0, 0, 1]])  # Red for False, Blue for True
    cmap = plt.matplotlib.colors.ListedColormap(colors)

    # Plot the array
    plt.imshow(sim.mask2.transpose(), cmap=cmap, origin='lower', extent=[0, sim.num_x, 0, sim.num_y])

    # Add colorbar for reference
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.set_ticklabels(['False', 'True'])

    # Show the plot
    plt.title('sim.mask2 Visualization')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.show()
    
    
    
def equilibrium(sim, rho, u):
    # Function for evaluating the equilibrium distribution, across the entire
    # lattice, for a given input fluid density and velocity field.
    
    u_dot_c = np.einsum('ijk,lk->ijl', u, sim.c)   #3D
    u_dot_u = np.einsum('ijk,ijk->ij', u, u)       #2D
    
    u_dot_u = u_dot_u[...,np.newaxis]
    rho = rho[..., np.newaxis]       #makes it 3D
    
    #equilibrium distribution of the liquid = feq
    feq = sim.w*(1+(u_dot_c/(sim.cs**2))+(u_dot_c**2/(2*sim.cs**4))-(u_dot_u/(2*sim.cs**2)))*rho
    # Return a numpy array of shape (sim.num_x, sim.num_y, sim.num_v)
    return feq


def fluid_density(sim, f):
    # Calculate the fluid density from the distribution f.
    rho = np.where(sim.mask, 0.0001, np.einsum('ijk->ij', f))
    
    # Returns a numpy array of fluid density, of shape (sim.num_x, sim.num_y).
    return rho



def fluid_velocity(sim, f, rho):
    # Calculate the fluid velocity from the distribution f and the fluid
    # density rho.
    
    inv_rho = 1/rho
    f_over_rho = np.einsum('ijk,ij->ijk', f, inv_rho)
    u = np.where(sim.mask[...,np.newaxis], 0, np.einsum('ijk,kl->ijl', f_over_rho,sim.c))
    # Returns a numpy array of shape (sim.num_x, sim.num_y, 2), of fluid
    # velocities.
    return u


def fluid_vorticity(sim, u):
    vor = (np.roll(u[:,:,1], -1, 0) - np.roll(u[:,:,1], 1, 0) -
           np.roll(u[:,:,0], -1, 1) + np.roll(u[:,:,0], 1, 1))
    return vor


def collision(sim, f, feq):
    # Perform the collision step, updating the distribution f, using the
    # equilibrium distribution provided in feq.
    delta_t = 1
    f = (f*(1-(delta_t/sim.tau))) + (feq*(delta_t/sim.tau))
    
    return f


def stream_and_reflect(sim, f):
    # Perform the streaming and boundary reflection step.
    
    delta_t = 1
    momentum_point=np.zeros((sim.num_x,sim.num_y,9))
    
    for i in range(len(sim.c)):
        momentum_point[:,:,i] = np.where(sim.mask2, 0, np.where(np.roll(sim.mask2, sim.c[i,:], axis=(0,1)), u[:,:,0]*(f[:,:,i]+f[:,:,sim.reflection[i]]), 0))               #should i roll by -sim.c as opposite direction as line below
        f[:,:,i] = np.where(sim.mask, 0, np.where(np.roll(sim.mask, sim.c[i,:], axis=(0,1)), f[:,:,sim.reflection[i]], np.roll(f[:,:,i], sim.c[i,:]*delta_t, axis=(0,1))))
    momentum_total = np.einsum('ijk->',momentum_point)
    
    return f, momentum_total
    

def initial_turbulence(sim):
    # Create the initial flow. Also initialise the mask, indicating the
    # presence of the obstacle. 
    
    #Read the mask from the text file
    print("""\nHi, welcome to this program. Here are your obstacle options: \n 
          a =  Airfoil used in plane wings\n
          b =  Parachute\n 
          c =  Cybertruck\n
          d =  Circle\n
          e =  Multiple circles\n
          f =  Slipstreamed Cybertruck, Spacing: 1 px
          g =  Slipstreamed Cybertruck, Spacing: 300 px
          h =  Slipstreamed Cybertruck, Spacing: 257 px
          i =  Slipstreamed Cybertruck, Spacing: 214 px
          j =  Slipstreamed Cybertruck, Spacing: 171 px
          k =  Slipstreamed Cybertruck, Spacing: 128 px
          l =  Slipstreamed Cybertruck, Spacing: 85 px
          m =  Slipstreamed Cybertruck, Spacing: 42 px
          """)
    choice = input('\nWould you like to enter part a, b, c, d, e, f, g, h ,i, j, k, l or m: ').lower()
    
    #Adding some randomness to initial flow y-direction
    random1=-1
    random2=1
    
    #dimensions for 100x75 grid template
    xleft=50
    xright=150
    ytop=138
    ybottom=63
    
    if choice == 'a':
        mask_data = r"C:\Users\Luke Johnson\OneDrive\Documents\Uni\Physics DLM Labs\Year 3\masks\plane wing mask with spaces.txt"
        random1=0 #adds an incline to the wing by adjusting the oncoming fluid
        random2=2
    elif choice =='b':
        mask_data = r"C:\Users\Luke Johnson\OneDrive\Documents\Uni\Physics DLM Labs\Year 3\masks\parachute mask with spaces.txt"
    elif choice =='c':
        mask_data = r"C:\Users\Luke Johnson\OneDrive\Documents\Uni\Physics DLM Labs\Year 3\masks\cybertruck mask with spaces.txt"
        xleft=50
        xright=250
        ytop=75
        ybottom=0
    elif choice in ['d', 'e']:
        mask_data = r"C:\Users\Luke Johnson\OneDrive\Documents\Uni\Physics DLM Labs\Year 3\masks\circle mask with spaces.txt"
        xleft=84
        xright=116
        ytop=116
        ybottom=84
    elif choice in ['f','g','h','i','j','k','l','m']:
        #for all slipstreaming cars, front car is setup here
        mask_data = r"C:\Users\Luke Johnson\OneDrive\Documents\Uni\Physics DLM Labs\Year 3\masks\cybertruck mask with spaces.txt"
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
        sim.mask[250:450, 0:75] = numbers      #adding 2nd car behind first one separation 1px
        sim.mask2[250:450, 0:75] = numbers     #ensuring that the foce is only being calculated for the 2nd car and not on the whole mask (inclusing the road etc)
        sim.mask[:,190:200]=True #creating a tarmac road and tunnel as an obstacle
        sim.mask[:,0:10]=True
        
    elif choice =='g':
        sim.mask[550:750, 0:75] = numbers      #2nd car separation 300px
        sim.mask2[550:750, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
    
    elif choice =='h':
        sim.mask[507:707, 0:75] = numbers      #2nd car separation 257px
        sim.mask2[507:707, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='i':
        sim.mask[464:664, 0:75] = numbers      #2nd car separation 214px
        sim.mask2[464:664, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='j':
        sim.mask[421:621, 0:75] = numbers      #2nd car separation 171px
        sim.mask2[421:621, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='k':
        sim.mask[378:578, 0:75] = numbers      #2nd car separation 128px
        sim.mask2[378:578, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    elif choice =='l':
        sim.mask[335:535, 0:75] = numbers      #2nd car separation 85px
        sim.mask2[335:535, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
    
    elif choice =='m':
        sim.mask[292:492, 0:75] = numbers      #2nd car separation 42px
        sim.mask2[292:492, 0:75] = numbers
        sim.mask[:,190:200]=True
        sim.mask[:,0:10]=True
        
    else:
        sim.mask2[xleft:xright, ybottom:ytop] = numbers
        
    #initialise the flow
    ux_initial = np.full((sim.num_x, sim.num_y), sim.u0)
    ux = np.where(sim.mask,0,ux_initial)
    
    r = np.random.uniform(random1,random2,(sim.num_x,sim.num_y)) #change numbers here to alter angle of incoming fluid. keep a range of 2 between them to lead to faster turbulant flow
    uy = np.where(sim.mask, 0, np.full((sim.num_x, sim.num_y), (1/10)*sim.u0*r))
    u = np.stack((ux, uy), axis=2) # sets the velocity of the fluid as u0 in the x-direction
    
    rho_initial = np.full((sim.num_x, sim.num_y), sim.rho0)
    rho = np.where(sim.mask,0.0001,rho_initial)
    
    
    # Returns numpy arrays of density and velocity data, of shape (sim.num_x,
    # sim.num_y) and (sim.num_x, sim.num_y, 2) respectively.    
    
    return rho, u


# Initialise parameters
# CHANGE PARAMETER VALUES HERE.
sim = Parameters(3200, 200, 0.500001, 0.18) #u0: use 0.01 for unseparated flow, 0.06 for Foppl vortices, 0.18  for Vortex shedding
t_steps = 24000
t_plot = 500

# Initialize density and velocity fields.
initial_rho, initial_u = initial_turbulence(sim)

# Create the initial distribution by finding the equilibrium for the flow
# calculated above.
f = equilibrium(sim, initial_rho, initial_u)

# We could just copy initial_rho, initial_v and f into rho, v and feq.
rho = fluid_density(sim, f)
u = fluid_velocity(sim, f, rho)
feq = equilibrium(sim, rho, u)
vor = fluid_vorticity(sim, u)
plot_solution(sim, 0, u, vor, t_plot)

# Finally evolve the distribution in time, using the 'collision' and
# 'streaming_reflect' functions.
force_array = np.zeros((t_steps)) #initialising the array which will store the force throughout the whole simulation
for t in range(1, t_steps + 1):
    # Perform collision step, using the calculated density and velocity data.
    f = collision(sim, f, feq)

    # Streaming and reflection
    f, momentum_total = stream_and_reflect(sim, f)
    force_array[t-1] = momentum_total
    # Calculate density and velocity data, for next time around
    rho = fluid_density(sim, f)
    u = fluid_velocity(sim, f, rho)
    feq = equilibrium(sim, rho, u)
    #print('reynolds number: ', sim.Re)
    if (t % t_plot == 0):
        vor = fluid_vorticity(sim, u)
        plot_solution(sim, t, u, vor, t_plot)

np.savetxt('forces_distance_1car.csv', force_array)
#edit each time file creation names here and in plot_solution() function   