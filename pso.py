import random
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
import matplotlib.animation as animation
from mpl_toolkits import mplot3d

# Fitness function 
# Assuming problem can be achieved by the following function
# f(x1, x2) = (x1+2*-x2+3)^2 + (2*x1+x2-8)^2
# The objective is to find a minimum which is 0

particles = [[random.uniform(-100.0, 100.0) for _ in range(2)] for _ in range(100)]

def fitness_function(x1, x2):
    # f1 = x1+2*-x2+3
    # f2 = 2*x1+x2-8
    # z = f1**2 + f2**2
    f1 = x1 + 2 * (-x2) + 3
    f2 = 2 * x1 + x2 - 8
    f3 = x1**3 + x2**3 - 20
    z = f1**2 + f2**2 + f3**2
    return z

def update_velocity(particle, velocity, pbest, gbest, w_min=0.5, max=1.0, c=0.1):
    # Initialising new velocity array
    num_particle = len(particle)
    new_velocity = np.array([0.0 for i in range(num_particle)])

    # Randomly generate r1, r2 and inertia weight from normal distribution
    r1 = random.uniform(0, max)
    r2 = random.uniform(0, max)
    w = random.uniform(w_min, max)
    c1 = c
    c2 = c

    # Calculate new velocity
    for i in range(num_particle):
        new_velocity[i] = w*velocity[i] + c1*r1*(pbest[i] - particle[i]) + c2*r2*(gbest[i] - particle[i])

    return new_velocity

def update_position(particle, velocity):
    # Move particle by adding velocity
    new_particle = particle + velocity
    return new_particle

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z') 

ax.view_init(-140, 60)

x = np.linspace(-100.0, 100.0, 80)
y = np.linspace(-100.0, 100.0, 80)

X, Y = np.meshgrid(x, y)
Z = fitness_function(X, Y)
ax.plot_wireframe(X, Y, Z, color='g', linewidth=0.2)

# Collect the particle positions for each generation
particle_positions = []

# Initialize an empty scatter plot for particles
scatter = ax.scatter([], [], [], c='b') 

def pso_2d(population, dimension, positon_min, postion_max, generation, fitness_criterion):
    # Initialisation
    # Population
    global particle
    # Particle's best position
    pbest_position = particles

    # Fitness
    pbest_fitness = [fitness_function(p[0], p[1]) for p in particles]

    # Index of the best particle
    gbest_index = np.argmin(pbest_fitness)

    # Global best particle position
    gbest_position = pbest_position[gbest_index]

    # Velocity (starting from 0 speed)
    velocity = [[0.0 for _ in range(dimension)] for _ in range(population)]

    # Plotting preparation
    # fig = plt.figure(figsize=(10, 10))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_zlabel('z') 

    # x = np.linspace(position_min, position_max, 80)
    # y = np.linspace(position_min, position_max, 80)

    # X, Y = np.meshgrid(x, y)
    # Z = fitness_function(X, Y)
    # ax.plot_wireframe(X, Y, Z, color='g', linewidth=0.2)

    # # Animation placeholder
    # images = []

    # Loop for the number of generation
    for t in range(generation):

        # Stop if the average fitness value reached a predefined success criterion
        if np.average(pbest_fitness) <= fitness_criterion:
            break
        else:
            for n in range(population):

                # Update the velocity of each particle
                velocity[n] = update_velocity(particles[n], velocity[n], pbest_position[n], gbest_position)

                # Move the particles to the new position
                particles[n] = update_position(particles[n], velocity[n])

            # Calculate he fitness value
            pbest_fitness = [fitness_function(p[0], p[1]) for p in particles]

            # Find the index of the best particle
            gbest_index = np.argmin(pbest_fitness)

            # Update the position of the best particle
            gbest_position = pbest_position[gbest_index]

        # Append the particle positions for this generation
        particle_positions.append([[particles[n][0], particles[n][1], fitness_function(particles[n][0], particles[n][1])] for n in range(population)])

        
        # x = [particles[n][0] for n in range(population)]
        # y = [particles[n][1] for n in range(population)]
        # z = [fitness_function(particles[n][0], particles[n][1]) for n in range(population)]

        # # Add plot for each generation (within the generation for-loop)
        # image = ax.scatter(x, y, z, c='b')
        # images.append(image)

    # Print the results
    print('Global Best Position: ', gbest_position)
    print('Best Fitness Value: ', min(pbest_fitness))
    print('Average Particle Best Fitness Value: ', np.average(pbest_fitness))
    print('Number of Generation: ', t)

    # Generating the animation and saving
    # animated_image = animation.ArtistAnimation(fig=fig, artists=images)
    # animated_image.save(filename='./pso_simple.gif', writer='pillow') 

def init():
    scatter._offsets3d = ([], [], [])
    return scatter,

def animate(t):
    global scatter
    positions = particle_positions[t]
    scatter.remove()  # Clear the previous scatter plot
    # Update the scatter plot for the current generation
    scatter = ax.scatter(
        [pos[0] for pos in positions],
        [pos[1] for pos in positions],
        [pos[2] for pos in positions], c='b')
    return scatter


if __name__=='__main__':
    population = 100
    dimension = 2
    position_min = -100.0
    position_max = 100.0
    generation = 400
    fitness_criterion = 10e-4

    pso_2d(population, dimension, position_min, position_max, generation, fitness_criterion)
    
    ani = animation.FuncAnimation(fig, animate, init_func=init, frames=min(len(particle_positions), generation), blit=True)
    ani.save('./pso_simple_1.gif', writer='pillow')