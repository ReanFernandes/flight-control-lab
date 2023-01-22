import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(X_mpc):
    #define 3d plot to plot trajectory of the drone
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot the trajectory of the drone
    ax.plot(X_mpc[:,0], X_mpc[:,1], X_mpc[:,2])
    #set the title of the plot
    ax.set_title('Trajectory of the drone')
    #set the label of the x axis
    ax.set_xlabel('x [m]')
    #set the label of the y axis
    ax.set_ylabel('y [m]')
    #set the label of the z axis
    ax.set_zlabel('z [m]')
    #show the plot
    plt.show()
    