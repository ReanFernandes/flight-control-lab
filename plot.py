import matplotlib.pyplot as plt
import numpy as np


def plot_drone_trajectory(X_mpc, function_type, numerical_method, x_init, x_desired):
    #define 3d plot to plot trajectory of the drone
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    #plot the trajectory of the drone
    ax.plot(X_mpc[:,0], X_mpc[:,1], X_mpc[:,2], 'o')
    #set the title of the plot
    ax.set_title('Trajectory of the drone')
    #set the label of the x axis
    ax.set_xlabel('x [m]')
    #set the label of the y axis
    ax.set_ylabel('y [m]')
    #set the label of the z axis
    ax.set_zlabel('z [m]')
    #save the figure to a png file in the folder drone_trajectory
    if numerical_method == 'DMS':
        plt.savefig('drone_trajectory/DMS/'+ x_init + ' to ' + x_desired + '_' + function_type + '_.png')
    
    elif numerical_method == 'DC':
        plt.savefig('drone_trajectory/DC/'+ x_init + ' to ' + x_desired + '_' + function_type + '_.png')
    
def plot_state_trajectory(X_mpc, U_mpc, deviation, step, function_type, numerical_method):
    
    #define the figure
    fig = plt.figure()
    #define the number of subplots
    if function_type == 'force_control':
        n = 7
        # plot the trajectories
        ax1 = fig.add_subplot(n,1,1)
        ax1.plot(X_mpc[:,0], label = 'x')
        ax1.plot(X_mpc[:,1], label = 'y')
        ax1.plot(X_mpc[:,2], label = 'z')
        ax1.set_title('Pose')
        ax1.set_ylabel('m')
        ax1.legend()
        ax2 = fig.add_subplot(n,1,2)
        ax2.plot(X_mpc[:,3], label = 'phi')
        ax2.plot(X_mpc[:,4], label = 'theta')
        ax2.plot(X_mpc[:,5], label = 'psi')
        ax2.set_title('Orientation')
        ax2.set_ylabel('rad')
        ax2.legend()
        ax3 = fig.add_subplot(n,1,3)
        ax3.plot(X_mpc[:,6], label = 'vx')
        ax3.plot(X_mpc[:,7], label = 'vy')
        ax3.plot(X_mpc[:,8], label = 'vz')
        ax3.set_title('Velocity')
        ax3.set_ylabel('m/s')
        ax3.legend()
        ax4 = fig.add_subplot(n,1,4)
        ax4.plot(X_mpc[:,9], label = 'phi_dot')
        ax4.plot(X_mpc[:,10], label = 'theta_dot')
        ax4.plot(X_mpc[:,11], label = 'psi_dot')
        ax4.set_title('Angular velocity')
        ax4.set_ylabel('rad/s')
        ax4.legend()
        ax5 = fig.add_subplot(n,1,5)
        ax5.plot(U_mpc[:,0], label = 'Thrust')
        ax5.set_title('Thrust')
        ax5.set_ylabel('N')
        ax5.legend()
        ax6 = fig.add_subplot(n,1,6)
        ax6.plot(U_mpc[:,1], label = 'Mx')
        ax6.plot(U_mpc[:,2], label = 'My')
        ax6.plot(U_mpc[:,3], label = 'Mz')
        ax6.set_title('Moments')
        ax6.set_ylabel('N.m')
        ax6.legend()
        ax7 = fig.add_subplot(n,1,7)
        ax7.plot(deviation, label = 'deviation')
        ax7.set_title('Deviation')
        ax7.set_ylabel('m')
        ax7.legend()
        #save the figure to a png file in the folder state_trajectory
        if numerical_method == 'DMS':
            plt.savefig('state_trajectory/DMS/'+ step + '_' + function_type + '_.png')
        elif numerical_method == 'DC':
            plt.savefig('state_trajectory/DC/'+ step + '_' + function_type + '_.png')
    
    elif function_type == 'rpm_control':
        n = 6
        # plot the trajectories
        ax1 = fig.add_subplot(n,1,1)
        ax1.plot(X_mpc[:,0], label = 'x')
        ax1.plot(X_mpc[:,1], label = 'y')
        ax1.plot(X_mpc[:,2], label = 'z')
        ax1.set_title('Pose')
        ax1.set_ylabel('m')
        ax1.legend()
        ax2 = fig.add_subplot(n,1,2)
        ax2.plot(X_mpc[:,3], label = 'phi')
        ax2.plot(X_mpc[:,4], label = 'theta')
        ax2.plot(X_mpc[:,5], label = 'psi')
        ax2.set_title('Orientation')
        ax2.set_ylabel('rad')
        ax2.legend()
        ax3 = fig.add_subplot(n,1,3)
        ax3.plot(X_mpc[:,6], label = 'vx')
        ax3.plot(X_mpc[:,7], label = 'vy')
        ax3.plot(X_mpc[:,8], label = 'vz')
        ax3.set_title('Velocity')
        ax3.set_ylabel('m/s')
        ax3.legend()
        ax4 = fig.add_subplot(n,1,4)
        ax4.plot(X_mpc[:,9], label = 'phi_dot')
        ax4.plot(X_mpc[:,10], label = 'theta_dot')
        ax4.plot(X_mpc[:,11], label = 'psi_dot')
        ax4.set_title('Angular velocity')
        ax4.set_ylabel('rad/s')
        ax4.legend()
        ax5 = fig.add_subplot(n,1,5)
        ax5.plot(U_mpc[:,0], label = 'rpm1')
        ax5.plot(U_mpc[:,1], label = 'rpm2')
        ax5.plot(U_mpc[:,2], label = 'rpm3')
        ax5.plot(U_mpc[:,3], label = 'rpm4')
        ax5.set_title('RPM')
        ax5.set_ylabel('rpm')
        ax5.legend()
        ax6 = fig.add_subplot(n,1,6)
        ax6.plot(deviation, label = 'deviation')
        ax6.set_title('Deviation')
        ax6.set_ylabel('m')
        ax6.legend()
        #save the figure to a png file in the folder state_trajectory
        if numerical_method == 'DMS':
            plt.savefig('state_trajectory/DMS/'+ step + '_' + function_type + '_.png')
        elif numerical_method == 'DC':
            plt.savefig('state_trajectory/DC/'+ step + '_' + function_type + '_.png')


    