from casadi import *
import numpy as np
from MPC import MPC
import matplotlib.pyplot as plt
from plot_trajectory import plot_trajectory

def main():
    # define the parameters 
    N = 5 # number of horizon 
    T = 0.01 # time horizon 0.1 for dms, 0.01 for colloc
    Tf = 10 # MPC simulation time

    Q = np.diag([120,   #x
                120,    #y
                120,    #z
                1e-2,   #phi
                1e-2,   #theta
                1e-2,   #psi
                7e-1,   #vx
                1,      #vy
                4,      #vz
                1e-5,   #phi_dot
                1e-5,   #theta_dot
                10])   #psi_dot

    R = np.diag([1, 1, 1, 1])* 6

    # define the starting point
    x_init = np.array([1, 0, 1])
    x_desired = np.array([9, 3, 7])

    # options for ipopt
    nlpopts = {'ipopt': {'print_level': 0, 'max_iter':100}, 'print_time' : 0}

    # Thrust controlled MPC 
    function_type = "force_control"

    # Method 1: Multiple shooting
    # numerical_method = "multiple_shooting"
    # X_mpc_dms, U_mpc_dms, deviation, _  = MPC(None, Q, R, function_type, x_init, x_desired, N, T, Tf, numerical_method, nlpopts)

    # #plot the results in Figure 1, containing 7 subplots, 4 for the state and 2 for the control and 1 for the deviation, with all subplots sharing same x axis
    
    # fig, axs = plt.subplots(7, sharex=True)
    # #plot the pose in first subplot, X_mpc_dms[:,:3] is the x,y,z information
    # axs[0].plot(X_mpc_dms[:,:3])
    # axs[0].set_title('Position [m]')
    # axs[0].legend(['x', 'y', 'z'])
    # #plot the euler angles in second subplot, X_mpc_dms[:,3:6] is the phi, theta, psi information
    # axs[1].plot(X_mpc_dms[:,3:6])
    # axs[1].set_title('Euler angles [rad]')
    # axs[1].legend(['phi', 'theta', 'psi'])
    # #plot the velocity in third subplot, X_mpc_dms[:,6:9] is the vx, vy, vz information
    # axs[2].plot(X_mpc_dms[:,6:9])
    # axs[2].set_title('Velocity [m/s]')
    # axs[2].legend(['vx', 'vy', 'vz'])
    # #plot the angular velocity in fourth subplot, X_mpc_dms[:,9:12] is the phi_dot, theta_dot, psi_dot information
    # axs[3].plot(X_mpc_dms[:,9:12])
    # axs[3].set_title('Angular velocity [rad/s]')
    # axs[3].legend(['phi_dot', 'theta_dot', 'psi_dot'])
    # #plot the control input in fifth subplot, U_mpc_dms[:,0] is the thrust information
    # axs[4].plot(U_mpc_dms[:,0])
    # axs[4].set_title('Thrust [N]')
    # #plot the control input in sixth subplot, U_mpc_dms[:,1:4] is the moment information
    # axs[5].plot(U_mpc_dms[:,1:4])
    # axs[5].set_title('Moment [N*m]')
    # axs[5].legend(['Mx', 'My', 'Mz'])
    # #plot the deviation in seventh subplot
    # axs[6].plot(deviation)
    # axs[6].set_title('Euclidean distance from target[m]')
    # axs[6].legend(['x', 'y', 'z'])
    # plt.savefig('state plot_' + function_type + '_'+numerical_method + '.png')
    # plot_trajectory(X_mpc_dms, function_type, numerical_method)

    # Method 2: Direct collocation
    numerical_method = "direct_collocation"
    degree = 2
    X_mpc_dc, U_mpc_dc, deviation_dc, _  = MPC(degree ,Q, R, function_type, x_init, x_desired, N, T, Tf, numerical_method, nlpopts)

    #plot the results in Figure 2, containing 7 subplots, 4 for the state and 2 for the control and 1 for the deviation, with all subplots sharing same x axis. Ignore the 
    
    fig2, axs = plt.subplots(7, sharex=True)
    #plot the pose in first subplot, X_mpc_dc[:,:3] is the x,y,z information
    axs[0].plot(X_mpc_dc[:,:3])
    axs[0].set_title('Position [m]')
    axs[0].legend(['x', 'y', 'z'])
    #plot the euler angles in second subplot, X_mpc_dc[:,3:6] is the phi, theta, psi information
    axs[1].plot(X_mpc_dc[:,3:6])
    axs[1].set_title('Euler angles [rad]')
    axs[1].legend(['phi', 'theta', 'psi'])
    #plot the velocity in third subplot, X_mpc_dc[:,6:9] is the vx, vy, vz information
    axs[2].plot(X_mpc_dc[:,6:9])
    axs[2].set_title('Velocity [m/s]')
    axs[2].legend(['vx', 'vy', 'vz'])
    #plot the angular velocity in fourth subplot, X_mpc_dc[:,9:12] is the phi_dot, theta_dot, psi_dot information
    axs[3].plot(X_mpc_dc[:,9:12])
    axs[3].set_title('Angular velocity [rad/s]')
    axs[3].legend(['phi_dot', 'theta_dot', 'psi_dot'])
    #plot the control input as a  function in fifth subplot, U_mpc_dc[:,0] is the thrust information
    axs[4].plot(U_mpc_dc[:,0], '-')
    axs[4].set_title('Thrust [N]')

    #plot the control input as a  function in sixth subplot, U_mpc_dc[:,1:4] is the moment information
    axs[5].plot(U_mpc_dc[:,1:4])
    axs[5].set_title('Moment [N*m]')
    axs[5].legend(['Mx', 'My', 'Mz'])
    #plot the deviation in seventh subplot
    axs[6].plot(deviation_dc)
    axs[6].set_title('Euclidean distance from target[m]')
    axs[6].legend(['x', 'y', 'z'])
    #open in different figure
    
    plt.savefig('state plot_' + function_type + '_'+numerical_method + '.png')
    plot_trajectory(X_mpc_dc, function_type, numerical_method)
    plt.show()

    # RPM controlled MPC 
    function_type = "rpm_control"

    # # Method 1: Direct multiple shooting
    # numerical_method = "multiple_shooting"
    # X_mpc_dms, U_mpc_dms, deviation, _  = MPC(None,Q, R, function_type, x_init, x_desired, N, T, Tf, numerical_method, nlpopts)

    # #plot the results in Figure 1, containing 7 subplots, 4 for the state and 2 for the control and 1 for the deviation, with all subplots sharing same x axis
    
    # fig, axs = plt.subplots(7, sharex=True)
    # #plot the pose in first subplot, X_mpc_dms[:,:3] is the x,y,z information
    # axs[0].plot(X_mpc_dms[:,:3])
    # axs[0].set_title('Position [m]')
    # axs[0].legend(['x', 'y', 'z'])
    # #plot the euler angles in second subplot, X_mpc_dms[:,3:6] is the phi, theta, psi information
    # axs[1].plot(X_mpc_dms[:,3:6])
    # axs[1].set_title('Euler angles [rad]')
    # axs[1].legend(['phi', 'theta', 'psi'])
    # #plot the velocity in third subplot, X_mpc_dms[:,6:9] is the vx, vy, vz information
    # axs[2].plot(X_mpc_dms[:,6:9])
    # axs[2].set_title('Velocity [m/s]')
    # axs[2].legend(['vx', 'vy', 'vz'])
    # #plot the angular velocity in fourth subplot, X_mpc_dms[:,9:12] is the phi_dot, theta_dot, psi_dot information
    # axs[3].plot(X_mpc_dms[:,9:12])
    # axs[3].set_title('Angular velocity [rad/s]')
    # axs[3].legend(['phi_dot', 'theta_dot', 'psi_dot'])
    # #plot the control input in fifth subplot, U_mpc_dms[:,0] is the thrust information
    # axs[4].plot(U_mpc_dms[:,0])
    # axs[4].set_title('Thrust [N]')
    # #plot the control input in sixth subplot, U_mpc_dms[:,1:4] is the moment information
    # axs[5].plot(U_mpc_dms[:,1:4])
    # axs[5].set_title('Moment [N*m]')
    # axs[5].legend(['Mx', 'My', 'Mz'])
    # #plot the deviation in seventh subplot
    # axs[6].plot(deviation)
    # axs[6].set_title('Euclidean distance from target[m]')
    # axs[6].legend(['x', 'y', 'z'])
    # plt.show()

     # Method 2: Direct collocation
    # numerical_method = "direct_collocation"
    # degree = 2
    # X_mpc_dc, U_mpc_dc, deviation_dc, _ = MPC(degree ,Q, R, function_type, x_init, x_desired, N, T, Tf, numerical_method, nlpopts)

    # #plot the results in Figure 2, containing 7 subplots, 4 for the state and 2 for the control and 1 for the deviation, with all subplots sharing same x axis. Ignore the 
    
    # fig2, axs = plt.subplots(7, sharex=True)
    # #plot the pose in first subplot, X_mpc_dc[:,:3] is the x,y,z information
    # axs[0].plot(X_mpc_dc[:,:3])
    # axs[0].set_title('Position [m]')
    # axs[0].legend(['x', 'y', 'z'])
    # #plot the euler angles in second subplot, X_mpc_dc[:,3:6] is the phi, theta, psi information
    # axs[1].plot(X_mpc_dc[:,3:6])
    # axs[1].set_title('Euler angles [rad]')
    # axs[1].legend(['phi', 'theta', 'psi'])
    # #plot the velocity in third subplot, X_mpc_dc[:,6:9] is the vx, vy, vz information
    # axs[2].plot(X_mpc_dc[:,6:9])
    # axs[2].set_title('Velocity [m/s]')
    # axs[2].legend(['vx', 'vy', 'vz'])
    # #plot the angular velocity in fourth subplot, X_mpc_dc[:,9:12] is the phi_dot, theta_dot, psi_dot information
    # axs[3].plot(X_mpc_dc[:,9:12])
    # axs[3].set_title('Angular velocity [rad/s]')
    # axs[3].legend(['phi_dot', 'theta_dot', 'psi_dot'])
    # #plot the control input as a  function in fifth subplot, U_mpc_dc[:,0] is the thrust information
    # axs[4].plot(U_mpc_dc[:,0], '-')
    # axs[4].set_title('Thrust [N]')

    # #plot the control input as a  function in sixth subplot, U_mpc_dc[:,1:4] is the moment information
    # axs[5].plot(U_mpc_dc[:,1:4])
    # axs[5].set_title('Moment [N*m]')
    # axs[5].legend(['Mx', 'My', 'Mz'])
    # #plot the deviation in seventh subplot
    # axs[6].plot(deviation_dc)
    # axs[6].set_title('Euclidean distance from target[m]')
    # axs[6].legend(['x', 'y', 'z'])
    # #open in different figure
    # plot_trajectory(X_mpc_dc)
    # plt.show()

if __name__ == "__main__":
    main()      
