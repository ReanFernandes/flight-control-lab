from casadi import *
import numpy as np
from MPC import *
import matplotlib.pyplot as plt
from plot import *
def main():
    # define the parameters 
    # Multiple Shooting Thrust control
    N_dms_force = 15
    T_dms_force = 0.001
    Tf_dms_force = 15
    nlpopts_dms_force = {'ipopt': {'print_level': 0, 'max_iter':20}, 'print_time' : 0}

    # Direct Collocation Thrust control
    N_dc_force = 5
    T_dc_force = 0.1
    Tf_dc_force = 15
    nlpopts_dc_force = {'ipopt': {'print_level': 0, 'max_iter':50}, 'print_time' : 0}
    degree = 3



    # Multiple Shooting
    N_dms_rpm = 15
    T_dms_rpm = 0.25
    Tf_dms_rpm = 1000
    nlpopts_dms_rpm = {'ipopt': {'print_level': 0, 'max_iter':20}, 'print_time' : 0}

    # Direct Collocation
    N_dc_rpm = 5
    T_dc_rpm = 0.5
    Tf_dc_rpm = 1000
    nlpopts_dc_rpm = {'ipopt': {'print_level': 0, 'max_iter':200}, 'print_time' : 0}
    degree = 2


    Q = np.diag([120,   #x
                120,    #y
                120,    #z
                1e-2,   #phi
                1e-2,   #theta
                1,   #psi
                7e-1,   #vx
                1,      #vy
                4,      #vz
                1e-5,   #phi_dot
                1e-5,   #theta_dot
                10])   #psi_dot

    R = np.diag([1, 1, 1, 1])* 0.06

    # define the starting point
    x_init = np.array([0,0,1])
    x_desired = np.array([1,1,7])

    # Thrust controlled MPC 
    function_type = "force_control"

    # Multiple Shooting
    # X_mpc_dms_force, U_mpc_dms_force, deviation, step = MPC_multiple_shooting(Q, R, function_type, x_init, x_desired, N_dms_force, T_dms_force, Tf_dms_force, nlpopts_dms_force)
    # plot_state_trajectory(X_mpc_dms_force, U_mpc_dms_force, deviation, step, function_type, 'DMS', x_init, x_desired)
    # plot_drone_trajectory(X_mpc_dms_force, function_type, 'DMS', x_init, x_desired,step)

    # # Direct Collocation
    # X_mpc_dc_force, U_mpc_dc_force, deviation, step = MPC_collocation(degree,Q, R, function_type, x_init, x_desired, N_dc_force, T_dc_force, Tf_dc_force, nlpopts_dc_force)
    # plot_state_trajectory(X_mpc_dc_force, U_mpc_dc_force, deviation, step, function_type, 'DC', x_init, x_desired)
    # plot_drone_trajectory(X_mpc_dc_force, function_type, 'DC', x_init, x_desired, step)

    # RPM controlled MPC
    function_type = "rpm_control"

    # # Multiple Shooting
    # X_mpc_dms_rpm, U_mpc_dms_rpm, deviation, step = MPC_multiple_shooting(Q, R, function_type, x_init, x_desired, N_dms_rpm, T_dms_rpm, Tf_dms_rpm, nlpopts_dms_rpm)
    # plot_state_trajectory(X_mpc_dms_rpm, U_mpc_dms_rpm, deviation, step, function_type, 'DMS', x_init, x_desired)
    # plot_drone_trajectory(X_mpc_dms_rpm, function_type, 'DMS', x_init, x_desired,step)

    # # Direct Collocation
    X_mpc_dc_rpm, U_mpc_dc_rpm, deviation, step = MPC_collocation(degree, Q, R, function_type, x_init, x_desired, N_dc_rpm, T_dc_rpm, Tf_dc_rpm, nlpopts_dc_rpm)
    plot_state_trajectory(X_mpc_dc_rpm, U_mpc_dc_rpm, deviation, step, function_type, 'DC', x_init, x_desired)
    plot_drone_trajectory(X_mpc_dc_rpm, function_type, 'DC', x_init, x_desired, step)



    
if __name__ == "__main__":
    main()      
