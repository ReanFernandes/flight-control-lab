from casadi import *
import numpy as np
from MPC import *
import matplotlib.pyplot as plt
from plot import *
def main():
    # define the parameters 
    # Multiple Shooting
    N_dms = 5
    T_dms = 0.5
    Tf_dms = 1000
    nlpopts_dms = {'ipopt': {'print_level': 0, 'max_iter':100}, 'print_time' : 0}

    # Direct Collocation
    N_dc = 5
    T_dc = 0.5
    Tf_dc = 1000
    nlpopts_dc = {'ipopt': {'print_level': 0, 'max_iter':200}, 'print_time' : 0}
    degree = 2


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

    R = np.diag([1, 1, 1, 1])* 0.6

    # define the starting point
    x_init = np.array([1, 3, 9])
    x_desired = np.array([8, 5, 1])

    # Thrust controlled MPC 
    function_type = "force_control"

    # Multiple Shooting
    X_mpc_dms_force, U_mpc_dms_force, deviation, step = MPC_multiple_shooting(Q, R, function_type, x_init, x_desired, N_dms, T_dms, Tf_dms, nlpopts_dms)
    plot_state_trajectory(X_mpc_dms_force, U_mpc_dms_force, deviation, step, function_type, 'DMS', x_init, x_desired)
    plot_drone_trajectory(X_mpc_dms_force, function_type, 'DMS', x_init, x_desired,step)

    # Direct Collocation
    X_mpc_dc_force, U_mpc_dc_force, deviation, step = MPC_collocation(degree,Q, R, function_type, x_init, x_desired, N_dc, T_dc, Tf_dc, nlpopts_dc)
    plot_state_trajectory(X_mpc_dc_force, U_mpc_dc_force, deviation, step, function_type, 'DC', x_init, x_desired)
    plot_drone_trajectory(X_mpc_dc_force, function_type, 'DC', x_init, x_desired, step)

    # # RPM controlled MPC
    # function_type = "rpm_control"

    # # Multiple Shooting
    # X_mpc_dms_rpm, U_mpc_dms_rpm, deviation, step = MPC_multiple_shooting(Q, R, function_type, x_init, x_desired, N_dms, T_dms, Tf_dms, nlpopts_dms)
    # plot_state_trajectory(X_mpc_dms_rpm, U_mpc_dms_rpm, deviation, step, function_type, 'DMS', x_init, x_desired)
    # plot_drone_trajectory(X_mpc_dms_rpm, function_type, 'DMS', x_init, x_desired)

    # # Direct Collocation
    # X_mpc_dc_rpm, U_mpc_dc_rpm, deviation, step = MPC_collocation(degree, Q, R, function_type, x_init, x_desired, N_dc, T_dc, Tf_dc, nlpopts_dc)
    # plot_state_trajectory(X_mpc_dc_rpm, U_mpc_dc_rpm, deviation, step, function_type, 'DC', x_init, x_desired)
    # plot_drone_trajectory(X_mpc_dc_rpm, function_type, 'DC', x_init, x_desired)



    
if __name__ == "__main__":
    main()      
