from casadi import *
import numpy as np
from MPC import *
import matplotlib.pyplot as plt
from plot import *
def main():
    # define the parameters 
    # Multiple Shooting
    N_dms = 10
    T_dms = 0.1
    Tf_dms = 10
    nlpopts_dms = {'ipopt': {'print_level': 0, 'max_iter':100}, 'print_time' : 0}

    # Direct Collocation
    N_dc = 10
    T_dc = 0.1
    Tf_dc = 10
    nlpopts_dc = {'ipopt': {'print_level': 0, 'max_iter':100}, 'print_time' : 0}


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

    # Thrust controlled MPC 
    function_type = "force_control"

    # Multiple Shooting
    X_mpc_dms_force, U_mpc_dms_force, deviation, step = MPC_multiple_shooting(Q, R, function_type, x_init, x_desired, N_dms, T_dms, Tf_dms, nlpopts_dms)
    plot_state_trajectory(X_mpc_dms_force, U_mpc_dms_force, deviation, step, function_type, 'DMS')



    
if __name__ == "__main__":
    main()      
