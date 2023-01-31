from casadi import *
import numpy as np
from MPC import *
import matplotlib.pyplot as plt
from plot import *
def main():
    # define the parameters 
    # Multiple Shooting Thrust control
    N_dms_force = 150
    T_dms_force = 0.5
    Tf_dms_force = 0.01
    nlpopts_dms_force = {'ipopt': {'print_level': 0, 'max_iter':500}, 'print_time' : 0}

    # Direct Collocation Thrust control
    N_dc_force = 150
    T_dc_force = 0.5
    Tf_dc_force = 0.01
    nlpopts_dc_force = {'ipopt': {'print_level': 0, 'max_iter':500}, 'print_time' : 0}
    degree = 2



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

    R = np.diag([1, 1, 1, 1])* 10

    # define the starting point
    x_init = np.array([3,4,5])
    x_desired = np.array([1,1,0])

    x0 = np.concatenate((x_init, np.zeros(nx-3)))
    x_ref = np.concatenate((x_desired, np.zeros(nx-3)))

    # Thrust controlled MPC 
    function_type = "force_control"

    # run the open loop solver for dms force control
    X_mpc_dms = np.zeros((N_dms_force+1, nx))
    U_mpc_dms = np.zeros((N_dms_force, nu))
    solver, w0, lbw, ubw, lbg, ubg = NLP_multiple_shooting( Q, R, function_type, N_dms_force, T_dms_force, nlpopts_dms_force)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(x0, x_ref))
    w_opt = sol['x'].full().flatten()
    for i in range(N_dms_force):
        X_mpc_dms[i,:] = w_opt[i*(nx+nu):i*(nx+nu)+nx]
        print("Pose in step " + str(i) + " is: " + str(X_mpc_dms[i,:]))
        U_mpc_dms[i,:] = w_opt[i*(nx+nu)+nx:i*(nx+nu)+nx+nu]
        print("Control in step " + str(i) + " is: " + str(U_mpc_dms[i,:]))
    deviation = []
    for i in range(N_dms_force):
        deviation.append(np.linalg.norm(X_mpc_dms[i,:3] - x_desired))
    plot_drone_trajectory(X_mpc_dms, function_type,"DMS",x_init, x_desired, N_dms_force)
    plot_state_trajectory(X_mpc_dms, U_mpc_dms,deviation,N_dms_force, function_type, "DMS", x_init, x_desired)

    # run the open loop solver for dc force control
    X_mpc_dc = np.zeros((N_dc_force+1, nx))
    U_mpc_dc = np.zeros((N_dc_force, nu))
    solver, w0, lbw, ubw, lbg, ubg, _, _ = NLP_direct_collocation( degree, Q, R, function_type, N_dc_force, T_dc_force, nlpopts_dc_force)
    sol = solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg, p=vertcat(x0, x_ref))
    w_opt = sol['x'].full().flatten()
    for i in range(N_dc_force):
        X_mpc_dc[i,:] = w_opt[(nx + nu + nx * degree) * i:(nx + nu + nx * degree) * i + nx]
        print("Pose in step " + str(i) + " is: " + str(X_mpc_dc[i,:]))
        if i == N_dc_force-1:
            U_mpc_dc 
            break
        U_mpc_dc[i,:] = w_opt[(nx + nu + nx * degree) * (i) + nx : (nx + nu + nx * degree) * (i) + nx + nu]
        print("Control in step " + str(i) + " is: " + str(U_mpc_dc[i,:]))

    deviation = []
    for i in range(N_dc_force):
        deviation.append(np.linalg.norm(X_mpc_dc[i,:3] - x_desired))

    plot_drone_trajectory(X_mpc_dc, function_type,"DC",x_init, x_desired, N_dc_force)
    plot_state_trajectory(X_mpc_dc, U_mpc_dc,deviation,N_dc_force, function_type, "DC", x_init, x_desired)

if __name__ == "__main__":
    main()        